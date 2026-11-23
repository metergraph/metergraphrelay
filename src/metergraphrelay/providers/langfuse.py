from __future__ import annotations

import base64
import email.utils
import hashlib
import json
import math
import os
import re
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from typing import Any, Callable, Iterable

from ..reliability import reliability_fields
from .. import __version__
from .cache_shape import CACHE_EXCLUSIVE_INPUT_PROVIDERS
from ..http_limits import ResponseTooLarge, read_bounded
from ..billing_evidence import reported_cost
from ..capture_contract import capture_response_text, capture_row
from ..finish_reason import extract_stop_reason
from ..import_identity import ImportContext, canonical_import_event_id

DEFAULT_LANGFUSE_HOST = "https://cloud.langfuse.com"
OBSERVATIONS_PATH = "/api/public/v2/observations"
PAGE_LIMIT = 1000
MAX_PAGES_PER_PULL = 10_000
MAX_ITEMS_PER_PULL = 100_000
GENERATION_TYPE = "GENERATION"
# Langfuse's "generation-like" observation types: the ones its ingestion lets
# carry a model, usage and a cost (langfuse packages/shared/src/domain/
# observations.ts, GenerationLikeObservationTypes, MIT). SPAN and EVENT never
# do, so they are not requested at all.
COST_BEARING_TYPES: tuple[str, ...] = (
    "GENERATION",
    "EMBEDDING",
    "AGENT",
    "TOOL",
    "CHAIN",
    "RETRIEVER",
    "EVALUATOR",
    "GUARDRAIL",
)
# Field groups of the v2 Observations API. core+basic+time cover
# id/type/name/traceId/startTime/endTime/completionStartTime/level/
# statusMessage/parentObservationId/sessionId/userId/version; io covers
# input/output; usage covers usageDetails, totalCost and costDetails; model
# covers the model name and modelParameters; metadata covers the explicit
# provider hints used for provider resolution; prompt covers promptName and
# promptVersion; metrics covers latency and timeToFirstToken; trace_context
# denormalizes traceName/tags/environment/release onto each observation.
# Requesting all of them up front avoids silently missing a field the
# normalize step depends on.
RESPONSE_FIELDS = "core,basic,time,io,usage,model,metadata,prompt,metrics,trace_context"

# Rate limits and gateway errors are a matter of when, not if, on a large
# pull. The policy matches the hosted importer's: honour Retry-After when
# Langfuse sends one, else double from two seconds, never wait more than two
# minutes at a time, and give up after six retries.
RETRY_STATUSES = frozenset({429, 502, 503, 504})
MAX_RETRIES = 6
RETRY_BASE_SECONDS = 2.0
MAX_RETRY_SECONDS = 120.0
REQUEST_TIMEOUT_SECONDS = 10


def _selector_conditions(
    trace_names: list[str],
    tags: list[str],
    *,
    user_id: str | None = None,
    session_id: str | None = None,
    metadata: Iterable[tuple[str, str]] = (),
) -> list[dict[str, Any]]:
    conditions: list[dict[str, Any]] = []
    if trace_names:
        conditions.append(
            {
                "type": "stringOptions",
                "column": "traceName",
                "operator": "any of",
                "value": list(trace_names),
            }
        )
    if tags:
        conditions.append(
            {
                "type": "arrayOptions",
                "column": "tags",
                "operator": "all of",
                "value": list(tags),
            }
        )
    if user_id:
        conditions.append(
            {"type": "string", "column": "userId", "operator": "=", "value": user_id}
        )
    if session_id:
        conditions.append(
            {"type": "string", "column": "sessionId", "operator": "=", "value": session_id}
        )
    for key, value in metadata:
        conditions.append(
            {
                "type": "stringObject",
                "column": "metadata",
                "key": key,
                "operator": "=",
                "value": value,
            }
        )
    return conditions


def _encode_filter(conditions: list[dict[str, Any]]) -> str:
    return json.dumps(conditions)


def build_filter(trace_names: list[str], tags: list[str]) -> str | None:
    conditions = _selector_conditions(trace_names, tags)
    if not conditions:
        return None
    return _encode_filter(conditions)


def build_base_params(
    *,
    until: str,
    since: str | None,
    trace_names: list[str],
    tags: list[str],
    environment: str | None,
    observation_types: Iterable[str] = (GENERATION_TYPE,),
    user_id: str | None = None,
    session_id: str | None = None,
    metadata: Iterable[tuple[str, str]] = (),
) -> dict[str, str]:
    """Query parameters for one pull, before the page limit and cursor.

    ``observation_types`` defaults to GENERATION alone, which is what callers
    written before other types were importable expect; pass
    ``COST_BEARING_TYPES`` (and keep only rows that pass
    ``is_importable_observation``) to import every observation that can carry
    a cost.
    """
    types = list(observation_types)
    if not types:
        raise ValueError("observation_types must name at least one type")
    params: dict[str, str] = {"fields": RESPONSE_FIELDS}
    conditions = _selector_conditions(
        trace_names, tags, user_id=user_id, session_id=session_id, metadata=metadata
    )
    if conditions or len(types) > 1:
        # Langfuse's structured `filter` param takes precedence over the
        # simpler type/environment/fromStartTime/toStartTime query params:
        # whenever a selector is present, or more than one type is wanted
        # (the simple `type` param takes one), those constraints must be
        # folded into this same filter array, or they are silently ignored.
        conditions.append(
            {
                "type": "stringOptions",
                "column": "type",
                "operator": "any of",
                "value": types,
            }
        )
        if since:
            conditions.append(
                {
                    "type": "datetime",
                    "column": "startTime",
                    "operator": ">=",
                    "value": since,
                }
            )
        conditions.append(
            {
                "type": "datetime",
                "column": "startTime",
                "operator": "<",
                "value": until,
            }
        )
        if environment:
            conditions.append(
                {
                    "type": "stringOptions",
                    "column": "environment",
                    "operator": "any of",
                    "value": [environment],
                }
            )
        params["filter"] = _encode_filter(conditions)
    else:
        params["type"] = types[0]
        params["toStartTime"] = until
        if since:
            params["fromStartTime"] = since
        if environment:
            params["environment"] = environment
    return params


class LangfuseAPIError(Exception):
    """Raised when Langfuse's API returns an error response or an unusable body."""


def _auth_header(public_key: str, secret_key: str) -> str:
    token = base64.b64encode(f"{public_key}:{secret_key}".encode()).decode()
    return f"Basic {token}"


def _retry_after_seconds(value: str | None, now: Callable[[], float]) -> float | None:
    """Seconds named by a Retry-After header: delta-seconds or an HTTP date."""
    if not value:
        return None
    value = value.strip()
    if value.isascii() and value.isdecimal():
        return float(value)
    try:
        parsed = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError, OverflowError, IndexError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return max(0.0, parsed.timestamp() - now())


def retry_delay(
    retry_after: str | None, attempt: int, *, now: Callable[[], float] = time.time
) -> float:
    """Seconds to wait before retry ``attempt`` (0-based): Retry-After, else doubling."""
    seconds = _retry_after_seconds(retry_after, now)
    if seconds is None:
        seconds = RETRY_BASE_SECONDS * (2 ** attempt)
    return min(seconds, MAX_RETRY_SECONDS)


def fetch_observations_page(
    base_url: str,
    *,
    public_key: str,
    secret_key: str,
    params: dict[str, str],
    sleep: Callable[[float], object] | None = time.sleep,
) -> dict[str, Any]:
    """One page of observations.

    A rate limit or gateway error (``RETRY_STATUSES``) is retried up to
    ``MAX_RETRIES`` times after ``retry_delay``; ``sleep=None`` disables the
    retry. Any other HTTP error fails at once.
    """
    query = urllib.parse.urlencode(params)
    url = f"{base_url.rstrip('/')}{OBSERVATIONS_PATH}"
    if query:
        url = f"{url}?{query}"
    request = urllib.request.Request(
        url,
        headers={
            "Authorization": _auth_header(public_key, secret_key),
            "Accept": "application/json",
        },
        method="GET",
    )
    attempt = 0
    while True:
        try:
            with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
                body = read_bounded(response)
            break
        except urllib.error.HTTPError as exc:
            if sleep is not None and exc.code in RETRY_STATUSES and attempt < MAX_RETRIES:
                headers = exc.headers
                retry_after = headers.get("Retry-After") if headers is not None else None
                delay = retry_delay(retry_after, attempt)
                exc.close()
                print(
                    f"Langfuse answered HTTP {exc.code}; retry {attempt + 1} of "
                    f"{MAX_RETRIES} in {delay:.0f} s",
                    file=sys.stderr,
                )
                sleep(delay)
                attempt += 1
                continue
            raise LangfuseAPIError(
                f"Langfuse API request failed: HTTP {exc.code} {exc.reason}"
            ) from exc
        except urllib.error.URLError as exc:
            raise LangfuseAPIError(f"Langfuse API request failed: {exc.reason}") from exc
        except ResponseTooLarge as exc:
            raise LangfuseAPIError(str(exc)) from exc
    try:
        payload = json.loads(body)
    except (json.JSONDecodeError, RecursionError) as exc:
        raise LangfuseAPIError(f"Langfuse API returned invalid JSON: {exc}") from exc
    data = payload.get("data") if isinstance(payload, dict) else None
    meta = payload.get("meta") if isinstance(payload, dict) else None
    if not isinstance(data, list) or not isinstance(meta, dict):
        raise LangfuseAPIError(
            "Langfuse API response missing/malformed 'data'/'meta': unsupported "
            "deployment or unexpected response shape (self-hosted v4+ with the v2 "
            "Observations API is required)"
        )
    return payload


# ---------------------------------------------------------------------------
# Provider resolution
#
# metergraph prices a call from its catalog, keyed on (provider, model
# spelling). Langfuse records whatever model string the caller's integration
# passed, so the relay resolves the provider, and for a gateway-style name the
# bare model, before the row leaves. Order, most to least specific:
#
#   1. an explicit provider in the observation's metadata (``provider``,
#      LangChain's ``ls_provider``, OTel ``gen_ai.system`` /
#      ``gen_ai.provider.name``), folded through ``_PROVIDER_SPELLINGS``;
#   2. the model string's own shape: a Bedrock ARN or model id
#      (``us.anthropic.claude-...``), a Vertex resource path
#      (``publishers/google/models/gemini-...``) or Anthropic-on-Vertex
#      version (``claude-...@20241022``), a gateway prefix
#      (``openai/gpt-4o``, LiteLLM and OpenRouter style);
#   3. a model family prefix (``gpt-``, ``o4-``, ``claude-``, ``mistral-``...).
#
# Anything else is ``unknown``, which the catalog resolves by model alone when
# the spelling is unambiguous. Approaches cross-checked against LiteLLM's
# get_llm_provider and its Bedrock cross-region prefix list
# (litellm/llms/bedrock/common_utils.py, MIT).
# ---------------------------------------------------------------------------

# Provider spellings seen in Langfuse metadata (LangChain ls_provider, OTel
# gen_ai.system, LiteLLM custom_llm_provider) folded onto metergraph's names.
# A spelling not listed passes through lowercased.
_PROVIDER_SPELLINGS: dict[str, str] = {
    "openai": "openai",
    "azure": "azure",
    "azure_openai": "azure",
    "azure-openai": "azure",
    "azure_ai": "azure",
    "az.ai.openai": "azure",
    "az.ai.inference": "azure",
    "azure.ai.openai": "azure",
    "azure.ai.inference": "azure",
    "anthropic": "anthropic",
    "bedrock": "bedrock",
    "bedrock_converse": "bedrock",
    "amazon_bedrock": "bedrock",
    "amazon-bedrock": "bedrock",
    "aws_bedrock": "bedrock",
    "aws-bedrock": "bedrock",
    "aws.bedrock": "bedrock",
    "aws": "bedrock",
    "google": "google",
    "gemini": "google",
    "google_genai": "google",
    "google-genai": "google",
    "google_ai_studio": "google",
    "gcp.gemini": "google",
    "gcp.gen_ai": "google",
    "vertex": "vertex-ai",
    "vertexai": "vertex-ai",
    "vertex_ai": "vertex-ai",
    "vertex-ai": "vertex-ai",
    "google_vertexai": "vertex-ai",
    "google_anthropic_vertex": "vertex-ai",
    "vertex_ai_beta": "vertex-ai",
    "gcp.vertex_ai": "vertex-ai",
    "mistral": "mistral",
    "mistralai": "mistral",
    "mistral_ai": "mistral",
    "meta": "meta",
    "meta-llama": "meta",
    "meta_llama": "meta",
    "cohere": "cohere",
    "cohere_chat": "cohere",
    "deepseek": "deepseek",
    "xai": "x-ai",
    "x_ai": "x-ai",
    "x-ai": "x-ai",
    "grok": "x-ai",
    "perplexity": "perplexity-ai",
    "perplexity-ai": "perplexity-ai",
    "groq": "groq",
    "together": "together",
    "together_ai": "together",
    "togetherai": "together",
    "fireworks": "fireworks",
    "fireworks_ai": "fireworks",
    "moonshot": "moonshotai",
    "moonshotai": "moonshotai",
    "alibaba": "alibaba",
    "dashscope": "alibaba",
    "qwen": "alibaba",
    "openrouter": "openrouter",
    "ollama": "ollama",
}
# Spellings that say nothing about who served the call. Resolution goes on to
# the model name instead.
_UNINFORMATIVE_PROVIDERS = frozenset(
    {"", "unknown", "none", "null", "litellm", "proxy", "langfuse", "custom"}
)
# Metadata keys that carry a provider, most explicit first. Langfuse v4
# flattens nested metadata to dotted keys; older deployments keep the nesting,
# which _metadata_value also reads.
_PROVIDER_METADATA_KEYS = (
    "provider",
    "ls_provider",
    "gen_ai.provider.name",
    "gen_ai.system",
    "custom_llm_provider",
)
# Bedrock cross-region inference prefixes and the model publishers Bedrock
# namespaces ids under (``us.anthropic.claude-...``, ``meta.llama3-...``).
_BEDROCK_MODEL_ID = re.compile(
    r"^(?:(?:global|us|us-gov|eu|apac|jp|au|ca|sa)\.)?"
    r"(?:anthropic|amazon|meta|mistral|cohere|ai21|deepseek|qwen|openai|writer|"
    r"twelvelabs|stability|moonshotai|google|minimax|nvidia|luma)\.[a-z0-9]"
)
_VERTEX_RESOURCE = re.compile(r"(?:^|/)publishers/[^/]+/models/([^/]+)$")
# Anthropic models on Vertex are versioned with "@" (claude-3-5-sonnet-v2@20241022).
_VERTEX_ANTHROPIC = re.compile(r"^claude-[^@]+@\d{8}$")
# Gateway prefixes whose remainder is a provider's own model name. Only these
# are stripped: a prefix not listed (an org or a fine-tune path) stays part of
# the model.
_GATEWAY_PREFIXES: dict[str, str] = {
    "openai": "openai",
    "azure": "azure",
    "anthropic": "anthropic",
    "bedrock": "bedrock",
    "google": "google",
    "gemini": "google",
    "vertex_ai": "vertex-ai",
    "vertex-ai": "vertex-ai",
    "mistral": "mistral",
    "mistralai": "mistral",
    "meta-llama": "meta",
    "cohere": "cohere",
    "deepseek": "deepseek",
    "x-ai": "x-ai",
    "xai": "x-ai",
    "groq": "groq",
    "together_ai": "together",
    "fireworks_ai": "fireworks",
    "moonshotai": "moonshotai",
    "qwen": "alibaba",
}
_PROVIDER_MODEL_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = tuple(
    (re.compile(pattern), provider)
    for pattern, provider in (
        (r"^(?:gpt-|chatgpt-|o\d+(?:-|$)|text-embedding-(?:3|ada)|davinci-|babbage-|"
         r"omni-moderation|text-moderation|whisper-|tts-|dall-e|computer-use-|codex-)", "openai"),
        (r"^claude-", "anthropic"),
        (r"^(?:gemini-|gemma-|text-embedding-00|text-multilingual-embedding-|"
         r"imagen-|veo-)", "google"),
        (r"^(?:mistral-|mixtral-|codestral-|ministral-|pixtral-|magistral-|"
         r"devstral-|open-mistral-|open-mixtral-|voxtral-)", "mistral"),
        (r"^(?:llama-?\d|llama-guard|meta-llama-)", "meta"),
        (r"^(?:command-|command$|embed-english-|embed-multilingual-|rerank-)", "cohere"),
        (r"^deepseek-", "deepseek"),
        (r"^grok-", "x-ai"),
        (r"^(?:qwen|qwq-)", "alibaba"),
        (r"^(?:kimi-|moonshot-)", "moonshotai"),
    )
)


def _resolve_model_name(observation: dict[str, Any]) -> str | None:
    # Live v4 Observations API evidence: model comes back on `model`
    # (e.g. "gpt-4o-mini", "claude-3-5-haiku-latest"), with `providedModelName`
    # observed as None in practice. `providedModelName` is kept as a fallback
    # for robustness, not as the primary source.
    for key in ("model", "providedModelName"):
        value = observation.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _metadata_value(metadata: dict[str, Any], dotted_key: str) -> Any:
    if dotted_key in metadata:
        return metadata[dotted_key]
    node: Any = metadata
    for part in dotted_key.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def _fold_provider(spelling: str) -> str:
    key = spelling.strip().lower()
    return _PROVIDER_SPELLINGS.get(key, key)


def _explicit_provider(observation: dict[str, Any]) -> str | None:
    metadata = observation.get("metadata")
    if not isinstance(metadata, dict):
        return None
    for key in _PROVIDER_METADATA_KEYS:
        value = _metadata_value(metadata, key)
        if not isinstance(value, str):
            continue
        folded = _fold_provider(value)
        if folded not in _UNINFORMATIVE_PROVIDERS:
            return folded
    return None


def _provider_from_model(model: str) -> tuple[str, str]:
    """(provider, model spelling to price) from the model string alone."""
    lowered = model.strip().lower()
    if lowered.startswith("arn:aws:bedrock:"):
        return "bedrock", model
    if _BEDROCK_MODEL_ID.match(lowered):
        return "bedrock", model
    vertex = _VERTEX_RESOURCE.search(model)
    if vertex:
        return "vertex-ai", vertex.group(1)
    if _VERTEX_ANTHROPIC.match(lowered):
        return "vertex-ai", model
    prefix, separator, rest = model.partition("/")
    if separator and rest:
        gateway_provider = _GATEWAY_PREFIXES.get(prefix.strip().lower())
        if gateway_provider is not None:
            if gateway_provider == "bedrock":
                return "bedrock", rest
            inner_provider, inner_model = _provider_from_model(rest)
            if gateway_provider == "vertex-ai" or inner_provider == "unknown":
                return gateway_provider, inner_model
            # "openai/gpt-4o": the prefix and the family agree, or the
            # family is more specific (a Bedrock id behind a prefix).
            return (inner_provider if inner_provider == "bedrock" else gateway_provider), inner_model
    for pattern, provider in _PROVIDER_MODEL_PATTERNS:
        if pattern.match(lowered):
            return provider, model
    return "unknown", model


def resolve_provider_and_model(observation: dict[str, Any]) -> tuple[str, str | None]:
    """The provider and the model spelling a row should carry.

    The model differs from Langfuse's only when a gateway prefix or a Vertex
    resource path is stripped; the original is kept on the row as
    ``langfuse_model``.
    """
    model = _resolve_model_name(observation)
    explicit = _explicit_provider(observation)
    if model is None:
        return explicit or "unknown", None
    inferred, priced_model = _provider_from_model(model)
    if explicit is None:
        return inferred, priced_model
    if explicit in {"openrouter", "vercel", "gateway"}:
        # A router names itself; the creator prefix in the model is the
        # catalog's to resolve, so the model stays as recorded.
        return explicit, model
    if inferred == explicit or inferred == "unknown":
        return explicit, priced_model
    if explicit in {"bedrock", "vertex-ai", "azure"}:
        # A cloud that serves another publisher's models: the cloud is the
        # channel that billed the call, the model name the publisher's.
        return explicit, priced_model
    return explicit, model


def infer_provider(observation: dict[str, Any]) -> str:
    return resolve_provider_and_model(observation)[0]


def _is_chat_message_list(value: Any) -> bool:
    return isinstance(value, list) and bool(value) and all(
        isinstance(item, dict) and "role" in item and "content" in item
        for item in value
    )


_JSON_PARSE_FAILED = object()


def _try_parse_json(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return _JSON_PARSE_FAILED


def _map_content(input_value: Any) -> tuple[str | None, str | None]:
    if input_value is None:
        return None, None
    if _is_chat_message_list(input_value):
        return json.dumps(input_value), None
    if isinstance(input_value, str):
        parsed = _try_parse_json(input_value)
        if _is_chat_message_list(parsed):
            return json.dumps(parsed), None
        return None, input_value
    return None, json.dumps(input_value)


def _response_text(output_value: Any) -> str | None:
    # No response_json field exists in metergraph's native row — JSON-shaped
    # output is serialized into this single field instead of split like input.
    if output_value is None:
        return None
    if isinstance(output_value, str):
        return output_value
    return json.dumps(output_value)


# Langfuse stores usageDetails FLAT: its ingestion normalizes provider usage
# into one level of integer buckets and drops nested objects. Which keys appear
# depends on the integration that recorded the observation:
#
#   openai wrapper   -- ingestion renames prompt/completion to "input"/"output"
#                       and flattens details as input_cached_tokens /
#                       output_reasoning_tokens, each SUBTRACTED from its parent.
#   langchain        -- the callback handler flattens usage_metadata details as
#                       input_cache_read / input_cache_creation /
#                       output_reasoning, likewise subtracted.
#   gateway fallthrough -- a response the wrapper schema rejects keeps the raw
#                       prompt_tokens / completion_tokens, inclusive totals with
#                       the details dropped.
#   native SDK       -- whatever the caller passed to update(usage_details=...),
#                       e.g. Anthropic's input_tokens + cache_read_input_tokens.
#   gemini           -- promptTokenCount / candidatesTokenCount / thoughts_token_count.
#
# Langfuse's own contract is that every key is a non-overlapping bucket and the
# total is their sum. metergraph's convention is that input_tokens is the total
# with cache_read / cache_write / reasoning as subsets of it, so the named total
# key has every other *input* bucket added back (Langfuse itself sums by the
# "input" substring). Priority-tier buckets are pricing markers that overlap
# the parent, not exclusive sub-buckets, so they are left out of the sum.
_INPUT_TOTAL_KEYS = (
    "input", "input_tokens", "prompt_tokens", "promptTokenCount", "prompt_token_count",
)
_OUTPUT_TOTAL_KEYS = (
    "output", "output_tokens", "completion_tokens", "candidatesTokenCount",
    "candidates_token_count",
)
_CACHE_READ_KEYS = (
    "input_cached_tokens", "input_cache_read", "cache_read_input_tokens",
    "cachedContentTokenCount", "cached_content_token_count",
)
_CACHE_WRITE_KEYS = (
    "input_cache_creation_tokens", "input_cache_creation", "cache_creation_input_tokens",
)
_REASONING_KEYS = (
    "output_reasoning_tokens", "output_reasoning", "thoughtsTokenCount",
    "thoughts_token_count",
)
_NON_BUCKET_MARKERS = ("priority",)

# Providers whose own usage reports input *excluding* the cached tokens, which
# it lists separately. metergraph prices cache handling per publisher, so a row
# has to arrive in the shape that publisher's API uses: for one of these,
# folding the cache buckets into input makes those tokens part of the input
# total as well as a bucket of their own, and they are billed twice -- once at
# the input rate and once at the cache rate, an order of magnitude apart.
#
# For OpenAI and Google the opposite is true: their own totals include the
# cached tokens, so the buckets Langfuse subtracted do have to be added back.
_CACHE_EXCLUSIVE_INPUT_PROVIDERS = CACHE_EXCLUSIVE_INPUT_PROVIDERS


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _first_int_key(source: dict[str, Any], keys: tuple[str, ...]) -> tuple[str | None, int | None]:
    for key in keys:
        value = source.get(key)
        if _is_int(value):
            return key, value
    return None, None


def _first_int(source: dict[str, Any], keys: tuple[str, ...]) -> int | None:
    return _first_int_key(source, keys)[1]


def _total_with_buckets(
    usage_details: dict[str, Any],
    total_keys: tuple[str, ...],
    marker: str,
    *,
    skip_keys: tuple[str, ...] = (),
) -> int | None:
    """The named total plus every other integer bucket carrying ``marker``.

    ``skip_keys`` names buckets the total already excludes by the provider's own
    convention, so adding them would count those tokens twice.
    """
    total_key, total = _first_int_key(usage_details, total_keys)
    if total_key is None:
        return None
    skipped = {key.lower() for key in skip_keys}
    for key, value in usage_details.items():
        if key == total_key or not _is_int(value):
            continue
        lowered = key.lower()
        if marker not in lowered or lowered in skipped:
            continue
        if any(skip in lowered for skip in _NON_BUCKET_MARKERS):
            continue
        total += value
    return total


def map_usage_details(
    usage_details: Any, *, provider: str | None = None
) -> dict[str, int | None]:
    """Map one observation's usageDetails onto metergraph token fields.

    ``provider`` decides whether the cache buckets belong inside the input
    total, because that is a property of the provider's own usage shape rather
    than of Langfuse's.
    """
    if not isinstance(usage_details, dict):
        return {
            "input_tokens": None,
            "output_tokens": None,
            "cache_read_tokens": None,
            "cache_write_tokens": None,
            "reasoning_tokens": None,
        }
    cache_exclusive = (
        str(provider or "").strip().lower() in _CACHE_EXCLUSIVE_INPUT_PROVIDERS
    )
    return {
        "input_tokens": _total_with_buckets(
            usage_details,
            _INPUT_TOTAL_KEYS,
            "input",
            skip_keys=(_CACHE_READ_KEYS + _CACHE_WRITE_KEYS) if cache_exclusive else (),
        ),
        "output_tokens": _total_with_buckets(usage_details, _OUTPUT_TOTAL_KEYS, "output"),
        "cache_read_tokens": _first_int(usage_details, _CACHE_READ_KEYS),
        "cache_write_tokens": _first_int(usage_details, _CACHE_WRITE_KEYS),
        "reasoning_tokens": _first_int(usage_details, _REASONING_KEYS),
    }


def _positive_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value > 0
    )


def is_importable_observation(observation: Any) -> bool:
    """Whether an observation is a call metergraph should record.

    A GENERATION always is, as it was before other types were importable. Any
    other type (EMBEDDING, and the v4 AGENT/TOOL/CHAIN/RETRIEVER/EVALUATOR/
    GUARDRAIL types) is when it names a model and carries usage or a cost:
    those are the ones Langfuse itself prices. Structural spans without a
    model are left out.
    """
    if not isinstance(observation, dict):
        return False
    kind = observation.get("type")
    if kind == GENERATION_TYPE:
        return True
    if kind not in COST_BEARING_TYPES or _resolve_model_name(observation) is None:
        return False
    usage = observation.get("usageDetails")
    # Usage units need not be whole tokens (Langfuse allows audio seconds and
    # other fractional units), so any positive finite number counts.
    has_usage = isinstance(usage, dict) and any(
        _positive_number(value) for value in usage.values()
    )
    return has_usage or _positive_number(observation.get("totalCost"))


def _parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _seconds_to_ms(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(value) or value < 0:
        return None
    return int(round(value * 1000))


def _duration_ms(observation: dict[str, Any], seconds_key: str, start_key: str,
                 end_key: str) -> int | None:
    """Langfuse's own figure in seconds, else the gap between two timestamps."""
    reported = _seconds_to_ms(observation.get(seconds_key))
    if reported is not None:
        return reported
    start = _parse_timestamp(observation.get(start_key))
    end = _parse_timestamp(observation.get(end_key))
    if start is None or end is None or end < start:
        return None
    return int(round((end - start).total_seconds() * 1000))


def _nonempty_string(value: Any) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


# metergraph keeps a call's customer metadata in ``tags``: a flat map of string
# keys to string values, at most 32 pairs, keys up to 200 and values up to
# 1000 characters. Anything outside that is dropped by the server pair by pair,
# so the relay shapes the map to fit and decides what is kept first.
MAX_TAG_PAIRS = 32
MAX_TAG_KEY_LENGTH = 200
MAX_TAG_VALUE_LENGTH = 1000
LANGFUSE_TAG_PREFIX = "langfuse.tag."
LANGFUSE_TAG_VALUE = "true"
LANGFUSE_METADATA_PREFIX = "langfuse.metadata."
# The same key names the metergraph SDK's Langfuse exporter writes, so a call
# lands identically whichever path brought it in.
# Metadata the Langfuse and LangChain SDKs write themselves. It describes the
# instrumentation, not the call, and would crowd out the customer's own keys.
_SDK_METADATA_PREFIXES = ("resourceAttributes.", "scope.", "lc_versions.", "ls_")
_SDK_METADATA_KEYS = frozenset({"is_langchain_root", "resourceAttributes", "scope",
                                "lc_versions"})


def _metadata_tag_value(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return json.dumps(value) if math.isfinite(value) else None
    try:
        return json.dumps(value, separators=(",", ":"), sort_keys=True)
    except (TypeError, ValueError):
        return None


def _flatten_metadata(metadata: dict[str, Any], prefix: str = "") -> Iterable[tuple[str, Any]]:
    for key, value in metadata.items():
        if not isinstance(key, str):
            continue
        dotted = f"{prefix}{key}"
        if isinstance(value, dict) and value:
            yield from _flatten_metadata(value, f"{dotted}.")
        else:
            yield dotted, value


def _customer_metadata(metadata: Any) -> dict[str, str]:
    """Langfuse metadata as metergraph customer metadata (string to string)."""
    if not isinstance(metadata, dict):
        return {}
    kept: dict[str, str] = {}
    for key, value in _flatten_metadata(metadata):
        if key in _SDK_METADATA_KEYS or key.startswith(_SDK_METADATA_PREFIXES):
            continue
        text = _metadata_tag_value(value)
        if text is None or not key or len(key) > MAX_TAG_KEY_LENGTH:
            continue
        if len(text) > MAX_TAG_VALUE_LENGTH:
            continue
        kept[key] = text
    return kept


def build_tags(observation: dict[str, Any], *, name: str | None = None) -> dict[str, str]:
    """The row's ``tags`` map, in priority order, at most 32 pairs.

    1. Each Langfuse tag as its own pair, ``langfuse.tag.<tag>: "true"``, so a
       metadata filter or group-by on one tag works.
    2. ``name``, the trace name, when ``--route`` replaced it as the route.
    3. ``langfuse.release``, ``langfuse.version``, ``langfuse.prompt.name``
       and ``langfuse.prompt.version``.
    4. The customer's metadata as ``langfuse.metadata.<key>``, in key order.
    """
    tags: dict[str, str] = {}
    langfuse_tags = observation.get("tags")
    if isinstance(langfuse_tags, list):
        for tag in langfuse_tags:
            if not isinstance(tag, str) or not tag:
                continue
            key = f"{LANGFUSE_TAG_PREFIX}{tag}"
            if len(key) <= MAX_TAG_KEY_LENGTH:
                tags[key] = LANGFUSE_TAG_VALUE
    if name:
        tags["name"] = name
    prompt_version = observation.get("promptVersion")
    for key, value in (
        ("langfuse.release", _nonempty_string(observation.get("release"))),
        ("langfuse.version", _nonempty_string(observation.get("version"))),
        ("langfuse.prompt.name", _nonempty_string(observation.get("promptName"))),
        ("langfuse.prompt.version", str(prompt_version) if _is_int(prompt_version) else None),
    ):
        if value is not None and len(value) <= MAX_TAG_VALUE_LENGTH:
            tags[key] = value
    for key, value in sorted(_customer_metadata(observation.get("metadata")).items()):
        prefixed = f"{LANGFUSE_METADATA_PREFIX}{key}"
        if len(prefixed) <= MAX_TAG_KEY_LENGTH:
            tags.setdefault(prefixed, value)
    if len(tags) > MAX_TAG_PAIRS:
        tags = dict(list(tags.items())[:MAX_TAG_PAIRS])
    return tags


def _cost_details(value: Any) -> dict[str, str] | None:
    """Langfuse's per-bucket cost (``costDetails``) as decimal strings."""
    if not isinstance(value, dict):
        return None
    details: dict[str, str] = {}
    for key, amount in value.items():
        reported = reported_cost(amount, source="langfuse")
        if isinstance(key, str) and reported:
            details[key] = reported["reported_cost_usd"]
    return details or None


def _model_parameters(value: Any) -> dict[str, Any] | None:
    return value if isinstance(value, dict) and value else None


def normalize_observation(
    observation: dict[str, Any],
    *,
    route_override: str | None,
    import_context: ImportContext | None = None,
) -> dict:
    trace_name = observation.get("traceName") or None
    own_name = observation.get("name") or None
    name_fallback = trace_name or own_name

    if route_override:
        route = route_override
        name_consumed = False
    else:
        route = name_fallback or ""
        name_consumed = True

    tags = build_tags(
        observation, name=name_fallback if not name_consumed else None
    )

    error = observation.get("level") == "ERROR"
    raw_status_message = observation.get("statusMessage")
    error_type = (
        raw_status_message if error and isinstance(raw_status_message, str) else None
    )

    provider, model = resolve_provider_and_model(observation)
    langfuse_model = _resolve_model_name(observation)
    usage = map_usage_details(observation.get("usageDetails"), provider=provider)
    request_json, request_text = _map_content(observation.get("input"))
    response_text = _response_text(observation.get("output"))

    stop_reason = extract_stop_reason(observation)
    prompt_version = observation.get("promptVersion")

    row = {
        "ts": observation["startTime"],
        "source": "langfuse",
        "sdk": "metergraphrelay",
        "sdk_version": __version__,
        "provider": provider,
        "model": model,
        "status": "error" if error else "success",
        "input_tokens": usage["input_tokens"],
        "output_tokens": usage["output_tokens"],
        "cache_read_tokens": usage["cache_read_tokens"],
        "cache_write_tokens": usage["cache_write_tokens"],
        "reasoning_tokens": usage["reasoning_tokens"],
        "cost_usd": observation.get("totalCost"),
        **reported_cost(observation.get("totalCost"), source="langfuse.observation.totalCost"),
        "error": error,
        "error_type": error_type,
        "request_id": observation["id"],
        "tags": tags,
        "route": route,
        "content_opted_in": True,
        "request_json": request_json,
        "request_text": request_text,
        "response_text": capture_response_text(response_text),
        "trace_id": observation["traceId"],
        "span_id": observation["id"],
        "parent_span_id": observation.get("parentObservationId"),
        "session_id": observation.get("sessionId"),
        "environment": observation.get("environment"),
    }
    # Attribution and timing Langfuse reports for this observation. Each is
    # left off the row when Langfuse has no value, as an absent field and a
    # null mean the same thing to the ingest API.
    optional = {
        "trace_name": trace_name,
        "observation_type": observation.get("type"),
        "latency_ms": _duration_ms(observation, "latency", "startTime", "endTime"),
        "ttft_ms": _duration_ms(
            observation, "timeToFirstToken", "startTime", "completionStartTime"
        ),
        "user_id": _nonempty_string(observation.get("userId")),
        "release": _nonempty_string(observation.get("release")),
        "version": _nonempty_string(observation.get("version")),
        "prompt_name": _nonempty_string(observation.get("promptName")),
        "prompt_version": prompt_version if _is_int(prompt_version) else None,
        "model_parameters": _model_parameters(observation.get("modelParameters")),
    }
    row.update({key: value for key, value in optional.items() if value is not None})
    if langfuse_model is not None and langfuse_model != model:
        row["langfuse_model"] = langfuse_model
    cost_details = _cost_details(observation.get("costDetails"))
    if cost_details is not None and "reported_cost_usd" in row:
        row["reported_cost_details"] = cost_details
    if stop_reason is not None:
        row["stop_reason"] = stop_reason
    if import_context is not None:
        # The observation id is Langfuse's stable identity for this generation,
        # so an overlap re-pull deduplicates on the server instead of double
        # counting.
        row["import_source"] = import_context.source
        row["import_source_scope"] = import_context.source_scope
        row["import_event_id"] = canonical_import_event_id(observation.get("id"))
    row.update(reliability_fields(provider=provider, response=observation.get("output"),
        error=None, http_status_code=None, finish_reason=stop_reason))
    return capture_row(row)


# The fields that hold provider content on a row. A content-blind row has none
# of them and says so with content_opted_in = False.
CONTENT_FIELDS = ("request_json", "request_text", "response_text", "tool_calls")


def strip_content(row: dict) -> dict:
    for field in CONTENT_FIELDS:
        row.pop(field, None)
    row["content_opted_in"] = False
    return row


def default_import_scope(base_url: str, public_key: str) -> str:
    """A stable ``import_source_scope`` for a ``pull`` of one Langfuse project.

    A Langfuse API key pair belongs to exactly one project, so the host plus
    the public key names the project. Only a digest of them goes on the row:
    the scope is an identity for deduplication, not a place to carry a key.
    Re-pulling the same project gives the same scope, so the server's
    (source, scope, observation id) claim drops a re-pulled observation
    instead of storing it twice.
    """
    host = base_url.strip().rstrip("/").lower()
    digest = hashlib.sha256(f"{host}\n{public_key.strip()}".encode()).hexdigest()
    return f"pull:{digest[:32]}"


def _cleanup_temp_file(tmp_path: str) -> None:
    try:
        os.remove(tmp_path)
    except OSError:
        pass


def pull_langfuse(
    *,
    base_url: str,
    public_key: str,
    secret_key: str,
    count: int,
    since: str | None,
    until: str,
    trace_names: list[str],
    tags: list[str],
    environment: str | None,
    route: str | None,
    output_path: str,
    import_context: ImportContext | None = None,
    on_progress: Callable[[], None] | None = None,
    user_id: str | None = None,
    session_id: str | None = None,
    metadata: Iterable[tuple[str, str]] = (),
    observation_types: Iterable[str] = COST_BEARING_TYPES,
    include_content: bool = True,
    sleep: Callable[[float], object] = time.sleep,
    source_scope: str | None = None,
) -> tuple[int, int]:
    if import_context is None:
        # Every pulled row carries an import identity, not only sync's: without
        # one, pushing an overlapping pull again stores each observation twice.
        import_context = ImportContext(
            source="langfuse",
            source_scope=source_scope or default_import_scope(base_url, public_key),
        )
    base_params = build_base_params(
        until=until,
        since=since,
        trace_names=trace_names,
        tags=tags,
        environment=environment,
        observation_types=observation_types,
        user_id=user_id,
        session_id=session_id,
        metadata=list(metadata),
    )
    imported = 0
    skipped = 0
    cursor: str | None = None
    used_cursors: set[str] = set()
    page_count = 0
    item_count = 0

    def wait(seconds: float) -> None:
        # A rate-limit wait is progress as far as a sync lease is concerned:
        # renew after each one, so a long Retry-After cannot expire it.
        sleep(seconds)
        if on_progress is not None:
            on_progress()

    output_dir = os.path.dirname(output_path) or "."
    fd, tmp_path = tempfile.mkstemp(
        dir=output_dir, prefix=f".{os.path.basename(output_path)}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w") as f:
            while imported < count:
                page_count += 1
                if page_count > MAX_PAGES_PER_PULL:
                    raise LangfuseAPIError(
                        f"Langfuse pull exceeded the maximum of {MAX_PAGES_PER_PULL} pages"
                    )
                page_params = dict(base_params)
                page_params["limit"] = str(min(PAGE_LIMIT, count - imported))
                if cursor:
                    if cursor in used_cursors:
                        raise LangfuseAPIError(
                            "Langfuse API returned a repeated pagination cursor "
                            "(non-advancing pagination), aborting to avoid an "
                            f"infinite loop: {cursor!r}"
                        )
                    used_cursors.add(cursor)
                    page_params["cursor"] = cursor
                payload = fetch_observations_page(
                    base_url,
                    public_key=public_key,
                    secret_key=secret_key,
                    params=page_params,
                    sleep=wait,
                )
                if on_progress is not None:
                    on_progress()  # a page fetch is progress too
                observations = payload["data"]
                if not observations:
                    break
                item_count += len(observations)
                if item_count > MAX_ITEMS_PER_PULL:
                    raise LangfuseAPIError(
                        f"Langfuse pull exceeded the maximum of {MAX_ITEMS_PER_PULL} items"
                    )
                for observation in observations:
                    if imported >= count:
                        break
                    if isinstance(observation, dict) and not is_importable_observation(
                        observation
                    ):
                        # A span with no model and no cost: structure, not a call.
                        continue
                    try:
                        row = normalize_observation(
                            observation,
                            route_override=route,
                            import_context=import_context,
                        )
                        if not include_content:
                            strip_content(row)
                        line = json.dumps(row)
                    except (KeyError, TypeError, AttributeError) as exc:
                        skipped += 1
                        if on_progress is not None:
                            on_progress()
                        obs_id = (
                            observation.get("id", "<unknown>")
                            if isinstance(observation, dict)
                            else "<unknown>"
                        )
                        print(
                            f"Warning: skipping malformed observation {obs_id}: {exc}",
                            file=sys.stderr,
                        )
                        continue
                    f.write(line + "\n")
                    imported += 1
                    if on_progress is not None:
                        on_progress()
                raw_cursor = payload.get("meta", {}).get("cursor")
                if raw_cursor is not None and not (
                    isinstance(raw_cursor, str) and raw_cursor
                ):
                    raise LangfuseAPIError(
                        "Langfuse API returned a malformed pagination cursor: "
                        f"{raw_cursor!r}"
                    )
                cursor = raw_cursor
                if not cursor:
                    break
        os.replace(tmp_path, output_path)
    finally:
        # On success, tmp_path has already been moved to output_path by
        # os.replace above, so this is a harmless no-op (_cleanup_temp_file
        # silently ignores a missing path); on any failure, this removes the
        # still-present temp file before the exception propagates.
        _cleanup_temp_file(tmp_path)
    return imported, skipped
