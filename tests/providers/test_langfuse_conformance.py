"""The Langfuse normalizer against real Langfuse Cloud responses.

Every observation here was recorded by a live Langfuse run (see
tests/fixtures/langfuse_v2/README.md); the manifest holds what the seeder sent
and what Langfuse reported back. Nothing in these tests is hand-written JSON.
"""

import email.message
import io
import json
import urllib.error
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

import pytest

from metergraphrelay.providers.langfuse import (
    COST_BEARING_TYPES,
    MAX_RETRIES,
    LangfuseAPIError,
    build_base_params,
    default_import_scope,
    fetch_observations_page,
    is_importable_observation,
    normalize_observation,
    pull_langfuse,
    resolve_provider_and_model,
    retry_delay,
)

FIXTURE_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "langfuse_v2"
OBSERVATIONS = json.loads((FIXTURE_DIR / "observations.json").read_text())
MANIFEST = json.loads((FIXTURE_DIR / "manifest.json").read_text())
BY_ID = {observation["id"]: observation for observation in OBSERVATIONS["data"]}
COST_KEYS = sorted(
    key for key, spec in MANIFEST["observations"].items() if spec.get("cost_bearing")
)
STRUCTURE_KEYS = sorted(
    key for key, spec in MANIFEST["observations"].items() if not spec.get("cost_bearing")
)


def observation(key: str) -> dict:
    return BY_ID[MANIFEST["observations"][key]["id"]]


def row(key: str) -> dict:
    return normalize_observation(observation(key), route_override=None)


def test_fixture_covers_every_manifest_observation():
    assert set(BY_ID) == {spec["id"] for spec in MANIFEST["observations"].values()}


# --- G2: which observations are imported -------------------------------------


@pytest.mark.parametrize("key", COST_KEYS)
def test_every_cost_bearing_observation_is_importable(key):
    assert is_importable_observation(observation(key))


@pytest.mark.parametrize("key", STRUCTURE_KEYS)
def test_structural_observations_without_a_model_are_not_imported(key):
    assert not is_importable_observation(observation(key))


def test_embedding_is_imported_with_its_type():
    embedding = row("embedding")
    assert embedding["observation_type"] == "EMBEDDING"
    assert embedding["provider"] == "openai"
    assert embedding["input_tokens"] == 1200


def test_a_non_generation_type_needs_usage_or_cost():
    tool = dict(observation("lookup_tool"), model="gpt-4o-mini")
    assert not is_importable_observation(tool)
    assert is_importable_observation(dict(tool, totalCost=0.001))
    assert is_importable_observation(dict(tool, usageDetails={"input": 3}))
    assert is_importable_observation(dict(tool, usageDetails={"audio_seconds": 1.5}))
    assert not is_importable_observation(dict(tool, usageDetails={"input": 0, "x": True}))
    assert not is_importable_observation(dict(observation("llm_step"), model="gpt-4o", totalCost=1.0))


# --- G1: provider resolution and Langfuse's cost ----------------------------


@pytest.mark.parametrize("key", COST_KEYS)
def test_provider_is_one_the_seeder_expected(key):
    expected = MANIFEST["observations"][key]["expected_providers"]
    assert row(key)["provider"] in expected


def test_bedrock_model_id_keeps_its_catalog_spelling():
    bedrock = row("bedrock_claude")
    assert bedrock["provider"] == "bedrock"
    assert bedrock["model"] == "us.anthropic.claude-3-5-haiku-20241022-v1:0"
    assert "langfuse_model" not in bedrock


def test_gateway_prefix_is_stripped_and_the_original_kept():
    gateway = row("gateway_openai")
    assert (gateway["provider"], gateway["model"]) == ("openai", "gpt-4o")
    assert gateway["langfuse_model"] == "openai/gpt-4o"


def test_no_cost_bearing_row_is_unknown():
    assert [key for key in COST_KEYS if row(key)["provider"] == "unknown"] == []


@pytest.mark.parametrize("key", COST_KEYS)
def test_langfuse_cost_is_always_carried_as_reported(key):
    readback = MANIFEST["langfuse_readback"][key]["totalCost"]
    normalized = row(key)
    assert Decimal(normalized["reported_cost_usd"]) == Decimal(str(readback))
    assert normalized["reported_cost_source"] == "langfuse.observation.totalCost"
    assert normalized["cost_usd"] == readback


def test_reported_cost_total_matches_langfuse():
    relay_total = sum(Decimal(row(key)["reported_cost_usd"]) for key in COST_KEYS)
    langfuse_total = sum(
        Decimal(str(MANIFEST["langfuse_readback"][key]["totalCost"])) for key in COST_KEYS
    )
    assert relay_total == langfuse_total


def test_cost_details_are_carried_as_decimal_strings():
    details = row("openai_cached_reasoning")["reported_cost_details"]
    assert details and all(isinstance(value, str) for value in details.values())
    assert Decimal(details["total"]) == Decimal("0.000252")


# --- G5: latency and time to first token -------------------------------------


@pytest.mark.parametrize(
    "key", [k for k in COST_KEYS if MANIFEST["observations"][k].get("latency_ms_min")]
)
def test_latency_and_ttft_match_langfuse(key):
    readback = MANIFEST["langfuse_readback"][key]
    spec = MANIFEST["observations"][key]
    normalized = row(key)
    assert normalized["latency_ms"] == round(readback["latency"] * 1000)
    assert normalized["ttft_ms"] == round(readback["timeToFirstToken"] * 1000)
    assert normalized["latency_ms"] >= spec["latency_ms_min"]
    assert normalized["ttft_ms"] >= spec["ttft_ms_min"]


def test_latency_falls_back_to_timestamps_when_langfuse_omits_it():
    source = dict(observation("o4_mini"), latency=None, timeToFirstToken=None)
    normalized = normalize_observation(source, route_override=None)
    assert normalized["latency_ms"] == 405
    assert normalized["ttft_ms"] == 151


def test_zero_length_observation_has_zero_latency_not_none():
    assert row("embedding")["latency_ms"] == 0
    assert "ttft_ms" not in row("embedding")


# --- G5b and G6: tags and attribution ----------------------------------------


@pytest.mark.parametrize("key", COST_KEYS)
def test_every_langfuse_tag_lands_as_a_string_pair(key):
    trace = MANIFEST["traces"][MANIFEST["observations"][key]["trace"]]
    tags = row(key)["tags"]
    for tag in trace["tags"]:
        assert tags[f"langfuse.tag.{tag}"] == "true"
    assert all(isinstance(value, str) for value in tags.values())


def test_customer_metadata_lands_and_sdk_metadata_does_not():
    tags = row("openai_cached_reasoning")["tags"]
    assert tags["langfuse.metadata.customer_tier"] == "gold"
    assert tags["langfuse.metadata.feature"] == "support-triage"
    assert tags["langfuse.metadata.nested.attempt"] == "1"
    assert tags["langfuse.metadata.nested.region"] == "us-west-2"
    sdk_written = ("resourceAttributes.", "scope.", "ls_", "lc_versions.", "is_langchain_root")
    for key in (*tags, *row("langchain_generation")["tags"]):
        assert not key.removeprefix("langfuse.metadata.").startswith(sdk_written), key


def test_release_version_and_prompt_are_filterable_tags():
    tags = row("openai_cached_reasoning")["tags"]
    assert tags["langfuse.release"] == MANIFEST["release"]
    assert tags["langfuse.version"] == MANIFEST["traces"]["main"]["version"]
    assert tags["langfuse.prompt.name"] == "mg-conformance-prompt"
    assert tags["langfuse.prompt.version"] == "1"


def test_attribution_fields_carry_through():
    main = MANIFEST["traces"]["main"]
    normalized = row("openai_cached_reasoning")
    assert normalized["user_id"] == main["user_id"]
    assert normalized["session_id"] == main["session_id"]
    assert normalized["release"] == MANIFEST["release"]
    assert normalized["version"] == main["version"]
    assert normalized["trace_name"] == main["name"]
    assert normalized["prompt_name"] == "mg-conformance-prompt"
    assert normalized["prompt_version"] == 1
    assert normalized["model_parameters"] == MANIFEST["observations"][
        "openai_cached_reasoning"
    ]["model_parameters"]


def test_error_observation_keeps_its_status_message():
    spec = MANIFEST["observations"]["error_call"]
    normalized = row("error_call")
    assert normalized["status"] == "error" and normalized["error"] is True
    assert normalized["error_type"] == spec["status_message"]


def test_tags_never_exceed_the_server_limit_and_langfuse_tags_come_first():
    metadata = {f"k{index:03d}": "v" for index in range(60)}
    source = dict(observation("o4_mini"), metadata=metadata, tags=["zz-last", "aa"])
    tags = normalize_observation(source, route_override="r")["tags"]
    assert len(tags) == 32
    assert tags["langfuse.tag.zz-last"] == "true" and tags["langfuse.tag.aa"] == "true"
    assert tags["name"] == "conformance-workflow"
    assert tags["langfuse.release"] == MANIFEST["release"]
    assert "langfuse.metadata.k000" in tags and "langfuse.metadata.k059" not in tags


# --- pulling the fixture as one page -----------------------------------------


def test_pull_imports_every_cost_bearing_observation_and_nothing_else(tmp_path):
    output = tmp_path / "rows.jsonl"
    with patch(
        "metergraphrelay.providers.langfuse.fetch_observations_page",
        return_value={"data": OBSERVATIONS["data"], "meta": {"cursor": None}},
    ) as fetch:
        imported, skipped = pull_langfuse(
            base_url="https://cloud.langfuse.com", public_key="pk", secret_key="sk",
            count=1000, since=None, until="2026-01-02T00:00:00Z", trace_names=[],
            tags=[], environment=None, route=None, output_path=str(output),
        )
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert (imported, skipped) == (len(COST_KEYS), 0)
    assert {r["request_id"] for r in rows} == {
        MANIFEST["observations"][key]["id"] for key in COST_KEYS
    }
    sent = json.loads(fetch.call_args.kwargs["params"]["filter"])
    assert sent[0] == {
        "type": "stringOptions", "column": "type", "operator": "any of",
        "value": list(COST_BEARING_TYPES),
    }


def test_pull_with_exclude_content_strips_input_and_output(tmp_path):
    output = tmp_path / "rows.jsonl"
    with patch(
        "metergraphrelay.providers.langfuse.fetch_observations_page",
        return_value={"data": OBSERVATIONS["data"], "meta": {"cursor": None}},
    ):
        pull_langfuse(
            base_url="https://cloud.langfuse.com", public_key="pk", secret_key="sk",
            count=1000, since=None, until="2026-01-02T00:00:00Z", trace_names=[],
            tags=[], environment=None, route=None, output_path=str(output),
            include_content=False,
        )
    for line in output.read_text().splitlines():
        normalized = json.loads(line)
        assert normalized["content_opted_in"] is False
        for field in ("request_json", "request_text", "response_text", "tool_calls"):
            assert field not in normalized


def _pull_rows(tmp_path, name, data, **overrides):
    output = tmp_path / f"{name}.jsonl"
    kwargs = dict(
        base_url="https://cloud.langfuse.com", public_key="pk", secret_key="sk",
        count=1000, since=None, until="2026-01-02T00:00:00Z", trace_names=[],
        tags=[], environment=None, route=None, output_path=str(output),
    )
    kwargs.update(overrides)
    with patch(
        "metergraphrelay.providers.langfuse.fetch_observations_page",
        return_value={"data": data, "meta": {"cursor": None}},
    ):
        pull_langfuse(**kwargs)
    return [json.loads(line) for line in output.read_text().splitlines()]


def _server_claims(*pushes):
    """The server's dedup gate: the first row with an identity wins."""
    stored = {}
    for rows in pushes:
        for pushed in rows:
            identity = (
                pushed["import_source"], pushed["import_source_scope"],
                pushed["import_event_id"],
            )
            stored.setdefault(identity, pushed)
    return stored


def test_a_re_pull_of_the_same_window_dedups_on_the_import_identity(tmp_path):
    first = _pull_rows(tmp_path, "first", OBSERVATIONS["data"])
    # The re-sync sees the same observations again, one of them edited late
    # (Langfuse updated its cost), plus nothing new.
    edited = [dict(obs) for obs in OBSERVATIONS["data"]]
    for obs in edited:
        if obs["id"] == MANIFEST["observations"]["o4_mini"]["id"]:
            obs["totalCost"] = (obs.get("totalCost") or 0) + 1
    second = _pull_rows(tmp_path, "second", edited)

    for pushed in first + second:
        assert pushed["import_source"] == "langfuse"
        # The event id is Langfuse's observation id, the same id span_id
        # and request_id carry, so scorers that match on span_id still do.
        assert pushed["import_event_id"] == pushed["span_id"] == pushed["request_id"]
    assert {r["import_source_scope"] for r in first + second} == {
        default_import_scope("https://cloud.langfuse.com", "pk")
    }
    stored = _server_claims(first, second)
    assert len(stored) == len(first) == len(COST_KEYS)
    assert sorted(r["span_id"] for r in stored.values()) == sorted(
        MANIFEST["observations"][key]["id"] for key in COST_KEYS
    )


def test_the_default_scope_is_per_project_and_carries_no_key():
    scope = default_import_scope("https://cloud.langfuse.com/", "pk-lf-secretish")
    assert scope == default_import_scope("HTTPS://cloud.langfuse.com", " pk-lf-secretish ")
    assert scope != default_import_scope("https://cloud.langfuse.com", "pk-lf-other")
    assert scope != default_import_scope("https://us.cloud.langfuse.com", "pk-lf-secretish")
    assert "pk-lf" not in scope and "langfuse.com" not in scope
    assert 1 <= len(scope) <= 255


def test_an_explicit_source_scope_matches_sync_rows(tmp_path):
    rows = _pull_rows(tmp_path, "scoped", OBSERVATIONS["data"], source_scope="prod-project")
    assert {r["import_source_scope"] for r in rows} == {"prod-project"}


def test_build_base_params_sends_attribution_selectors_in_the_filter():
    params = build_base_params(
        until="2026-01-02T00:00:00Z", since=None, trace_names=[], tags=[],
        environment=None, user_id="user-conf-1", session_id="s-1",
        metadata=[("feature", "support-triage")],
    )
    conditions = json.loads(params["filter"])
    assert {"type": "string", "column": "userId", "operator": "=", "value": "user-conf-1"} in conditions
    assert {"type": "string", "column": "sessionId", "operator": "=", "value": "s-1"} in conditions
    assert {
        "type": "stringObject", "column": "metadata", "key": "feature",
        "operator": "=", "value": "support-triage",
    } in conditions
    assert "type" not in params and "toStartTime" not in params


def test_build_base_params_default_is_unchanged_for_existing_callers():
    params = build_base_params(
        until="2026-01-02T00:00:00Z", since="2026-01-01T00:00:00Z", trace_names=[],
        tags=[], environment="conformance",
    )
    assert params["type"] == "GENERATION"
    assert params["fromStartTime"] == "2026-01-01T00:00:00Z"
    assert "filter" not in params
    assert "prompt" in params["fields"] and "metrics" in params["fields"]


# --- G11: retries -------------------------------------------------------------


class _Response(io.BytesIO):
    headers: dict[str, str] = {}

    def __enter__(self):
        return self

    def __exit__(self, *_exc) -> None:
        self.close()


def _http_error(code: int, retry_after: str | None = None) -> urllib.error.HTTPError:
    headers = email.message.Message()
    if retry_after is not None:
        headers["Retry-After"] = retry_after
    return urllib.error.HTTPError("https://x", code, "Too Many Requests", headers, io.BytesIO(b""))


def _page() -> _Response:
    return _Response(json.dumps({"data": [], "meta": {}}).encode())


def test_rate_limit_is_retried_after_retry_after():
    waits: list[float] = []
    with patch(
        "urllib.request.urlopen",
        side_effect=[_http_error(429, "7"), _http_error(503), _page()],
    ) as urlopen:
        payload = fetch_observations_page(
            "https://cloud.langfuse.com", public_key="pk", secret_key="sk", params={},
            sleep=waits.append,
        )
    assert payload == {"data": [], "meta": {}}
    assert urlopen.call_count == 3
    assert waits == [7.0, 4.0]  # Retry-After, then doubling from 2 s at attempt 1


def test_rate_limit_gives_up_after_max_retries():
    waits: list[float] = []
    with patch(
        "urllib.request.urlopen",
        side_effect=[_http_error(429) for _ in range(MAX_RETRIES + 1)],
    ):
        with pytest.raises(LangfuseAPIError, match="HTTP 429"):
            fetch_observations_page(
                "https://cloud.langfuse.com", public_key="pk", secret_key="sk", params={},
                sleep=waits.append,
            )
    assert len(waits) == MAX_RETRIES


def test_other_errors_and_disabled_retry_fail_at_once():
    waits: list[float] = []
    with patch("urllib.request.urlopen", side_effect=[_http_error(500)]):
        with pytest.raises(LangfuseAPIError, match="HTTP 500"):
            fetch_observations_page(
                "https://x", public_key="pk", secret_key="sk", params={}, sleep=waits.append,
            )
    with patch("urllib.request.urlopen", side_effect=[_http_error(429)]):
        with pytest.raises(LangfuseAPIError, match="HTTP 429"):
            fetch_observations_page("https://x", public_key="pk", secret_key="sk", params={}, sleep=None)
    assert waits == []


def test_retry_delay_reads_http_dates_and_caps_the_wait():
    assert retry_delay("Thu, 01 Jan 2026 00:00:30 GMT", 0, now=lambda: 1767225600.0) == 30.0
    assert retry_delay("9999", 0) == 120.0
    assert retry_delay(None, 3) == 16.0
    assert retry_delay("garbage", 0) == 2.0


def test_pull_renews_progress_after_a_rate_limit_wait(tmp_path):
    ticks: list[str] = []
    with patch(
        "urllib.request.urlopen",
        side_effect=[_http_error(429, "1"), _page()],
    ):
        pull_langfuse(
            base_url="https://x", public_key="pk", secret_key="sk", count=10, since=None,
            until="2026-01-02T00:00:00Z", trace_names=[], tags=[], environment=None,
            route=None, output_path=str(tmp_path / "o.jsonl"),
            on_progress=lambda: ticks.append("tick"), sleep=lambda _seconds: ticks.append("sleep"),
        )
    assert ticks[:2] == ["sleep", "tick"]


# --- provider spellings beyond the fixture ------------------------------------


@pytest.mark.parametrize(
    ("model", "metadata", "expected"),
    [
        ("o4-mini", None, ("openai", "o4-mini")),
        ("o3", None, ("openai", "o3")),
        ("text-embedding-3-large", None, ("openai", "text-embedding-3-large")),
        ("eu.anthropic.claude-sonnet-4-20250514-v1:0", None,
         ("bedrock", "eu.anthropic.claude-sonnet-4-20250514-v1:0")),
        ("meta.llama3-1-70b-instruct-v1:0", None, ("bedrock", "meta.llama3-1-70b-instruct-v1:0")),
        ("arn:aws:bedrock:us-east-1:111111111111:inference-profile/us.anthropic.claude-3-5-haiku",
         None, ("bedrock",
                "arn:aws:bedrock:us-east-1:111111111111:inference-profile/us.anthropic.claude-3-5-haiku")),
        ("bedrock/us.anthropic.claude-3-5-haiku-20241022-v1:0", None,
         ("bedrock", "us.anthropic.claude-3-5-haiku-20241022-v1:0")),
        ("anthropic/claude-3-5-haiku-20241022", None, ("anthropic", "claude-3-5-haiku-20241022")),
        ("vertex_ai/gemini-2.0-flash", None, ("vertex-ai", "gemini-2.0-flash")),
        ("projects/p/locations/us-central1/publishers/google/models/gemini-2.5-pro", None,
         ("vertex-ai", "gemini-2.5-pro")),
        ("claude-3-5-sonnet-v2@20241022", None, ("vertex-ai", "claude-3-5-sonnet-v2@20241022")),
        ("gemini-2.5-flash", None, ("google", "gemini-2.5-flash")),
        ("codestral-latest", None, ("mistral", "codestral-latest")),
        ("mistral/mistral-large-latest", None, ("mistral", "mistral-large-latest")),
        ("llama-3.1-8b-instant", None, ("meta", "llama-3.1-8b-instant")),
        ("meta-llama/Llama-3.3-70B-Instruct", None, ("meta", "Llama-3.3-70B-Instruct")),
        ("command-r-plus", None, ("cohere", "command-r-plus")),
        ("grok-3", None, ("x-ai", "grok-3")),
        ("my-finetune", None, ("unknown", "my-finetune")),
        ("acme/my-finetune", None, ("unknown", "acme/my-finetune")),
        # Explicit metadata wins, in any of the spellings integrations write.
        ("claude-3-5-haiku-20241022", {"ls_provider": "amazon_bedrock"},
         ("bedrock", "claude-3-5-haiku-20241022")),
        ("gpt-4o", {"gen_ai.system": "az.ai.openai"}, ("azure", "gpt-4o")),
        ("gpt-4o", {"gen_ai": {"system": "openai"}}, ("openai", "gpt-4o")),
        ("llama-3.1-8b-instant", {"ls_provider": "groq"}, ("groq", "llama-3.1-8b-instant")),
        ("gemini-2.0-flash", {"ls_provider": "google_vertexai"}, ("vertex-ai", "gemini-2.0-flash")),
        ("openai/gpt-4o", {"provider": "openrouter"}, ("openrouter", "openai/gpt-4o")),
        ("gpt-4o", {"ls_provider": "litellm"}, ("openai", "gpt-4o")),
    ],
)
def test_provider_resolution(model, metadata, expected):
    source = {"model": model}
    if metadata is not None:
        source["metadata"] = metadata
    assert resolve_provider_and_model(source) == expected
