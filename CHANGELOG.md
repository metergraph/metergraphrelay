# Changelog

## 0.8.0

Langfuse import (`pull langfuse` and `sync langfuse`):

- **Every cost-bearing observation.** Besides `GENERATION`, any `EMBEDDING`,
  `AGENT`, `TOOL`, `CHAIN`, `RETRIEVER`, `EVALUATOR` or `GUARDRAIL`
  observation that names a model and carries usage or a cost is imported.
  Each row carries `observation_type`.
- **Provider resolution.** Resolved from the observation's metadata
  (`provider`, `ls_provider`, `gen_ai.system`, `gen_ai.provider.name`) or the
  model name: Bedrock model ids and ARNs, Vertex resource paths and
  Anthropic-on-Vertex versions, gateway prefixes and more model families
  (`o4-`, Mistral, Llama, Cohere, Grok, Qwen, Kimi). A stripped gateway prefix
  keeps the original model as `langfuse_model`.
- **Langfuse's cost** is kept on every row as `reported_cost_usd`, with the
  per-bucket `reported_cost_details`.
- **Tags land.** Each Langfuse tag is one `tags` pair,
  `langfuse.tag.<tag>: "true"`. They used to be sent as one list, which
  metergraph drops because it keeps only string values. Trace metadata
  lands in `tags` too, scalar values as strings, without SDK-written keys.
- **Timing and attribution.** Rows carry `latency_ms`, `ttft_ms`, `user_id`,
  `trace_name`, `release`, `version`, `prompt_name`, `prompt_version` and
  `model_parameters` when Langfuse has them.
- **Rate limits.** HTTP 429, 502, 503 and 504 are retried up to six times,
  after `Retry-After` when present, else with doubling waits capped at two
  minutes. `sync` renews its lease after each wait.
- **Selectors.** `--user-id`, `--session-id` and repeatable
  `--metadata KEY=VALUE` on `pull` and `sync`.
- **Content.** `pull langfuse --exclude-content` leaves input and output in
  Langfuse (`--include-content`, the default, says so explicitly). The
  `sync langfuse` help no longer claims content is uploaded with no opt-out.

For library callers: `build_base_params` still requests `GENERATION` alone
unless given `observation_types`; pass `COST_BEARING_TYPES` and keep rows with
`is_importable_observation` to import the other types.
