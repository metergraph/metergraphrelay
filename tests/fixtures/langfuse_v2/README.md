# Langfuse v2 conformance fixture

Real responses from Langfuse Cloud for one seeded conformance run (Langfuse
Python SDK 5.2.0), scrubbed before they were committed. Content is synthetic;
ids are hashes of stable labels and timestamps are shifted to start at
2026-01-01T00:00:00Z, while durations are real.

- `observations.json`: `GET /api/public/v2/observations` with every field group
  (`core,basic,time,io,metadata,model,usage,prompt,metrics,trace_context`),
  all 18 observations of the run and every observation type, in the API's own
  `{"data": [...], "meta": {...}}` shape so a test can serve it as one page.
- `manifest.json`: what the seeder sent (`observations.<key>`, `traces.<key>`)
  and what Langfuse reported back (`langfuse_readback.<key>`: `totalCost`,
  `latency`, `timeToFirstToken`, `usageDetails`). Tests use it as the expected
  values.

Vendor updates unchanged. Do not hand-edit these files: a field that changes
upstream should show up as a diff here.
