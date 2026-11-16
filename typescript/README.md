# Metergraph Relay for Node.js

A small TypeScript library for forwarding normalized LLM telemetry from a database or export job. Node.js 22.12+; no runtime dependencies. This is a sender, not a port of the Python provider pull/sync commands.

```sh
npm install metergraphrelay
```

```ts
import { MetergraphRelay } from 'metergraphrelay';

const relay = new MetergraphRelay({
  token: process.env.METERGRAPH_APP_TOKEN!,
  baseUrl: process.env.METERGRAPH_INGEST_URL!,
});

await relay.send([{
  ts: '2026-11-16T00:00:00Z',
  provider: 'openai',
  model: 'your-model',
  request_id: 'provider-request-id',
  input_tokens: 1200,
  output_tokens: 200,
  cache_read_tokens: 1000,
  tags: { workflow: 'daily-report', workflowRunId: 'run-123' },
  content_opted_in: false,
}], { idempotencyKey: 'persisted-export-batch-id' });
// Advance the DB checkpoint only after a successful acknowledgment.
```

`send()` submits one batch of at most 2,000 rows / 6 MiB and returns `{accepted, ignored}`. It throws on HTTP errors, invalid or incomplete acknowledgments, timeout (30 seconds by default), or cancellation. An acknowledgment confirms ingest acceptance, not completion of downstream analysis. An `ignored` count is reported explicitly; consult the ingest endpoint's deduplication policy before treating ignored records as delivered.

The application owns mapping, DB queries, durable checkpoints and retries. For retry safety, persist a batch ID and its exact body together and reuse the same `idempotencyKey`; never reuse a key for different records. Idempotency support depends on the target server. There are no hidden retries or background tasks. Pass an `AbortSignal` in `send()` options for cancellation. Call from a Temporal **activity**, rather than workflow code.

Rows use the same normalized `/v1/ingest` schema as the Python relay. Extra fields are forwarded as supplied, but this does not guarantee the server interprets every field. Native provider payloads need mapping before submission. Do not send credentials or unredacted headers. Content is caller-controlled; omit it unless explicitly opted in. Unknown usage counts should be omitted rather than set to null.

## Development and publishing

```sh
npm ci
npm test
npm pack --dry-run
```

Publish this directory with `npm publish --access public`. Python and npm versions are independent. Prefer npm releases over GitHub dependencies for stable versions, generated JavaScript, and bundled TypeScript declarations. After the first release, configure npm trusted publishing for this repository rather than keeping a long-lived publishing token.
