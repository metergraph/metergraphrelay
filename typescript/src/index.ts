/** A normalized /v1/ingest row. Provider-specific fields can be preserved. */
export interface RelayRow {
  ts: string;
  provider: string;
  model: string;
  input_tokens?: number;
  output_tokens?: number;
  cache_read_tokens?: number;
  request_id?: string;
  tags?: Record<string, unknown>;
  content_opted_in?: boolean;
  [field: string]: unknown;
}
export interface RelayOptions {
  token: string;
  baseUrl: string;
  timeoutMs?: number;
  fetch?: typeof globalThis.fetch;
}
export interface SendOptions { idempotencyKey?: string; signal?: AbortSignal }
export interface SendResult { accepted: number; ignored: number }

/** Sends one batch. The caller owns persistence, retries and checkpoints. */
export class MetergraphRelay {
  private readonly options: RelayOptions;
  private readonly url: string;
  constructor(options: RelayOptions) {
    if (!options.token.trim()) throw new Error('A Metergraph ingest token is required');
    const url = new URL(options.baseUrl);
    if (url.username || url.password || url.search || url.hash) throw new Error('baseUrl must not contain credentials, a query or a fragment');
    if (url.protocol !== 'https:' && !(url.protocol === 'http:' && ['localhost','127.0.0.1','[::1]'].includes(url.hostname))) {
      throw new Error('Use HTTPS except for a local development server');
    }
    if (!Number.isInteger(options.timeoutMs ?? 30000) || (options.timeoutMs ?? 30000) <= 0) throw new Error('timeoutMs must be a positive integer');
    this.options = {...options};
    this.url = url.toString().replace(/\/$/, '') + '/v1/ingest';
  }
  async send(rows: readonly RelayRow[], options: SendOptions = {}): Promise<SendResult> {
    options.signal?.throwIfAborted();
    if (!rows.length) return {accepted:0, ignored:0};
    if (rows.length > 2000) throw new Error('A batch can contain at most 2000 rows');
    for (const row of rows) {
      if (!row || typeof row !== 'object' || !row.ts || !row.provider || !row.model) throw new Error('Every row needs ts, provider and model');
    }
    const body = JSON.stringify({schema_version:1, rows, meta:{}});
    if (new TextEncoder().encode(body).byteLength > 6*1024*1024) throw new Error('Batch exceeds 6291456 bytes');
    const headers: Record<string,string> = {'Authorization':`Bearer ${this.options.token}`, 'Content-Type':'application/json'};
    if (options.idempotencyKey !== undefined) {
      if (!options.idempotencyKey.trim()) throw new Error('idempotencyKey must not be empty');
      headers['Idempotency-Key'] = options.idempotencyKey;
    }
    const timeout = AbortSignal.timeout(this.options.timeoutMs ?? 30000);
    const response = await (this.options.fetch ?? globalThis.fetch)(this.url, {
      method:'POST', headers, body, redirect:'error',
      signal: options.signal ? AbortSignal.any([options.signal, timeout]) : timeout,
    });
    if (!response.ok) { await response.body?.cancel(); throw new Error(`Metergraph ingest HTTP ${response.status}`); }
    let acknowledgment: unknown;
    try { acknowledgment = await response.json(); } catch { throw new Error('Invalid Metergraph ingest acknowledgment'); }
    const result = acknowledgment as Partial<SendResult> | null;
    const accepted = result?.accepted;
    const ignored = result?.ignored ?? 0;
    if (!Number.isInteger(accepted) || !Number.isInteger(ignored) || accepted! < 0 || ignored < 0 || accepted! + ignored !== rows.length) {
      throw new Error('Incomplete or invalid Metergraph ingest acknowledgment; do not advance the checkpoint');
    }
    return {accepted:accepted!, ignored};
  }
}
