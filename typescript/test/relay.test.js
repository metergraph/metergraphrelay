import {test} from 'node:test';
import assert from 'node:assert/strict';
import {MetergraphRelay} from '../dist/index.js';
const rows = [{ts:'2026-11-16T00:00:00Z',provider:'openai',model:'test-model',input_tokens:10,output_tokens:2}];
const create = (fetch) => new MetergraphRelay({token:'test-token',baseUrl:'https://example.com',fetch});
test('sends the ingest envelope and explicit idempotency key', async () => {
  const relay = create(async (url, options) => {
    assert.equal(url, 'https://example.com/v1/ingest');
    assert.equal(options.headers.Authorization, 'Bearer test-token');
    assert.equal(options.headers['Idempotency-Key'], 'batch-1');
    assert.equal(options.redirect, 'error');
    assert.deepEqual(JSON.parse(options.body), {schema_version:1,rows,meta:{}});
    return Response.json({accepted:1}, {status:202});
  });
  assert.deepEqual(await relay.send(rows,{idempotencyKey:'batch-1'}), {accepted:1,ignored:0});
});
test('rejects HTTP errors without leaking the response body', async () => {
  await assert.rejects(create(async () => new Response('secret', {status:503})).send(rows), /HTTP 503/);
});
for (const ack of [{}, {accepted:'1'}, {accepted:2}, {accepted:-1}, {accepted:0,ignored:0}]) {
  test(`rejects unusable acknowledgment ${JSON.stringify(ack)}`, async () => {
    await assert.rejects(create(async () => Response.json(ack)).send(rows), /acknowledgment/);
  });
}
test('returns ignored rows explicitly', async () => {
  assert.deepEqual(await create(async () => Response.json({accepted:0,ignored:1})).send(rows), {accepted:0,ignored:1});
});
test('does not send empty batches or oversized batches', async () => {
  const relay = create(async () => {throw new Error('must not fetch');});
  assert.deepEqual(await relay.send([]), {accepted:0,ignored:0});
  await assert.rejects(relay.send(Array(2001).fill(rows[0])), /2000/);
  await assert.rejects(relay.send([{...rows[0],tags:{large:'x'.repeat(6*1024*1024)}}]), /bytes/);
});
test('propagates cancellation', async () => {
  const controller = new AbortController(); controller.abort();
  await assert.rejects(create(async () => {throw new Error('must not fetch');}).send(rows,{signal:controller.signal}), {name:'AbortError'});
});
test('rejects insecure remote destinations and empty credentials', () => {
  assert.throws(() => new MetergraphRelay({token:' ',baseUrl:'https://example.com'}), /token/);
  assert.throws(() => new MetergraphRelay({token:'x',baseUrl:'http://example.com'}), /HTTPS/);
});
test('sends over HTTP and times out a stalled server', async () => {
  const {createServer} = await import('node:http');
  const server = createServer(async (req,res) => {
    let body=''; for await (const part of req) body+=part;
    assert.equal(JSON.parse(body).rows[0].provider,'openai');
    if (req.headers['idempotency-key']==='stall') return;
    res.writeHead(202, {'Content-Type':'application/json'});
    res.end(JSON.stringify({accepted:1}));
  });
  await new Promise(resolve => server.listen(0,'127.0.0.1',resolve));
  try {
    const relay=new MetergraphRelay({token:'test',baseUrl:`http://127.0.0.1:${server.address().port}`,timeoutMs:100});
    assert.equal((await relay.send(rows)).accepted,1);
    await assert.rejects(relay.send(rows,{idempotencyKey:'stall'}),{name:'TimeoutError'});
  } finally {server.closeAllConnections(); await new Promise(resolve=>server.close(resolve));}
});
test('preserves cancellation while reading the acknowledgment', async () => {
  const controller=new AbortController();
  const relay=create(async () => {
    controller.abort();
    return {ok:true,json:async()=>{throw controller.signal.reason;}};
  });
  await assert.rejects(relay.send(rows,{signal:controller.signal}),{name:'AbortError'});
});
