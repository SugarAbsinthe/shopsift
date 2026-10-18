const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const ts = require('typescript');

function client(fetch) {
  const timers = new Map();
  let nextTimer = 0;
  const source = fs.readFileSync(path.join(__dirname, '../src/api/client.ts'), 'utf8');
  const compiled = ts.transpileModule(source, {
    compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 },
  }).outputText;
  const context = {
    exports: {}, fetch, AbortController, TextDecoder,
    setTimeout(fn) { const id = ++nextTimer; timers.set(id, fn); return id; },
    clearTimeout(id) { timers.delete(id); },
  };
  vm.runInNewContext(compiled, context);
  return { api: context.exports, timers };
}

function response(text) {
  const bytes = new TextEncoder().encode(text);
  return new Response(new ReadableStream({
    start(controller) {
      // Splitting individual UTF-8 bytes also tests boundaries inside Chinese.
      for (let i = 0; i < bytes.length; i += 3) controller.enqueue(bytes.slice(i, i + 3));
      controller.close();
    },
  }), { headers: { 'content-type': 'text/event-stream' } });
}

function callbacks() {
  const events = [];
  return {
    events,
    onToken: value => events.push(['token', value]),
    onStatus: value => events.push(['status', value]),
    onStage: value => events.push(['stage', value]),
    onDone: value => events.push(['done', value.answer]),
    onError: value => events.push(['error', value]),
  };
}

test('SSE decodes chunked UTF-8, CRLF and terminal answer without accepting later events', async () => {
  const c = client(async () => response(
    ': heartbeat\r\n\r\nevent:status\r\ndata:{"message":"查询中"}\r\n\r\n' +
    'event: token\r\ndata: {"content":"你好"}\r\n\r\n' +
    'event: done\r\ndata: {"answer":"完整结果"}\r\n\r\n' +
    'event: token\r\ndata: {"content":"late"}\r\n\r\n',
  ));
  const cb = callbacks();
  await c.api.sendMessageStream('id', 'question', [], cb);
  assert.deepEqual(cb.events, [['status', '查询中'], ['token', '你好'], ['done', '完整结果']]);
  assert.equal(c.timers.size, 0);
});

test('stream EOF before a terminal event is an error, not successful completion', async () => {
  const c = client(async () => response('event: token\ndata: {"content":"partial"}\n\n'));
  const cb = callbacks();
  await assert.rejects(c.api.sendMessageStream('id', 'q', [], cb), /提前中断/);
  assert.deepEqual(cb.events, [['token', 'partial']]);
  assert.equal(c.timers.size, 0);
});

test('error event terminates and cancels the reader without waiting for server EOF', async () => {
  let cancelled = false;
  const c = client(async () => new Response(new ReadableStream({
    start(controller) { controller.enqueue(new TextEncoder().encode('event: error\ndata: {"message":"failed"}\n\n')); },
    cancel() { cancelled = true; },
  }), { headers: { 'content-type': 'text/event-stream' } }));
  const cb = callbacks();
  await c.api.sendMessageStream('id', 'q', [], cb);
  assert.deepEqual(cb.events, [['error', 'failed']]);
  assert.equal(cancelled, true);
});

test('complete final frame without a trailing newline is preserved', async () => {
  const c = client(async () => response('event: done\ndata: {"answer":"done"}'));
  const cb = callbacks();
  await c.api.sendMessageStream('id', 'q', [], cb);
  assert.deepEqual(cb.events, [['done', 'done']]);
});

test('HTTP errors and unexpected HTML responses remain visible', async () => {
  for (const [reply, expected] of [
    [new Response('{"detail":"unavailable"}', { status: 503 }), /unavailable/],
    [new Response('<html>proxy</html>', { headers: { 'content-type': 'text/html' } }), /未返回对话流/],
  ]) {
    const c = client(async () => reply);
    await assert.rejects(c.api.sendMessageStream('id', 'q', [], callbacks()), expected);
    assert.equal(c.timers.size, 0);
  }
});

test('malformed payloads cannot be accepted as a completed answer', async () => {
  for (const payload of ['bad-json', '{}', 'null']) {
    const c = client(async () => response(`event: done\ndata: ${payload}\n\n`));
    const cb = callbacks();
    await assert.rejects(c.api.sendMessageStream('id', 'q', [], cb), /格式错误/);
    assert.deepEqual(cb.events, []);
  }
});

test('timeout aborts pending fetch and caller cancellation stays distinct', async () => {
  for (const action of ['timeout', 'cancel']) {
    const c = client((_url, init) => new Promise((_resolve, reject) => {
      init.signal.addEventListener('abort', () => reject(new DOMException('aborted', 'AbortError')));
    }));
    const controller = new AbortController();
    const sending = c.api.sendMessageStream('id', 'q', [], callbacks(), controller.signal);
    if (action === 'timeout') [...c.timers.values()][0](); else controller.abort();
    await assert.rejects(sending, action === 'timeout' ? /超时/ : { name: 'AbortError' });
    assert.equal(c.timers.size, 0);
  }
});
