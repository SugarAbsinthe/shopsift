// Exercise the real hook against controlled async API completions. The small
// hook harness isolates request ownership; this is not a browser rendering test.
const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const ts = require('typescript');

function deferred() {
  let resolve, reject;
  const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
}

function harness(overrides = {}) {
  let state, reducer, mounted = false, refIndex = 0;
  const refs = [], cleanups = [], streams = [], saved = [];
  const api = {
    listConversations: async () => [],
    getMessages: async () => [],
    deleteConversation: async () => {},
    saveMessage: async (...args) => { saved.push(args); },
    sendMessageStream: (_id, _q, _history, callbacks, signal) => {
      const completion = deferred();
      streams.push({ callbacks, signal, ...completion });
      return completion.promise;
    },
    ...overrides,
  };
  const react = {
    useReducer(fn, initial) {
      reducer = fn;
      if (state === undefined) state = initial;
      return [state, action => { state = reducer(state, action); }];
    },
    useRef(initial) {
      const slot = refIndex++;
      if (!refs[slot]) refs[slot] = { current: initial };
      return refs[slot];
    },
    useCallback: fn => fn,
    useEffect(effect) {
      if (!mounted) {
        const cleanup = effect();
        if (typeof cleanup === 'function') cleanups.push(cleanup);
      }
    },
  };
  const source = fs.readFileSync(path.join(__dirname, '../src/hooks/useChat.ts'), 'utf8');
  const compiled = ts.transpileModule(source, {
    compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 },
  }).outputText;
  const context = {
    exports: {}, AbortController,
    require(name) {
      if (name === 'react') return react;
      if (name === '../api/client') return api;
      throw new Error(`Unexpected dependency: ${name}`);
    },
  };
  vm.runInNewContext(compiled, context, { filename: 'useChat.js' });
  function render() {
    refIndex = 0;
    const hook = context.exports.useChat();
    mounted = true;
    return hook;
  }
  render();
  return { render, state: () => state, streams, saved, unmount: () => cleanups.forEach(fn => fn()) };
}

const done = (fields = {}) => ({
  answer: 'verified answer', stage: 'search', user_profile: 'profile',
  product_context: 'products', tool_rounds: 0, ...fields,
});

test('terminal answer replaces preview and only authoritative text is saved', async () => {
  const h = harness();
  await h.render().selectConversation('A');
  const sending = h.render().sendMessage('推荐产品');
  const stream = h.streams[0];
  stream.callbacks.onToken('wrong preview');
  stream.callbacks.onDone(done());
  assert.equal(h.state().messages.at(-1).content, 'verified answer');
  assert.equal(h.saved.find(args => args[1] === 'assistant')[2], 'verified answer');
  stream.resolve();
  await sending;
  assert.equal(h.state().isLoading, false);
  stream.callbacks.onToken('after completion');
  assert.equal(h.state().messages.at(-1).content, 'verified answer');
});

test('switch resets metadata and ignores every late callback from the old stream', async () => {
  const h = harness({ getMessages: async id => id === 'B' ? [{ role: 'assistant', content: 'B-answer' }] : [] });
  await h.render().selectConversation('A');
  const sending = h.render().sendMessage('A-question');
  const stream = h.streams[0];
  stream.callbacks.onDone(done());
  await h.render().selectConversation('B');
  assert.equal(stream.signal.aborted, true);
  const before = JSON.stringify(h.state());
  stream.callbacks.onToken('late A');
  stream.callbacks.onStage('summary');
  stream.callbacks.onStatus('late status');
  stream.callbacks.onToolStart(['late tool']);
  stream.callbacks.onToolEnd([], ['error']);
  stream.callbacks.onDone(done({ answer: 'late answer' }));
  stream.callbacks.onError('late error');
  stream.reject(new Error('late rejection'));
  await sending;
  assert.equal(JSON.stringify(h.state()), before);
  assert.equal(h.state().lastProfile, '');
  assert.equal(h.state().lastProductContext, '');
  assert.equal(h.state().lastStage, 'discovery');
  assert.equal(h.state().messages[0].content, 'B-answer');
  assert.equal(h.saved.filter(args => args[1] === 'assistant').length, 1);
});

test('rapid A to B to C selections ignore out-of-order history and errors', async () => {
  const loads = { A: deferred(), B: deferred(), C: deferred() };
  const h = harness({ getMessages: id => loads[id].promise });
  const a = h.render().selectConversation('A');
  const b = h.render().selectConversation('B');
  const c = h.render().selectConversation('C');
  loads.C.resolve([{ role: 'assistant', content: 'C-answer' }]);
  await c;
  loads.A.resolve([{ role: 'assistant', content: 'A-answer' }]);
  loads.B.reject(new Error('late history error'));
  await Promise.all([a, b]);
  assert.equal(h.state().currentConvId, 'C');
  assert.equal(h.state().messages[0].content, 'C-answer');
});

test('replaced request cannot clear loading or write into the newer request', async () => {
  const h = harness();
  await h.render().selectConversation('A');
  const first = h.render().sendMessage('first');
  const second = h.render().sendMessage('second');
  const oldStream = h.streams[0], newStream = h.streams[1];
  oldStream.callbacks.onToken('old');
  oldStream.callbacks.onDone(done());
  oldStream.resolve();
  await first;
  assert.equal(h.state().isLoading, true);
  assert.equal(h.state().messages.at(-1).content, '');
  assert.equal(h.saved.filter(args => args[1] === 'assistant').length, 0);
  newStream.callbacks.onDone(done({ answer: 'new' }));
  newStream.resolve();
  await second;
  assert.equal(h.state().messages.at(-1).content, 'new');
  assert.equal(h.state().isLoading, false);
});

test('empty final profile and context clear previous values', async () => {
  const h = harness();
  await h.render().selectConversation('A');
  const first = h.render().sendMessage('first');
  h.streams[0].callbacks.onDone(done());
  h.streams[0].resolve();
  await first;
  const second = h.render().sendMessage('清空画像');
  h.streams[1].callbacks.onDone(done({ user_profile: '', product_context: '' }));
  h.streams[1].resolve();
  await second;
  assert.equal(h.state().lastProfile, '');
  assert.equal(h.state().lastProductContext, '');
});

test('cancel and unmount suppress late completion and persistence', async () => {
  for (const action of ['cancel', 'unmount']) {
    const h = harness();
    await h.render().selectConversation('A');
    const sending = h.render().sendMessage('question');
    if (action === 'cancel') h.render().cancel(); else h.unmount();
    const before = JSON.stringify(h.state());
    h.streams[0].callbacks.onDone(done());
    h.streams[0].callbacks.onToken('late');
    h.streams[0].resolve();
    await sending;
    assert.equal(h.streams[0].signal.aborted, true);
    assert.equal(JSON.stringify(h.state()), before);
    assert.equal(h.saved.filter(args => args[1] === 'assistant').length, 0);
  }
});

test('deleting the selected conversation clears state and invalidates its stream', async () => {
  const h = harness();
  await h.render().selectConversation('A');
  const sending = h.render().sendMessage('question');
  await h.render().removeConversation('A');
  h.streams[0].callbacks.onDone(done());
  h.streams[0].resolve();
  await sending;
  assert.equal(h.state().currentConvId, null);
  assert.equal(h.state().messages.length, 0);
  assert.equal(h.saved.filter(args => args[1] === 'assistant').length, 0);
});

test('history still loading cannot replace a newly sent turn', async () => {
  const history = deferred();
  const h = harness({ getMessages: () => history.promise });
  const selecting = h.render().selectConversation('A');
  const sending = h.render().sendMessage('question');
  history.resolve([{ role: 'assistant', content: 'old history' }]);
  await selecting;
  h.streams[0].callbacks.onDone(done());
  h.streams[0].resolve();
  await sending;
  assert.equal(h.state().messages[0].content, 'question');
  assert.equal(h.state().messages.at(-1).content, 'verified answer');
});

test('reloaded history contains the authoritative final answer', async () => {
  const records = [];
  const h = harness({
    saveMessage: async (_id, role, content) => records.push({ role, content }),
    getMessages: async id => id === 'A' ? [...records] : [],
  });
  await h.render().selectConversation('A');
  const sending = h.render().sendMessage('question');
  h.streams[0].callbacks.onToken('unverified preview');
  h.streams[0].callbacks.onDone(done());
  h.streams[0].resolve();
  await sending;
  await h.render().selectConversation('B');
  await h.render().selectConversation('A');
  assert.equal(h.state().messages.at(-1).content, 'verified answer');
});

test('late conversation creation cannot override a later selection', async () => {
  const creation = deferred();
  const h = harness({ createConversation: () => creation.promise });
  const creating = h.render().newConversation();
  await h.render().selectConversation('B');
  creation.resolve({ id: 'new', title: 'new conversation' });
  await creating;
  assert.equal(h.state().currentConvId, 'B');
  assert.equal(h.state().conversations[0].id, 'new');
});
