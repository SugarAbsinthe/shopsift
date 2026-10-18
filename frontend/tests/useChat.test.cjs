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
  const refs = [], cleanups = [], effects = [], streams = [], saved = [];
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
        effects.push(effect);
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
  return {
    render, state: () => state, streams, saved,
    unmount: () => cleanups.forEach(fn => fn()),
    replayEffects() {
      cleanups.splice(0).forEach(fn => fn());
      effects.forEach(effect => {
        const cleanup = effect();
        if (typeof cleanup === 'function') cleanups.push(cleanup);
      });
    },
  };
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
  stream.resolve();
  await sending;
  assert.equal(h.saved.find(args => args[1] === 'assistant')[2], 'verified answer');
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

test('first send creates and selects a conversation after StrictMode effect replay', async () => {
  let creations = 0;
  const h = harness({ createConversation: async () => { creations++; return { id: 'new' }; } });
  h.replayEffects();
  const sending = h.render().sendMessage('first question');
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(creations, 1);
  assert.equal(h.state().currentConvId, 'new');
  assert.equal(h.state().messages[0].content, 'first question');
  assert.equal(h.streams.length, 1);
  h.streams[0].callbacks.onDone(done());
  h.streams[0].resolve();
  await sending;
  assert.equal(h.state().isLoading, false);
});

test('rapid initial clicks create only one conversation and one stream', async () => {
  const creation = deferred();
  let creations = 0;
  const h = harness({ createConversation: () => { creations++; return creation.promise; } });
  const first = h.render().sendMessage('first');
  const second = h.render().sendMessage('second');
  assert.equal(h.state().isLoading, true);
  creation.resolve({ id: 'new' });
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(creations, 1);
  assert.equal(h.streams.length, 1);
  assert.equal(h.state().messages[0].content, 'first');
  h.streams[0].resolve();
  await Promise.all([first, second]);
});

test('initial list arriving late does not remove the newly created conversation', async () => {
  const list = deferred();
  const h = harness({ listConversations: () => list.promise, createConversation: async () => ({ id: 'new' }) });
  await h.render().newConversation();
  list.resolve([{ id: 'older' }]);
  await new Promise(resolve => setImmediate(resolve));
  assert.deepEqual(Array.from(h.state().conversations, c => c.id), ['new', 'older']);
});

test('failed initial creation displays the error without starting a stream', async () => {
  const h = harness({ createConversation: async () => { throw new Error('service unavailable'); } });
  await h.render().sendMessage('first');
  assert.equal(h.streams.length, 0);
  assert.equal(h.state().error, 'service unavailable');
  assert.equal(h.state().isLoading, false);
});

test('cancel or unmount while creating prevents a late send', async () => {
  for (const action of ['cancel', 'unmount']) {
    const creation = deferred();
    const h = harness({ createConversation: () => creation.promise });
    const sending = h.render().sendMessage('first');
    if (action === 'cancel') h.render().cancel(); else h.unmount();
    creation.resolve({ id: 'new' });
    await sending;
    assert.equal(h.streams.length, 0);
    assert.equal(h.state().currentConvId, null);
  }
});

test('stream progress is visible and cleared on completion without hiding errors', async () => {
  const h = harness();
  await h.render().selectConversation('A');
  const sending = h.render().sendMessage('question');
  const stream = h.streams[0];
  stream.callbacks.onStatus('retrieving');
  assert.equal(h.state().status, 'retrieving');
  stream.callbacks.onError('service failed');
  stream.callbacks.onStatus('cleanup');
  assert.equal(h.state().error, 'service failed');
  stream.resolve();
  await sending;
  assert.equal(h.state().status, null);
});

test('cancelling new conversation creation preserves the previously selected chat', async () => {
  const creation = deferred();
  const h = harness({ createConversation: () => creation.promise });
  await h.render().selectConversation('A');
  const creating = h.render().newConversation();
  h.render().cancel();
  creation.resolve({ id: 'new' });
  await creating;
  assert.equal(h.state().currentConvId, 'A');
  const sending = h.render().sendMessage('continue A');
  assert.equal(h.streams.length, 1);
  h.streams[0].resolve();
  await sending;
});

test('persistence failure remains visible after streaming progress and completion', async () => {
  const h = harness({ saveMessage: async () => { throw new Error('disk unavailable'); } });
  await h.render().selectConversation('A');
  const sending = h.render().sendMessage('question');
  await Promise.resolve();
  h.streams[0].callbacks.onStatus('working');
  h.streams[0].callbacks.onDone(done());
  h.streams[0].resolve();
  await sending;
  assert.match(h.state().error, /消息保存失败/);
  assert.equal(h.state().messages.at(-1).content, 'verified answer');
  assert.equal(h.state().isLoading, false);
});

test('cancelled creation does not block retry or release the newer creation gate', async () => {
  const old = deferred(), next = deferred();
  let creations = 0;
  const h = harness({ createConversation: () => ++creations === 1 ? old.promise : next.promise });
  const first = h.render().sendMessage('cancelled question');
  h.render().cancel();
  const retry = h.render().sendMessage('retry');
  assert.equal(creations, 2);
  old.resolve({ id: 'old' });
  await first;
  assert.equal(h.state().currentConvId, null);
  assert.equal(h.state().isLoading, true);
  await h.render().sendMessage('duplicate click');
  assert.equal(creations, 2);
  next.resolve({ id: 'next' });
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(h.state().currentConvId, 'next');
  assert.equal(h.streams.length, 1);
  h.streams[0].callbacks.onDone(done());
  h.streams[0].resolve();
  await retry;
  assert.equal(h.saved[0][2], 'retry');
  assert.equal(h.state().isLoading, false);
});

test('old creation returning after retry cannot steal focus or stop its stream', async () => {
  const old = deferred();
  let creations = 0;
  const h = harness({ createConversation: () => ++creations === 1 ? old.promise : Promise.resolve({ id: 'next' }) });
  const first = h.render().sendMessage('cancelled question');
  h.render().cancel();
  const retry = h.render().sendMessage('retry');
  await new Promise(resolve => setImmediate(resolve));
  old.resolve({ id: 'old' });
  await first;
  assert.equal(h.state().currentConvId, 'next');
  assert.equal(h.state().isLoading, true);
  assert.equal(h.streams[0].signal.aborted, false);
  h.streams[0].resolve();
  await retry;
});

test('slow user save preserves user then assistant order when history is reopened', async () => {
  const userSave = deferred();
  const records = [], attempts = [];
  const h = harness({
    saveMessage: async (_id, role, content) => {
      attempts.push(role);
      if (role === 'user') await userSave.promise;
      records.push({ role, content });
    },
    getMessages: async id => id === 'A' ? [...records] : [],
  });
  await h.render().selectConversation('A');
  const sending = h.render().sendMessage('question');
  h.streams[0].callbacks.onDone(done());
  h.streams[0].resolve();
  await new Promise(resolve => setImmediate(resolve));
  assert.deepEqual(attempts, ['user']);
  assert.equal(h.state().isLoading, true);
  userSave.resolve();
  await sending;
  assert.deepEqual(records.map(record => record.role), ['user', 'assistant']);
  await h.render().selectConversation('B');
  await h.render().selectConversation('A');
  assert.deepEqual(Array.from(h.state().messages, message => message.role), ['user', 'assistant']);
});

test('failed user save never creates an orphan assistant record', async () => {
  const attempts = [];
  const h = harness({ saveMessage: async (_id, role) => {
    attempts.push(role);
    if (role === 'user') throw new Error('disk unavailable');
  } });
  await h.render().selectConversation('A');
  const sending = h.render().sendMessage('question');
  h.streams[0].callbacks.onDone(done());
  h.streams[0].resolve();
  await sending;
  assert.deepEqual(attempts, ['user']);
  assert.match(h.state().error, /消息保存失败/);
});

test('queued assistant save stays with its original conversation after switching', async () => {
  const userSave = deferred();
  const saved = [];
  const h = harness({ saveMessage: async (id, role) => {
    if (role === 'user') await userSave.promise;
    saved.push([id, role]);
    if (role === 'assistant') throw new Error('late save failure');
  } });
  await h.render().selectConversation('A');
  const sending = h.render().sendMessage('question');
  h.streams[0].callbacks.onDone(done());
  h.streams[0].resolve();
  await h.render().selectConversation('B');
  const before = JSON.stringify(h.state());
  userSave.resolve();
  await sending;
  assert.deepEqual(saved, [['A', 'user'], ['A', 'assistant']]);
  assert.equal(JSON.stringify(h.state()), before);
});
