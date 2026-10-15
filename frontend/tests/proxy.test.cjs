const assert = require("node:assert/strict");
const http = require("node:http");
const path = require("node:path");
const { test } = require("node:test");

const root = path.resolve(__dirname, "..");

test("local proxy defaults to localhost:8000", async () => {
  const { resolveConfig } = await import("vite");
  const previous = process.env.API_PROXY_TARGET;
  delete process.env.API_PROXY_TARGET;
  try {
    const config = await resolveConfig({ root }, "serve");
    assert.equal(config.server.proxy["/api"].target, "http://localhost:8000");
  } finally {
    if (previous === undefined) delete process.env.API_PROXY_TARGET;
    else process.env.API_PROXY_TARGET = previous;
  }
});

test("configured proxy forwards REST and SSE before stream completion", { timeout: 10000 }, async () => {
  const { createServer } = await import("vite");
  let finishStream;
  const backend = http.createServer((req, res) => {
    if (req.url === "/api/chat/stream") {
      assert.equal(req.method, "POST");
      res.writeHead(200, { "content-type": "text/event-stream" });
      res.write('event: token\ndata: {"content":"hello"}\n\n');
      finishStream = () => res.end('event: done\ndata: {"answer":"hello"}\n\n');
      return;
    }
    res.writeHead(200, { "content-type": "application/json" });
    res.end(JSON.stringify({ path: req.url }));
  });
  await new Promise((resolve) => backend.listen(0, "127.0.0.1", resolve));
  const previous = process.env.API_PROXY_TARGET;
  process.env.API_PROXY_TARGET = `http://127.0.0.1:${backend.address().port}`;
  let frontend;
  let proxyServer;
  try {
    frontend = await createServer({
      root, logLevel: "silent",
      server: { middlewareMode: true, hmr: false, watch: null },
    });
    // Vite can resolve port 0 back to 5173. Bind the middleware host directly
    // to an OS-assigned port so this test can run alongside a development app.
    proxyServer = http.createServer(frontend.middlewares);
    await new Promise((resolve) => proxyServer.listen(0, "127.0.0.1", resolve));
    const base = `http://127.0.0.1:${proxyServer.address().port}`;
    const response = await fetch(`${base}/api/conversations`);
    assert.deepEqual(await response.json(), { path: "/api/conversations" });

    const stream = await fetch(`${base}/api/chat/stream`, { method: "POST", body: "{}" });
    assert.equal(stream.headers.get("content-type"), "text/event-stream");
    assert.equal(stream.headers.get("x-accel-buffering"), "no");
    const reader = stream.body.getReader();
    const first = await reader.read();
    assert.equal(first.done, false);
    assert.match(new TextDecoder().decode(first.value), /event: token/);
    finishStream();
    let remaining = "";
    for (;;) {
      const chunk = await reader.read();
      if (chunk.done) break;
      remaining += new TextDecoder().decode(chunk.value);
    }
    assert.match(remaining, /event: done/);
  } finally {
    finishStream?.();
    if (proxyServer?.listening) {
      proxyServer.closeAllConnections();
      await new Promise((resolve) => proxyServer.close(resolve));
    }
    if (frontend) await frontend.close();
    backend.closeAllConnections();
    await new Promise((resolve) => backend.close(resolve));
    if (previous === undefined) delete process.env.API_PROXY_TARGET;
    else process.env.API_PROXY_TARGET = previous;
  }
});
