const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const ts = require('typescript');
const React = require('react');
const { renderToStaticMarkup } = require('react-dom/server');

async function renderProducts(content) {
  const [{ default: Markdown }, { default: gfm }] = await Promise.all([
    import('react-markdown'), import('remark-gfm'),
  ]);
  const source = fs.readFileSync(path.join(__dirname, '../src/components/DebugPanel.tsx'), 'utf8');
  const compiled = ts.transpileModule(source, {
    compilerOptions: { module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX, target: ts.ScriptTarget.ES2022 },
  }).outputText;
  const context = { exports: {}, require(name) {
    if (name === 'react') return { ...React, useState: () => ['products', () => {}] };
    if (name === 'react-markdown') return { __esModule: true, default: Markdown };
    if (name === 'remark-gfm') return { __esModule: true, default: gfm };
    if (name === 'lucide-react') return { X: () => null };
    return require(name);
  }};
  vm.runInNewContext(compiled, context);
  return renderToStaticMarkup(React.createElement(context.exports.default, {
    stage: 'search', toolRounds: 0, productContext: content, userProfile: '', visible: true, onClose() {},
  }));
}

test('product debug context renders Markdown headings and source links', async () => {
  const html = await renderProducts('### 商品\n\n价格快照：[官方来源](https://example.com/item)\n\n|规格|值|\n|---|---|\n|内存|16GB|');
  assert.match(html, /<h3>商品<\/h3>/);
  assert.match(html, /href="https:\/\/example.com\/item"/);
  assert.match(html, /<table>/);
});

test('product debug context does not insert raw source HTML', async () => {
  const html = await renderProducts('<img src="x" onerror="alert(1)"><script>alert(1)</script>');
  assert.doesNotMatch(html, /<img\b|<script\b/);
});

test('product debug context rejects JavaScript source links', async () => {
  const html = await renderProducts('[不安全链接](javascript:alert%281%29)');
  assert.doesNotMatch(html, /href="javascript:/);
});
