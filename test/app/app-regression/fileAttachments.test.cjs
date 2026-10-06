// Chat file attachment regression tests (#1984).
//
// Exercises the TypeScript helpers in fileAttachments.ts directly (classify,
// language detection, binary sniffing, prompt wrapping, request conversion)
// and asserts the panel/preview wiring contract via source inspection.

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

const repoRoot = path.resolve(__dirname, '..', '..', '..');
const appRoot = path.join(repoRoot, 'src', 'app');

// ── TypeScript loader (same pattern as routerCollections.test.cjs) ─────────

let ts = null;
try { ts = require(path.join(appRoot, 'node_modules', 'typescript')); }
catch (_) {
  try { ts = require('typescript'); } catch (_2) { ts = null; }
}

if (!ts) {
  module.exports = {
    tests: [{
      name: 'file attachments suite',
      run: () => ({ skip: true, reason: "typescript not installed - run 'npm ci' in src/app first" }),
    }],
  };
  return;
}

if (!require.extensions['.ts']) {
  require.extensions['.ts'] = function loadTypeScript(module, filename) {
    const source = fs.readFileSync(filename, 'utf8');
    const output = ts.transpileModule(source, {
      compilerOptions: {
        esModuleInterop: true, module: ts.ModuleKind.CommonJS,
        moduleResolution: ts.ModuleResolutionKind.NodeJs, target: ts.ScriptTarget.ES2020,
      },
      fileName: filename,
    }).outputText;
    module._compile(output, filename);
  };
}

const fileAttachments = require(
  path.join(appRoot, 'src', 'renderer', 'utils', 'fileAttachments.ts'),
);

const panelSourcePath = path.join(
  appRoot, 'src', 'renderer', 'components', 'panels', 'LLMChatPanel.tsx',
);

function readPanelSource() {
  return fs.readFileSync(panelSourcePath, 'utf8');
}

const f = (name, type = '', size = 10) => ({ name, type, size });

const tests = [
  {
    name: 'classifyFile routes by MIME first',
    run() {
      assert.equal(fileAttachments.classifyFile(f('blob', 'image/png')), 'image');
      assert.equal(fileAttachments.classifyFile(f('clip', 'audio/mpeg')), 'audio');
      assert.equal(fileAttachments.classifyFile(f('notes', 'text/plain')), 'text');
      assert.equal(fileAttachments.classifyFile(f('data', 'application/json')), 'text');
    },
  },
  {
    name: 'classifyFile recognizes code/text extensions without a MIME',
    run() {
      for (const name of ['main.py', 'app.tsx', 'index.js', 'style.css', 'query.sql', 'script.sh', 'config.yaml', 'data.csv', 'lib.rs', 'Main.java']) {
        assert.equal(fileAttachments.classifyFile(f(name, '')), 'text', `${name} should classify as text`);
      }
    },
  },
  {
    name: 'classifyFile recognizes special extensionless text filenames',
    run() {
      for (const name of ['Dockerfile', 'Makefile', 'LICENSE', '.gitignore', '.env']) {
        assert.equal(fileAttachments.classifyFile(f(name, '')), 'text', `${name} should classify as text`);
      }
    },
  },
  {
    name: 'classifyFile rejects binaries and PDFs',
    run() {
      assert.equal(fileAttachments.classifyFile(f('report.pdf', 'application/pdf')), 'unsupported');
      assert.equal(fileAttachments.classifyFile(f('archive.zip', 'application/zip')), 'unsupported');
      assert.equal(fileAttachments.classifyFile(f('movie.mp4', 'video/mp4')), 'unsupported');
      assert.equal(fileAttachments.classifyFile(f('model.gguf', '')), 'unsupported');
    },
  },
  {
    name: 'languageForFilename maps extensions to fence languages',
    run() {
      assert.equal(fileAttachments.languageForFilename('main.py'), 'python');
      assert.equal(fileAttachments.languageForFilename('App.tsx'), 'tsx');
      assert.equal(fileAttachments.languageForFilename('data.JSON'), 'json');
      assert.equal(fileAttachments.languageForFilename('Dockerfile'), 'docker');
      assert.equal(fileAttachments.languageForFilename('README'), 'markdown');
      assert.equal(fileAttachments.languageForFilename('unknown.qqq', 'text/plain'), 'text');
    },
  },
  {
    name: 'isProbablyBinaryText detects NUL and replacement chars',
    run() {
      assert.equal(fileAttachments.isProbablyBinaryText('hello world\nline two'), false);
      assert.equal(fileAttachments.isProbablyBinaryText('bad\u0000bytes'), true);
      assert.equal(fileAttachments.isProbablyBinaryText('bad\ufffdbytes'), true);
    },
  },
  {
    name: 'formatFileSize renders B/KB/MB',
    run() {
      assert.equal(fileAttachments.formatFileSize(512), '512 B');
      assert.equal(fileAttachments.formatFileSize(2048), '2.0 KB');
      assert.equal(fileAttachments.formatFileSize(5 * 1024 * 1024), '5.0 MB');
    },
  },
  {
    name: 'wrapFileForPrompt emits a fenced block with filename header',
    run() {
      const wrapped = fileAttachments.wrapFileForPrompt({
        id: 'file-1', filename: 'main.py', content: 'print("hi")', language: 'python', sizeBytes: 11,
      });
      assert.ok(wrapped.includes('Attached file: main.py'), 'header missing');
      assert.ok(wrapped.includes('```python'), 'language fence missing');
      assert.ok(wrapped.includes('print("hi")'), 'content missing');
    },
  },
  {
    name: 'wrapFileForPrompt lengthens the fence around backtick-heavy content',
    run() {
      const wrapped = fileAttachments.wrapFileForPrompt({
        id: 'file-2', filename: 'notes.md', content: 'nested ``` fence', language: 'markdown', sizeBytes: 16,
      });
      assert.ok(wrapped.includes('````markdown'), 'fence must outgrow the content backtick run');
      assert.ok(wrapped.includes('nested ``` fence'), 'content must stay intact');
      assert.ok(wrapped.endsWith('\n````'), 'closing fence must match the opening one');
    },
  },
  {
    name: 'createFileAttachmentId mints unique sequential ids',
    run() {
      const a = fileAttachments.createFileAttachmentId();
      const b = fileAttachments.createFileAttachmentId();
      assert.notEqual(a, b);
      assert.match(a, /^file-\d+$/);
    },
  },
  {
    name: 'convertContentForRequest passes strings through',
    run() {
      assert.equal(fileAttachments.convertContentForRequest('plain'), 'plain');
    },
  },
  {
    name: 'convertContentForRequest collapses text+file to a single string',
    run() {
      const converted = fileAttachments.convertContentForRequest([
        { type: 'text', text: 'review this' },
        { type: 'file', file: { id: 'file-3', filename: 'a.py', content: 'x=1', language: 'python', sizeBytes: 3 } },
      ]);
      assert.equal(typeof converted, 'string');
      assert.ok(converted.startsWith('review this'), 'user text must come first');
      assert.ok(converted.includes('Attached file: a.py'));
      assert.ok(converted.includes('x=1'));
    },
  },
  {
    name: 'convertContentForRequest keeps binary parts after merged text',
    run() {
      const converted = fileAttachments.convertContentForRequest([
        { type: 'text', text: 'look' },
        { type: 'file', file: { id: 'file-4', filename: 'b.md', content: '# hi', language: 'markdown', sizeBytes: 4 } },
        { type: 'image_url', image_url: { url: 'data:image/png;base64,AAA' } },
      ]);
      assert.ok(Array.isArray(converted));
      assert.equal(converted[0].type, 'text');
      assert.ok(converted[0].text.includes('Attached file: b.md'));
      assert.equal(converted[1].type, 'image_url');
      assert.ok(converted.every((part) => part.type !== 'file'), 'file parts must never reach the wire');
    },
  },
  {
    name: 'FILE_INPUT_ACCEPT covers text globs and code extensions',
    run() {
      const accept = fileAttachments.FILE_INPUT_ACCEPT;
      assert.ok(accept.includes('text/*'));
      assert.ok(accept.includes('.py'));
      assert.ok(accept.includes('.json'));
      assert.ok(!accept.includes('.pdf'), 'PDFs are out of scope for #1984');
    },
  },
  {
    name: 'panel wires ingestFiles into input, drop, and paste paths',
    run() {
      const source = readPanelSource();
      assert.ok(source.includes('ingestFiles(event.dataTransfer.files'), 'drop path must ingest files');
      assert.ok(source.includes('ingestFiles(event.clipboardData.files'), 'paste path must ingest files');
      assert.ok(source.includes('handleFileInputChange(e'), 'file input must route through the shared handler');
      assert.ok(source.includes('classifyFile(file)'), 'ingestion must auto-detect the file kind');
      assert.ok(source.includes('MAX_FILE_SIZE_BYTES'), 'size guard missing');
      assert.ok(source.includes('isProbablyBinaryText'), 'binary guard missing');
    },
  },
  {
    name: 'panel converts file parts before requests leave the renderer',
    run() {
      const source = readPanelSource();
      assert.ok(source.includes('convertContentForRequest(content)'), 'streaming request must convert file parts');
      assert.ok(source.includes('wrapFileForPrompt(item.file)'), 'collection loop must inline file content');
    },
  },
  {
    name: 'panel gates image/audio drop routing on model capability',
    run() {
      const source = readPanelSource();
      assert.ok(source.includes('vision-capable model'), 'dropped images must be rejected without a vision model');
      assert.ok(source.includes('Audio attachments are not supported'), 'dropped audio must be rejected without audio support');
    },
  },
  {
    name: 'preview chips key on stable attachment ids',
    run() {
      const previewSource = fs.readFileSync(
        path.join(appRoot, 'src', 'renderer', 'components', 'FilePreviewList.tsx'),
        'utf8',
      );
      assert.ok(previewSource.includes('key={file.id}'), 'FilePreviewList must key chips on file.id');
      assert.ok(readPanelSource().includes('id: createFileAttachmentId()'), 'panel must mint ids at ingestion');
    },
  },
];

module.exports = { tests };
