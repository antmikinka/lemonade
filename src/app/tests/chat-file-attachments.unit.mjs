// Behavioral unit tests for the pure attachment helpers. Complements the
// source-contract suite; runs on Node >= 22.18 (native TypeScript stripping).

import assert from 'node:assert/strict';
import {
  DOCUMENT_INPUT_ACCEPT,
  FILE_INPUT_ACCEPT,
  MAX_FILE_SIZE_BYTES,
  classifyFile,
  composePromptWithFiles,
  decodeTextFile,
  formatFileSize,
  isProbablyBinaryText,
  languageForFilename,
  wrapFileForPrompt,
} from '../src/features/chatAttachments/fileAttachments.ts';

// ── classifyFile ───────────────────────────────────────────────────────────

assert.equal(classifyFile({ name: 'report.pdf', type: 'application/pdf' }), 'pdf');
assert.equal(classifyFile({ name: 'REPORT.PDF', type: '' }), 'pdf', 'extension sniffing must be case-insensitive');
assert.equal(classifyFile({ name: 'scan.pdf', type: 'application/octet-stream' }), 'pdf', 'extension must win over a generic MIME');
assert.equal(classifyFile({ name: 'a.png', type: 'image/png' }), 'image');
assert.equal(classifyFile({ name: 'a.wav', type: 'audio/wav' }), 'audio');
assert.equal(classifyFile({ name: 'clip.mp4', type: 'video/mp4' }), 'unsupported');
assert.equal(classifyFile({ name: 'notes.md', type: '' }), 'text');
assert.equal(classifyFile({ name: 'Dockerfile', type: '' }), 'text');
assert.equal(classifyFile({ name: '.env', type: '' }), 'text');
assert.equal(classifyFile({ name: 'data', type: 'text/plain' }), 'text');
assert.equal(classifyFile({ name: 'archive.zip', type: 'application/zip' }), 'unsupported');
assert.equal(classifyFile({ name: 'tool.exe', type: '' }), 'unsupported', 'unknown extensions must not be read as text');

// ── languageForFilename ────────────────────────────────────────────────────

assert.equal(languageForFilename('main.py'), 'python');
assert.equal(languageForFilename('Dockerfile'), 'docker');
assert.equal(languageForFilename('README'), 'markdown');
assert.equal(languageForFilename('notes.unknown'), 'text');
assert.equal(languageForFilename('file.bin', 'text/markdown'), 'markdown');

// ── Binary sniffing ────────────────────────────────────────────────────────

assert.equal(isProbablyBinaryText('hello world'), false);
assert.equal(isProbablyBinaryText('before\u0000after'), true);
assert.equal(isProbablyBinaryText('bad\ufffdbyte'), true);

// ── decodeTextFile ─────────────────────────────────────────────────────────

const utf16le = new Uint8Array([0xff, 0xfe, ...Buffer.from('h\u00e9llo', 'utf16le')]);
assert.equal(decodeTextFile(utf16le), 'h\u00e9llo', 'UTF-16LE BOM must decode, not read as binary');
const utf16be = new Uint8Array([0xfe, 0xff, 0x00, 0x68, 0x00, 0x69]);
assert.equal(decodeTextFile(utf16be), 'hi');
assert.equal(decodeTextFile(new TextEncoder().encode('plain utf-8')), 'plain utf-8');

// ── Prompt wrapping ────────────────────────────────────────────────────────

const plain = wrapFileForPrompt({ filename: 'a.py', language: 'python', content: 'print(1)', size: 8 });
assert.equal(plain, 'Attached file: a.py\n```python\nprint(1)\n```');

const md = wrapFileForPrompt({ filename: 'R.md', language: 'markdown', content: 'use ```bash\nfence``` inside', size: 20 });
assert.ok(md.includes('````markdown'), 'content with a 3-backtick run must be fenced with 4 backticks');
assert.ok(md.endsWith('\n````'), 'the closing fence must match the opening fence length');
assert.equal(md.split('````').length - 1, 2, 'exactly one open and one close fence');

const plainFile = { filename: 'a.py', language: 'python', content: 'print(1)', size: 8 };
assert.equal(composePromptWithFiles('hi', []), 'hi');
assert.equal(composePromptWithFiles('', [plainFile]), plain);
assert.equal(composePromptWithFiles('hi', [plainFile]), `hi\n\n${plain}`);

// ── Formatting and accept lists ────────────────────────────────────────────

assert.equal(formatFileSize(512), '512 B');
assert.equal(formatFileSize(2048), '2.0 KB');
assert.equal(formatFileSize(2 * 1024 * 1024), '2.0 MB');
assert.equal(MAX_FILE_SIZE_BYTES, 1024 * 1024);
assert.ok(FILE_INPUT_ACCEPT.includes('.py') && FILE_INPUT_ACCEPT.startsWith('text/*'));
assert.ok(DOCUMENT_INPUT_ACCEPT.endsWith(',application/pdf,.pdf'));

console.log('Chat file attachment unit checks passed.');
