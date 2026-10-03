// Behavioral unit tests for the conversation export helpers. Complements the
// source-contract suite; runs on Node >= 22.18 (native TypeScript stripping).

import assert from 'node:assert/strict';
import {
  conversationExportFilename,
  conversationToMarkdown,
} from '../src/features/chatHistory/conversationExport.ts';

const exportedAt = new Date('2026-10-03T12:00:00.000Z');

// ── conversationToMarkdown ─────────────────────────────────────────────────

const basic = conversationToMarkdown({
  title: 'Trip planning',
  modelName: 'Llama-3.2-1B-Instruct',
  updatedAt: 0,
  messages: [
    { role: 'user', content: 'Where should I go?' },
    { role: 'assistant', content: 'Try **Kyoto**.' },
  ],
}, exportedAt);

assert.equal(basic, [
  '# Trip planning',
  '',
  '- Model: Llama-3.2-1B-Instruct',
  '- Exported: 2026-10-03T12:00:00.000Z',
  '',
  '---',
  '',
  '## You',
  '',
  'Where should I go?',
  '',
  '---',
  '',
  '## Llama-3.2-1B-Instruct',
  '',
  'Try **Kyoto**.',
  '',
].join('\n'), 'a modelled conversation must serialize with model metadata and role sections');

const anonymous = conversationToMarkdown({
  title: 'Quick question',
  modelName: null,
  updatedAt: 0,
  messages: [{ role: 'assistant', content: 'Hi!' }],
}, exportedAt);

assert.match(anonymous, /^# Quick question\n\n- Exported: /,
  'a conversation without a model must omit the Model metadata line');
assert.match(anonymous, /## Assistant\n\nHi!/,
  'assistant sections must fall back to a generic heading without a model name');

const withError = conversationToMarkdown({
  title: 'Broken run',
  modelName: 'Model-X',
  updatedAt: 0,
  messages: [
    { role: 'user', content: 'do a thing' },
    { role: 'assistant', content: 'Error: backend exploded\nsecond line', isError: true },
  ],
}, exportedAt);

assert.match(withError, /_\(This reply failed\.\)_\n\n> Error: backend exploded\n> second line/,
  'failed replies must be marked and blockquoted so they never read as real answers');

const emptyContent = conversationToMarkdown({
  title: 'Mostly empty',
  modelName: null,
  updatedAt: 0,
  messages: [
    { role: 'user', content: '   ' },
    { role: 'assistant', content: '' },
  ],
}, exportedAt);

assert.match(emptyContent, /_No messages to export\._/,
  'content-free conversations must degrade to an explicit empty marker');

const untitled = conversationToMarkdown({
  title: '   ',
  modelName: undefined,
  updatedAt: 0,
  messages: [{ role: 'user', content: 'hi' }],
}, exportedAt);

assert.match(untitled, /^# Untitled conversation\n/,
  'a blank title must fall back to a readable heading');

assert.equal(
  conversationToMarkdown({ title: 'T', updatedAt: 0, messages: [{ role: 'user', content: 'hi' }] }, exportedAt),
  conversationToMarkdown({ title: 'T', updatedAt: 0, messages: [{ role: 'user', content: 'hi' }] }, exportedAt),
  'exports must be deterministic for a fixed timestamp');

// ── conversationExportFilename ─────────────────────────────────────────────

assert.equal(conversationExportFilename('Trip planning'), 'Trip planning.md');
assert.equal(conversationExportFilename('a\\b/c:d*e?f"g<h>i|j'), 'a-b-c-d-e-f-g-h-i-j.md',
  'every Windows-invalid path character must become a dash');
assert.equal(conversationExportFilename('  ..--Weird  title--.. '), 'Weird title.md',
  'leading/trailing junk must strip and inner whitespace collapse');
assert.equal(conversationExportFilename('report.'), 'report.md',
  'a trailing dot must not survive into the filename');
assert.equal(conversationExportFilename('CON'), 'conversation-CON.md',
  'Windows reserved stems must be prefixed');
assert.equal(conversationExportFilename('lpt9'), 'conversation-lpt9.md');
assert.equal(conversationExportFilename(''), 'conversation.md', 'empty titles need a fallback');
assert.equal(conversationExportFilename('\u0000\u0001'), 'conversation.md',
  'control characters must not produce a filename');
assert.equal(conversationExportFilename('x'.repeat(200)).length, 83,
  'long titles must cap the stem at 80 characters plus .md');
assert.equal(conversationExportFilename('x'.repeat(200)), `${'x'.repeat(80)}.md`);

console.log('Conversation export unit checks passed.');
