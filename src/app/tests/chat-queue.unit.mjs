// Behavioral unit tests for the pure queue helpers. Complements the
// source-contract suite; runs on Node >= 22.18 (native TypeScript stripping).

import assert from 'node:assert/strict';
import {
  MAX_QUEUED_MESSAGES,
  createQueuedMessageId,
  summarizeQueuedItem,
  withQueueCap,
} from '../src/features/chatQueue.ts';

// ── createQueuedMessageId ──────────────────────────────────────────────────

const first = createQueuedMessageId();
const second = createQueuedMessageId();
assert.match(first, /^queued-/, 'queued message ids must be namespaced');
assert.notEqual(first, second, 'ids minted in the same millisecond must still be unique');

// ── withQueueCap ───────────────────────────────────────────────────────────

assert.equal(MAX_QUEUED_MESSAGES, 10, 'the queue must stay bounded at 10 messages');

let queue = [];
for (let i = 0; i < MAX_QUEUED_MESSAGES; i += 1) {
  const result = withQueueCap(queue, { id: createQueuedMessageId(), text: `follow-up ${i}` });
  assert.equal(result.dropped, false, `slot ${i} must accept`);
  queue = result.kept;
}
assert.equal(queue.length, MAX_QUEUED_MESSAGES);

const overflow = withQueueCap(queue, { id: 'overflow', text: 'one too many' });
assert.equal(overflow.dropped, true, 'the 11th message must be rejected');
assert.equal(overflow.kept, queue, 'a rejection must return the original queue untouched');
assert.ok(!overflow.kept.some(item => item.id === 'overflow'));

const room = withQueueCap(queue.slice(0, 3), { id: 'fits', text: 'still fits' });
assert.equal(room.dropped, false);
assert.equal(room.kept.length, 4, 'accepted messages append in order');
assert.equal(room.kept[3].id, 'fits');

// ── summarizeQueuedItem ────────────────────────────────────────────────────

assert.equal(summarizeQueuedItem({ id: 'a', text: 'plain question' }), 'plain question');
assert.equal(
  summarizeQueuedItem({ id: 'a', text: `  ${'x'.repeat(80)}  ` }).length,
  58,
  'long drafts must truncate to a 57-char snippet plus ellipsis',
);
assert.ok(summarizeQueuedItem({ id: 'a', text: 'y'.repeat(80) }).endsWith('…'));
assert.equal(summarizeQueuedItem({ id: 'a', text: 'x'.repeat(60) }).length, 60, 'exactly 60 chars need no ellipsis');

assert.equal(
  summarizeQueuedItem({ id: 'a', text: '', files: [{ filename: 'alpha.txt', language: 'text', content: '', size: 1 }] }),
  'File: alpha.txt',
  'a file-only message must name the file',
);
assert.equal(summarizeQueuedItem({ id: 'a', text: '', images: ['data:1'] }), 'Image');
assert.equal(summarizeQueuedItem({ id: 'a', text: '', images: ['data:1', 'data:2'] }), 'Images (2)');
assert.equal(
  summarizeQueuedItem({ id: 'a', text: '', audioFiles: [{ name: 'voice.wav' }] }),
  'Audio: voice.wav',
);
assert.equal(summarizeQueuedItem({ id: 'a', text: '   ' }), 'Empty message');
assert.equal(
  summarizeQueuedItem({ id: 'a', text: 'text wins', files: [{ filename: 'f.md', language: 'markdown', content: '', size: 1 }] }),
  'text wins',
  'text takes precedence over attachment summaries',
);

console.log('Chat queue unit checks passed.');
