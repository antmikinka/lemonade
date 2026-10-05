// Behavioral unit tests for the auto-title helpers. Complements the
// source-contract suite; runs on Node >= 22.18 (native TypeScript stripping).

import assert from 'node:assert/strict';
import {
  AUTO_TITLE_MAX_TOKENS,
  AUTO_TITLE_TEMPERATURE,
  buildTitleRequest,
  sanitizeAutoTitle,
} from '../src/features/chatHistory/conversationTitle.ts';

// ── sanitizeAutoTitle ──────────────────────────────────────────────────────

assert.equal(sanitizeAutoTitle('Kyoto Weekend Trip Planning'), 'Kyoto Weekend Trip Planning');
assert.equal(sanitizeAutoTitle('"Kyoto Weekend Trip"'), 'Kyoto Weekend Trip',
  'straight wrapping quotes must strip');
assert.equal(sanitizeAutoTitle('“Kyoto Weekend Trip”'), 'Kyoto Weekend Trip',
  'curly wrapping quotes must strip');
assert.equal(sanitizeAutoTitle("'Kyoto Weekend Trip'"), 'Kyoto Weekend Trip',
  'single wrapping quotes must strip');
assert.equal(sanitizeAutoTitle('```\nKyoto Trip\n```'), 'Kyoto Trip',
  'fenced code blocks must not leak into titles');
assert.equal(sanitizeAutoTitle('Title: Kyoto Weekend Trip'), 'Kyoto Weekend Trip',
  'a leading Title: label must strip');
assert.equal(sanitizeAutoTitle('title: Kyoto trip.'), 'Kyoto trip',
  'label and trailing period must both strip');
assert.equal(sanitizeAutoTitle('Kyoto trip!\nSecond line is ignored'), 'Kyoto trip',
  'only the first non-empty line may survive and exclamation must strip');
assert.equal(sanitizeAutoTitle('Kyoto   weekend\t trip'), 'Kyoto weekend trip',
  'inner whitespace must collapse');
assert.equal(sanitizeAutoTitle('Kyoto weekend trip...'), 'Kyoto weekend trip',
  'repeated trailing punctuation must strip');
assert.equal(sanitizeAutoTitle('x'.repeat(80)), 'x'.repeat(60),
  'overlong titles must cap at 60 characters');
assert.equal(sanitizeAutoTitle(''), '');
assert.equal(sanitizeAutoTitle('   '), '', 'whitespace-only output must reject');
assert.equal(sanitizeAutoTitle('\n\n'), '', 'empty-line output must reject');
assert.equal(sanitizeAutoTitle('A'), '', 'a single character is not a title');
assert.equal(sanitizeAutoTitle('```Kyoto Trip```'), 'Kyoto Trip',
  'an inline-fenced title must unwrap, not vanish');
assert.equal(sanitizeAutoTitle('""'), '', 'empty quotes are not a title');

// ── buildTitleRequest ──────────────────────────────────────────────────────

assert.equal(AUTO_TITLE_MAX_TOKENS, 24, 'the naming request must stay tiny');
assert.equal(AUTO_TITLE_TEMPERATURE, 0.2, 'naming must be near-deterministic');

const request = buildTitleRequest('Plan a trip to Kyoto', 'Day 1: Fushimi Inari.');

assert.deepEqual(request.map(m => m.role), ['system', 'user'],
  'the naming request must be a system instruction plus one evidence turn');
assert.match(request[0].content, /ONLY a title of 3 to 6 words/,
  'the system prompt must forbid prose around the title');
assert.match(request[1].content, /^User: Plan a trip to Kyoto\nAssistant: Day 1: Fushimi Inari\.$/,
  'both sides of the first exchange must reach the model');

const clipped = buildTitleRequest('x'.repeat(500), 'y'.repeat(500));
const [userLine, assistantLine] = clipped[1].content.split('\n');
assert.equal(userLine.length, 'User: '.length + 400 + 1,
  'the user excerpt must clip at 400 characters plus an ellipsis');
assert.equal(assistantLine.length, 'Assistant: '.length + 400 + 1,
  'the assistant excerpt must clip at 400 characters plus an ellipsis');

const folded = buildTitleRequest('line one\n\n  line two', 'reply');
assert.match(folded[1].content, /^User: line one line two\n/,
  'source whitespace must collapse so the naming prompt stays compact');

console.log('Conversation title unit checks passed.');
