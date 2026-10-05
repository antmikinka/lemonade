// Auto-title wiring contract checks: the naming request must be exactly-once,
// race-safe against manual renames, and silent on failure.

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

const root = path.resolve(__dirname, '..');
const titleSource = fs.readFileSync(
  path.join(root, 'src/features/chatHistory/conversationTitle.ts'), 'utf8');
const chatViewSource = fs.readFileSync(path.join(root, 'src/components/ChatView.tsx'), 'utf8');
const packageSource = fs.readFileSync(path.join(root, 'package.json'), 'utf8');
const workflowSource = fs.readFileSync(
  path.join(root, '../../.github/workflows/gui3-renderer-tests.yml'), 'utf8');

// ── Title module ───────────────────────────────────────────────────────────

assert.doesNotMatch(titleSource, /^import |require\(/m,
  'the title module must stay dependency-free so the Debian web-app build can inline it');
assert.match(titleSource, /export const AUTO_TITLE_MAX_TOKENS = 24;/,
  'the naming request must stay bounded to a few tokens');
assert.match(titleSource, /export const AUTO_TITLE_TEMPERATURE = 0\.2;/,
  'naming must be near-deterministic');
assert.match(titleSource, /role: 'system',/,
  'the instruction must ride in a system turn');
assert.match(titleSource, /const quoted = \(value: string\): boolean =>/,
  'quote stripping must cover straight, curly, and single quotes');

// ── ChatView wiring ────────────────────────────────────────────────────────

assert.match(chatViewSource, /import \{\s*AUTO_TITLE_MAX_TOKENS,\s*AUTO_TITLE_TEMPERATURE,\s*buildTitleRequest,\s*sanitizeAutoTitle,\s*\} from '\.\.\/features\/chatHistory\/conversationTitle';/,
  'ChatView must consume the title module statically — it is small and dependency-free');
assert.match(chatViewSource, /const pendingAutoTitleRef = useRef<Map<string, string>>\(new Map\(\)\);/,
  'first-exchange detection must ride in a ref because handleStreamDone cannot read conversation state');
assert.match(chatViewSource, /setActiveId\(convoId\);\s*pendingAutoTitleRef\.current\.set\(convoId, userMessage\.content\);/,
  'a brand-new conversation must arm the seed at send time');
assert.match(chatViewSource, /if \(currentMessages\.length === 0 && !existingConvo\?\.customTitle\) \{\s*pendingAutoTitleRef\.current\.set\(convoId, userMessage\.content\);\s*\}/,
  'an existing conversation must arm only when this send is its first exchange and no manual title exists');
assert.match(chatViewSource, /if \(messageIndex === 0 && !convo\.customTitle\) \{\s*pendingAutoTitleRef\.current\.set\(activeId, text\);\s*\}/,
  'editing the first message must re-arm the seed with the revised text');
assert.match(chatViewSource, /const autoTitleSeed = pendingAutoTitleRef\.current\.get\(convoId\);\s*pendingAutoTitleRef\.current\.delete\(convoId\);\s*if \(autoTitleSeed !== undefined && model\?\.name && assistantContent\) \{\s*void runAutoTitle\(convoId, autoTitleSeed, assistantContent, model\.name\);\s*\}/,
  'the seed must be consumed get+delete on stream success only, and only with a real model and reply');
assert.match(chatViewSource, /const raw = await api\.chatCompletionOnce\(modelName, buildTitleRequest\(userText, assistantText\), \{\s*max_tokens: AUTO_TITLE_MAX_TOKENS,\s*temperature: AUTO_TITLE_TEMPERATURE,[\s\S]*?enable_thinking: false,\s*\}\);/,
  'the naming request must override sampling with the tiny bounded params and disable thinking');
assert.match(chatViewSource, /updateConversation\(convoId, c => \(c\.customTitle \? c : \{ \.\.\.c, title \}\)\);/,
  'a manual rename landing mid-flight must always win the apply');
assert.match(chatViewSource, /\} catch \{\s*\/\/ A failed naming request keeps the derived snippet title\.\s*\}/,
  'naming failures must be silent — the derived title is the fallback');
assert.match(chatViewSource, /delete streamModelsRef\.current\[id\];\s*pendingAutoTitleRef\.current\.delete\(id\);/,
  'deleting a conversation must drop its stale seed');

const streamErrorBody = /const handleStreamError = useCallback\([\s\S]*?\}, \[appendAssistantMessage\]\);/.exec(chatViewSource)?.[0] || '';
assert.ok(streamErrorBody && !streamErrorBody.includes('pendingAutoTitleRef'),
  'stream errors must not consume the seed so a later retry can still title');
const stopBody = /const handleStop = useCallback\([\s\S]*?\}, \[/.exec(chatViewSource)?.[0] || '';
assert.ok(stopBody && !stopBody.includes('pendingAutoTitleRef'),
  'manual stop must not consume the seed either');

// ── Test + CI wiring ───────────────────────────────────────────────────────

assert.match(packageSource, /"test:conversation-title": "node tests\/conversation-title\.runtime\.cjs && node tests\/conversation-title\.unit\.mjs"/,
  'the title suites must join the package scripts');
assert.match(workflowSource, /npm run test:conversation-title/,
  'the fork CI gate must run the title suites');

console.log('Conversation title contract checks passed.');
