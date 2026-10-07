// Chat message-queue contract checks.
//
// The queue lets a busy composer enqueue follow-ups that drain one at a time
// whenever the conversation goes idle. The contracts that matter: enqueue
// never loses or double-sends a message, the drain resumes after every stream
// end (complete, error, manual stop), the queue is capped, and it stays
// transient — persistence and export must never see it.

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

const root = path.resolve(__dirname, '..');
const chatQueueSource = fs.readFileSync(path.join(root, 'src/features/chatQueue.ts'), 'utf8');
const chatViewSource = fs.readFileSync(path.join(root, 'src/components/ChatView.tsx'), 'utf8');
const stylesSource = fs.readFileSync(path.join(root, 'src/styles/styles.css'), 'utf8');
const exportSource = fs.readFileSync(
  path.join(root, 'src/features/chatHistory/conversationExport.ts'), 'utf8');
const packageSource = fs.readFileSync(path.join(root, 'package.json'), 'utf8');

// ── Queue module ───────────────────────────────────────────────────────────

assert.match(chatQueueSource, /import type \{ AttachedFile \} from '\.\/chatAttachments\/fileAttachments';/,
  'the queue module may only carry type imports so any bundler can inline it');
assert.match(chatQueueSource, /export const MAX_QUEUED_MESSAGES = 10;/,
  'the queue must stay bounded so a runaway paste cannot grow it forever');
assert.match(chatQueueSource, /if \(queue\.length >= MAX_QUEUED_MESSAGES\) return \{ kept: queue, dropped: true \};/,
  'a full queue must reject the new item and return the original queue untouched');
assert.match(chatQueueSource, /return `queued-\$\{Date\.now\(\)\.toString\(36\)\}-\$\{queuedMessageCounter\}`;/,
  'queued ids must stay unique within a millisecond via the session counter');

// ── Enqueue path ───────────────────────────────────────────────────────────

assert.match(chatViewSource, /if \(!canSubmitContent \|\| isAttaching \|\| !currentModelSnapshot\) return;/,
  'invalid or mid-extraction submissions must still hard-return, never enqueue');
assert.match(chatViewSource, /if \(isBusy \|\| \(activeId !== null && drainLockRef\.current\.has\(activeId\)\)\) \{/,
  'a busy composer — or a drain between items — must route the message into the queue');
assert.match(chatViewSource, /if \(currentCapability !== 'chat' \|\| !activeId\) return;/,
  'only chat completions with a conversation can queue; other modes keep the old drop behaviour');
assert.match(chatViewSource, /const \{ kept, dropped \} = withQueueCap\(queuedRef\.current\[activeId\] \|\| \[\], payload\);/,
  'enqueue must go through the capped helper');
assert.match(chatViewSource, /if \(dropped\) \{\s*\/\/ The draft stays in the composer — a full queue must not eat it\.\s*setQueueNotice\(`Queue is full — the limit is \$\{MAX_QUEUED_MESSAGES\} messages\. Send or remove a queued message first\.`\);\s*return;\s*\}/,
  'a rejected message must explain itself and stay in the composer as a draft');
assert.match(chatViewSource, /queuedRef\.current = \{ \.\.\.queuedRef\.current, \[activeId\]: kept \};\s*setQueued\(queuedRef\.current\);\s*setQueueNotice\(null\);\s*clearComposer\(\);\s*return;/,
  'ref and state must update together, then the composer clears for the next draft');

// ── Drain loop ─────────────────────────────────────────────────────────────

const drainBody = /const drainQueue = useCallback\(async \(convoId: string\) => \{[\s\S]*?\n  \}, \[\]\);/.exec(chatViewSource)?.[0] || '';
assert.ok(drainBody, 'drainQueue must exist as a stable ref-only callback');
assert.match(drainBody, /if \(drainLockRef\.current\.has\(convoId\)\) return;\s*drainLockRef\.current\.add\(convoId\);/,
  'the per-conversation lock must make overlapping drains impossible');
assert.match(drainBody, /if \(connectionRef\.current !== 'connected'\) break;/,
  'a lost server must pause the drain instead of burning the queue into error replies');
assert.match(drainBody, /const \[next, \.\.\.rest\] = queuedRef\.current\[convoId\] \|\| \[\];\s*if \(!next\) break;/,
  'the drain must be strictly FIFO and stop on an empty queue');
assert.match(drainBody, /try \{\s*await submitMessageRef\.current\(convoId, next\);\s*\} catch \{/,
  'one failing item must not strand the rest of the queue');
assert.match(drainBody, /\} finally \{[\s\S]*?drainLockRef\.current\.delete\(convoId\);\s*\}/,
  'the lock releases only when the loop exits, so an enqueue during the final stream is never orphaned');

assert.match(chatViewSource, /if \(!activeId \|\| isBusy \|\| connectionStatus !== 'connected'\) return;\s*if \(\(queued\[activeId\]\?\.length \|\| 0\) === 0\) return;\s*void drainQueue\(activeId\);\s*\}, \[activeId, isBusy, connectionStatus, queued, drainQueue\]\);/,
  'the idle effect must own drain kicks — isBusy flips false after completion, error, AND manual stop');

const stopBody = /const handleStop = useCallback\([\s\S]*?\}, \[/.exec(chatViewSource)?.[0] || '';
assert.ok(stopBody && !stopBody.includes('drainQueue'),
  'handleStop must not kick drains directly; the idle effect is the single drain trigger');

assert.match(chatViewSource, /if \(!activeId \|\| isBusy \|\| drainLockRef\.current\.has\(activeId\)\) return;/,
  'retry must treat a draining conversation as busy or it would double-stream');
assert.match(chatViewSource, /if \(!text \|\| !activeId \|\| isBusy \|\| drainLockRef\.current\.has\(activeId\)\) return;/,
  'editing a message must treat a draining conversation as busy too');

const deleteBody = /const handleDeleteConversation = useCallback\(\(id: string\) => \{[\s\S]*?\}, \[/.exec(chatViewSource)?.[0] || '';
assert.ok(deleteBody, 'handleDeleteConversation must exist');
assert.match(deleteBody, /drainLockRef\.current\.delete\(id\);\s*delete queuedRef\.current\[id\];/,
  'deleting a conversation must drop its queue and release its drain lock');

assert.match(chatViewSource, /if \(currentCapability === 'chat'\) return;\s*drainLockRef\.current\.clear\(\);\s*queuedRef\.current = \{\};\s*setQueued\(\{\}\);/,
  'losing chat capability must clear every queue — those messages could never drain');

// ── Transience: persistence and export must never see the queue ────────────

const conversationInterface = /interface Conversation \{[^}]*\}/.exec(chatViewSource)?.[0] || '';
assert.ok(conversationInterface, 'the Conversation interface must exist');
assert.ok(!/queue/i.test(conversationInterface),
  'the queue must never become a Conversation field — persistence and export stay untouched');
assert.doesNotMatch(chatViewSource, /queued\?:/,
  'no persisted model may gain an optional queue field');
assert.doesNotMatch(exportSource, /chatQueue|queued/i,
  'conversation export must know nothing about the queue');
assert.match(chatViewSource, /const queuedRef = useRef<Record<string, QueuedChatMessage\[\]>>\(\{\}\);/,
  'queue state must live beside the composer, not in the conversation store');

// ── Composer UI ────────────────────────────────────────────────────────────

assert.match(chatViewSource, /aria-label="Queued messages"/,
  'the queue strip must be discoverable for screen readers');
assert.match(chatViewSource, /aria-label="Clear queued messages"/,
  'clearing the whole queue must be a labelled action');
assert.match(chatViewSource, /aria-label=\{`Remove queued message \$\{index \+ 1\}`\}/,
  'each queued item must expose an accessible remove button');
assert.match(chatViewSource, /className="composer__queue-notice" role="status"/,
  'the cap-full rejection must announce politely');
assert.match(chatViewSource, /const queueAcceptingMode = modeSupportsChatCompletions && !!activeId;/,
  'queueing is only offered where drains can happen');
assert.match(chatViewSource, /disabled=\{isBusy && !queueAcceptingMode\}/,
  'the textarea must stay editable during a stream so follow-ups can be queued');
assert.match(chatViewSource, /disabled=\{isStreaming \? !canQueueSubmit : !canSubmit\}/,
  'Send must stay usable beside Stop while streaming');
assert.match(chatViewSource, /aria-label=\{isStreaming \? 'Queue message' : 'Send'\}/,
  'the send button must announce that it queues while a stream runs');

// ── Styling contracts ──────────────────────────────────────────────────────

assert.match(stylesSource, /\.composer__queue-notice \{[\s\S]*?color: var\(--danger\);/,
  'the queue-full notice must use the danger colour');
assert.match(stylesSource, /\.composer__queue-snippet \{[\s\S]*?text-overflow: ellipsis;/,
  'long queued drafts must truncate instead of blowing out the strip');

// ── Test wiring ────────────────────────────────────────────────────────────

assert.match(packageSource, /"test:chat-queue": "node tests\/chat-queue\.runtime\.cjs && node tests\/chat-queue\.unit\.mjs"/,
  'the queue suites must join the package scripts');

console.log('Chat queue contract checks passed.');
