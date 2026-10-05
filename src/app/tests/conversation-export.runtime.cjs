// Conversation export contract checks: copy-as-Markdown + download-.md.
//
// Export spans a pure feature module, the shared WorkspaceListRow, and the
// ChatView rail wiring, so this suite pins the behavioural contracts at the
// source level: serializer guarantees, row action stacking, clipboard and
// download plumbing, toast feedback, and attachment folding.

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

const root = path.resolve(__dirname, '..');
const exportSource = fs.readFileSync(
  path.join(root, 'src/features/chatHistory/conversationExport.ts'), 'utf8');
const chatViewSource = fs.readFileSync(path.join(root, 'src/components/ChatView.tsx'), 'utf8');
const workspacePanelsSource = fs.readFileSync(path.join(root, 'src/components/WorkspacePanels.tsx'), 'utf8');
const stylesSource = fs.readFileSync(path.join(root, 'src/styles/styles.css'), 'utf8');

// ── Export module ──────────────────────────────────────────────────────────

assert.match(exportSource, /export function conversationToMarkdown\(\s*convo: ExportableConversation,\s*exportedAt: Date = new Date\(\),\s*\): string \{/,
  'the serializer must accept an injectable timestamp so exports are testable and deterministic');
assert.match(exportSource, /exportedAt\.toISOString\(\)/,
  'export metadata must use an unambiguous ISO timestamp');
assert.match(exportSource, /const speaker = message\.modelName\?\.trim\(\) \|\| assistantLabel;/,
  'a message-level model must win over the conversation-level label so mid-conversation switches stay attributable');
assert.match(exportSource, /message\.role === 'user' \? '## You' : `## \$\{speaker\}`/,
  'sections must be headed by role, with the model name standing in for Assistant');
assert.match(exportSource, /_\(This reply failed\.\)_/,
  'failed replies must be visibly marked in the export');
assert.match(exportSource, /if \(!content && !message\.isError\) continue;/,
  'content-free messages must be skipped instead of exporting empty sections');
assert.match(exportSource, /_No messages to export\._/,
  'an empty conversation must still produce a valid document');
assert.match(exportSource, /const WINDOWS_RESERVED_STEMS = \/\^\(con\|prn\|aux\|nul\|com\[1-9\]\|lpt\[1-9\]\)\$\/i;/,
  'Windows reserved stems must be guarded or downloads would fail on Windows');
assert.match(exportSource, /WINDOWS_RESERVED_STEMS\.test\(stem\.split\('\.'\)\[0\]\)/,
  'the reserved check must run on the part before the first dot — Windows rejects "CON.md" too');
assert.match(exportSource, /\.replace\(\/\[\\\\\/:\*\?"<>\|\\u0000-\\u001f\]\+\/g, '-'\)/,
  'filenames must strip every path-invalid and control character');
assert.match(exportSource, /\.slice\(0, MAX_EXPORT_FILENAME_STEM\)/,
  'filename stems must stay bounded');
assert.match(exportSource, /if \(!stem\) return 'conversation\.md';/,
  'a fully-sanitized-away title must fall back to a stable filename');
assert.match(exportSource, /\.replace\(\/\^\[\.\\s…-\]\+\/, ''\)/,
  'a snippet truncation ellipsis must strip from the stem start');
assert.match(exportSource, /\.replace\(\/\[\.\\s…-\]\+\$\/, ''\);/,
  'a snippet truncation ellipsis must strip from the stem end or the .md reads as missing');
assert.match(exportSource, /new Blob\(\[markdown\], \{ type: 'text\/markdown;charset=utf-8' \}\)/,
  'downloads must be typed as UTF-8 markdown');
assert.match(exportSource, /window\.setTimeout\(\(\) => URL\.revokeObjectURL\(url\), 1000\);/,
  'the object URL must be revoked after the download starts');

// ── ChatView wiring ────────────────────────────────────────────────────────

assert.match(chatViewSource, /import \{\s*conversationExportFilename,\s*conversationToMarkdown,\s*downloadMarkdownFile,\s*type ExportableConversation,\s*\} from '\.\.\/features\/chatHistory\/conversationExport';/,
  'ChatView must consume the export module statically — it is small and dependency-free');
assert.match(chatViewSource, /function buildConversationExport\(c: Conversation\): ExportableConversation \{/,
  'export construction must be a pure module-level mapping');
assert.match(chatViewSource, /title: c\.title \|\| deriveTitle\(c\.messages\),/,
  'exports must use the same visible title as the rail row');
assert.match(chatViewSource, /content: m\.files\?\.length \? composePromptWithFiles\(m\.content, m\.files\) : m\.content,/,
  'user attachments must fold into the export exactly like history replay');
assert.match(chatViewSource, /modelName: m\.model\?\.name \|\| null,/,
  'per-message model snapshots must reach the serializer for mid-conversation model switches');
assert.match(chatViewSource, /const \[exportNotice, setExportNotice\] = useState<\{ message: string; error: boolean \} \| null>\(null\);/,
  'export feedback must live in a dismissible notice state');
assert.match(chatViewSource, /exportNoticeTimerRef\.current = window\.setTimeout\(\(\) => \{\s*exportNoticeTimerRef\.current = null;\s*setExportNotice\(null\);\s*\}, 3000\);/,
  'the notice must auto-clear so it never becomes stale chrome');
assert.match(chatViewSource, /if \(exportNoticeTimerRef\.current !== null\) window\.clearTimeout\(exportNoticeTimerRef\.current\);\s*\}, \[\]\);/,
  'unmounting mid-notice must clear the pending timer');
assert.match(chatViewSource, /await copyTextToClipboard\(conversationToMarkdown\(exportData\)\);/,
  'copy must go through the shared clipboard helper so plain-HTTP LAN deployments work');
assert.match(chatViewSource, /showExportNotice\('Could not copy the conversation to the clipboard', true\);/,
  'a rejected clipboard write must surface an error, not fail silently');
assert.match(chatViewSource, /const filename = conversationExportFilename\(exportData\.title\);\s*downloadMarkdownFile\(conversationToMarkdown\(exportData\), filename\);/,
  'download must derive the filename from the same title the copy path shows');
assert.match(chatViewSource, /showExportNotice\(`Downloaded \$\{filename\}`\);/,
  'a successful download must name the file it produced');
assert.match(chatViewSource, /extraActions=\{\[\s*\{\s*icon: 'copy',\s*label: `Copy conversation as Markdown: \$\{convTitle\}`,/,
  'every rail row must offer copy-as-Markdown');
assert.match(chatViewSource, /icon: 'download',\s*label: `Download conversation as Markdown: \$\{convTitle\}`,/,
  'every rail row must offer download-.md');
assert.match(chatViewSource, /onClick: \(\) => \{ void handleCopyConversationMarkdown\(c\); \},/,
  'the async copy handler must be explicitly voided in the action slot');
assert.match(chatViewSource, /className=\{`chat__toast\$\{exportNotice\.error \? ' chat__toast--error' : ''\}`\}\s*role="status"\s*aria-live="polite"/,
  'export feedback must announce politely to assistive tech');

// ── WorkspaceListRow extra actions ─────────────────────────────────────────

assert.match(workspacePanelsSource, /extraActions\?: WorkspaceListRowAction\[\];/,
  'WorkspaceListRow must accept stacked extra actions without disturbing the single-action API');
assert.match(workspacePanelsSource, /className=\{`workspace-list-row__action workspace-list-row__action--extra\$\{extra\.active \? ' workspace-list-row__action--active' : ''\}`\}/,
  'extras must reuse the row action treatment so hover/focus visibility comes for free');
assert.match(workspacePanelsSource, /style=\{\{ insetInlineEnd: `calc\(var\(--workspace-list-row-action\) \* \$\{extraActions\.length - index\}\)` \}\}/,
  'extras must stack leftward from the primary action using the shared width token');
assert.match(workspacePanelsSource, /onClick=\{event => \{ event\.stopPropagation\(\); extra\.onClick\(\); \}\}/,
  'clicking an extra must not select the row');
assert.match(workspacePanelsSource, /aria-label=\{extra\.label\}\s*title=\{extra\.label\}\s*tabIndex=\{selectable \? -1 : 0\}/,
  'extras must follow the listbox tab-stop contract of the primary action');
assert.match(workspacePanelsSource, /'--workspace-list-row-action-span': `calc\(var\(--workspace-list-row-action\) \* \$\{extraActions\.length \+ 1\}\)`/,
  'a row with extras must reserve title space for the whole action stack, not just the primary');
assert.match(workspacePanelsSource, /aria-keyshortcuts=\{selectable && index === 0 \? 'ArrowRight' : undefined\}/,
  'the ArrowRight hint belongs on the first action actually reached');
assert.match(workspacePanelsSource, /aria-keyshortcuts=\{selectable && !extraActions\?\.length \? 'ArrowRight' : undefined\}/,
  'the primary only advertises ArrowRight when no extra sits before it');

const extrasBeforePrimary = workspacePanelsSource.indexOf('extraActions?.map') < workspacePanelsSource.indexOf('{action && (action.pointerOnly ? (');
assert.ok(extrasBeforePrimary,
  'extras must render before the primary action so ArrowRight/Tab walk commands left to right');

// With every action at tabIndex=-1, Tab can never reach them — the list key
// handler is the only door, and it must open onto the whole stack.
const arrowRightWalksStack = /case 'ArrowRight': \{[\s\S]*?querySelectorAll<HTMLElement>\('button\.workspace-list-row__action'\)[\s\S]*?actions\[focused \+ 1\]\.focus\(\);/.test(workspacePanelsSource);
assert.ok(arrowRightWalksStack,
  'ArrowRight must walk rightward through every action in the row, stopping at the last');
const arrowLeftWalksBack = /case 'ArrowLeft': \{[\s\S]*?if \(focused > 0\) actions\[focused - 1\]\.focus\(\);\s*else options\[current\]\.focus\(\);/.test(workspacePanelsSource);
assert.ok(arrowLeftWalksBack,
  'ArrowLeft must walk back through the stack and land on the row itself');

// ── Styling contracts ──────────────────────────────────────────────────────

assert.match(stylesSource, /\.backends__toast,\s*\.manager__toast,\s*\.chat__toast \{\s*position: fixed;\s*bottom: var\(--space-6\);\s*left: 50%;/,
  'the export toast must join the established bottom-center pill toast block, not duplicate it');
assert.match(stylesSource, /\.chat__toast \{[\s\S]*?animation: toast-in var\(--duration-base\) var\(--ease-out\);/,
  'the export toast must reuse the shared toast-in animation');
assert.match(stylesSource, /\.chat__toast \{\s*max-width: min\(480px, calc\(100vw - var\(--space-8\)\)\);\s*\}/,
  'only the chat-specific width cap may live in a standalone rule');
assert.match(stylesSource, /\.chat__toast--error \{\s*color: var\(--danger\);/,
  'export failures must use the danger colour');
assert.match(stylesSource, /\.workspace-list-row__action--extra \{\s*border-radius: var\(--radius-md\);\s*\}/,
  'inboard extras must round all corners, unlike the edge-flush primary');
assert.match(stylesSource, /padding-inline-end: var\(--workspace-list-row-action-span, var\(--workspace-list-row-action\)\);/,
  'the title reservation must widen for stacked actions while defaulting to the single-action token');

console.log('Conversation export contract checks passed.');
