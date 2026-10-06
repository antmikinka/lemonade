// Chat file/PDF attachment contract checks.
//
// Attachments live inside ChatView (a 5k-line React component) plus the
// chatAttachments feature module, so this suite pins the behavioural
// contracts at the source level: classification, prompt folding, PDF
// extraction wiring, capability gating, persistence stripping, and UI chips.

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

const root = path.resolve(__dirname, '..');
const chatViewSource = fs.readFileSync(path.join(root, 'src/components/ChatView.tsx'), 'utf8');
const stylesSource = fs.readFileSync(path.join(root, 'src/styles/styles.css'), 'utf8');
const fileAttachmentsSource = fs.readFileSync(
  path.join(root, 'src/features/chatAttachments/fileAttachments.ts'), 'utf8');
const pdfTextSource = fs.readFileSync(
  path.join(root, 'src/features/chatAttachments/pdfText.ts'), 'utf8');
const packageJson = JSON.parse(fs.readFileSync(path.join(root, 'package.json'), 'utf8'));

// ── Classification module ──────────────────────────────────────────────────

assert.match(fileAttachmentsSource, /export type FileKind = 'image' \| 'audio' \| 'pdf' \| 'text' \| 'unsupported';/,
  'classifyFile must distinguish pdf, text, image, audio, and unsupported kinds');
assert.match(fileAttachmentsSource, /if \(mime === 'application\/pdf' \|\| extensionOf\(file\.name\) === 'pdf'\) return 'pdf';/,
  'PDF detection must cover both MIME and extension so mislabelled uploads still extract');
assert.match(fileAttachmentsSource, /export const MAX_FILE_SIZE_BYTES = 1024 \* 1024;/,
  'attachments must stay bounded at 1 MB');
assert.match(fileAttachmentsSource, /export const MAX_FILE_ATTACHMENTS = 4;/,
  'the number of concurrent document attachments must be capped');
assert.match(fileAttachmentsSource, /sample\.includes\('\\u0000'\) \|\| sample\.includes\('\\ufffd'\)/,
  'binary sniffing must reject NUL bytes and replacement characters');
assert.match(fileAttachmentsSource, /return '`'\.repeat\(Math\.max\(3, longest \+ 1\)\);/,
  'the fence must outgrow any backtick run inside the content or markdown files would escape it');
assert.match(fileAttachmentsSource, /const fence = fenceFor\(file\.content\);\s*return `Attached file: \$\{file\.filename\}\\n\$\{fence\}\$\{file\.language\}\\n\$\{file\.content\}\\n\$\{fence\}`;/,
  'wrapFileForPrompt must fence file content with its language using the dynamic fence');
assert.match(fileAttachmentsSource, /new TextDecoder\('utf-16le'\)/,
  'UTF-16 text files must decode instead of being rejected as binary');
assert.match(fileAttachmentsSource, /export function composePromptWithFiles\(text: string, files: AttachedFile\[\]\): string \{\s*return \[text\.trim\(\), \.\.\.files\.map\(wrapFileForPrompt\)\]\.filter\(Boolean\)\.join\('\\n\\n'\);/,
  'files must fold into plain prompt text so every chat backend accepts them');
assert.match(fileAttachmentsSource, /export const DOCUMENT_INPUT_ACCEPT = `\$\{FILE_INPUT_ACCEPT\},application\/pdf,\.pdf`;/,
  'the document picker must accept PDFs alongside text/code');

// ── PDF extraction module ──────────────────────────────────────────────────

assert.match(pdfTextSource, /import\(\/\* webpackChunkName: "pdfjs" \*\/ 'pdfjs-dist'\)/,
  'pdfjs-dist must load as a lazy chunk, never in the main bundle');
assert.match(pdfTextSource, /GlobalWorkerOptions\.workerSrc = new URL\(\s*'pdfjs-dist\/build\/pdf\.worker\.min\.mjs',\s*import\.meta\.url,\s*\)\.toString\(\);/,
  'the pdf worker must be bundled as an asset URL so extraction works offline in Tauri');
assert.match(pdfTextSource, /export const MAX_PDF_PAGES = 50;/,
  'PDF extraction must cap page count so huge documents cannot stall the composer');
assert.match(pdfTextSource, /await doc\.destroy\(\);/,
  'extracted documents must release their worker resources');
assert.match(pdfTextSource, /pdfjsPromise = null;\s*throw err;/,
  'a failed pdfjs chunk fetch must not poison later attempts');
assert.ok(packageJson.dependencies['pdfjs-dist'], 'pdfjs-dist must be a runtime dependency of the app');

// ── Web-app packaging (npm ci + Debian system-modules builds) ──────────────

const webAppRoot = path.join(root, '..', 'web-app');
const webAppWebpackSource = fs.readFileSync(path.join(webAppRoot, 'webpack.config.js'), 'utf8');
const webAppLock = JSON.parse(fs.readFileSync(path.join(webAppRoot, 'package-lock.json'), 'utf8'));

assert.ok(webAppLock.packages['node_modules/pdfjs-dist'],
  'the web-app lockfile must carry pdfjs-dist or npm ci fails in packaging builds');
assert.match(webAppWebpackSource, /'pdfjs-dist\$': path\.resolve\(__dirname, 'system-stubs\/pdfjs-dist\.ts'\)/,
  'system-modules builds must stub pdfjs-dist — Debian ships no v4 worker');
assert.match(webAppWebpackSource, /'pdfjs-dist\/build\/pdf\.worker\.min\.mjs\$': path\.resolve\(__dirname, 'system-stubs\/pdfjs-worker-stub\.mjs'\)/,
  'the pdf worker URL must also resolve to a stub in system-modules builds');
assert.ok(fs.existsSync(path.join(webAppRoot, 'system-stubs', 'pdfjs-dist.ts')),
  'the pdfjs-dist system stub must exist');
assert.ok(fs.existsSync(path.join(webAppRoot, 'system-stubs', 'pdfjs-worker-stub.mjs')),
  'the pdf worker stub asset must exist');

// ── Message model and persistence ──────────────────────────────────────────

assert.match(chatViewSource, /interface Message \{[\s\S]*?files\?: AttachedFile\[\];[\s\S]*?\}/,
  'Message must carry transient document attachments');
assert.match(chatViewSource, /images: undefined,\s*files: undefined,/,
  'saveConversations must strip file payloads so nothing leaks into localStorage');

// ── Capability gating ──────────────────────────────────────────────────────

assert.match(chatViewSource, /const acceptsFileAttachments = modeSupportsChatCompletions && currentCapability === 'chat';/,
  'document attachments must be limited to chat-completions mode');
assert.match(chatViewSource, /const canAttach = acceptsImageAttachments \|\| acceptsAudioAttachments \|\| acceptsFileAttachments;/,
  'the attach button must enable for document-capable models');
assert.match(chatViewSource, /const fileAccept = acceptsFileAttachments\s*\? \[mediaAccept, DOCUMENT_INPUT_ACCEPT\]\.filter\(Boolean\)\.join\(','\)\s*: mediaAccept;/,
  'the file input accept list must gain document types only when files are allowed');

// ── Attachment ingestion ───────────────────────────────────────────────────

assert.match(chatViewSource, /const documents = files\.filter\(isDocumentAttachment\);/,
  'addAttachments must route text/PDF uploads through the document path');
assert.match(chatViewSource, /if \(file\.size > MAX_FILE_SIZE_BYTES\) \{/,
  'oversized files must be rejected with an explanation');
assert.match(chatViewSource, /const \{ extractPdfText, MAX_PDF_PAGES \} = await import\(\s*\/\* webpackChunkName: "pdf-attachments" \*\/ '\.\.\/features\/chatAttachments\/pdfText'\s*\);/,
  'PDF extraction must load lazily only when a PDF is attached');
assert.match(chatViewSource, /has no extractable text \(it may be a scanned document\)/,
  'scanned PDFs must report why nothing was attached');
assert.match(chatViewSource, /if \(isProbablyBinaryText\(text\)\) \{/,
  'binary files disguised as text must be skipped');
assert.match(chatViewSource, /setPendingFiles\(prev => \[\.\.\.prev, \.\.\.accepted\]\.slice\(0, MAX_FILE_ATTACHMENTS\)\);/,
  'accepted documents must append up to the cap');
assert.match(chatViewSource, /if \(acceptsFileAttachments\) \{\s*const file = item\.getAsFile\(\);\s*if \(file && isDocumentAttachment\(file\)\) files\.push\(file\);/,
  'pasting documents from the OS clipboard must attach them');
assert.match(chatViewSource, /files = files\.filter\(f => !isDocumentAttachment\(f\) && classifyFile\(f\) !== 'unsupported'\);/,
  'a mixed drop must keep routing its images/audio after documents are extracted');
assert.match(chatViewSource, /if \(files\.length === 0\) return;\s*\}\s*\} else if \(documents\.length > 0 \|\| unsupported\.length > 0\) \{\s*\n[\s\S]*?not attachable in this mode\.`\);\s*\n\s*if \(files\.length === 0\) return;\s*\n\s*\}\s*\n\s*if \(isOpenMossTts/,
  'document-only drops must stop before the media routing branches, and modes without a document sink must name the rejected files');
assert.match(chatViewSource, /const text = decodeTextFile\(new Uint8Array\(await file\.arrayBuffer\(\)\)\);/,
  'text decoding must go through the BOM-sniffing helper');
assert.match(chatViewSource, /unsupported file type/,
  'unsupported drops must explain themselves instead of vanishing');

// ── Capability-switch and in-flight guards ─────────────────────────────────

assert.match(chatViewSource, /if \(currentCapability !== 'chat'\) \{\s*setPendingFiles\(\[\]\);\s*setFileAttachmentError\(null\);\s*\}/,
  'switching away from chat must clear document chips so they are never attached-but-not-sent');
assert.match(chatViewSource, /const hasFiles = pendingFiles\.length > 0 && acceptsFileAttachments;/,
  'handleSend must not smuggle stale documents into non-chat send paths');
assert.match(chatViewSource, /if \(!canSubmitContent \|\| isBusy \|\| isAttaching \|\| !currentModelSnapshot\) return;/,
  'sending mid-extraction must be blocked so the attachment cannot reappear after send');
assert.match(chatViewSource, /setIsAttaching\(true\);\s*try \{/,
  'extraction must flag itself in flight');
assert.match(chatViewSource, /\} finally \{\s*setIsAttaching\(false\);\s*\}/,
  'the in-flight flag must clear even when extraction throws');
assert.match(chatViewSource, /&& !acceptsFileAttachments\s*&& pendingImages\.length >= MAX_IMAGES;/,
  'a full image budget must not block document attachments');
assert.match(chatViewSource, /\? 'Images, text, and PDF files'/,
  'vision-only chat models must not be advertised as accepting audio');

// ── Request composition ────────────────────────────────────────────────────

assert.match(chatViewSource, /let requestText = files \? composePromptWithFiles\(text, files\) : text;/,
  'the outgoing prompt must fold attached files into plain text');
assert.match(chatViewSource, /const text = m\.files\?\.length \? composePromptWithFiles\(m\.content, m\.files\) : m\.content;/,
  'history replay must re-fold file content for earlier turns');
assert.match(chatViewSource, /files: hasFiles \? \[\.\.\.pendingFiles\] : undefined,/,
  'the stored user message must keep its attachments for retries');
assert.match(chatViewSource, /if \(documentFiles\.length > 0\) return `File: \$\{documentFiles\[0\]\.filename\}`\.slice\(0, 50\);/,
  'a file-only message must still produce a sensible conversation title');
assert.match(chatViewSource, /setPendingFiles\(\[\]\);\s*setFileAttachmentError\(null\);/,
  'sending must clear the pending documents and any error banner');

// ── Composer and transcript UI ─────────────────────────────────────────────

assert.match(chatViewSource, /aria-label="Document attachments"/,
  'the pending document chip list must be labelled for screen readers');
assert.match(chatViewSource, /aria-label=\{`Remove \$\{file\.filename\}`\}/,
  'each document chip must expose an accessible remove button');
assert.match(chatViewSource, /className="composer__file-error" role="status"/,
  'attachment errors must announce politely instead of failing silently');
assert.match(chatViewSource, /\{message\.files\?\.map\(\(file, i\) => \(/,
  'sent messages must show their attached documents in the transcript');
assert.match(chatViewSource, /'Text, code, and PDF files'/,
  'the Add files menu must advertise document support');

// ── Styling contracts ──────────────────────────────────────────────────────

assert.match(stylesSource, /\.composer__file-error \{[\s\S]*?color: var\(--danger\);/,
  'attachment errors must use the danger colour');

console.log('Chat file/PDF attachment contract checks passed.');
