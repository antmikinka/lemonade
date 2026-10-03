// Conversation rail contract checks: manual rename and search filtering.
//
// The rail lives inside ChatView (a 5k-line React component), so this suite
// pins the behavioural contracts at the source level: custom-title
// persistence, auto-title guards, rename interaction, and search filtering.

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

const root = path.resolve(__dirname, '..');
const chatViewSource = fs.readFileSync(path.join(root, 'src/components/ChatView.tsx'), 'utf8');
const stylesSource = fs.readFileSync(path.join(root, 'src/styles/styles.css'), 'utf8');

// ── Custom titles survive storage round-trips ──────────────────────────────

assert.match(chatViewSource, /interface Conversation \{[\s\S]*?customTitle\?: boolean;[\s\S]*?\}/,
  'Conversation must carry a customTitle marker');
assert.match(chatViewSource, /customTitle: obj\.customTitle === true \? true : undefined/,
  'normalizeConversation must whitelist customTitle or reloads would drop renames');
assert.match(chatViewSource, /const stripped = convos\.map\(c => \(\{\s*\.\.\.c,/,
  'saveConversations must keep spreading the whole conversation so customTitle persists');

// ── Auto-titling never overwrites a manual rename ──────────────────────────

const firstMessageGuards = chatViewSource.match(/messages\.length === 0 && !\w+\.customTitle \? titleFromInput/g) || [];
assert.equal(firstMessageGuards.length, 2,
  'both first-message auto-title paths must check customTitle');
assert.match(chatViewSource, /title: messageIndex === 0 && !c\.customTitle \? titleFromInput/,
  'editing the first message must not clobber a manual rename');

// ── Rename interaction ─────────────────────────────────────────────────────

assert.match(chatViewSource, /const startRenameConversation = useCallback\(\(id: string, currentTitle: string\)/,
  'rename entry point must seed the draft from the current title');
assert.match(chatViewSource, /const commitRenameConversation = useCallback\(\(\) => \{[\s\S]*?renameDraft\.trim\(\)\.slice\(0, 120\)/,
  'commit must trim and bound the title length');
assert.match(chatViewSource, /c\.id === id && c\.title !== nextTitle\s*\? \{ \.\.\.c, title: nextTitle, customTitle: true \}/,
  'commit must set customTitle and leave updatedAt alone so renames do not reorder the rail');
assert.match(chatViewSource, /if \(!nextTitle\) return;/,
  'an empty rename must be discarded, not applied');
assert.match(chatViewSource, /const isRenaming = renamingId === c\.id;/,
  'rows must render the rename field in place of the title');
assert.match(chatViewSource, /className="rail__rename-input"[\s\S]*?onBlur=\{commitRenameConversation\}/,
  'leaving the rename field must commit the edit');
assert.match(chatViewSource, /if \(event\.key === 'Enter'\) \{\s*event\.preventDefault\(\);\s*commitRenameConversation\(\);\s*\} else if \(event\.key === 'Escape'\) \{\s*event\.preventDefault\(\);\s*cancelRenameConversation\(\);\s*\}/,
  'Enter commits and Escape cancels the rename');
assert.match(chatViewSource, /event\.stopPropagation\(\);\s*if \(event\.key === 'Enter'\)/,
  'rename keystrokes must not bubble into listbox row navigation');
assert.match(chatViewSource, /if \(event\.key === 'F2'\) \{[\s\S]*?startRenameConversation\(c\.id, convTitle\)/,
  'F2 on a focused row must start the rename');
assert.match(chatViewSource, /ariaKeyShortcuts="F2"/,
  'the F2 rename shortcut must be announced to assistive tech');
assert.match(chatViewSource, /onDoubleClick=\{event => \{\s*event\.stopPropagation\(\);\s*startRenameConversation\(c\.id, convTitle\);\s*\}\}/,
  'double-clicking the title must start the rename without reselecting');
assert.match(chatViewSource, /setRenamingId\(prev => \(prev === id \? null : prev\)\);/,
  'deleting a conversation mid-rename must clear the rename state');
assert.match(chatViewSource, /const closeMobileSheet = useCallback\(\(\) => \{\s*setMobileSheetOpen\(false\);\s*setRenamingId\(null\);/,
  'closing the mobile sheet must abandon an in-flight rename');
assert.match(chatViewSource, /maxLength=\{120\}/,
  'the rename field must surface the 120-char bound instead of truncating silently');
assert.match(chatViewSource, /selectable=\{!isRenaming\}/,
  'role="option" hides children from AT; the row must become a listitem while renaming');
assert.match(chatViewSource, /idx === 0 && \(!activeId \|\| !visibleConversations\.some\(v => v\.id === activeId\)\)/,
  'filtering out the active conversation must still leave the listbox keyboard-reachable');

// ── Search filtering ───────────────────────────────────────────────────────

assert.match(chatViewSource, /const \[railQuery, setRailQuery\] = useState\(''\);/,
  'the rail must own a search query state');
assert.match(chatViewSource, /const visibleConversations = useMemo\(\(\) => \{[\s\S]*?title\.includes\(q\) \|\| model\.includes\(q\)/,
  'search must match conversation title and model name, case-insensitively');
assert.match(chatViewSource, /const q = railQuery\.trim\(\)\.toLowerCase\(\);\s*if \(!q\) return conversations;/,
  'an empty query must short-circuit to the unfiltered list');

const filteredMaps = chatViewSource.match(/visibleConversations\.map\(\(c, idx\) =>/g) || [];
assert.equal(filteredMaps.length, 2,
  'both the desktop rail and the mobile bottom sheet must render the filtered list');

assert.match(chatViewSource, /aria-label="Search conversations"/,
  'the search field must be labelled for screen readers');
assert.match(chatViewSource, /No conversations match/,
  'a filtered-out rail must explain itself instead of rendering blank');
assert.match(chatViewSource, /aria-label="Clear conversation search"/,
  'the clear button must be labelled');
assert.match(chatViewSource, /if \(event\.key === 'Escape' && railQuery\) \{\s*event\.preventDefault\(\);[\s\S]*?event\.stopPropagation\(\);\s*setRailQuery\(''\);\s*\}/,
  'Escape in the search field must clear the query without closing the mobile sheet');

// ── Styling contracts ──────────────────────────────────────────────────────

assert.match(stylesSource, /\.rail__search-wrap \{[\s\S]*?border: 1px solid var\(--border-subtle\)/,
  'the search field must use the standard subtle border treatment');
assert.match(stylesSource, /\.rail__search-wrap:focus-within \{\s*border-color: var\(--accent-fg\);\s*\}/,
  'the search field must highlight on focus');
assert.match(stylesSource, /\.rail__rename-input \{[\s\S]*?font: inherit;/,
  'the rename field must inherit row typography so the title does not jump');
assert.match(stylesSource, /\.rail__title-text \{[\s\S]*?text-overflow: ellipsis;/,
  'long titles must ellipsize in the rail');
assert.match(stylesSource, /\.chat:not\(\.rail-expanded\) \.rail__list,\s*\.chat:not\(\.rail-expanded\) \.rail > \.rail__search-wrap,/,
  'the search field must hide with the collapsed rail but stay available in the mobile sheet');

console.log('Conversation rail rename/search contract checks passed.');
