export const AUTO_TITLE_MAX_TOKENS = 24;
export const AUTO_TITLE_TEMPERATURE = 0.2;

const TITLE_SOURCE_CHARS = 400;
const MAX_AUTO_TITLE_CHARS = 60;

export interface AutoTitleMessage {
  role: 'system' | 'user';
  content: string;
}

function clip(text: string): string {
  const collapsed = text.replace(/\s+/g, ' ').trim();
  if (collapsed.length <= TITLE_SOURCE_CHARS) return collapsed;
  return `${collapsed.slice(0, TITLE_SOURCE_CHARS).trimEnd()}…`;
}

export function buildTitleRequest(userText: string, assistantText: string): AutoTitleMessage[] {
  return [
    {
      role: 'system',
      content: 'You name chat conversations. Reply with ONLY a title of 3 to 6 words. No quotes, no trailing punctuation, no explanation.',
    },
    {
      role: 'user',
      content: `User: ${clip(userText)}\nAssistant: ${clip(assistantText)}`,
    },
  ];
}

export function sanitizeAutoTitle(raw: string): string {
  // Fence markers unwrap rather than delete: a wrapped title lives inside them.
  const unfenced = raw.replace(/```/g, ' ');
  let title = unfenced
    .split('\n')
    .map(line => line.trim())
    .find(line => line) ?? '';
  const quoted = (value: string): boolean => (
    (value.startsWith('"') && value.endsWith('"'))
    || (value.startsWith('“') && value.endsWith('”'))
    || (value.startsWith("'") && value.endsWith("'"))
  );
  while (title.length >= 2 && quoted(title)) title = title.slice(1, -1).trim();
  title = title
    .replace(/^title\s*:\s*/i, '')
    .replace(/\s+/g, ' ')
    .trim()
    .replace(/[.!?]+$/, '')
    .slice(0, MAX_AUTO_TITLE_CHARS)
    .trimEnd();
  return title.length < 2 ? '' : title;
}
