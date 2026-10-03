export interface ExportableMessage {
  role: 'user' | 'assistant';
  content: string;
  isError?: boolean;
}

export interface ExportableConversation {
  title: string;
  modelName?: string | null;
  updatedAt: number;
  messages: ExportableMessage[];
}

// Windows rejects these as file stems regardless of extension, and the title
// comes from user input or model output.
const WINDOWS_RESERVED_STEMS = /^(con|prn|aux|nul|com[1-9]|lpt[1-9])$/i;

const MAX_EXPORT_FILENAME_STEM = 80;

function blockquote(text: string): string {
  return text
    .split('\n')
    .map(line => (line ? `> ${line}` : '>'))
    .join('\n');
}

export function conversationToMarkdown(
  convo: ExportableConversation,
  exportedAt: Date = new Date(),
): string {
  const assistantLabel = convo.modelName?.trim() || 'Assistant';
  const sections: string[] = [];

  for (const message of convo.messages) {
    const content = message.content.trim();
    if (!content && !message.isError) continue;
    const heading = message.role === 'user' ? '## You' : `## ${assistantLabel}`;
    if (message.isError) {
      sections.push(`${heading}\n\n_(This reply failed.)_\n\n${blockquote(content)}`);
    } else {
      sections.push(`${heading}\n\n${content}`);
    }
  }

  const metadata = convo.modelName?.trim()
    ? `- Model: ${convo.modelName.trim()}\n- Exported: ${exportedAt.toISOString()}`
    : `- Exported: ${exportedAt.toISOString()}`;
  const body = sections.length ? sections.join('\n\n---\n\n') : '_No messages to export._';

  return `# ${convo.title.trim() || 'Untitled conversation'}\n\n${metadata}\n\n---\n\n${body}\n`;
}

export function conversationExportFilename(title: string): string {
  const stem = title
    .replace(/[\\/:*?"<>|\u0000-\u001f]+/g, '-')
    .replace(/\s+/g, ' ')
    .replace(/^[.\s-]+/, '')
    .slice(0, MAX_EXPORT_FILENAME_STEM)
    .replace(/[.\s-]+$/, '');
  if (!stem) return 'conversation.md';
  if (WINDOWS_RESERVED_STEMS.test(stem)) return `conversation-${stem}.md`;
  return `${stem}.md`;
}

export function downloadMarkdownFile(markdown: string, filename: string): void {
  const url = URL.createObjectURL(new Blob([markdown], { type: 'text/markdown;charset=utf-8' }));
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = filename;
  anchor.style.display = 'none';
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}
