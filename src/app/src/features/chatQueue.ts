import type { AttachedFile } from './chatAttachments/fileAttachments';

export interface QueuedChatMessage {
  id: string;
  text: string;
  images?: string[];
  files?: AttachedFile[];
  audioFiles?: File[];
  audioName?: string;
}

export const MAX_QUEUED_MESSAGES = 10;

let queuedMessageCounter = 0;

export function createQueuedMessageId(): string {
  queuedMessageCounter += 1;
  return `queued-${Date.now().toString(36)}-${queuedMessageCounter}`;
}

export function withQueueCap(
  queue: QueuedChatMessage[],
  item: QueuedChatMessage,
): { kept: QueuedChatMessage[]; dropped: boolean } {
  if (queue.length >= MAX_QUEUED_MESSAGES) return { kept: queue, dropped: true };
  return { kept: [...queue, item], dropped: false };
}

export function summarizeQueuedItem(item: QueuedChatMessage): string {
  const text = item.text.trim();
  if (text) return text.length > 60 ? `${text.slice(0, 57)}…` : text;
  if (item.files?.length) return `File: ${item.files[0].filename}`;
  if (item.images?.length) {
    return item.images.length > 1 ? `Images (${item.images.length})` : 'Image';
  }
  if (item.audioFiles?.length) return `Audio: ${item.audioFiles[0].name}`;
  return 'Empty message';
}
