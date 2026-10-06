// Text/code/PDF attachments for the chat composer. Files are folded into the
// user prompt as fenced plain text so every chat-completions backend accepts
// them without content-part support.

export interface AttachedFile {
  // Optional: conversations saved before attachments carried ids still load.
  id?: string;
  filename: string;
  language: string;
  content: string;
  size: number;
}

export const MAX_FILE_SIZE_BYTES = 1024 * 1024;
export const MAX_FILE_ATTACHMENTS = 4;

let nextAttachmentId = 0;

// Session counter rather than crypto.randomUUID: stable React keys without a
// platform dependency (webkit2gtk availability varies by distro).
export function createAttachmentId(): string {
  nextAttachmentId += 1;
  return `file-${nextAttachmentId}`;
}

const LANGUAGE_BY_EXTENSION: Record<string, string> = {
  txt: 'text',
  text: 'text',
  log: 'text',
  md: 'markdown',
  markdown: 'markdown',
  mdx: 'markdown',
  json: 'json',
  jsonc: 'json',
  jsonl: 'json',
  ndjson: 'json',
  yaml: 'yaml',
  yml: 'yaml',
  toml: 'toml',
  ini: 'ini',
  cfg: 'ini',
  conf: 'ini',
  env: 'ini',
  xml: 'xml',
  html: 'html',
  htm: 'html',
  xhtml: 'html',
  vue: 'html',
  svelte: 'html',
  css: 'css',
  scss: 'scss',
  sass: 'sass',
  less: 'less',
  js: 'javascript',
  mjs: 'javascript',
  cjs: 'javascript',
  jsx: 'jsx',
  ts: 'typescript',
  mts: 'typescript',
  cts: 'typescript',
  tsx: 'tsx',
  py: 'python',
  pyi: 'python',
  ipynb: 'json',
  rb: 'ruby',
  php: 'php',
  c: 'c',
  h: 'c',
  cpp: 'cpp',
  cxx: 'cpp',
  cc: 'cpp',
  hpp: 'cpp',
  hxx: 'cpp',
  cs: 'csharp',
  java: 'java',
  kt: 'kotlin',
  kts: 'kotlin',
  swift: 'swift',
  go: 'go',
  rs: 'rust',
  zig: 'zig',
  nim: 'nim',
  r: 'r',
  lua: 'lua',
  pl: 'perl',
  pm: 'perl',
  sh: 'bash',
  bash: 'bash',
  zsh: 'bash',
  fish: 'fish',
  bat: 'batch',
  cmd: 'batch',
  ps1: 'powershell',
  psm1: 'powershell',
  sql: 'sql',
  graphql: 'graphql',
  gql: 'graphql',
  proto: 'protobuf',
  tf: 'terraform',
  csv: 'csv',
  tsv: 'csv',
  tex: 'latex',
  bib: 'bibtex',
  dart: 'dart',
  ex: 'elixir',
  exs: 'elixir',
  erl: 'erlang',
  hs: 'haskell',
  clj: 'clojure',
  scala: 'scala',
  groovy: 'groovy',
  gradle: 'groovy',
  dockerfile: 'docker',
  makefile: 'makefile',
  mk: 'makefile',
  cmake: 'cmake',
  gitignore: 'text',
  gitattributes: 'text',
  editorconfig: 'ini',
  license: 'text',
  readme: 'markdown',
  notes: 'text',
};

// Filenames (lowercased, no path) that are text even without a known extension.
const SPECIAL_TEXT_FILENAMES = new Set([
  'dockerfile',
  'makefile',
  'license',
  'readme',
  'notice',
  'changelog',
  '.gitignore',
  '.gitattributes',
  '.editorconfig',
  '.env',
]);

const TEXT_MIME_PREFIXES = ['text/'];
const TEXT_MIME_TYPES = new Set([
  'application/json',
  'application/ld+json',
  'application/manifest+json',
  'application/xml',
  'application/rss+xml',
  'application/atom+xml',
  'application/javascript',
  'application/ecmascript',
  'application/typescript',
  'application/x-yaml',
  'application/yaml',
  'application/x-sh',
  'application/x-shellscript',
  'application/x-httpd-php',
  'application/sql',
  'application/graphql',
  'application/toml',
  'application/x-toml',
  'application/csv',
  'application/x-ndjson',
  'application/x-empty',
]);

export type FileKind = 'image' | 'audio' | 'pdf' | 'text' | 'unsupported';

function extensionOf(filename: string): string {
  const base = filename.toLowerCase();
  const dot = base.lastIndexOf('.');
  if (dot <= 0 || dot === base.length - 1) return '';
  return base.slice(dot + 1);
}

export function classifyFile(file: { name: string; type?: string }): FileKind {
  const mime = (file.type || '').toLowerCase();
  if (mime === 'application/pdf' || extensionOf(file.name) === 'pdf') return 'pdf';
  if (mime.startsWith('image/')) return 'image';
  if (mime.startsWith('audio/')) return 'audio';
  if (mime.startsWith('video/')) return 'unsupported';
  if (TEXT_MIME_PREFIXES.some((prefix) => mime.startsWith(prefix))) return 'text';
  if (mime && TEXT_MIME_TYPES.has(mime)) return 'text';

  const base = file.name.toLowerCase();
  if (SPECIAL_TEXT_FILENAMES.has(base)) return 'text';
  const ext = extensionOf(file.name);
  if (ext && LANGUAGE_BY_EXTENSION[ext]) return 'text';
  // Browsers report text/plain for unknown-but-readable files; anything with an
  // unrecognized extension and no MIME stays unsupported to avoid attaching
  // binaries that FileReader would mangle.
  return 'unsupported';
}

export function isDocumentAttachment(file: { name: string; type?: string }): boolean {
  const kind = classifyFile(file);
  return kind === 'text' || kind === 'pdf';
}

export function languageForFilename(filename: string, mime?: string): string {
  const base = filename.toLowerCase();
  if (SPECIAL_TEXT_FILENAMES.has(base)) {
    if (base === 'dockerfile') return 'docker';
    if (base === 'makefile') return 'makefile';
    if (base === 'readme') return 'markdown';
    return 'text';
  }
  const ext = extensionOf(filename);
  if (ext && LANGUAGE_BY_EXTENSION[ext]) return LANGUAGE_BY_EXTENSION[ext];
  if (mime === 'text/markdown') return 'markdown';
  if (mime === 'text/csv') return 'csv';
  if (mime === 'text/html') return 'html';
  if (mime === 'text/css') return 'css';
  return 'text';
}

export function isProbablyBinaryText(text: string): boolean {
  // NUL bytes never appear in the text formats we accept; U+FFFD means the
  // decoder already hit invalid byte sequences.
  const sample = text.slice(0, 8192);
  return sample.includes('\u0000') || sample.includes('\ufffd');
}

export function formatFileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function fenceFor(content: string): string {
  // Markdown files routinely contain ``` lines; a fence only survives if it is
  // longer than the longest backtick run inside the content.
  let longest = 0;
  for (const run of content.match(/`+/g) || []) {
    if (run.length > longest) longest = run.length;
  }
  return '`'.repeat(Math.max(3, longest + 1));
}

export function wrapFileForPrompt(file: AttachedFile): string {
  const fence = fenceFor(file.content);
  return `Attached file: ${file.filename}\n${fence}${file.language}\n${file.content}\n${fence}`;
}

export function decodeTextFile(bytes: Uint8Array): string {
  // UTF-16 text (PowerShell redirection, Notepad "Unicode") decodes to NUL
  // bytes under UTF-8 and would be mistaken for binary without the BOM check.
  if (bytes.length >= 2 && bytes[0] === 0xff && bytes[1] === 0xfe) {
    return new TextDecoder('utf-16le').decode(bytes.subarray(2));
  }
  if (bytes.length >= 2 && bytes[0] === 0xfe && bytes[1] === 0xff) {
    return new TextDecoder('utf-16be').decode(bytes.subarray(2));
  }
  return new TextDecoder('utf-8').decode(bytes);
}

export function composePromptWithFiles(text: string, files: AttachedFile[]): string {
  return [text.trim(), ...files.map(wrapFileForPrompt)].filter(Boolean).join('\n\n');
}

// Accept list for the attach-files input. Broad text globs plus explicit
// extensions so the OS picker shows plaintext/code files by default.
export const FILE_INPUT_ACCEPT = [
  'text/*',
  'application/json',
  'application/xml',
  'application/javascript',
  'application/typescript',
  'application/x-yaml',
  'application/yaml',
  'application/sql',
  'application/toml',
  ...Object.keys(LANGUAGE_BY_EXTENSION).map((ext) => `.${ext}`),
].join(',');

export const DOCUMENT_INPUT_ACCEPT = `${FILE_INPUT_ACCEPT},application/pdf,.pdf`;
