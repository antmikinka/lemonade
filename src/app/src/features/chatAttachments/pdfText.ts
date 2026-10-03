// Client-side PDF text extraction. pdfjs-dist runs in the webview (Tauri and
// browser builds alike), unlike Node-only parsers such as pdf-parse.

export const MAX_PDF_PAGES = 50;

export interface PdfExtractionResult {
  text: string;
  pages: number;
  truncated: boolean;
}

let pdfjsPromise: Promise<typeof import('pdfjs-dist')> | null = null;

function loadPdfjs(): Promise<typeof import('pdfjs-dist')> {
  if (!pdfjsPromise) {
    pdfjsPromise = import(/* webpackChunkName: "pdfjs" */ 'pdfjs-dist').then(pdfjs => {
      pdfjs.GlobalWorkerOptions.workerSrc = new URL(
        'pdfjs-dist/build/pdf.worker.min.mjs',
        import.meta.url,
      ).toString();
      return pdfjs;
    }).catch(err => {
      // A failed chunk fetch must not poison later attempts until page reload.
      pdfjsPromise = null;
      throw err;
    });
  }
  return pdfjsPromise;
}

export async function extractPdfText(file: File): Promise<PdfExtractionResult> {
  const pdfjs = await loadPdfjs();
  const data = new Uint8Array(await file.arrayBuffer());
  const doc = await pdfjs.getDocument({ data, isEvalSupported: false }).promise;
  try {
    const pageCount = Math.min(doc.numPages, MAX_PDF_PAGES);
    const pages: string[] = [];
    for (let i = 1; i <= pageCount; i += 1) {
      const page = await doc.getPage(i);
      try {
        const content = await page.getTextContent();
        const lines: string[] = [];
        let line = '';
        for (const item of content.items) {
          if (!('str' in item)) continue;
          line += item.str;
          if (item.hasEOL) {
            lines.push(line.trimEnd());
            line = '';
          }
        }
        if (line.trimEnd()) lines.push(line.trimEnd());
        pages.push(lines.filter(l => l.length > 0).join('\n'));
      } finally {
        page.cleanup();
      }
    }
    return {
      text: pages.join('\n\n').trim(),
      pages: doc.numPages,
      truncated: doc.numPages > MAX_PDF_PAGES,
    };
  } finally {
    await doc.destroy();
  }
}
