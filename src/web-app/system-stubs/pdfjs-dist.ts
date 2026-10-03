// System-package build stub: Debian/Ubuntu do not ship a pdfjs-dist new enough
// for the v4 worker layout (build/pdf.worker.min.mjs), so distro web-app builds
// degrade gracefully instead of offering PDF text extraction. The rejection
// surfaces through ChatView's attachment error banner.

export const GlobalWorkerOptions: { workerSrc: string } = { workerSrc: '' };

export function getDocument(): { promise: Promise<never> } {
  return {
    promise: Promise.reject(
      new Error('PDF attachments are not supported in this build'),
    ),
  };
}

export default { GlobalWorkerOptions, getDocument };
