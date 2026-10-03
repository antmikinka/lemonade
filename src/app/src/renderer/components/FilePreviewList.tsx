import React from 'react';
import { UploadedFile } from '../utils/chatTypes';
import { formatFileSize } from '../utils/fileAttachments';
import { FileTextIcon } from './Icons';

interface FilePreviewListProps {
  files: UploadedFile[];
  onRemove?: (index: number) => void;
  className?: string;
}

const FilePreviewList: React.FC<FilePreviewListProps> = ({
  files,
  onRemove,
  className = 'file-preview-container',
}) => {
  if (files.length === 0) return null;

  return (
    <div className={className}>
      {files.map((file, index) => (
        <div key={index} className="file-preview-item" title={`${file.filename} (${formatFileSize(file.sizeBytes)})`}>
          <span className="file-preview-icon">
            <FileTextIcon />
          </span>
          <span className="file-preview-name">{file.filename}</span>
          <span className="file-preview-meta">{formatFileSize(file.sizeBytes)}</span>
          {file.language !== 'text' && (
            <span className="file-preview-lang">{file.language}</span>
          )}
          {onRemove && (
            <button
              className="file-remove-button"
              onClick={() => onRemove(index)}
              title="Remove file"
            >
              ×
            </button>
          )}
        </div>
      ))}
    </div>
  );
};

export default FilePreviewList;
