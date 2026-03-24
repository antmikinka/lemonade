/**
 * Backlog Item component for displaying draggable backlog items.
 *
 * @module components/backlog/BacklogItem
 */

import React, { useCallback } from 'react';
import { PrioritizationItem } from '../../services/prioritization/types';
import { Badge } from '../common';

/**
 * Props for the BacklogItem component.
 */
export interface BacklogItemProps {
  /** The backlog item to display */
  item: PrioritizationItem;
  /** Whether the item is selected */
  isSelected?: boolean;
  /** Whether the item is being dragged */
  isDragging?: boolean;
  /** Whether drag is enabled */
  draggable?: boolean;
  /** Item selection handler */
  onSelect?: (item: PrioritizationItem) => void;
  /** Item click handler */
  onClick?: (item: PrioritizationItem) => void;
  /** Drag start handler */
  onDragStart?: (item: PrioritizationItem) => void;
  /** Drag end handler */
  onDragEnd?: () => void;
  /** Custom class name */
  className?: string;
  /** Show item description */
  showDescription?: boolean;
  /** Show category badge */
  showCategory?: boolean;
  /** Custom action buttons */
  actions?: React.ReactNode;
}

/**
 * Backlog Item component for displaying individual backlog items.
 *
 * This component displays a single backlog item with support for:
 * - Selection state
 * - Drag and drop
 * - Category badges
 * - Custom actions
 *
 * @param props - BacklogItem component props
 * @returns Rendered backlog item
 *
 * @example
 * ```tsx
 * <BacklogItem
 *   item={{ id: '1', title: 'Feature A', description: 'Description', createdAt: new Date() }}
 *   isSelected={true}
 *   onSelect={handleSelect}
 *   onDragStart={handleDragStart}
 * />
 * ```
 */
export function BacklogItem({
  item,
  isSelected = false,
  isDragging = false,
  draggable = true,
  onSelect,
  onClick,
  onDragStart,
  onDragEnd,
  className = '',
  showDescription = true,
  showCategory = true,
  actions,
}: BacklogItemProps): React.JSX.Element {
  const handleSelect = useCallback(() => {
    onSelect?.(item);
  }, [item, onSelect]);

  const handleClick = useCallback(() => {
    onClick?.(item);
  }, [item, onClick]);

  const handleDragStartInternal = useCallback(
    (e: React.DragEvent) => {
      if (draggable && onDragStart) {
        e.dataTransfer.setData('text/plain', item.id);
        e.dataTransfer.effectAllowed = 'move';
        onDragStart(item);
      }
    },
    [draggable, item, onDragStart]
  );

  const handleDragEndInternal = useCallback(() => {
    onDragEnd?.();
  }, [onDragEnd]);

  const combinedClasses = `backlog-item ${isSelected ? 'backlog-item-selected' : ''} ${
    isDragging ? 'backlog-item-dragging' : ''
  } ${className}`.trim();

  return (
    <div
      className={combinedClasses}
      onClick={handleClick}
      onKeyDown={(e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          handleClick();
        }
      }}
      draggable={draggable}
      onDragStart={handleDragStartInternal}
      onDragEnd={handleDragEndInternal}
      role="button"
      tabIndex={0}
      aria-selected={isSelected}
    >
      <div className="backlog-item-content">
        {/* Drag Handle */}
        {draggable && (
          <div className="backlog-item-drag-handle" aria-hidden="true">
            <svg viewBox="0 0 24 24" fill="currentColor">
              <circle cx="9" cy="6" r="1.5" />
              <circle cx="15" cy="6" r="1.5" />
              <circle cx="9" cy="12" r="1.5" />
              <circle cx="15" cy="12" r="1.5" />
              <circle cx="9" cy="18" r="1.5" />
              <circle cx="15" cy="18" r="1.5" />
            </svg>
          </div>
        )}

        {/* Selection Checkbox */}
        <div className="backlog-item-checkbox">
          <input
            type="checkbox"
            checked={isSelected}
            onChange={handleSelect}
            onClick={(e) => e.stopPropagation()}
            aria-label={`Select ${item.title}`}
          />
        </div>

        {/* Item Info */}
        <div className="backlog-item-info">
          <h4 className="backlog-item-title">{item.title}</h4>
          {showDescription && item.description && (
            <p className="backlog-item-description">{item.description}</p>
          )}
          <div className="backlog-item-meta">
            {showCategory && item.category && (
              <Badge variant="default" size="sm">
                {item.category}
              </Badge>
            )}
            <span className="backlog-item-date">
              Created {new Date(item.createdAt).toLocaleDateString()}
            </span>
          </div>
        </div>

        {/* Actions */}
        {actions && (
          <div className="backlog-item-actions">{actions}</div>
        )}
      </div>
    </div>
  );
}

export default BacklogItem;
