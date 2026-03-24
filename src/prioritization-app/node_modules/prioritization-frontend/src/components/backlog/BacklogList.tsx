/**
 * Backlog List component for displaying a list of backlog items.
 * Supports drag and drop reordering.
 *
 * @module components/backlog/BacklogList
 */

import React, { useState, useCallback, useRef } from 'react';
import { PrioritizationItem } from '../../services/prioritization/types';
import { BacklogItem } from './BacklogItem';
import { Card, Button } from '../common';

/**
 * Props for the BacklogList component.
 */
export interface BacklogListProps {
  /** List title */
  title?: string;
  /** Backlog items to display */
  items: PrioritizationItem[];
  /** Currently selected item IDs */
  selectedItems?: string[];
  /** Item selection handler */
  onItemSelect?: (item: PrioritizationItem) => void;
  /** Item click handler */
  onItemClick?: (item: PrioritizationItem) => void;
  /** Reorder handler (when item is dropped) */
  onReorder?: (fromIndex: number, toIndex: number) => void;
  /** Add new item handler */
  onAddItem?: () => void;
  /** Clear all items handler */
  onClearAll?: () => void;
  /** Enable drag and drop reordering */
  enableReorder?: boolean;
  /** Show item descriptions */
  showDescriptions?: boolean;
  /** Show category badges */
  showCategories?: boolean;
  /** Custom action buttons */
  actions?: React.ReactNode;
  /** Custom class name */
  className?: string;
  /** Empty state message */
  emptyMessage?: string;
  /** Maximum number of items to show */
  limit?: number;
}

/**
 * Backlog List component for displaying and managing backlog items.
 *
 * This component displays a list of backlog items with support for:
 * - Selection
 * - Drag and drop reordering
 * - Add/remove actions
 * - Empty state
 *
 * @param props - BacklogList component props
 * @returns Rendered backlog list
 *
 * @example
 * ```tsx
 * <BacklogList
 *   items={backlogItems}
 *   selectedItems={selectedIds}
 *   onItemSelect={handleSelect}
 *   onReorder={handleReorder}
 *   onAddItem={handleAdd}
 * />
 * ```
 */
export function BacklogList({
  title = 'Backlog',
  items,
  selectedItems = [],
  onItemSelect,
  onItemClick,
  onReorder,
  onAddItem,
  onClearAll,
  enableReorder = true,
  showDescriptions = true,
  showCategories = true,
  actions,
  className = '',
  emptyMessage = 'No items in backlog',
  limit,
}: BacklogListProps): React.JSX.Element {
  const [draggedItem, setDraggedItem] = useState<PrioritizationItem | null>(null);
  const [dragOverIndex, setDragOverIndex] = useState<number | null>(null);
  const dragOverElement = useRef<HTMLDivElement | null>(null);

  // Apply limit if specified
  const displayItems = limit ? items.slice(0, limit) : items;

  const handleDragStart = useCallback((item: PrioritizationItem) => {
    setDraggedItem(item);
  }, []);

  const handleDragEnd = useCallback(() => {
    setDraggedItem(null);
    setDragOverIndex(null);
  }, []);

  const handleDragOver = useCallback(
    (e: React.DragEvent, index: number) => {
      e.preventDefault();
      if (!enableReorder || !draggedItem) return;

      const currentItem = displayItems[index];
      if (currentItem.id !== draggedItem.id) {
        setDragOverIndex(index);
      }
    },
    [enableReorder, draggedItem, displayItems]
  );

  const handleDrop = useCallback(
    (e: React.DragEvent, toIndex: number) => {
      e.preventDefault();
      if (!enableReorder || !draggedItem || dragOverIndex === null) return;

      const fromIndex = displayItems.findIndex((item) => item.id === draggedItem.id);
      if (fromIndex !== -1 && fromIndex !== toIndex) {
        onReorder?.(fromIndex, toIndex);
      }

      setDraggedItem(null);
      setDragOverIndex(null);
    },
    [enableReorder, draggedItem, dragOverIndex, displayItems, onReorder]
  );

  const handleDragLeave = useCallback(() => {
    setDragOverIndex(null);
  }, []);

  const handleSelectAll = useCallback(() => {
    // Select all visible items
    displayItems.forEach((item) => {
      if (!selectedItems.includes(item.id)) {
        onItemSelect?.(item);
      }
    });
  }, [displayItems, selectedItems, onItemSelect]);

  const handleDeselectAll = useCallback(() => {
    // Deselect all items
    selectedItems.forEach((id) => {
      const item = displayItems.find((i) => i.id === id);
      if (item) {
        onItemSelect?.(item);
      }
    });
  }, [displayItems, selectedItems, onItemSelect]);

  return (
    <Card
      title={title}
      variant="default"
      className={`backlog-list ${className}`}
      headerAction={
        <div className="backlog-list-header-actions">
          {items.length > 0 && (
            <>
              <Button
                variant="secondary"
                size="sm"
                onClick={handleSelectAll}
                className="backlog-select-all"
              >
                Select All
              </Button>
              <Button
                variant="secondary"
                size="sm"
                onClick={handleDeselectAll}
                className="backlog-deselect-all"
              >
                Deselect All
              </Button>
            </>
          )}
          {onAddItem && (
            <Button variant="primary" size="sm" onClick={onAddItem}>
              Add Item
            </Button>
          )}
          {items.length > 0 && onClearAll && (
            <Button variant="danger" size="sm" onClick={onClearAll}>
              Clear All
            </Button>
          )}
          {actions}
        </div>
      }
    >
      {items.length === 0 ? (
        <div className="backlog-list-empty">
          <svg
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.5"
            className="backlog-empty-icon"
            aria-hidden="true"
          >
            <path d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
          </svg>
          <p>{emptyMessage}</p>
          {onAddItem && (
            <Button variant="primary" onClick={onAddItem}>
              Add Your First Item
            </Button>
          )}
        </div>
      ) : (
        <div className="backlog-items-container">
          {displayItems.map((item, index) => (
            <div
              key={item.id}
              ref={dragOverElement}
              className={`backlog-item-wrapper ${
                dragOverIndex === index ? 'backlog-drag-over' : ''
              }`}
              onDragOver={(e) => handleDragOver(e, index)}
              onDrop={(e) => handleDrop(e, index)}
              onDragLeave={handleDragLeave}
            >
              <BacklogItem
                item={item}
                isSelected={selectedItems.includes(item.id)}
                isDragging={draggedItem?.id === item.id}
                draggable={enableReorder}
                onSelect={onItemSelect}
                onClick={onItemClick}
                onDragStart={handleDragStart}
                onDragEnd={handleDragEnd}
                showDescription={showDescriptions}
                showCategory={showCategories}
              />
            </div>
          ))}
        </div>
      )}

      {limit && items.length > limit && (
        <div className="backlog-list-footer">
          <p>
            Showing {displayItems.length} of {items.length} items
          </p>
        </div>
      )}
    </Card>
  );
}

export default BacklogList;
