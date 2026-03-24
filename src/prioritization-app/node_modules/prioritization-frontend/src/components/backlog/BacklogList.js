import { jsx as _jsx, jsxs as _jsxs, Fragment as _Fragment } from "react/jsx-runtime";
/**
 * Backlog List component for displaying a list of backlog items.
 * Supports drag and drop reordering.
 *
 * @module components/backlog/BacklogList
 */
import { useState, useCallback, useRef } from 'react';
import { BacklogItem } from './BacklogItem';
import { Card, Button } from '../common';
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
export function BacklogList({ title = 'Backlog', items, selectedItems = [], onItemSelect, onItemClick, onReorder, onAddItem, onClearAll, enableReorder = true, showDescriptions = true, showCategories = true, actions, className = '', emptyMessage = 'No items in backlog', limit, }) {
    const [draggedItem, setDraggedItem] = useState(null);
    const [dragOverIndex, setDragOverIndex] = useState(null);
    const dragOverElement = useRef(null);
    // Apply limit if specified
    const displayItems = limit ? items.slice(0, limit) : items;
    const handleDragStart = useCallback((item) => {
        setDraggedItem(item);
    }, []);
    const handleDragEnd = useCallback(() => {
        setDraggedItem(null);
        setDragOverIndex(null);
    }, []);
    const handleDragOver = useCallback((e, index) => {
        e.preventDefault();
        if (!enableReorder || !draggedItem)
            return;
        const currentItem = displayItems[index];
        if (currentItem.id !== draggedItem.id) {
            setDragOverIndex(index);
        }
    }, [enableReorder, draggedItem, displayItems]);
    const handleDrop = useCallback((e, toIndex) => {
        e.preventDefault();
        if (!enableReorder || !draggedItem || dragOverIndex === null)
            return;
        const fromIndex = displayItems.findIndex((item) => item.id === draggedItem.id);
        if (fromIndex !== -1 && fromIndex !== toIndex) {
            onReorder?.(fromIndex, toIndex);
        }
        setDraggedItem(null);
        setDragOverIndex(null);
    }, [enableReorder, draggedItem, dragOverIndex, displayItems, onReorder]);
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
    return (_jsxs(Card, { title: title, variant: "default", className: `backlog-list ${className}`, headerAction: _jsxs("div", { className: "backlog-list-header-actions", children: [items.length > 0 && (_jsxs(_Fragment, { children: [_jsx(Button, { variant: "secondary", size: "sm", onClick: handleSelectAll, className: "backlog-select-all", children: "Select All" }), _jsx(Button, { variant: "secondary", size: "sm", onClick: handleDeselectAll, className: "backlog-deselect-all", children: "Deselect All" })] })), onAddItem && (_jsx(Button, { variant: "primary", size: "sm", onClick: onAddItem, children: "Add Item" })), items.length > 0 && onClearAll && (_jsx(Button, { variant: "danger", size: "sm", onClick: onClearAll, children: "Clear All" })), actions] }), children: [items.length === 0 ? (_jsxs("div", { className: "backlog-list-empty", children: [_jsx("svg", { viewBox: "0 0 24 24", fill: "none", stroke: "currentColor", strokeWidth: "1.5", className: "backlog-empty-icon", "aria-hidden": "true", children: _jsx("path", { d: "M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" }) }), _jsx("p", { children: emptyMessage }), onAddItem && (_jsx(Button, { variant: "primary", onClick: onAddItem, children: "Add Your First Item" }))] })) : (_jsx("div", { className: "backlog-items-container", children: displayItems.map((item, index) => (_jsx("div", { ref: dragOverElement, className: `backlog-item-wrapper ${dragOverIndex === index ? 'backlog-drag-over' : ''}`, onDragOver: (e) => handleDragOver(e, index), onDrop: (e) => handleDrop(e, index), onDragLeave: handleDragLeave, children: _jsx(BacklogItem, { item: item, isSelected: selectedItems.includes(item.id), isDragging: draggedItem?.id === item.id, draggable: enableReorder, onSelect: onItemSelect, onClick: onItemClick, onDragStart: handleDragStart, onDragEnd: handleDragEnd, showDescription: showDescriptions, showCategory: showCategories }) }, item.id))) })), limit && items.length > limit && (_jsx("div", { className: "backlog-list-footer", children: _jsxs("p", { children: ["Showing ", displayItems.length, " of ", items.length, " items"] }) }))] }));
}
export default BacklogList;
