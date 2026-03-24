import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
/**
 * Backlog Item component for displaying draggable backlog items.
 *
 * @module components/backlog/BacklogItem
 */
import { useCallback } from 'react';
import { Badge } from '../common';
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
export function BacklogItem({ item, isSelected = false, isDragging = false, draggable = true, onSelect, onClick, onDragStart, onDragEnd, className = '', showDescription = true, showCategory = true, actions, }) {
    const handleSelect = useCallback(() => {
        onSelect?.(item);
    }, [item, onSelect]);
    const handleClick = useCallback(() => {
        onClick?.(item);
    }, [item, onClick]);
    const handleDragStartInternal = useCallback((e) => {
        if (draggable && onDragStart) {
            e.dataTransfer.setData('text/plain', item.id);
            e.dataTransfer.effectAllowed = 'move';
            onDragStart(item);
        }
    }, [draggable, item, onDragStart]);
    const handleDragEndInternal = useCallback(() => {
        onDragEnd?.();
    }, [onDragEnd]);
    const combinedClasses = `backlog-item ${isSelected ? 'backlog-item-selected' : ''} ${isDragging ? 'backlog-item-dragging' : ''} ${className}`.trim();
    return (_jsx("div", { className: combinedClasses, onClick: handleClick, onKeyDown: (e) => {
            if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                handleClick();
            }
        }, draggable: draggable, onDragStart: handleDragStartInternal, onDragEnd: handleDragEndInternal, role: "button", tabIndex: 0, "aria-selected": isSelected, children: _jsxs("div", { className: "backlog-item-content", children: [draggable && (_jsx("div", { className: "backlog-item-drag-handle", "aria-hidden": "true", children: _jsxs("svg", { viewBox: "0 0 24 24", fill: "currentColor", children: [_jsx("circle", { cx: "9", cy: "6", r: "1.5" }), _jsx("circle", { cx: "15", cy: "6", r: "1.5" }), _jsx("circle", { cx: "9", cy: "12", r: "1.5" }), _jsx("circle", { cx: "15", cy: "12", r: "1.5" }), _jsx("circle", { cx: "9", cy: "18", r: "1.5" }), _jsx("circle", { cx: "15", cy: "18", r: "1.5" })] }) })), _jsx("div", { className: "backlog-item-checkbox", children: _jsx("input", { type: "checkbox", checked: isSelected, onChange: handleSelect, onClick: (e) => e.stopPropagation(), "aria-label": `Select ${item.title}` }) }), _jsxs("div", { className: "backlog-item-info", children: [_jsx("h4", { className: "backlog-item-title", children: item.title }), showDescription && item.description && (_jsx("p", { className: "backlog-item-description", children: item.description })), _jsxs("div", { className: "backlog-item-meta", children: [showCategory && item.category && (_jsx(Badge, { variant: "default", size: "sm", children: item.category })), _jsxs("span", { className: "backlog-item-date", children: ["Created ", new Date(item.createdAt).toLocaleDateString()] })] })] }), actions && (_jsx("div", { className: "backlog-item-actions", children: actions }))] }) }));
}
export default BacklogItem;
