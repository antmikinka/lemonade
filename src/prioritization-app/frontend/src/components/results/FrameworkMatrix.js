import { jsx as _jsx, jsxs as _jsxs, Fragment as _Fragment } from "react/jsx-runtime";
import { Card } from '../common';
/**
 * Default quadrant labels for Value-Effort matrix.
 */
const VALUE_EFFORT_QUADRANTS = [
    'Fill In', // Low Value, Low Effort (bottom-left)
    'Avoid', // Low Value, High Effort (bottom-right)
    'Quick Win', // High Value, Low Effort (top-left)
    'Major Project', // High Value, High Effort (top-right)
];
/**
 * Default quadrant labels for Eisenhower matrix.
 */
const EISENHOWER_QUADRANTS = [
    'Eliminate', // Not Urgent, Not Important (bottom-left)
    'Delegate', // Urgent, Not Important (bottom-right)
    'Schedule', // Not Urgent, Important (top-left)
    'Do First', // Urgent, Important (top-right)
];
/**
 * Framework Matrix component for 2x2 visualizations.
 *
 * This component renders an interactive 2x2 matrix for visualizing
 * prioritization frameworks like Value-Effort or Eisenhower Matrix.
 *
 * @param props - FrameworkMatrix component props
 * @returns Rendered matrix component
 *
 * @example
 * ```tsx
 * <FrameworkMatrix
 *   type="value-effort"
 *   items={[
 *     { id: '1', title: 'Feature A', x: 3, y: 8 },
 *     { id: '2', title: 'Feature B', x: 7, y: 5 },
 *   ]}
 *   onItemClick={handleItemClick}
 * />
 * ```
 */
export function FrameworkMatrix({ type = 'value-effort', items, xLabel, yLabel, quadrantLabels, showGrid = true, showLabels = true, onItemClick, className = '', title, }) {
    // Determine default labels based on matrix type
    const defaultXLabel = type === 'value-effort' ? 'Effort' : 'Urgent';
    const defaultYLabel = type === 'value-effort' ? 'Value' : 'Important';
    const defaultQuadrants = type === 'value-effort'
        ? VALUE_EFFORT_QUADRANTS
        : EISENHOWER_QUADRANTS;
    const effectiveXLabel = xLabel || defaultXLabel;
    const effectiveYLabel = yLabel || defaultYLabel;
    const effectiveQuadrants = quadrantLabels || defaultQuadrants;
    return (_jsx(Card, { title: title || `${type === 'value-effort' ? 'Value vs Effort' : 'Eisenhower'} Matrix`, variant: "default", className: `framework-matrix ${className}`, children: _jsxs("div", { className: "matrix-container", children: [_jsxs("div", { className: "matrix-grid", children: [_jsxs("div", { className: "matrix-quadrants", children: [_jsx("div", { className: "matrix-quadrant matrix-quadrant-top-left", children: _jsx("span", { className: "matrix-quadrant-label", children: effectiveQuadrants[2] }) }), _jsx("div", { className: "matrix-quadrant matrix-quadrant-top-right", children: _jsx("span", { className: "matrix-quadrant-label", children: effectiveQuadrants[3] }) }), _jsx("div", { className: "matrix-quadrant matrix-quadrant-bottom-left", children: _jsx("span", { className: "matrix-quadrant-label", children: effectiveQuadrants[0] }) }), _jsx("div", { className: "matrix-quadrant matrix-quadrant-bottom-right", children: _jsx("span", { className: "matrix-quadrant-label", children: effectiveQuadrants[1] }) })] }), showGrid && (_jsxs(_Fragment, { children: [_jsx("div", { className: "matrix-grid-line matrix-grid-line-vertical" }), _jsx("div", { className: "matrix-grid-line matrix-grid-line-horizontal" })] })), showLabels && (_jsxs(_Fragment, { children: [_jsxs("div", { className: "matrix-axis-label matrix-axis-label-top", children: [effectiveYLabel, ": High"] }), _jsxs("div", { className: "matrix-axis-label matrix-axis-label-bottom", children: [effectiveYLabel, ": Low"] }), _jsxs("div", { className: "matrix-axis-label matrix-axis-label-left", children: [effectiveXLabel, ": Low"] }), _jsxs("div", { className: "matrix-axis-label matrix-axis-label-right", children: [effectiveXLabel, ": High"] })] })), _jsx("div", { className: "matrix-items", children: items.map((item) => {
                                // Normalize coordinates to percentage (0-100)
                                // Assuming x and y are on a 1-10 scale
                                const normalizedX = Math.min(100, Math.max(0, ((item.x - 1) / 9) * 100));
                                const normalizedY = Math.min(100, Math.max(0, ((item.y - 1) / 9) * 100));
                                return (_jsxs("div", { className: `matrix-item ${onItemClick ? 'matrix-item-clickable' : ''}`, style: {
                                        left: `${normalizedX}%`,
                                        bottom: `${normalizedY}%`,
                                        backgroundColor: item.color || 'var(--color-accent)',
                                    }, onClick: () => onItemClick?.(item), role: onItemClick ? 'button' : undefined, tabIndex: onItemClick ? 0 : undefined, "aria-label": `${item.title}: ${effectiveXLabel} ${item.x}, ${effectiveYLabel} ${item.y}`, children: [_jsx("span", { className: "matrix-item-dot" }), _jsx("span", { className: "matrix-item-label", children: item.title })] }, item.id));
                            }) })] }), items.length > 0 && (_jsx("div", { className: "matrix-legend", children: _jsxs("span", { className: "matrix-legend-title", children: ["Items: ", items.length] }) }))] }) }));
}
export default FrameworkMatrix;
