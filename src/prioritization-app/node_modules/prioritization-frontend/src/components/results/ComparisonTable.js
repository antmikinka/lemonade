import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { Card, Badge } from '../common';
/**
 * Comparison Table component for comparing multiple items.
 *
 * This component displays a sortable table comparing multiple prioritization
 * items with their scores, categories, and other details.
 *
 * @param props - ComparisonTable component props
 * @returns Rendered comparison table
 *
 * @example
 * ```tsx
 * <ComparisonTable
 *   items={[
 *     {
 *       id: '1',
 *       title: 'Feature A',
 *       description: 'Description A',
 *       results: { RICE: { score: 45.5, ... } },
 *     },
 *   ]}
 *   onItemSelected={handleSelect}
 *   selectedItemId="1"
 * />
 * ```
 */
export function ComparisonTable({ title = 'Comparison', items, columns, selectedItemId, onItemSelected, sortConfig, onSortChange, className = '', showRowNumbers = true, showDescriptions = true, emptyMessage = 'No items to compare', }) {
    // Default columns if not provided
    const defaultColumns = [
        {
            key: 'title',
            header: 'Item',
            width: '250px',
            render: (_, item) => (_jsxs("div", { className: "table-item-cell", children: [_jsx("span", { className: "table-item-title", children: item.title }), showDescriptions && item.description && (_jsx("span", { className: "table-item-description", children: item.description }))] })),
        },
        {
            key: 'score',
            header: 'Score',
            width: '100px',
            sortable: true,
            render: (_, item) => {
                const score = getScoreFromResults(item.results);
                return score !== null ? (_jsx("span", { className: "table-score", children: score.toFixed(2) })) : (_jsx("span", { className: "table-no-data", children: "-" }));
            },
        },
        {
            key: 'category',
            header: 'Category',
            width: '150px',
            render: (_, item) => {
                const category = getCategoryFromResults(item.results);
                return category ? (_jsx(Badge, { variant: getCategoryBadgeVariant(category), size: "sm", children: category })) : (_jsx("span", { className: "table-no-data", children: "-" }));
            },
        },
    ];
    const effectiveColumns = columns || defaultColumns;
    // Handle sort
    const handleSort = (column) => {
        if (!column.sortable || !onSortChange)
            return;
        const newDirection = sortConfig?.key === column.key && sortConfig.direction === 'desc'
            ? 'asc'
            : 'desc';
        onSortChange({ key: column.key, direction: newDirection });
    };
    return (_jsx(Card, { title: title, variant: "default", className: `comparison-table ${className}`, children: _jsx("div", { className: "table-container", children: items.length === 0 ? (_jsx("div", { className: "table-empty", children: _jsx("p", { children: emptyMessage }) })) : (_jsxs("table", { className: "table", children: [_jsx("thead", { className: "table-head", children: _jsxs("tr", { className: "table-row", children: [showRowNumbers && (_jsx("th", { className: "table-header table-header-number", children: "#" })), effectiveColumns.map((column) => (_jsxs("th", { className: `table-header ${column.sortable ? 'table-header-sortable' : ''}`, style: { width: column.width }, onClick: () => handleSort(column), role: column.sortable ? 'button' : undefined, tabIndex: column.sortable ? 0 : undefined, "aria-sort": sortConfig?.key === column.key
                                        ? sortConfig.direction === 'asc'
                                            ? 'ascending'
                                            : 'descending'
                                        : 'none', children: [column.header, column.sortable && (_jsx("span", { className: "table-sort-icon", children: sortConfig?.key === column.key ? (sortConfig.direction === 'asc' ? (_jsx("svg", { viewBox: "0 0 24 24", fill: "none", stroke: "currentColor", strokeWidth: "2", children: _jsx("path", { d: "M12 19V5M5 12l7-7 7 7" }) })) : (_jsx("svg", { viewBox: "0 0 24 24", fill: "none", stroke: "currentColor", strokeWidth: "2", children: _jsx("path", { d: "M12 5v14M5 12l7 7 7-7" }) }))) : (_jsx("svg", { viewBox: "0 0 24 24", fill: "none", stroke: "currentColor", strokeWidth: "2", children: _jsx("path", { d: "M7 15l5 5 5-5M7 9l5-5 5 5" }) })) }))] }, column.key)))] }) }), _jsx("tbody", { className: "table-body", children: items.map((item, index) => (_jsxs("tr", { className: `table-row ${selectedItemId === item.id ? 'table-row-selected' : ''}`, onClick: () => onItemSelected?.(item.id), role: onItemSelected ? 'button' : undefined, tabIndex: onItemSelected ? 0 : undefined, children: [showRowNumbers && (_jsx("td", { className: "table-cell table-cell-number", children: index + 1 })), effectiveColumns.map((column) => (_jsx("td", { className: "table-cell", style: { width: column.width }, children: column.render
                                        ? column.render(item[column.key], item)
                                        : item[column.key] }, column.key)))] }, item.id))) })] })) }) }));
}
/**
 * Extract score from results object.
 */
function getScoreFromResults(results) {
    if (!results)
        return null;
    // Try to get score from any available result
    const frameworks = [
        'RICE',
        'WSJF',
        'ICE',
        'P0P4',
        'MoSCoW',
        'Eisenhower',
        'ValueEffort',
        'Kano',
    ];
    for (const framework of frameworks) {
        const result = results[framework];
        if (result?.score !== undefined) {
            return result.score;
        }
    }
    return null;
}
/**
 * Extract category from results object.
 */
function getCategoryFromResults(results) {
    if (!results)
        return null;
    const frameworkCategories = {
        RICE: null,
        ICE: null,
        WSJF: null,
        MoSCoW: results.MoSCoW?.details?.category || null,
        Eisenhower: results.Eisenhower?.details?.quadrant || null,
        ValueEffort: results.ValueEffort?.details?.quadrant || null,
        P0P4: results.P0P4?.details?.priority || null,
        Kano: results.Kano?.details?.category || null,
    };
    for (const key of Object.keys(frameworkCategories)) {
        const category = frameworkCategories[key];
        if (category)
            return category;
    }
    return null;
}
/**
 * Get badge variant based on category.
 */
function getCategoryBadgeVariant(category) {
    const highPriorityCategories = [
        'Must have',
        'Do First',
        'Quick Win',
        'P0',
        'P1',
        'OneDimensional',
        'Attractive',
    ];
    const mediumPriorityCategories = [
        'Should have',
        'Schedule',
        'MajorProject',
        'P2',
        'MustBe',
    ];
    if (highPriorityCategories.some((c) => category.includes(c)))
        return 'success';
    if (mediumPriorityCategories.some((c) => category.includes(c)))
        return 'warning';
    return 'primary';
}
export default ComparisonTable;
