import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { Card, Badge } from '../common';
/**
 * Priority Ranking component for displaying sorted results.
 *
 * This component displays a ranked list of prioritization items,
 * showing their position, score, and optional trend indicators.
 *
 * @param props - PriorityRanking component props
 * @returns Rendered priority ranking
 *
 * @example
 * ```tsx
 * <PriorityRanking
 *   items={[
 *     { id: '1', title: 'Feature A', rank: 1, score: 95.5, category: 'Must have' },
 *     { id: '2', title: 'Feature B', rank: 2, score: 82.3, category: 'Should have' },
 *   ]}
 *   onItemSelected={handleSelect}
 *   highlightTop={3}
 * />
 * ```
 */
export function PriorityRanking({ title = 'Priority Ranking', items, framework, selectedItemId, onItemSelected, limit, showRankChange = false, showScores = true, showCategories = true, className = '', emptyMessage = 'No items to rank', highlightTop = 3, }) {
    // Apply limit if specified
    const displayItems = limit ? items.slice(0, limit) : items;
    return (_jsxs(Card, { title: title, variant: "default", className: `priority-ranking ${className}`, children: [items.length === 0 ? (_jsx("div", { className: "ranking-empty", children: _jsx("p", { children: emptyMessage }) })) : (_jsx("div", { className: "ranking-list", children: displayItems.map((item, index) => (_jsxs("div", { className: `ranking-item ${selectedItemId === item.id ? 'ranking-item-selected' : ''} ${index < highlightTop ? 'ranking-item-highlighted' : ''}`, onClick: () => onItemSelected?.(item.id), role: onItemSelected ? 'button' : undefined, tabIndex: onItemSelected ? 0 : undefined, children: [_jsxs("div", { className: "ranking-rank", children: [_jsx("span", { className: `ranking-rank-number ${index < highlightTop ? 'ranking-rank-highlight' : ''}`, style: { color: item.color }, children: item.rank }), showRankChange && item.rankChange && (_jsxs("span", { className: `ranking-rank-change ${item.rankChange > 0 ? 'ranking-rank-up' : 'ranking-rank-down'}`, children: [item.rankChange > 0 ? (_jsx("svg", { viewBox: "0 0 24 24", fill: "none", stroke: "currentColor", strokeWidth: "2", children: _jsx("path", { d: "M18 15l-6-6-6 6" }) })) : (_jsx("svg", { viewBox: "0 0 24 24", fill: "none", stroke: "currentColor", strokeWidth: "2", children: _jsx("path", { d: "M6 9l6 6 6-6" }) })), Math.abs(item.rankChange)] }))] }), _jsxs("div", { className: "ranking-info", children: [_jsx("h4", { className: "ranking-title", children: item.title }), item.description && (_jsx("p", { className: "ranking-description", children: item.description })), _jsxs("div", { className: "ranking-meta", children: [framework && (_jsx("span", { className: "ranking-framework", children: getFrameworkName(framework) })), showCategories && item.category && (_jsx(Badge, { variant: getCategoryBadgeVariant(item.category), size: "sm", children: item.category }))] })] }), showScores && (_jsxs("div", { className: "ranking-score", children: [_jsx("span", { className: "ranking-score-value", children: item.score !== undefined ? item.score.toFixed(1) : '-' }), _jsx("span", { className: "ranking-score-label", children: "pts" })] }))] }, item.id))) })), limit && items.length > limit && (_jsx("div", { className: "ranking-footer", children: _jsxs("p", { children: ["Showing ", displayItems.length, " of ", items.length, " items"] }) }))] }));
}
/**
 * Get framework display name.
 */
function getFrameworkName(framework) {
    const names = {
        RICE: 'RICE',
        MoSCoW: 'MoSCoW',
        ValueEffort: 'Value vs Effort',
        ICE: 'ICE',
        Eisenhower: 'Eisenhower',
        P0P4: 'P0-P4',
        WSJF: 'WSJF',
        Kano: 'Kano',
    };
    return names[framework] || framework;
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
export default PriorityRanking;
