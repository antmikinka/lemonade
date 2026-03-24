import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { Card, Badge } from '../common';
/**
 * Get a human-readable framework name.
 */
function getFrameworkName(framework) {
    const names = {
        RICE: 'RICE Score',
        MoSCoW: 'MoSCoW Category',
        ValueEffort: 'Value vs Effort',
        ICE: 'ICE Score',
        Eisenhower: 'Eisenhower Matrix',
        P0P4: 'Priority Level',
        WSJF: 'WSJF Score',
        Kano: 'Kano Category',
    };
    return names[framework] || framework;
}
/**
 * Get the appropriate badge variant based on score or category.
 */
function getBadgeVariant(framework, score, details) {
    if (framework === 'RICE' || framework === 'ICE' || framework === 'WSJF') {
        if (!score)
            return 'default';
        if (framework === 'RICE' && score >= 50)
            return 'success';
        if (framework === 'RICE' && score >= 20)
            return 'warning';
        if (framework === 'ICE' && score >= 500)
            return 'success';
        if (framework === 'ICE' && score >= 200)
            return 'warning';
        if (framework === 'WSJF' && score >= 10)
            return 'success';
        if (framework === 'WSJF' && score >= 5)
            return 'warning';
        return 'default';
    }
    if (framework === 'MoSCoW') {
        const category = details?.category;
        if (category === 'Must have')
            return 'error';
        if (category === 'Should have')
            return 'warning';
        if (category === 'Could have')
            return 'primary';
        return 'default';
    }
    if (framework === 'Eisenhower') {
        const quadrant = details?.quadrant;
        if (quadrant === 'Do First')
            return 'error';
        if (quadrant === 'Schedule')
            return 'warning';
        if (quadrant === 'Delegate')
            return 'primary';
        return 'default';
    }
    if (framework === 'P0P4') {
        const priority = details?.priority;
        if (priority === 'P0')
            return 'error';
        if (priority === 'P1')
            return 'warning';
        if (priority === 'P2')
            return 'primary';
        return 'default';
    }
    if (framework === 'ValueEffort') {
        const quadrant = details?.quadrant;
        if (quadrant === 'QuickWin')
            return 'success';
        if (quadrant === 'MajorProject')
            return 'warning';
        if (quadrant === 'FillIn')
            return 'primary';
        return 'default';
    }
    if (framework === 'Kano') {
        const category = details?.category;
        if (category === 'OneDimensional' || category === 'Attractive')
            return 'success';
        if (category === 'MustBe')
            return 'warning';
        return 'default';
    }
    return 'default';
}
/**
 * Results Card component for displaying prioritization results.
 *
 * This component displays the results of a prioritization calculation,
 * including the score, category/quadrant, and optional detailed breakdown.
 *
 * @param props - ResultsCard component props
 * @returns Rendered results card
 *
 * @example
 * ```tsx
 * <ResultsCard
 *   result={{
 *     framework: 'RICE',
 *     score: 45.5,
 *     details: { reach: 100, impact: 2, confidence: 0.8, effort: 3 },
 *   }}
 *   itemTitle="Add Dark Mode"
 *   showDetails
 * />
 * ```
 */
export function ResultsCard({ result, itemTitle, itemDescription, showDetails = true, className = '', actions, }) {
    const badgeVariant = getBadgeVariant(result.framework, result.score, result.details);
    const getCategoryDisplay = () => {
        const details = result.details;
        switch (result.framework) {
            case 'MoSCoW':
                return details.category || '';
            case 'Eisenhower':
                return details.quadrant || '';
            case 'ValueEffort':
                return details.quadrant || '';
            case 'P0P4':
                return details.priority || '';
            case 'Kano':
                return details.category || '';
            default:
                return '';
        }
    };
    const categoryDisplay = getCategoryDisplay();
    return (_jsxs(Card, { variant: "elevated", className: `results-card ${className}`, headerAction: actions, children: [_jsxs("div", { className: "results-header", children: [_jsx("div", { className: "results-framework", children: _jsx("span", { className: "results-framework-label", children: getFrameworkName(result.framework) }) }), categoryDisplay && (_jsx(Badge, { variant: badgeVariant, size: "lg", children: categoryDisplay }))] }), (itemTitle || result.score !== undefined) && (_jsxs("div", { className: "results-main", children: [itemTitle && _jsx("h4", { className: "results-title", children: itemTitle }), itemDescription && (_jsx("p", { className: "results-description", children: itemDescription })), result.score !== undefined && (_jsxs("div", { className: "results-score", children: [_jsx("span", { className: "results-score-value", children: result.score.toFixed(2) }), _jsx("span", { className: "results-score-label", children: "Score" })] }))] })), showDetails && Object.keys(result.details).length > 0 && (_jsxs("div", { className: "results-details", children: [_jsx("h5", { className: "results-details-title", children: "Breakdown" }), _jsx("dl", { className: "results-details-list", children: Object.entries(result.details).map(([key, value]) => {
                            if (typeof value === 'number') {
                                return (_jsxs("div", { className: "results-detail-item", children: [_jsx("dt", { className: "results-detail-label", children: formatLabel(key) }), _jsx("dd", { className: "results-detail-value", children: typeof value === 'number' && (value % 1 !== 0)
                                                ? value.toFixed(2)
                                                : value })] }, key));
                            }
                            return null;
                        }) })] }))] }));
}
/**
 * Format a camelCase key to a readable label.
 */
function formatLabel(key) {
    return key
        .replace(/([A-Z])/g, ' $1')
        .replace(/^./, (str) => str.toUpperCase());
}
export default ResultsCard;
