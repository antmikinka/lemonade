import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
/**
 * Value vs Effort Input Form component for the Prioritization Frameworks application.
 * Provides input fields for Value vs Effort matrix scoring.
 *
 * @module components/framework/ValueEffortInputForm
 */
import { useState, useCallback } from 'react';
import { Card, Select, Button } from '../common';
/**
 * Value/Effort level options.
 */
const levelOptions = [
    { value: 'high', label: 'High' },
    { value: 'medium', label: 'Medium' },
    { value: 'low', label: 'Low' },
];
/**
 * Numeric mapping for value/effort levels.
 */
const levelToNumber = {
    high: 3,
    medium: 2,
    low: 1,
};
/**
 * Value vs Effort Input Form component.
 *
 * This component provides a complete form for collecting Value vs Effort inputs:
 * - Value: Business value score (High/Medium/Low)
 * - Effort: Implementation effort (High/Medium/Low)
 *
 * The combination determines the quadrant:
 * - High Value + Low Effort = Quick Win
 * - High Value + High Effort = Major Project
 * - Low Value + Low Effort = Fill In
 * - Low Value + High Effort = Avoid
 *
 * @param props - ValueEffortInputForm props
 * @returns Rendered Value vs Effort input form
 *
 * @example
 * ```tsx
 * <ValueEffortInputForm
 *   value={{ value: 3, effort: 2 }}
 *   onChange={handleValueEffortChange}
 *   onSubmit={handleSubmit}
 * />
 * ```
 */
export function ValueEffortInputForm({ value = {}, onChange, onSubmit, isLoading = false, showSubmitButton = true, className = '', }) {
    const [localValue, setLocalValue] = useState(value);
    const currentValue = { ...value, ...localValue };
    // Convert numeric values to string labels for selects
    const getValueLabel = (val) => {
        if (val === undefined)
            return '';
        if (val >= 7)
            return 'high';
        if (val >= 4)
            return 'medium';
        return 'low';
    };
    const getNumericValue = (label) => {
        return levelToNumber[label] || 0;
    };
    const handleChange = useCallback((field) => (val) => {
        const newValue = typeof val === 'object' ? val.target.value : val;
        const numericValue = getNumericValue(newValue);
        const updated = { ...currentValue, [field]: numericValue };
        setLocalValue(updated);
        onChange?.(updated);
    }, [currentValue, onChange]);
    const handleSubmit = useCallback((e) => {
        e.preventDefault();
        if (currentValue.value && currentValue.effort) {
            onSubmit?.({
                value: currentValue.value,
                effort: currentValue.effort,
            });
        }
    }, [currentValue, onSubmit]);
    return (_jsx(Card, { title: "Value vs Effort Matrix", variant: "default", className: `value-effort-input-form ${className}`, children: _jsxs("form", { onSubmit: handleSubmit, className: "framework-form", children: [_jsxs("div", { className: "form-grid form-grid-2", children: [_jsx(Select, { label: "Business Value", options: levelOptions, value: getValueLabel(currentValue.value), onChange: (e) => handleChange('value')(e.target.value), placeholder: "Select value level", helperText: "How much business value does this deliver?" }), _jsx(Select, { label: "Implementation Effort", options: levelOptions, value: getValueLabel(currentValue.effort), onChange: (e) => handleChange('effort')(e.target.value), placeholder: "Select effort level", helperText: "How much effort is required to implement?" })] }), currentValue.value && currentValue.effort && (_jsxs("div", { className: "quadrant-preview", children: [_jsx("span", { className: "quadrant-label", children: "Expected Quadrant:" }), _jsx("span", { className: `quadrant-badge quadrant-${getQuadrant(currentValue.value, currentValue.effort).toLowerCase()}`, children: getQuadrant(currentValue.value, currentValue.effort) })] })), showSubmitButton && (_jsx("div", { className: "form-actions", children: _jsx(Button, { type: "submit", variant: "primary", isLoading: isLoading, disabled: !currentValue.value || !currentValue.effort, children: "Plot on Matrix" }) }))] }) }));
}
/**
 * Determines the quadrant based on value and effort scores.
 */
function getQuadrant(value, effort) {
    const isHighValue = value >= 7;
    const isHighEffort = effort >= 7;
    if (isHighValue && !isHighEffort)
        return 'Quick Win';
    if (isHighValue && isHighEffort)
        return 'Major Project';
    if (!isHighValue && !isHighEffort)
        return 'Fill In';
    return 'Avoid';
}
export default ValueEffortInputForm;
