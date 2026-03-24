import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
/**
 * RICE Input Form component for the Prioritization Frameworks application.
 * Provides input fields for Reach, Impact, Confidence, and Effort scoring.
 *
 * @module components/framework/RICEInputForm
 */
import { useState, useCallback } from 'react';
import { Card, Input, Slider, Button } from '../common';
/**
 * Impact level options for the select dropdown.
 */
const impactOptions = [
    { value: 3, label: '3 - Massive impact' },
    { value: 2, label: '2 - High impact' },
    { value: 1, label: '1 - Medium impact' },
    { value: 0.5, label: '0.5 - Low impact' },
    { value: 0.25, label: '0.25 - Minimal impact' },
];
/**
 * RICE Input Form component.
 *
 * This component provides a complete form for collecting RICE scoring inputs:
 * - Reach: Number of users/events affected
 * - Impact: Impact multiplier (standard scale)
 * - Confidence: Confidence level percentage
 * - Effort: Person-months or story points
 *
 * @param props - RICEInputForm props
 * @returns Rendered RICE input form
 *
 * @example
 * ```tsx
 * <RICEInputForm
 *   value={{ reach: 100, impact: 2, confidence: 80, effort: 3 }}
 *   onChange={handleRICEChange}
 *   onSubmit={handleSubmit}
 * />
 * ```
 */
export function RICEInputForm({ value = {}, onChange, onSubmit, isLoading = false, showSubmitButton = true, className = '', }) {
    const [localValue, setLocalValue] = useState(value);
    const currentValue = { ...value, ...localValue };
    const handleChange = useCallback((field) => (val) => {
        const newValue = typeof val === 'object' ? Number(val.target.value) : val;
        const updated = { ...currentValue, [field]: newValue };
        setLocalValue(updated);
        onChange?.(updated);
    }, [currentValue, onChange]);
    const handleSubmit = useCallback((e) => {
        e.preventDefault();
        if (currentValue.reach && currentValue.impact && currentValue.confidence && currentValue.effort) {
            onSubmit?.(currentValue);
        }
    }, [currentValue, onSubmit]);
    return (_jsx(Card, { title: "RICE Scoring", variant: "default", className: `rice-input-form ${className}`, children: _jsxs("form", { onSubmit: handleSubmit, className: "framework-form", children: [_jsxs("div", { className: "form-grid form-grid-2", children: [_jsx(Input, { label: "Reach", type: "number", value: currentValue.reach ?? '', onChange: handleChange('reach'), placeholder: "e.g., 100 users per month", helperText: "Number of users or events affected per period", min: 0, step: 1, fullWidth: true }), _jsxs("div", { className: "form-field", children: [_jsx("label", { className: "input-label", htmlFor: "rice-impact", children: "Impact" }), _jsxs("select", { id: "rice-impact", className: "select-field select-md", value: currentValue.impact ?? '', onChange: (e) => handleChange('impact')(Number(e.target.value)), children: [_jsx("option", { value: "", children: "Select impact level" }), impactOptions.map((opt) => (_jsx("option", { value: opt.value, children: opt.label }, opt.value)))] }), _jsx("p", { className: "input-helper-text", children: "Impact on each affected user" })] })] }), _jsx(Slider, { label: "Confidence", value: currentValue.confidence ?? 50, onChange: handleChange('confidence'), min: 0, max: 100, step: 5, showValue: true, valueFormat: "${value}%", helperText: "How confident are you in your estimates?", marks: [
                        { value: 0, label: 'Low' },
                        { value: 50, label: 'Medium' },
                        { value: 100, label: 'High' },
                    ] }), _jsx(Input, { label: "Effort", type: "number", value: currentValue.effort ?? '', onChange: handleChange('effort'), placeholder: "e.g., 3 person-months", helperText: "Estimated effort in person-months or story points", min: 0, step: 0.5, fullWidth: true }), showSubmitButton && (_jsx("div", { className: "form-actions", children: _jsx(Button, { type: "submit", variant: "primary", isLoading: isLoading, disabled: !currentValue.reach ||
                            !currentValue.impact ||
                            !currentValue.confidence ||
                            !currentValue.effort, children: "Calculate RICE Score" }) }))] }) }));
}
export default RICEInputForm;
