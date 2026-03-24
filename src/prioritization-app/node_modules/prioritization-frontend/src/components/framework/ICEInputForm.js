import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
/**
 * ICE Input Form component for the Prioritization Frameworks application.
 * Provides input sliders for Impact, Confidence, and Ease scoring.
 *
 * @module components/framework/ICEInputForm
 */
import { useState, useCallback } from 'react';
import { Card, Slider, Button } from '../common';
/**
 * ICE Input Form component.
 *
 * This component provides a complete form for collecting ICE scoring inputs:
 * - Impact: How much will this help? (1-10)
 * - Confidence: How confident are we? (1-10)
 * - Ease: How easy is this to implement? (1-10)
 *
 * ICE Score = Impact x Confidence x Ease
 *
 * @param props - ICEInputForm props
 * @returns Rendered ICE input form
 *
 * @example
 * ```tsx
 * <ICEInputForm
 *   value={{ impact: 7, confidence: 8, ease: 6 }}
 *   onChange={handleICEChange}
 *   onSubmit={handleSubmit}
 * />
 * ```
 */
export function ICEInputForm({ value = {}, onChange, onSubmit, isLoading = false, showSubmitButton = true, className = '', }) {
    const [localValue, setLocalValue] = useState(value);
    const currentValue = { ...value, ...localValue };
    const handleChange = useCallback((field) => (val) => {
        const updated = { ...currentValue, [field]: val };
        setLocalValue(updated);
        onChange?.(updated);
    }, [currentValue, onChange]);
    const handleSubmit = useCallback((e) => {
        e.preventDefault();
        if (currentValue.impact && currentValue.confidence && currentValue.ease) {
            onSubmit?.({
                impact: currentValue.impact,
                confidence: currentValue.confidence,
                ease: currentValue.ease,
            });
        }
    }, [currentValue, onSubmit]);
    // Calculate preview score
    const previewScore = currentValue.impact && currentValue.confidence && currentValue.ease
        ? (currentValue.impact * currentValue.confidence * currentValue.ease).toFixed(1)
        : null;
    return (_jsx(Card, { title: "ICE Scoring", variant: "default", className: `ice-input-form ${className}`, children: _jsxs("form", { onSubmit: handleSubmit, className: "framework-form", children: [_jsx("div", { className: "ice-intro", children: _jsx("p", { className: "ice-description", children: "ICE is a simplified prioritization framework that balances impact, confidence, and ease of implementation." }) }), _jsx(Slider, { label: "Impact", value: currentValue.impact ?? 5, onChange: handleChange('impact'), min: 1, max: 10, step: 1, showValue: true, helperText: "How much will this initiative help achieve your goals?", marks: [
                        { value: 1, label: 'Low' },
                        { value: 5, label: 'Medium' },
                        { value: 10, label: 'High' },
                    ] }), _jsx(Slider, { label: "Confidence", value: currentValue.confidence ?? 5, onChange: handleChange('confidence'), min: 1, max: 10, step: 1, showValue: true, helperText: "How confident are you in your impact estimate?", marks: [
                        { value: 1, label: 'Uncertain' },
                        { value: 5, label: 'Moderate' },
                        { value: 10, label: 'Certain' },
                    ] }), _jsx(Slider, { label: "Ease", value: currentValue.ease ?? 5, onChange: handleChange('ease'), min: 1, max: 10, step: 1, showValue: true, helperText: "How easy is this to implement? (Higher = easier)", marks: [
                        { value: 1, label: 'Hard' },
                        { value: 5, label: 'Moderate' },
                        { value: 10, label: 'Easy' },
                    ] }), previewScore && (_jsxs("div", { className: "score-preview", children: [_jsx("span", { className: "score-preview-label", children: "Estimated ICE Score:" }), _jsx("span", { className: "score-preview-value", children: previewScore }), _jsxs("span", { className: "score-preview-formula", children: ["(", currentValue.impact, " x ", currentValue.confidence, " x ", currentValue.ease, ")"] })] })), showSubmitButton && (_jsx("div", { className: "form-actions", children: _jsx(Button, { type: "submit", variant: "primary", isLoading: isLoading, disabled: !currentValue.impact || !currentValue.confidence || !currentValue.ease, children: "Calculate ICE Score" }) }))] }) }));
}
export default ICEInputForm;
