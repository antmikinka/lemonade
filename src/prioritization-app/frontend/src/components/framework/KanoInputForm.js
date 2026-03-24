import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
/**
 * Kano Input Form component for the Prioritization Frameworks application.
 * Provides input fields for Kano model analysis with functional/dysfunctional questions.
 *
 * @module components/framework/KanoInputForm
 */
import { useState, useCallback } from 'react';
import { Card, Slider, Button } from '../common';
/**
 * Rating labels for the Kano questions.
 */
const ratingLabels = ['Dislike', "Don't Like", 'Neutral', 'Like', 'Like Very Much'];
/**
 * Kano Input Form component.
 *
 * This component provides a complete form for collecting Kano model inputs:
 * - Functional score: How much do you LIKE having this feature?
 * - Dysfunctional score: How much do you LIKE NOT having this feature?
 * - Importance rating
 * - Satisfaction if present / Dissatisfaction if absent
 *
 * @param props - KanoInputForm props
 * @returns Rendered Kano input form
 *
 * @example
 * ```tsx
 * <KanoInputForm
 *   value={{ functionalScore: 4, dysfunctionalScore: 2, importance: 5 }}
 *   onChange={handleKanoChange}
 *   onSubmit={handleSubmit}
 * />
 * ```
 */
export function KanoInputForm({ value = {}, onChange, onSubmit, isLoading = false, showSubmitButton = true, className = '', }) {
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
        if (currentValue.functionalScore && currentValue.dysfunctionalScore) {
            onSubmit?.({
                functionalScore: currentValue.functionalScore,
                dysfunctionalScore: currentValue.dysfunctionalScore,
                importance: currentValue.importance,
                satisfactionIfPresent: currentValue.satisfactionIfPresent,
                dissatisfactionIfAbsent: currentValue.dissatisfactionIfAbsent,
            });
        }
    }, [currentValue, onSubmit]);
    return (_jsx(Card, { title: "Kano Model Analysis", variant: "default", className: `kano-input-form ${className}`, children: _jsxs("form", { onSubmit: handleSubmit, className: "framework-form", children: [_jsx("div", { className: "kano-intro", children: _jsx("p", { className: "kano-description", children: "The Kano model evaluates features from the customer perspective. Answer the questions below to classify this feature." }) }), _jsxs("div", { className: "kano-question", children: [_jsx("h4", { className: "kano-question-title", children: "Functional Question" }), _jsxs("p", { className: "kano-question-text", children: ["How do you feel about ", _jsx("strong", { children: "having" }), " this feature?"] }), _jsx(Slider, { label: "Rating", value: currentValue.functionalScore ?? 3, onChange: handleChange('functionalScore'), min: 1, max: 5, step: 1, showValue: true, marks: ratingLabels.map((label, index) => ({
                                value: index + 1,
                                label,
                            })) })] }), _jsxs("div", { className: "kano-question", children: [_jsx("h4", { className: "kano-question-title", children: "Dysfunctional Question" }), _jsxs("p", { className: "kano-question-text", children: ["How do you feel about ", _jsx("strong", { children: "not having" }), " this feature?"] }), _jsx(Slider, { label: "Rating", value: currentValue.dysfunctionalScore ?? 3, onChange: handleChange('dysfunctionalScore'), min: 1, max: 5, step: 1, showValue: true, marks: ratingLabels.map((label, index) => ({
                                value: index + 1,
                                label,
                            })) })] }), _jsxs("div", { className: "form-grid form-grid-3", children: [_jsx(Slider, { label: "Importance (Optional)", value: currentValue.importance ?? 3, onChange: handleChange('importance'), min: 1, max: 5, step: 1, showValue: true }), _jsx(Slider, { label: "Satisfaction if Present (Optional)", value: currentValue.satisfactionIfPresent ?? 3, onChange: handleChange('satisfactionIfPresent'), min: 1, max: 5, step: 1, showValue: true }), _jsx(Slider, { label: "Dissatisfaction if Absent (Optional)", value: currentValue.dissatisfactionIfAbsent ?? 3, onChange: handleChange('dissatisfactionIfAbsent'), min: 1, max: 5, step: 1, showValue: true })] }), showSubmitButton && (_jsx("div", { className: "form-actions", children: _jsx(Button, { type: "submit", variant: "primary", isLoading: isLoading, disabled: !currentValue.functionalScore || !currentValue.dysfunctionalScore, children: "Analyze with Kano Model" }) }))] }) }));
}
export default KanoInputForm;
