import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
/**
 * Eisenhower Input Form component for the Prioritization Frameworks application.
 * Provides input toggles for Eisenhower Matrix (Urgent vs Important).
 *
 * @module components/framework/EisenhowerInputForm
 */
import { useState, useCallback } from 'react';
import { Card, Slider, Button } from '../common';
/**
 * Toggle Switch component for binary inputs.
 */
function ToggleSwitch({ checked, onChange, label, description, }) {
    return (_jsxs("label", { className: "toggle-switch", children: [_jsx("input", { type: "checkbox", checked: checked, onChange: (e) => onChange(e.target.checked), className: "toggle-input", "aria-label": label }), _jsx("span", { className: "toggle-slider" }), _jsxs("div", { className: "toggle-content", children: [_jsx("span", { className: "toggle-label", children: label }), description && (_jsx("span", { className: "toggle-description", children: description }))] })] }));
}
/**
 * Eisenhower Input Form component.
 *
 * This component provides a complete form for collecting Eisenhower Matrix inputs:
 * - Urgent: Is this time-sensitive?
 * - Important: Does this have high impact?
 * - Urgency Level: Optional fine-grained urgency (1-10)
 * - Importance Level: Optional fine-grained importance (1-10)
 *
 * The combination determines the quadrant:
 * - Urgent + Important = Do First
 * - Not Urgent + Important = Schedule
 * - Urgent + Not Important = Delegate
 * - Not Urgent + Not Important = Eliminate
 *
 * @param props - EisenhowerInputForm props
 * @returns Rendered Eisenhower input form
 *
 * @example
 * ```tsx
 * <EisenhowerInputForm
 *   value={{ urgent: true, important: true, urgencyLevel: 8, importanceLevel: 9 }}
 *   onChange={handleEisenhowerChange}
 *   onSubmit={handleSubmit}
 * />
 * ```
 */
export function EisenhowerInputForm({ value = {}, onChange, onSubmit, isLoading = false, showSubmitButton = true, className = '', }) {
    const [localValue, setLocalValue] = useState(value);
    const currentValue = { ...value, ...localValue };
    const handleToggleChange = useCallback((field) => (val) => {
        const updated = { ...currentValue, [field]: val };
        setLocalValue(updated);
        onChange?.(updated);
    }, [currentValue, onChange]);
    const handleLevelChange = useCallback((field) => (val) => {
        const updated = { ...currentValue, [field]: val };
        setLocalValue(updated);
        onChange?.(updated);
    }, [currentValue, onChange]);
    const handleSubmit = useCallback((e) => {
        e.preventDefault();
        if (currentValue.urgent !== undefined && currentValue.important !== undefined) {
            onSubmit?.({
                urgent: currentValue.urgent,
                important: currentValue.important,
                urgencyLevel: currentValue.urgencyLevel,
                importanceLevel: currentValue.importanceLevel,
            });
        }
    }, [currentValue, onSubmit]);
    // Determine preview quadrant
    const getPreviewQuadrant = () => {
        if (currentValue.urgent === undefined || currentValue.important === undefined) {
            return null;
        }
        if (currentValue.urgent && currentValue.important)
            return 'Do First';
        if (!currentValue.urgent && currentValue.important)
            return 'Schedule';
        if (currentValue.urgent && !currentValue.important)
            return 'Delegate';
        return 'Eliminate';
    };
    const previewQuadrant = getPreviewQuadrant();
    return (_jsx(Card, { title: "Eisenhower Matrix", variant: "default", className: `eisenhower-input-form ${className}`, children: _jsxs("form", { onSubmit: handleSubmit, className: "framework-form", children: [_jsx("div", { className: "eisenhower-intro", children: _jsx("p", { className: "eisenhower-description", children: "The Eisenhower Matrix helps prioritize tasks by urgency and importance. Categorize each task to determine the best action." }) }), _jsxs("div", { className: "toggle-group", children: [_jsx(ToggleSwitch, { checked: currentValue.urgent ?? false, onChange: handleToggleChange('urgent'), label: "Urgent", description: "This task is time-sensitive and requires immediate attention" }), _jsx(ToggleSwitch, { checked: currentValue.important ?? false, onChange: handleToggleChange('important'), label: "Important", description: "This task has significant impact on goals and objectives" })] }), _jsxs("div", { className: "form-grid form-grid-2", children: [_jsx(Slider, { label: "Urgency Level (Optional)", value: currentValue.urgencyLevel ?? 5, onChange: handleLevelChange('urgencyLevel'), min: 1, max: 10, step: 1, showValue: true, helperText: "Fine-tune urgency within category" }), _jsx(Slider, { label: "Importance Level (Optional)", value: currentValue.importanceLevel ?? 5, onChange: handleLevelChange('importanceLevel'), min: 1, max: 10, step: 1, showValue: true, helperText: "Fine-tune importance within category" })] }), previewQuadrant && (_jsxs("div", { className: "quadrant-preview", children: [_jsx("span", { className: "quadrant-label", children: "Quadrant:" }), _jsx("span", { className: `quadrant-badge quadrant-${previewQuadrant.toLowerCase().replace(' ', '-')}`, children: previewQuadrant })] })), showSubmitButton && (_jsx("div", { className: "form-actions", children: _jsx(Button, { type: "submit", variant: "primary", isLoading: isLoading, disabled: currentValue.urgent === undefined || currentValue.important === undefined, children: "Categorize Task" }) }))] }) }));
}
export default EisenhowerInputForm;
