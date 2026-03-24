import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
/**
 * WSJF Input Form component for the Prioritization Frameworks application.
 * Provides input fields for Weighted Shortest Job First scoring.
 *
 * @module components/framework/WSJFInputForm
 */
import { useState, useCallback } from 'react';
import { Card, Slider, Button } from '../common';
/**
 * WSJF Input Form component.
 *
 * This component provides a complete form for collecting WSJF scoring inputs:
 * - User-Business Value: Value to users and business (1-10 or Fibonacci)
 * - Time Criticality: How time-sensitive is this? (1-10 or Fibonacci)
 * - Risk Reduction & Opportunity Enablement: Risk mitigation value (1-10 or Fibonacci)
 * - Job Size: Estimated size/effort (lower = smaller/faster)
 *
 * WSJF Score = Cost of Delay / Job Size
 * Cost of Delay = User-Business Value + Time Criticality + Risk Reduction
 *
 * @param props - WSJFInputForm props
 * @returns Rendered WSJF input form
 *
 * @example
 * ```tsx
 * <WSJFInputForm
 *   value={{
 *     userBusinessValue: 8,
 *     timeCriticality: 6,
 *     riskReductionOpportunity: 5,
 *     jobSize: 3,
 *   }}
 *   onChange={handleWSJFChange}
 *   onSubmit={handleSubmit}
 * />
 * ```
 */
export function WSJFInputForm({ value = {}, onChange, onSubmit, isLoading = false, showSubmitButton = true, className = '', }) {
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
        if (currentValue.userBusinessValue &&
            currentValue.timeCriticality &&
            currentValue.riskReductionOpportunity &&
            currentValue.jobSize) {
            onSubmit?.({
                userBusinessValue: currentValue.userBusinessValue,
                timeCriticality: currentValue.timeCriticality,
                riskReductionOpportunity: currentValue.riskReductionOpportunity,
                jobSize: currentValue.jobSize,
            });
        }
    }, [currentValue, onSubmit]);
    // Calculate preview values
    const costOfDelay = currentValue.userBusinessValue && currentValue.timeCriticality && currentValue.riskReductionOpportunity
        ? currentValue.userBusinessValue + currentValue.timeCriticality + currentValue.riskReductionOpportunity
        : null;
    const previewScore = costOfDelay && currentValue.jobSize
        ? (costOfDelay / currentValue.jobSize).toFixed(2)
        : null;
    return (_jsx(Card, { title: "WSJF (Weighted Shortest Job First)", variant: "default", className: `wsjf-input-form ${className}`, children: _jsxs("form", { onSubmit: handleSubmit, className: "framework-form", children: [_jsxs("div", { className: "wsjf-intro", children: [_jsx("p", { className: "wsjf-description", children: "WSJF is a prioritization model used in SAFe to sequence jobs (features, capabilities, etc.) to produce maximum economic benefit." }), _jsx("div", { className: "wsjf-formula", children: _jsx("code", { children: "WSJF = Cost of Delay / Job Size" }) })] }), _jsxs("div", { className: "cod-section", children: [_jsx("h4", { className: "section-title", children: "Cost of Delay Components" }), _jsx(Slider, { label: "User-Business Value", value: currentValue.userBusinessValue ?? 5, onChange: handleChange('userBusinessValue'), min: 1, max: 20, step: 1, showValue: true, helperText: "Value delivered to users and the business", marks: [
                                { value: 1, label: 'Low' },
                                { value: 10, label: 'Medium' },
                                { value: 20, label: 'High' },
                            ] }), _jsx(Slider, { label: "Time Criticality", value: currentValue.timeCriticality ?? 5, onChange: handleChange('timeCriticality'), min: 1, max: 20, step: 1, showValue: true, helperText: "How urgent is this? Does timing matter?", marks: [
                                { value: 1, label: 'Flexible' },
                                { value: 10, label: 'Time-boxed' },
                                { value: 20, label: 'Fixed deadline' },
                            ] }), _jsx(Slider, { label: "Risk Reduction & Opportunity Enablement", value: currentValue.riskReductionOpportunity ?? 5, onChange: handleChange('riskReductionOpportunity'), min: 1, max: 20, step: 1, showValue: true, helperText: "Does this reduce risk or enable future opportunities?", marks: [
                                { value: 1, label: 'Low RR/OE' },
                                { value: 10, label: 'Medium' },
                                { value: 20, label: 'High RR/OE' },
                            ] })] }), _jsxs("div", { className: "job-size-section", children: [_jsx("h4", { className: "section-title", children: "Job Size" }), _jsx(Slider, { label: "Job Size", value: currentValue.jobSize ?? 5, onChange: handleChange('jobSize'), min: 1, max: 20, step: 1, showValue: true, helperText: "Relative size/effort (smaller = faster to complete)", marks: [
                                { value: 1, label: 'Tiny' },
                                { value: 10, label: 'Medium' },
                                { value: 20, label: 'Large' },
                            ] })] }), previewScore && costOfDelay && (_jsxs("div", { className: "score-preview", children: [_jsx("span", { className: "score-preview-label", children: "Estimated WSJF Score:" }), _jsx("span", { className: "score-preview-value", children: previewScore }), _jsxs("span", { className: "score-preview-formula", children: ["(", currentValue.userBusinessValue, " + ", currentValue.timeCriticality, " + ", currentValue.riskReductionOpportunity, ") / ", currentValue.jobSize] })] })), showSubmitButton && (_jsx("div", { className: "form-actions", children: _jsx(Button, { type: "submit", variant: "primary", isLoading: isLoading, disabled: !currentValue.userBusinessValue ||
                            !currentValue.timeCriticality ||
                            !currentValue.riskReductionOpportunity ||
                            !currentValue.jobSize, children: "Calculate WSJF Score" }) }))] }) }));
}
export default WSJFInputForm;
