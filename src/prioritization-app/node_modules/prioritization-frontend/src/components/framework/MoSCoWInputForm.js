import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
/**
 * MoSCoW Input Form component for the Prioritization Frameworks application.
 * Provides input fields for MoSCoW categorization (Must, Should, Could, Won't have).
 *
 * @module components/framework/MoSCoWInputForm
 */
import { useState, useCallback } from 'react';
import { Card, Select, Button } from '../common';
/**
 * Business value level options.
 */
const businessValueOptions = [
    { value: 'critical', label: 'Critical - Core business requirement' },
    { value: 'high', label: 'High - Significant business value' },
    { value: 'medium', label: 'Medium - Moderate business value' },
    { value: 'low', label: 'Low - Minimal business value' },
];
/**
 * Risk level options.
 */
const riskLevelOptions = [
    { value: 'critical', label: 'Critical - Severe consequences' },
    { value: 'high', label: 'High - Major impact' },
    { value: 'medium', label: 'Medium - Moderate impact' },
    { value: 'low', label: 'Low - Minimal impact' },
];
/**
 * MoSCoW Input Form component.
 *
 * This component provides a complete form for collecting MoSCoW categorization inputs:
 * - Business Value: Criticality to business
 * - Legal Requirement: Compliance necessity
 * - Customer Request: Customer-driven requirement
 * - Risk if Not Delivered: Impact of omission
 *
 * @param props - MoSCoWInputForm props
 * @returns Rendered MoSCoW input form
 *
 * @example
 * ```tsx
 * <MoSCoWInputForm
 *   value={{ businessValue: 'high', legalRequirement: false, customerRequest: true, riskIfNotDelivered: 'medium' }}
 *   onChange={handleMoSCoWChange}
 *   onSubmit={handleSubmit}
 * />
 * ```
 */
export function MoSCoWInputForm({ value = {}, onChange, onSubmit, isLoading = false, showSubmitButton = true, className = '', }) {
    const [localValue, setLocalValue] = useState(value);
    const currentValue = { ...value, ...localValue };
    const handleChange = useCallback((field) => (val) => {
        const newValue = typeof val === 'object' ? val.target.type === 'checkbox' ? val.target.checked : val.target.value : val;
        const updated = { ...currentValue, [field]: newValue };
        setLocalValue(updated);
        onChange?.(updated);
    }, [currentValue, onChange]);
    const handleSubmit = useCallback((e) => {
        e.preventDefault();
        if (currentValue.businessValue && currentValue.riskIfNotDelivered !== undefined) {
            onSubmit?.({
                businessValue: currentValue.businessValue,
                legalRequirement: currentValue.legalRequirement ?? false,
                customerRequest: currentValue.customerRequest ?? false,
                riskIfNotDelivered: currentValue.riskIfNotDelivered,
            });
        }
    }, [currentValue, onSubmit]);
    return (_jsx(Card, { title: "MoSCoW Categorization", variant: "default", className: `moscow-input-form ${className}`, children: _jsxs("form", { onSubmit: handleSubmit, className: "framework-form", children: [_jsxs("div", { className: "form-grid form-grid-2", children: [_jsx(Select, { label: "Business Value", options: businessValueOptions, value: currentValue.businessValue ?? '', onChange: (e) => handleChange('businessValue')(e.target.value), placeholder: "Select business value level", helperText: "How critical is this to business success?" }), _jsx(Select, { label: "Risk if Not Delivered", options: riskLevelOptions, value: currentValue.riskIfNotDelivered ?? '', onChange: (e) => handleChange('riskIfNotDelivered')(e.target.value), placeholder: "Select risk level", helperText: "What is the impact of not delivering this?" })] }), _jsxs("div", { className: "checkbox-group", children: [_jsxs("label", { className: "checkbox-label", children: [_jsx("input", { type: "checkbox", checked: currentValue.legalRequirement ?? false, onChange: (e) => handleChange('legalRequirement')(e.target.checked), className: "checkbox-input" }), _jsxs("span", { className: "checkbox-text", children: [_jsx("strong", { children: "Legal/Compliance Requirement" }), _jsx("span", { className: "checkbox-description", children: "This item is required for legal or regulatory compliance" })] })] }), _jsxs("label", { className: "checkbox-label", children: [_jsx("input", { type: "checkbox", checked: currentValue.customerRequest ?? false, onChange: (e) => handleChange('customerRequest')(e.target.checked), className: "checkbox-input" }), _jsxs("span", { className: "checkbox-text", children: [_jsx("strong", { children: "Customer Request" }), _jsx("span", { className: "checkbox-description", children: "This item was specifically requested by customers" })] })] })] }), showSubmitButton && (_jsx("div", { className: "form-actions", children: _jsx(Button, { type: "submit", variant: "primary", isLoading: isLoading, disabled: !currentValue.businessValue || currentValue.riskIfNotDelivered === undefined, children: "Categorize with MoSCoW" }) }))] }) }));
}
export default MoSCoWInputForm;
