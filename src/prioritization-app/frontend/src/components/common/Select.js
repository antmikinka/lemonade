import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
/**
 * Select component module for Prioritization Frameworks application.
 * Provides reusable select dropdown components with labels and error handling.
 *
 * @module components/common/Select
 */
import { useId } from 'react';
/**
 * Reusable Select component with label, error handling, and grouped options.
 *
 * This component provides a consistent select style across the application
 * with support for labels, helper text, error messages, and option groups.
 *
 * @param props - Select component props
 * @returns Rendered select element
 *
 * @example
 * ```tsx
 * <Select
 *   label="Priority"
 *   options={[
 *     { value: 'high', label: 'High' },
 *     { value: 'medium', label: 'Medium' },
 *     { value: 'low', label: 'Low' },
 *   ]}
 *   defaultValue="medium"
 * />
 *
 * <Select
 *   label="Category"
 *   options={categories}
 *   groupBy="group"
 *   placeholder="Select a category"
 * />
 * ```
 */
export function Select({ label, helperText, error, required, size = 'md', options, placeholder, allowEmpty = true, emptyLabel = 'Select...', groupBy, fullWidth = true, className = '', id: providedId, ...props }) {
    const generatedId = useId();
    const selectId = providedId || generatedId;
    const errorId = `${selectId}-error`;
    const helperId = `${selectId}-helper`;
    const hasError = Boolean(error);
    const hasHelper = Boolean(helperText);
    const sizeClasses = `select-${size}`;
    const errorClasses = hasError ? 'select-error' : '';
    const widthClasses = fullWidth ? 'select-full-width' : '';
    const combinedClasses = `select-field ${sizeClasses} ${errorClasses} ${widthClasses} ${className}`.trim();
    // Group options if groupBy is specified
    const groupedOptions = groupBy
        ? options.reduce((acc, option) => {
            const group = option.group || 'Other';
            if (!acc[group]) {
                acc[group] = [];
            }
            acc[group].push(option);
            return acc;
        }, {})
        : null;
    return (_jsxs("div", { className: "select-wrapper", children: [label && (_jsxs("label", { htmlFor: selectId, className: "select-label", children: [label, required && _jsx("span", { className: "select-required", "aria-hidden": "true", children: "*" })] })), _jsx("div", { className: "select-container", children: _jsxs("select", { id: selectId, className: combinedClasses, "aria-required": required, "aria-invalid": hasError, "aria-describedby": [hasError ? errorId : null, hasHelper ? helperId : null]
                        .filter(Boolean)
                        .join(' ') || undefined, ...props, children: [allowEmpty && !placeholder && (_jsx("option", { value: "", children: emptyLabel })), placeholder && (_jsx("option", { value: "", disabled: true, children: placeholder })), groupedOptions ? (Object.entries(groupedOptions).map(([group, groupOptions]) => (_jsx("optgroup", { label: group, children: groupOptions.map((option) => (_jsx("option", { value: option.value, disabled: option.disabled, children: option.label }, option.value))) }, group)))) : (options.map((option) => (_jsx("option", { value: option.value, disabled: option.disabled, children: option.label }, option.value))))] }) }), hasError && (_jsx("p", { id: errorId, className: "select-error-message", role: "alert", children: error })), hasHelper && !hasError && (_jsx("p", { id: helperId, className: "select-helper-text", children: helperText }))] }));
}
export default Select;
