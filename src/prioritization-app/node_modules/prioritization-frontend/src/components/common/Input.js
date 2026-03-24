import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
/**
 * Input component module for Prioritization Frameworks application.
 * Provides reusable input field components with labels and error handling.
 *
 * @module components/common/Input
 */
import { useId } from 'react';
/**
 * Reusable Input component with label, error handling, and addons.
 *
 * This component provides a consistent input style across the application
 * with support for labels, helper text, error messages, and left/right addons.
 *
 * @param props - Input component props
 * @returns Rendered input element
 *
 * @example
 * ```tsx
 * <Input
 *   label="Project Name"
 *   placeholder="Enter project name"
 *   required
 * />
 *
 * <Input
 *   label="Email"
 *   type="email"
 *   error="Invalid email format"
 * />
 *
 * <Input
 *   label="Search"
 *   leftAddon={<SearchIcon />}
 *   placeholder="Search..."
 * />
 * ```
 */
export function Input({ type = 'text', label, helperText, error, required, size = 'md', leftAddon, rightAddon, rows = 4, fullWidth = true, className = '', id: providedId, ...props }) {
    const generatedId = useId();
    const inputId = providedId || generatedId;
    const errorId = `${inputId}-error`;
    const helperId = `${inputId}-helper`;
    const hasError = Boolean(error);
    const hasHelper = Boolean(helperText);
    const sizeClasses = `input-${size}`;
    const errorClasses = hasError ? 'input-error' : '';
    const widthClasses = fullWidth ? 'input-full-width' : '';
    const hasAddonClasses = (leftAddon || rightAddon) ? 'input-has-addon' : '';
    const combinedClasses = `input-field ${sizeClasses} ${errorClasses} ${widthClasses} ${hasAddonClasses} ${className}`.trim();
    return (_jsxs("div", { className: "input-wrapper", children: [label && (_jsxs("label", { htmlFor: inputId, className: "input-label", children: [label, required && _jsx("span", { className: "input-required", "aria-hidden": "true", children: "*" })] })), _jsxs("div", { className: "input-container", children: [leftAddon && _jsx("div", { className: "input-addon-left", children: leftAddon }), type === 'textarea' ? (_jsx("textarea", { id: inputId, className: combinedClasses, rows: rows, "aria-required": required, "aria-invalid": hasError, "aria-describedby": [hasError ? errorId : null, hasHelper ? helperId : null].filter(Boolean).join(' ') || undefined, ...props })) : (_jsx("input", { id: inputId, type: type, className: combinedClasses, "aria-required": required, "aria-invalid": hasError, "aria-describedby": [hasError ? errorId : null, hasHelper ? helperId : null].filter(Boolean).join(' ') || undefined, ...props })), rightAddon && _jsx("div", { className: "input-addon-right", children: rightAddon })] }), hasError && (_jsx("p", { id: errorId, className: "input-error-message", role: "alert", children: error })), hasHelper && !hasError && (_jsx("p", { id: helperId, className: "input-helper-text", children: helperText }))] }));
}
export default Input;
