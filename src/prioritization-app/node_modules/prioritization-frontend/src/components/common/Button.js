import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
/**
 * Reusable Button component with multiple variants and sizes.
 *
 * This component provides a consistent button style across the application
 * with support for different visual variants, sizes, loading states, and icons.
 *
 * @param props - Button component props
 * @returns Rendered button element
 *
 * @example
 * ```tsx
 * <Button variant="primary" onClick={handleSubmit}>
 *   Submit
 * </Button>
 *
 * <Button variant="secondary" leftIcon={<SaveIcon />}>
 *   Save Draft
 * </Button>
 *
 * <Button variant="danger" isLoading={isDeleting}>
 *   Delete
 * </Button>
 * ```
 */
export function Button({ children, variant = 'primary', size = 'md', isLoading = false, leftIcon, rightIcon, fullWidth = false, disabled, className = '', ...props }) {
    const baseClasses = 'btn';
    const variantClasses = `btn-${variant}`;
    const sizeClasses = `btn-${size}`;
    const widthClasses = fullWidth ? 'btn-full-width' : '';
    const loadingClasses = isLoading ? 'btn-loading' : '';
    const combinedClasses = `${baseClasses} ${variantClasses} ${sizeClasses} ${widthClasses} ${loadingClasses} ${className}`.trim();
    return (_jsxs("button", { className: combinedClasses, disabled: disabled || isLoading, "aria-busy": isLoading, ...props, children: [isLoading && (_jsx("span", { className: "btn-spinner", "aria-hidden": "true", children: _jsx("svg", { viewBox: "0 0 24 24", fill: "none", xmlns: "http://www.w3.org/2000/svg", children: _jsx("circle", { cx: "12", cy: "12", r: "10", stroke: "currentColor", strokeWidth: "3", strokeLinecap: "round", strokeDasharray: "32 32" }) }) })), leftIcon && _jsx("span", { className: "btn-icon-left", children: leftIcon }), children, rightIcon && _jsx("span", { className: "btn-icon-right", children: rightIcon })] }));
}
export default Button;
