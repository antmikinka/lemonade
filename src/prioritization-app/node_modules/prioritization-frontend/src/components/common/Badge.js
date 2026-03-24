import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
/**
 * Reusable Badge component for displaying status, categories, or tags.
 *
 * This component provides a consistent badge style across the application
 * with support for different colors, sizes, icons, and interactive states.
 *
 * @param props - Badge component props
 * @returns Rendered badge element
 *
 * @example
 * ```tsx
 * <Badge variant="success">Completed</Badge>
 *
 * <Badge variant="primary" dot>Active</Badge>
 *
 * <Badge variant="warning" leftIcon={<WarningIcon />}>
 *   Review Needed
 * </Badge>
 *
 * <Badge variant="error" onClick={onDismiss}>
 *   Error x
 * </Badge>
 * ```
 */
export function Badge({ children, variant = 'default', size = 'md', leftIcon, rightIcon, dot = false, onClick, className = '', style, }) {
    const variantClasses = `badge-${variant}`;
    const sizeClasses = `badge-${size}`;
    const interactiveClasses = onClick ? 'badge-interactive' : '';
    const dotClasses = dot ? 'badge-dot' : '';
    const hasIconClasses = (leftIcon || rightIcon) ? 'badge-has-icon' : '';
    const combinedClasses = `badge ${variantClasses} ${sizeClasses} ${interactiveClasses} ${dotClasses} ${hasIconClasses} ${className}`.trim();
    return (_jsxs("span", { className: combinedClasses, onClick: onClick, role: onClick ? 'button' : undefined, tabIndex: onClick ? 0 : undefined, style: style, children: [dot && _jsx("span", { className: "badge-dot-indicator", "aria-hidden": "true" }), leftIcon && _jsx("span", { className: "badge-icon-left", children: leftIcon }), _jsx("span", { className: "badge-content", children: children }), rightIcon && _jsx("span", { className: "badge-icon-right", children: rightIcon })] }));
}
export default Badge;
