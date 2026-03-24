import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
/**
 * Reusable Card component for containing content sections.
 *
 * This component provides a consistent container style across the application
 * with support for headers, footers, different variants, and interactive states.
 *
 * @param props - Card component props
 * @returns Rendered card element
 *
 * @example
 * ```tsx
 * <Card title="Project Details">
 *   <p>Content goes here...</p>
 * </Card>
 *
 * <Card variant="elevated" headerAction={<Button>Edit</Button>}>
 *   <p>Elevated card with actions</p>
 * </Card>
 *
 * <Card variant="interactive" onClick={handleClick}>
 *   <p>Clickable card</p>
 * </Card>
 * ```
 */
export function Card({ variant = 'default', title, headerAction, children, footer, className = '', onClick, disabled = false, style, }) {
    const variantClasses = `card-${variant}`;
    const interactiveClasses = onClick && !disabled ? 'card-interactive' : '';
    const disabledClasses = disabled ? 'card-disabled' : '';
    const combinedClasses = `card ${variantClasses} ${interactiveClasses} ${disabledClasses} ${className}`.trim();
    return (_jsxs("div", { className: combinedClasses, onClick: onClick, role: onClick ? 'button' : undefined, tabIndex: onClick && !disabled ? 0 : undefined, "aria-disabled": disabled, style: style, children: [(title || headerAction) && (_jsxs("div", { className: "card-header", children: [title && _jsx("h3", { className: "card-title", children: title }), headerAction && _jsx("div", { className: "card-actions", children: headerAction })] })), _jsx("div", { className: "card-content", children: children }), footer && _jsx("div", { className: "card-footer", children: footer })] }));
}
export default Card;
