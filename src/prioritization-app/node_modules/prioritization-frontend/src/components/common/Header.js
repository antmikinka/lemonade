import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { Select } from './Select';
import { Button } from './Button';
/**
 * Framework selector options for the dropdown.
 */
const frameworkOptions = [
    { value: 'RICE', label: 'RICE Scoring' },
    { value: 'MoSCoW', label: 'MoSCoW Method' },
    { value: 'Kano', label: 'Kano Model' },
    { value: 'ValueEffort', label: 'Value vs Effort' },
    { value: 'ICE', label: 'ICE Scoring' },
    { value: 'Eisenhower', label: 'Eisenhower Matrix' },
    { value: 'P0P4', label: 'P0-P4 Priority' },
    { value: 'WSJF', label: 'WSJF (Weighted Shortest Job First)' },
];
/**
 * Main application Header component with framework selector and actions.
 *
 * This component provides the primary navigation header with:
 * - Application title and branding
 * - Framework selector dropdown
 * - Theme toggle button
 * - Additional action buttons
 *
 * @param props - Header component props
 * @returns Rendered header element
 *
 * @example
 * ```tsx
 * <Header
 *   selectedFramework="RICE"
 *   onFrameworkChange={handleFrameworkChange}
 *   onThemeToggle={handleThemeToggle}
 *   isDarkTheme={true}
 * />
 * ```
 */
export function Header({ selectedFramework, onFrameworkChange, onThemeToggle, isDarkTheme = true, logo, actions, }) {
    return (_jsx("header", { className: "app-header-main", children: _jsxs("div", { className: "header-content", children: [_jsxs("div", { className: "header-brand", children: [logo || (_jsx("div", { className: "header-logo", children: _jsxs("svg", { viewBox: "0 0 40 40", fill: "none", xmlns: "http://www.w3.org/2000/svg", "aria-hidden": "true", children: [_jsx("rect", { width: "40", height: "40", rx: "8", fill: "var(--color-accent)" }), _jsx("path", { d: "M12 20L18 26L28 14", stroke: "var(--color-bg-primary)", strokeWidth: "3", strokeLinecap: "round", strokeLinejoin: "round" })] }) })), _jsxs("div", { className: "header-title-group", children: [_jsx("h1", { className: "header-title", children: "Prioritization Frameworks" }), _jsx("p", { className: "header-subtitle", children: "AI-Powered Decision Making Tools" })] })] }), _jsxs("div", { className: "header-actions", children: [_jsx("div", { className: "header-framework-selector", children: _jsx(Select, { label: "", options: frameworkOptions, value: selectedFramework || '', onChange: (e) => onFrameworkChange?.(e.target.value), placeholder: "Select Framework", allowEmpty: false, fullWidth: false, size: "md" }) }), onThemeToggle && (_jsx(Button, { variant: "secondary", size: "md", onClick: onThemeToggle, "aria-label": isDarkTheme ? 'Switch to light theme' : 'Switch to dark theme', className: "header-theme-toggle", children: isDarkTheme ? (_jsxs("svg", { viewBox: "0 0 24 24", fill: "none", stroke: "currentColor", strokeWidth: "2", "aria-hidden": "true", children: [_jsx("circle", { cx: "12", cy: "12", r: "5" }), _jsx("path", { d: "M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42" })] })) : (_jsx("svg", { viewBox: "0 0 24 24", fill: "none", stroke: "currentColor", strokeWidth: "2", "aria-hidden": "true", children: _jsx("path", { d: "M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" }) })) })), actions && _jsx("div", { className: "header-extra-actions", children: actions })] })] }) }));
}
export default Header;
