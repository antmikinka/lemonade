import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
/**
 * Slider component module for Prioritization Frameworks application.
 * Provides reusable range slider components for numeric inputs.
 *
 * @module components/common/Slider
 */
import { useId, useState, useCallback } from 'react';
/**
 * Reusable Slider component for numeric range inputs.
 *
 * This component provides a consistent slider style across the application
 * with support for labels, value display, custom marks, and various sizes.
 *
 * @param props - Slider component props
 * @returns Rendered slider element
 *
 * @example
 * ```tsx
 * <Slider
 *   label="Impact"
 *   min={1}
 *   max={10}
 *   defaultValue={5}
 *   onChange={(value) => console.log(value)}
 * />
 *
 * <Slider
 *   label="Confidence"
 *   min={0}
 *   max={100}
 *   showValue
 *   valueFormat="${value}%"
 *   marks={[
 *     { value: 0, label: 'Low' },
 *     { value: 50, label: 'Medium' },
 *     { value: 100, label: 'High' },
 *   ]}
 * />
 * ```
 */
export function Slider({ label, helperText, error, required, min = 0, max = 100, step = 1, value: controlledValue, defaultValue, onChange, size = 'md', showValue = false, valuePosition = 'right', valueFormat, marks, disabled = false, fullWidth = true, className = '', }) {
    const sliderId = useId();
    const errorId = `${sliderId}-error`;
    const helperId = `${sliderId}-helper`;
    const [uncontrolledValue, setUncontrolledValue] = useState(defaultValue ?? min);
    const hasError = Boolean(error);
    const hasHelper = Boolean(helperText);
    // Use controlled value if provided, otherwise use uncontrolled
    const currentValue = controlledValue !== undefined ? controlledValue : uncontrolledValue;
    const handleChange = useCallback((e) => {
        const newValue = Number(e.target.value);
        if (onChange) {
            onChange(newValue);
        }
        setUncontrolledValue(newValue);
    }, [onChange]);
    // Calculate percentage for thumb position
    const percentage = ((currentValue - min) / (max - min)) * 100;
    const sizeClasses = `slider-${size}`;
    const errorClasses = hasError ? 'slider-error' : '';
    const widthClasses = fullWidth ? 'slider-full-width' : '';
    const disabledClasses = disabled ? 'slider-disabled' : '';
    const hasMarksClasses = marks ? 'slider-has-marks' : '';
    const combinedClasses = `slider-wrapper ${sizeClasses} ${errorClasses} ${widthClasses} ${disabledClasses} ${hasMarksClasses} ${className}`.trim();
    const formatValue = (val) => {
        if (valueFormat) {
            return valueFormat.replace('${value}', String(val));
        }
        return String(val);
    };
    return (_jsxs("div", { className: combinedClasses, children: [_jsxs("div", { className: "slider-header", children: [label && (_jsxs("label", { htmlFor: sliderId, className: "slider-label", children: [label, required && _jsx("span", { className: "slider-required", "aria-hidden": "true", children: "*" })] })), showValue && valuePosition !== 'top' && (_jsx("span", { className: "slider-value", children: formatValue(currentValue) }))] }), showValue && valuePosition === 'top' && (_jsx("div", { className: "slider-value-top", children: formatValue(currentValue) })), _jsxs("div", { className: "slider-track-container", children: [_jsx("div", { className: "slider-track", style: {
                            backgroundImage: `linear-gradient(to right, var(--color-accent) ${percentage}%, var(--color-border) ${percentage}%)`,
                        }, children: _jsx("input", { id: sliderId, type: "range", min: min, max: max, step: step, value: currentValue, onChange: handleChange, disabled: disabled, className: "slider-input", "aria-required": required, "aria-valuemin": min, "aria-valuemax": max, "aria-valuenow": currentValue }) }), marks && marks.length > 0 && (_jsx("div", { className: "slider-marks", children: marks.map((mark) => {
                            const markPercentage = ((mark.value - min) / (max - min)) * 100;
                            return (_jsx("div", { className: "slider-mark", style: { left: `${markPercentage}%` }, children: _jsx("span", { className: "slider-mark-value", children: mark.label }) }, mark.value));
                        }) }))] }), _jsxs("div", { className: "slider-min-max", children: [_jsx("span", { className: "slider-min", children: min }), _jsx("span", { className: "slider-max", children: max })] }), hasError && (_jsx("p", { id: errorId, className: "slider-error-message", role: "alert", children: error })), hasHelper && !hasError && (_jsx("p", { id: helperId, className: "slider-helper-text", children: helperText }))] }));
}
export default Slider;
