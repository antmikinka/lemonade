/**
 * Slider component module for Prioritization Frameworks application.
 * Provides reusable range slider components for numeric inputs.
 *
 * @module components/common/Slider
 */

import React, { useId, useState, useCallback } from 'react';

/**
 * Slider size options.
 */
export type SliderSize = 'sm' | 'md' | 'lg';

/**
 * Props for the Slider component.
 */
export interface SliderProps {
  /** Label text for the slider */
  label?: string;
  /** Helper text displayed below the slider */
  helperText?: string;
  /** Error message displayed below the slider */
  error?: string;
  /** Whether the field is required */
  required?: boolean;
  /** Minimum value */
  min?: number;
  /** Maximum value */
  max?: number;
  /** Step increment */
  step?: number;
  /** Current value */
  value?: number;
  /** Default value for uncontrolled slider */
  defaultValue?: number;
  /** Value change handler */
  onChange?: (value: number) => void;
  /** Slider size */
  size?: SliderSize;
  /** Show value display */
  showValue?: boolean;
  /** Value display position */
  valuePosition?: 'left' | 'right' | 'top';
  /** Display format for value (e.g., '${value}', '${value}%') */
  valueFormat?: string;
  /** Custom marks to display on the slider */
  marks?: { value: number; label: string }[];
  /** Disabled state */
  disabled?: boolean;
  /** Full width slider */
  fullWidth?: boolean;
  /** Additional CSS class name */
  className?: string;
}

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
export function Slider({
  label,
  helperText,
  error,
  required,
  min = 0,
  max = 100,
  step = 1,
  value: controlledValue,
  defaultValue,
  onChange,
  size = 'md',
  showValue = false,
  valuePosition = 'right',
  valueFormat,
  marks,
  disabled = false,
  fullWidth = true,
  className = '',
}: SliderProps): React.JSX.Element {
  const sliderId = useId();
  const errorId = `${sliderId}-error`;
  const helperId = `${sliderId}-helper`;

  const [uncontrolledValue, setUncontrolledValue] = useState(defaultValue ?? min);

  const hasError = Boolean(error);
  const hasHelper = Boolean(helperText);

  // Use controlled value if provided, otherwise use uncontrolled
  const currentValue = controlledValue !== undefined ? controlledValue : uncontrolledValue;

  const handleChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const newValue = Number(e.target.value);
      if (onChange) {
        onChange(newValue);
      }
      setUncontrolledValue(newValue);
    },
    [onChange]
  );

  // Calculate percentage for thumb position
  const percentage = ((currentValue - min) / (max - min)) * 100;

  const sizeClasses = `slider-${size}`;
  const errorClasses = hasError ? 'slider-error' : '';
  const widthClasses = fullWidth ? 'slider-full-width' : '';
  const disabledClasses = disabled ? 'slider-disabled' : '';
  const hasMarksClasses = marks ? 'slider-has-marks' : '';

  const combinedClasses = `slider-wrapper ${sizeClasses} ${errorClasses} ${widthClasses} ${disabledClasses} ${hasMarksClasses} ${className}`.trim();

  const formatValue = (val: number): string => {
    if (valueFormat) {
      return valueFormat.replace('${value}', String(val));
    }
    return String(val);
  };

  return (
    <div className={combinedClasses}>
      <div className="slider-header">
        {label && (
          <label htmlFor={sliderId} className="slider-label">
            {label}
            {required && <span className="slider-required" aria-hidden="true">*</span>}
          </label>
        )}
        {showValue && valuePosition !== 'top' && (
          <span className="slider-value">{formatValue(currentValue)}</span>
        )}
      </div>

      {showValue && valuePosition === 'top' && (
        <div className="slider-value-top">{formatValue(currentValue)}</div>
      )}

      <div className="slider-track-container">
        <div
          className="slider-track"
          style={{
            backgroundImage: `linear-gradient(to right, var(--color-accent) ${percentage}%, var(--color-border) ${percentage}%)`,
          }}
        >
          <input
            id={sliderId}
            type="range"
            min={min}
            max={max}
            step={step}
            value={currentValue}
            onChange={handleChange}
            disabled={disabled}
            className="slider-input"
            aria-required={required}
            aria-valuemin={min}
            aria-valuemax={max}
            aria-valuenow={currentValue}
          />
        </div>

        {marks && marks.length > 0 && (
          <div className="slider-marks">
            {marks.map((mark) => {
              const markPercentage = ((mark.value - min) / (max - min)) * 100;
              return (
                <div
                  key={mark.value}
                  className="slider-mark"
                  style={{ left: `${markPercentage}%` }}
                >
                  <span className="slider-mark-value">{mark.label}</span>
                </div>
              );
            })}
          </div>
        )}
      </div>

      <div className="slider-min-max">
        <span className="slider-min">{min}</span>
        <span className="slider-max">{max}</span>
      </div>

      {hasError && (
        <p id={errorId} className="slider-error-message" role="alert">
          {error}
        </p>
      )}

      {hasHelper && !hasError && (
        <p id={helperId} className="slider-helper-text">
          {helperText}
        </p>
      )}
    </div>
  );
}

export default Slider;
