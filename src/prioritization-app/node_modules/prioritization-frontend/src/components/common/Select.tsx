/**
 * Select component module for Prioritization Frameworks application.
 * Provides reusable select dropdown components with labels and error handling.
 *
 * @module components/common/Select
 */

import React, { useId } from 'react';

/**
 * Select option interface.
 */
export interface SelectOption {
  /** Option value */
  value: string | number;
  /** Option display label */
  label: string;
  /** Whether the option is disabled */
  disabled?: boolean;
  /** Optional group label for grouping options */
  group?: string;
}

/**
 * Select size options.
 */
export type SelectSize = 'sm' | 'md' | 'lg';

/**
 * Props for the Select component.
 */
export interface SelectProps extends Omit<React.SelectHTMLAttributes<HTMLSelectElement>, 'size'> {
  /** Label text for the select */
  label?: string;
  /** Helper text displayed below the select */
  helperText?: string;
  /** Error message displayed below the select */
  error?: string;
  /** Whether the field is required */
  required?: boolean;
  /** Select size */
  size?: SelectSize;
  /** Array of options to display */
  options: SelectOption[];
  /** Placeholder option text (displays when no value selected) */
  placeholder?: string;
  /** Allow empty value */
  allowEmpty?: boolean;
  /** Empty value label text */
  emptyLabel?: string;
  /** Group options by this key */
  groupBy?: string;
  /** Full width select */
  fullWidth?: boolean;
}

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
export function Select({
  label,
  helperText,
  error,
  required,
  size = 'md',
  options,
  placeholder,
  allowEmpty = true,
  emptyLabel = 'Select...',
  groupBy,
  fullWidth = true,
  className = '',
  id: providedId,
  ...props
}: SelectProps): React.JSX.Element {
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
      }, {} as Record<string, SelectOption[]>)
    : null;

  return (
    <div className="select-wrapper">
      {label && (
        <label htmlFor={selectId} className="select-label">
          {label}
          {required && <span className="select-required" aria-hidden="true">*</span>}
        </label>
      )}

      <div className="select-container">
        <select
          id={selectId}
          className={combinedClasses}
          aria-required={required}
          aria-invalid={hasError}
          aria-describedby={
            [hasError ? errorId : null, hasHelper ? helperId : null]
              .filter(Boolean)
              .join(' ') || undefined
          }
          {...props}
        >
          {allowEmpty && !placeholder && (
            <option value="">{emptyLabel}</option>
          )}
          {placeholder && (
            <option value="" disabled>
              {placeholder}
            </option>
          )}

          {groupedOptions ? (
            Object.entries(groupedOptions).map(([group, groupOptions]) => (
              <optgroup key={group} label={group}>
                {groupOptions.map((option) => (
                  <option
                    key={option.value}
                    value={option.value}
                    disabled={option.disabled}
                  >
                    {option.label}
                  </option>
                ))}
              </optgroup>
            ))
          ) : (
            options.map((option) => (
              <option
                key={option.value}
                value={option.value}
                disabled={option.disabled}
              >
                {option.label}
              </option>
            ))
          )}
        </select>
      </div>

      {hasError && (
        <p id={errorId} className="select-error-message" role="alert">
          {error}
        </p>
      )}

      {hasHelper && !hasError && (
        <p id={helperId} className="select-helper-text">
          {helperText}
        </p>
      )}
    </div>
  );
}

export default Select;
