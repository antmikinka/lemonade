/**
 * Input component module for Prioritization Frameworks application.
 * Provides reusable input field components with labels and error handling.
 *
 * @module components/common/Input
 */

import React, { useId } from 'react';

/**
 * Input type options.
 */
export type InputType =
  | 'text'
  | 'email'
  | 'number'
  | 'password'
  | 'search'
  | 'tel'
  | 'url'
  | 'textarea';

/**
 * Input size options.
 */
export type InputSize = 'sm' | 'md' | 'lg';

/**
 * Props for the Input component.
 */
export interface InputProps extends Omit<React.InputHTMLAttributes<HTMLInputElement>, 'size'> {
  /** Input field type */
  type?: InputType;
  /** Label text for the input */
  label?: string;
  /** Helper text displayed below the input */
  helperText?: string;
  /** Error message displayed below the input */
  error?: string;
  /** Whether the field is required */
  required?: boolean;
  /** Input size */
  size?: InputSize;
  /** Left icon or addon element */
  leftAddon?: React.ReactNode;
  /** Right icon or addon element */
  rightAddon?: React.ReactNode;
  /** Number of rows for textarea type */
  rows?: number;
  /** Full width input */
  fullWidth?: boolean;
}

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
export function Input({
  type = 'text',
  label,
  helperText,
  error,
  required,
  size = 'md',
  leftAddon,
  rightAddon,
  rows = 4,
  fullWidth = true,
  className = '',
  id: providedId,
  ...props
}: InputProps): React.JSX.Element {
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

  return (
    <div className="input-wrapper">
      {label && (
        <label htmlFor={inputId} className="input-label">
          {label}
          {required && <span className="input-required" aria-hidden="true">*</span>}
        </label>
      )}

      <div className="input-container">
        {leftAddon && <div className="input-addon-left">{leftAddon}</div>}

        {type === 'textarea' ? (
          <textarea
            id={inputId}
            className={combinedClasses}
            rows={rows}
            aria-required={required}
            aria-invalid={hasError}
            aria-describedby={[hasError ? errorId : null, hasHelper ? helperId : null].filter(Boolean).join(' ') || undefined}
            {...(props as React.TextareaHTMLAttributes<HTMLTextAreaElement>)}
          />
        ) : (
          <input
            id={inputId}
            type={type}
            className={combinedClasses}
            aria-required={required}
            aria-invalid={hasError}
            aria-describedby={[hasError ? errorId : null, hasHelper ? helperId : null].filter(Boolean).join(' ') || undefined}
            {...props}
          />
        )}

        {rightAddon && <div className="input-addon-right">{rightAddon}</div>}
      </div>

      {hasError && (
        <p id={errorId} className="input-error-message" role="alert">
          {error}
        </p>
      )}

      {hasHelper && !hasError && (
        <p id={helperId} className="input-helper-text">
          {helperText}
        </p>
      )}
    </div>
  );
}

export default Input;
