/**
 * Button component module for Prioritization Frameworks application.
 * Provides reusable button components with multiple variants.
 *
 * @module components/common/Button
 */

import React from 'react';

/**
 * Button variant types.
 * - primary: Main action button with accent color
 * - secondary: Secondary action with border style
 * - danger: Destructive action with error color
 * - ghost: Minimal style for subtle actions
 */
export type ButtonVariant = 'primary' | 'secondary' | 'danger' | 'ghost';

/**
 * Button size options.
 * - sm: Small button for compact spaces
 * - md: Medium default size
 * - lg: Large button for prominent actions
 */
export type ButtonSize = 'sm' | 'md' | 'lg';

/**
 * Props for the Button component.
 */
export interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  /** Visual style variant of the button */
  variant?: ButtonVariant;
  /** Size of the button */
  size?: ButtonSize;
  /** Whether the button is in a loading state */
  isLoading?: boolean;
  /** Optional left icon element */
  leftIcon?: React.ReactNode;
  /** Optional right icon element */
  rightIcon?: React.ReactNode;
  /** Full width button */
  fullWidth?: boolean;
}

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
export function Button({
  children,
  variant = 'primary',
  size = 'md',
  isLoading = false,
  leftIcon,
  rightIcon,
  fullWidth = false,
  disabled,
  className = '',
  ...props
}: ButtonProps): React.JSX.Element {
  const baseClasses = 'btn';
  const variantClasses = `btn-${variant}`;
  const sizeClasses = `btn-${size}`;
  const widthClasses = fullWidth ? 'btn-full-width' : '';
  const loadingClasses = isLoading ? 'btn-loading' : '';

  const combinedClasses = `${baseClasses} ${variantClasses} ${sizeClasses} ${widthClasses} ${loadingClasses} ${className}`.trim();

  return (
    <button
      className={combinedClasses}
      disabled={disabled || isLoading}
      aria-busy={isLoading}
      {...props}
    >
      {isLoading && (
        <span className="btn-spinner" aria-hidden="true">
          <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeDasharray="32 32" />
          </svg>
        </span>
      )}
      {leftIcon && <span className="btn-icon-left">{leftIcon}</span>}
      {children}
      {rightIcon && <span className="btn-icon-right">{rightIcon}</span>}
    </button>
  );
}

export default Button;
