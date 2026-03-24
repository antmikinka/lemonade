/**
 * Badge component module for Prioritization Frameworks application.
 * Provides reusable badge components for status and category display.
 *
 * @module components/common/Badge
 */

import React from 'react';

/**
 * Badge color variants.
 * - default: Gray neutral badge
 * - primary: Blue accent badge
 * - success: Green success badge
 * - warning: Yellow warning badge
 * - error: Red error badge
 * - info: Cyan information badge
 */
export type BadgeVariant = 'default' | 'primary' | 'success' | 'warning' | 'error' | 'info';

/**
 * Badge size options.
 * - sm: Small badge for compact spaces
 * - md: Medium default size
 * - lg: Large badge for emphasis
 */
export type BadgeSize = 'sm' | 'md' | 'lg';

/**
 * Props for the Badge component.
 */
export interface BadgeProps {
  /** Badge text content */
  children: React.ReactNode;
  /** Visual style variant */
  variant?: BadgeVariant;
  /** Badge size */
  size?: BadgeSize;
  /** Left icon element */
  leftIcon?: React.ReactNode;
  /** Right icon element */
  rightIcon?: React.ReactNode;
  /** Dot indicator shown before content */
  dot?: boolean;
  /** Click handler for interactive badges */
  onClick?: () => void;
  /** Additional CSS class name */
  className?: string;
  /** Custom CSS styles */
  style?: React.CSSProperties;
}

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
export function Badge({
  children,
  variant = 'default',
  size = 'md',
  leftIcon,
  rightIcon,
  dot = false,
  onClick,
  className = '',
  style,
}: BadgeProps): React.JSX.Element {
  const variantClasses = `badge-${variant}`;
  const sizeClasses = `badge-${size}`;
  const interactiveClasses = onClick ? 'badge-interactive' : '';
  const dotClasses = dot ? 'badge-dot' : '';
  const hasIconClasses = (leftIcon || rightIcon) ? 'badge-has-icon' : '';

  const combinedClasses = `badge ${variantClasses} ${sizeClasses} ${interactiveClasses} ${dotClasses} ${hasIconClasses} ${className}`.trim();

  return (
    <span
      className={combinedClasses}
      onClick={onClick}
      role={onClick ? 'button' : undefined}
      tabIndex={onClick ? 0 : undefined}
      style={style}
    >
      {dot && <span className="badge-dot-indicator" aria-hidden="true" />}
      {leftIcon && <span className="badge-icon-left">{leftIcon}</span>}
      <span className="badge-content">{children}</span>
      {rightIcon && <span className="badge-icon-right">{rightIcon}</span>}
    </span>
  );
}

export default Badge;
