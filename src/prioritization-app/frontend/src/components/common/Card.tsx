/**
 * Card component module for Prioritization Frameworks application.
 * Provides a reusable container card with consistent styling.
 *
 * @module components/common/Card
 */

import React from 'react';

/**
 * Card variant types.
 * - default: Standard card style
 * - elevated: Card with shadow for emphasis
 * - bordered: Card with prominent border
 * - interactive: Card with hover effects for clickable content
 */
export type CardVariant = 'default' | 'elevated' | 'bordered' | 'interactive';

/**
 * Props for the Card component.
 */
export interface CardProps {
  /** Visual style variant of the card */
  variant?: CardVariant;
  /** Card title displayed in header */
  title?: string;
  /** Optional header action elements (buttons, icons) */
  headerAction?: React.ReactNode;
  /** Card content */
  children: React.ReactNode;
  /** Optional footer content */
  footer?: React.ReactNode;
  /** Additional CSS class name */
  className?: string;
  /** Click handler for interactive cards */
  onClick?: () => void;
  /** Whether the card is disabled */
  disabled?: boolean;
  /** Custom CSS styles */
  style?: React.CSSProperties;
}

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
export function Card({
  variant = 'default',
  title,
  headerAction,
  children,
  footer,
  className = '',
  onClick,
  disabled = false,
  style,
}: CardProps): React.JSX.Element {
  const variantClasses = `card-${variant}`;
  const interactiveClasses = onClick && !disabled ? 'card-interactive' : '';
  const disabledClasses = disabled ? 'card-disabled' : '';

  const combinedClasses = `card ${variantClasses} ${interactiveClasses} ${disabledClasses} ${className}`.trim();

  return (
    <div
      className={combinedClasses}
      onClick={onClick}
      role={onClick ? 'button' : undefined}
      tabIndex={onClick && !disabled ? 0 : undefined}
      aria-disabled={disabled}
      style={style}
    >
      {(title || headerAction) && (
        <div className="card-header">
          {title && <h3 className="card-title">{title}</h3>}
          {headerAction && <div className="card-actions">{headerAction}</div>}
        </div>
      )}
      <div className="card-content">{children}</div>
      {footer && <div className="card-footer">{footer}</div>}
    </div>
  );
}

export default Card;
