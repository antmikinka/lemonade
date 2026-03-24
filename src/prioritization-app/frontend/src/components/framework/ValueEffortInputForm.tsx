/**
 * Value vs Effort Input Form component for the Prioritization Frameworks application.
 * Provides input fields for Value vs Effort matrix scoring.
 *
 * @module components/framework/ValueEffortInputForm
 */

import React, { useState, useCallback } from 'react';
import { ValueEffortInput } from '../../services/prioritization/types';
import { Card, Select, Button } from '../common';

/**
 * Props for the ValueEffortInputForm component.
 */
export interface ValueEffortInputFormProps {
  /** Current input values */
  value?: Partial<ValueEffortInput>;
  /** Input change handler */
  onChange?: (input: ValueEffortInput) => void;
  /** Form submission handler */
  onSubmit?: (input: ValueEffortInput) => void;
  /** Whether the form is in loading state */
  isLoading?: boolean;
  /** Whether to show the submit button */
  showSubmitButton?: boolean;
  /** Custom class name */
  className?: string;
}

/**
 * Value/Effort level options.
 */
const levelOptions = [
  { value: 'high', label: 'High' },
  { value: 'medium', label: 'Medium' },
  { value: 'low', label: 'Low' },
];

/**
 * Numeric mapping for value/effort levels.
 */
const levelToNumber: Record<string, number> = {
  high: 3,
  medium: 2,
  low: 1,
};

/**
 * Value vs Effort Input Form component.
 *
 * This component provides a complete form for collecting Value vs Effort inputs:
 * - Value: Business value score (High/Medium/Low)
 * - Effort: Implementation effort (High/Medium/Low)
 *
 * The combination determines the quadrant:
 * - High Value + Low Effort = Quick Win
 * - High Value + High Effort = Major Project
 * - Low Value + Low Effort = Fill In
 * - Low Value + High Effort = Avoid
 *
 * @param props - ValueEffortInputForm props
 * @returns Rendered Value vs Effort input form
 *
 * @example
 * ```tsx
 * <ValueEffortInputForm
 *   value={{ value: 3, effort: 2 }}
 *   onChange={handleValueEffortChange}
 *   onSubmit={handleSubmit}
 * />
 * ```
 */
export function ValueEffortInputForm({
  value = {},
  onChange,
  onSubmit,
  isLoading = false,
  showSubmitButton = true,
  className = '',
}: ValueEffortInputFormProps): React.JSX.Element {
  const [localValue, setLocalValue] = useState<Partial<ValueEffortInput>>(value);

  const currentValue = { ...value, ...localValue };

  // Convert numeric values to string labels for selects
  const getValueLabel = (val?: number): string => {
    if (val === undefined) return '';
    if (val >= 7) return 'high';
    if (val >= 4) return 'medium';
    return 'low';
  };

  const getNumericValue = (label: string): number => {
    return levelToNumber[label] || 0;
  };

  const handleChange = useCallback(
    (field: keyof ValueEffortInput) =>
      (val: string | React.ChangeEvent<HTMLSelectElement>) => {
        const newValue = typeof val === 'object' ? val.target.value : val;
        const numericValue = getNumericValue(newValue);
        const updated = { ...currentValue, [field]: numericValue };
        setLocalValue(updated);
        onChange?.(updated as ValueEffortInput);
      },
    [currentValue, onChange]
  );

  const handleSubmit = useCallback(
    (e: React.FormEvent) => {
      e.preventDefault();
      if (currentValue.value && currentValue.effort) {
        onSubmit?.({
          value: currentValue.value,
          effort: currentValue.effort,
        });
      }
    },
    [currentValue, onSubmit]
  );

  return (
    <Card
      title="Value vs Effort Matrix"
      variant="default"
      className={`value-effort-input-form ${className}`}
    >
      <form onSubmit={handleSubmit} className="framework-form">
        <div className="form-grid form-grid-2">
          {/* Value Select */}
          <Select
            label="Business Value"
            options={levelOptions}
            value={getValueLabel(currentValue.value)}
            onChange={(e) => handleChange('value')(e.target.value)}
            placeholder="Select value level"
            helperText="How much business value does this deliver?"
          />

          {/* Effort Select */}
          <Select
            label="Implementation Effort"
            options={levelOptions}
            value={getValueLabel(currentValue.effort)}
            onChange={(e) => handleChange('effort')(e.target.value)}
            placeholder="Select effort level"
            helperText="How much effort is required to implement?"
          />
        </div>

        {/* Quadrant preview */}
        {currentValue.value && currentValue.effort && (
          <div className="quadrant-preview">
            <span className="quadrant-label">Expected Quadrant:</span>
            <span className={`quadrant-badge quadrant-${getQuadrant(currentValue.value, currentValue.effort).toLowerCase()}`}>
              {getQuadrant(currentValue.value, currentValue.effort)}
            </span>
          </div>
        )}

        {showSubmitButton && (
          <div className="form-actions">
            <Button
              type="submit"
              variant="primary"
              isLoading={isLoading}
              disabled={!currentValue.value || !currentValue.effort}
            >
              Plot on Matrix
            </Button>
          </div>
        )}
      </form>
    </Card>
  );
}

/**
 * Determines the quadrant based on value and effort scores.
 */
function getQuadrant(value: number, effort: number): string {
  const isHighValue = value >= 7;
  const isHighEffort = effort >= 7;

  if (isHighValue && !isHighEffort) return 'Quick Win';
  if (isHighValue && isHighEffort) return 'Major Project';
  if (!isHighValue && !isHighEffort) return 'Fill In';
  return 'Avoid';
}

export default ValueEffortInputForm;
