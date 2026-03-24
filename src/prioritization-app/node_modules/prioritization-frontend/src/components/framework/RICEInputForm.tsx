/**
 * RICE Input Form component for the Prioritization Frameworks application.
 * Provides input fields for Reach, Impact, Confidence, and Effort scoring.
 *
 * @module components/framework/RICEInputForm
 */

import React, { useState, useCallback } from 'react';
import { RICEInput } from '../../services/prioritization/types';
import { Card, Input, Slider, Button } from '../common';

/**
 * Props for the RICEInputForm component.
 */
export interface RICEInputFormProps {
  /** Current input values */
  value?: Partial<RICEInput>;
  /** Input change handler */
  onChange?: (input: RICEInput) => void;
  /** Form submission handler */
  onSubmit?: (input: RICEInput) => void;
  /** Whether the form is in loading state */
  isLoading?: boolean;
  /** Whether to show the submit button */
  showSubmitButton?: boolean;
  /** Custom class name */
  className?: string;
}

/**
 * Impact level options for the select dropdown.
 */
const impactOptions = [
  { value: 3, label: '3 - Massive impact' },
  { value: 2, label: '2 - High impact' },
  { value: 1, label: '1 - Medium impact' },
  { value: 0.5, label: '0.5 - Low impact' },
  { value: 0.25, label: '0.25 - Minimal impact' },
];

/**
 * RICE Input Form component.
 *
 * This component provides a complete form for collecting RICE scoring inputs:
 * - Reach: Number of users/events affected
 * - Impact: Impact multiplier (standard scale)
 * - Confidence: Confidence level percentage
 * - Effort: Person-months or story points
 *
 * @param props - RICEInputForm props
 * @returns Rendered RICE input form
 *
 * @example
 * ```tsx
 * <RICEInputForm
 *   value={{ reach: 100, impact: 2, confidence: 80, effort: 3 }}
 *   onChange={handleRICEChange}
 *   onSubmit={handleSubmit}
 * />
 * ```
 */
export function RICEInputForm({
  value = {},
  onChange,
  onSubmit,
  isLoading = false,
  showSubmitButton = true,
  className = '',
}: RICEInputFormProps): React.JSX.Element {
  const [localValue, setLocalValue] = useState<Partial<RICEInput>>(value);

  const currentValue = { ...value, ...localValue };

  const handleChange = useCallback(
    (field: keyof RICEInput) =>
      (val: number | React.ChangeEvent<HTMLInputElement>) => {
        const newValue = typeof val === 'object' ? Number(val.target.value) : val;
        const updated = { ...currentValue, [field]: newValue };
        setLocalValue(updated);
        onChange?.(updated as RICEInput);
      },
    [currentValue, onChange]
  );

  const handleSubmit = useCallback(
    (e: React.FormEvent) => {
      e.preventDefault();
      if (currentValue.reach && currentValue.impact && currentValue.confidence && currentValue.effort) {
        onSubmit?.(currentValue as RICEInput);
      }
    },
    [currentValue, onSubmit]
  );

  return (
    <Card
      title="RICE Scoring"
      variant="default"
      className={`rice-input-form ${className}`}
    >
      <form onSubmit={handleSubmit} className="framework-form">
        <div className="form-grid form-grid-2">
          {/* Reach Input */}
          <Input
            label="Reach"
            type="number"
            value={currentValue.reach ?? ''}
            onChange={handleChange('reach')}
            placeholder="e.g., 100 users per month"
            helperText="Number of users or events affected per period"
            min={0}
            step={1}
            fullWidth
          />

          {/* Impact Select */}
          <div className="form-field">
            <label className="input-label" htmlFor="rice-impact">
              Impact
            </label>
            <select
              id="rice-impact"
              className="select-field select-md"
              value={currentValue.impact ?? ''}
              onChange={(e) => handleChange('impact')(Number(e.target.value))}
            >
              <option value="">Select impact level</option>
              {impactOptions.map((opt) => (
                <option key={opt.value} value={opt.value}>
                  {opt.label}
                </option>
              ))}
            </select>
            <p className="input-helper-text">
              Impact on each affected user
            </p>
          </div>
        </div>

        {/* Confidence Slider */}
        <Slider
          label="Confidence"
          value={currentValue.confidence ?? 50}
          onChange={handleChange('confidence')}
          min={0}
          max={100}
          step={5}
          showValue
          valueFormat="${value}%"
          helperText="How confident are you in your estimates?"
          marks={[
            { value: 0, label: 'Low' },
            { value: 50, label: 'Medium' },
            { value: 100, label: 'High' },
          ]}
        />

        {/* Effort Input */}
        <Input
          label="Effort"
          type="number"
          value={currentValue.effort ?? ''}
          onChange={handleChange('effort')}
          placeholder="e.g., 3 person-months"
          helperText="Estimated effort in person-months or story points"
          min={0}
          step={0.5}
          fullWidth
        />

        {showSubmitButton && (
          <div className="form-actions">
            <Button
              type="submit"
              variant="primary"
              isLoading={isLoading}
              disabled={
                !currentValue.reach ||
                !currentValue.impact ||
                !currentValue.confidence ||
                !currentValue.effort
              }
            >
              Calculate RICE Score
            </Button>
          </div>
        )}
      </form>
    </Card>
  );
}

export default RICEInputForm;
