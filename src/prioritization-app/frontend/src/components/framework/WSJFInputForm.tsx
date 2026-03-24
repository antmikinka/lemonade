/**
 * WSJF Input Form component for the Prioritization Frameworks application.
 * Provides input fields for Weighted Shortest Job First scoring.
 *
 * @module components/framework/WSJFInputForm
 */

import React, { useState, useCallback } from 'react';
import { WSJFInput } from '../../services/prioritization/types';
import { Card, Slider, Button } from '../common';

/**
 * Props for the WSJFInputForm component.
 */
export interface WSJFInputFormProps {
  /** Current input values */
  value?: Partial<WSJFInput>;
  /** Input change handler */
  onChange?: (input: WSJFInput) => void;
  /** Form submission handler */
  onSubmit?: (input: WSJFInput) => void;
  /** Whether the form is in loading state */
  isLoading?: boolean;
  /** Whether to show the submit button */
  showSubmitButton?: boolean;
  /** Custom class name */
  className?: string;
}

/**
 * WSJF Input Form component.
 *
 * This component provides a complete form for collecting WSJF scoring inputs:
 * - User-Business Value: Value to users and business (1-10 or Fibonacci)
 * - Time Criticality: How time-sensitive is this? (1-10 or Fibonacci)
 * - Risk Reduction & Opportunity Enablement: Risk mitigation value (1-10 or Fibonacci)
 * - Job Size: Estimated size/effort (lower = smaller/faster)
 *
 * WSJF Score = Cost of Delay / Job Size
 * Cost of Delay = User-Business Value + Time Criticality + Risk Reduction
 *
 * @param props - WSJFInputForm props
 * @returns Rendered WSJF input form
 *
 * @example
 * ```tsx
 * <WSJFInputForm
 *   value={{
 *     userBusinessValue: 8,
 *     timeCriticality: 6,
 *     riskReductionOpportunity: 5,
 *     jobSize: 3,
 *   }}
 *   onChange={handleWSJFChange}
 *   onSubmit={handleSubmit}
 * />
 * ```
 */
export function WSJFInputForm({
  value = {},
  onChange,
  onSubmit,
  isLoading = false,
  showSubmitButton = true,
  className = '',
}: WSJFInputFormProps): React.JSX.Element {
  const [localValue, setLocalValue] = useState<Partial<WSJFInput>>(value);

  const currentValue = { ...value, ...localValue };

  const handleChange = useCallback(
    (field: keyof WSJFInput) => (val: number | React.ChangeEvent<HTMLInputElement>) => {
      const newValue = typeof val === 'object' ? Number(val.target.value) : val;
      const updated = { ...currentValue, [field]: newValue };
      setLocalValue(updated);
      onChange?.(updated as WSJFInput);
    },
    [currentValue, onChange]
  );

  const handleSubmit = useCallback(
    (e: React.FormEvent) => {
      e.preventDefault();
      if (
        currentValue.userBusinessValue &&
        currentValue.timeCriticality &&
        currentValue.riskReductionOpportunity &&
        currentValue.jobSize
      ) {
        onSubmit?.({
          userBusinessValue: currentValue.userBusinessValue,
          timeCriticality: currentValue.timeCriticality,
          riskReductionOpportunity: currentValue.riskReductionOpportunity,
          jobSize: currentValue.jobSize,
        });
      }
    },
    [currentValue, onSubmit]
  );

  // Calculate preview values
  const costOfDelay = currentValue.userBusinessValue && currentValue.timeCriticality && currentValue.riskReductionOpportunity
    ? currentValue.userBusinessValue + currentValue.timeCriticality + currentValue.riskReductionOpportunity
    : null;

  const previewScore = costOfDelay && currentValue.jobSize
    ? (costOfDelay / currentValue.jobSize).toFixed(2)
    : null;

  return (
    <Card
      title="WSJF (Weighted Shortest Job First)"
      variant="default"
      className={`wsjf-input-form ${className}`}
    >
      <form onSubmit={handleSubmit} className="framework-form">
        <div className="wsjf-intro">
          <p className="wsjf-description">
            WSJF is a prioritization model used in SAFe to sequence jobs
            (features, capabilities, etc.) to produce maximum economic benefit.
          </p>
          <div className="wsjf-formula">
            <code>WSJF = Cost of Delay / Job Size</code>
          </div>
        </div>

        {/* Cost of Delay Components */}
        <div className="cod-section">
          <h4 className="section-title">Cost of Delay Components</h4>

          <Slider
            label="User-Business Value"
            value={currentValue.userBusinessValue ?? 5}
            onChange={handleChange('userBusinessValue')}
            min={1}
            max={20}
            step={1}
            showValue
            helperText="Value delivered to users and the business"
            marks={[
              { value: 1, label: 'Low' },
              { value: 10, label: 'Medium' },
              { value: 20, label: 'High' },
            ]}
          />

          <Slider
            label="Time Criticality"
            value={currentValue.timeCriticality ?? 5}
            onChange={handleChange('timeCriticality')}
            min={1}
            max={20}
            step={1}
            showValue
            helperText="How urgent is this? Does timing matter?"
            marks={[
              { value: 1, label: 'Flexible' },
              { value: 10, label: 'Time-boxed' },
              { value: 20, label: 'Fixed deadline' },
            ]}
          />

          <Slider
            label="Risk Reduction & Opportunity Enablement"
            value={currentValue.riskReductionOpportunity ?? 5}
            onChange={handleChange('riskReductionOpportunity')}
            min={1}
            max={20}
            step={1}
            showValue
            helperText="Does this reduce risk or enable future opportunities?"
            marks={[
              { value: 1, label: 'Low RR/OE' },
              { value: 10, label: 'Medium' },
              { value: 20, label: 'High RR/OE' },
            ]}
          />
        </div>

        {/* Job Size */}
        <div className="job-size-section">
          <h4 className="section-title">Job Size</h4>

          <Slider
            label="Job Size"
            value={currentValue.jobSize ?? 5}
            onChange={handleChange('jobSize')}
            min={1}
            max={20}
            step={1}
            showValue
            helperText="Relative size/effort (smaller = faster to complete)"
            marks={[
              { value: 1, label: 'Tiny' },
              { value: 10, label: 'Medium' },
              { value: 20, label: 'Large' },
            ]}
          />
        </div>

        {/* Score Preview */}
        {previewScore && costOfDelay && (
          <div className="score-preview">
            <span className="score-preview-label">Estimated WSJF Score:</span>
            <span className="score-preview-value">{previewScore}</span>
            <span className="score-preview-formula">
              ({currentValue.userBusinessValue} + {currentValue.timeCriticality} + {currentValue.riskReductionOpportunity}) / {currentValue.jobSize}
            </span>
          </div>
        )}

        {showSubmitButton && (
          <div className="form-actions">
            <Button
              type="submit"
              variant="primary"
              isLoading={isLoading}
              disabled={
                !currentValue.userBusinessValue ||
                !currentValue.timeCriticality ||
                !currentValue.riskReductionOpportunity ||
                !currentValue.jobSize
              }
            >
              Calculate WSJF Score
            </Button>
          </div>
        )}
      </form>
    </Card>
  );
}

export default WSJFInputForm;
