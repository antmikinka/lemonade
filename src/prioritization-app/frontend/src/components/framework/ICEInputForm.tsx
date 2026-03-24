/**
 * ICE Input Form component for the Prioritization Frameworks application.
 * Provides input sliders for Impact, Confidence, and Ease scoring.
 *
 * @module components/framework/ICEInputForm
 */

import React, { useState, useCallback } from 'react';
import { ICEInput } from '../../services/prioritization/types';
import { Card, Slider, Button } from '../common';

/**
 * Props for the ICEInputForm component.
 */
export interface ICEInputFormProps {
  /** Current input values */
  value?: Partial<ICEInput>;
  /** Input change handler */
  onChange?: (input: ICEInput) => void;
  /** Form submission handler */
  onSubmit?: (input: ICEInput) => void;
  /** Whether the form is in loading state */
  isLoading?: boolean;
  /** Whether to show the submit button */
  showSubmitButton?: boolean;
  /** Custom class name */
  className?: string;
}

/**
 * ICE Input Form component.
 *
 * This component provides a complete form for collecting ICE scoring inputs:
 * - Impact: How much will this help? (1-10)
 * - Confidence: How confident are we? (1-10)
 * - Ease: How easy is this to implement? (1-10)
 *
 * ICE Score = Impact x Confidence x Ease
 *
 * @param props - ICEInputForm props
 * @returns Rendered ICE input form
 *
 * @example
 * ```tsx
 * <ICEInputForm
 *   value={{ impact: 7, confidence: 8, ease: 6 }}
 *   onChange={handleICEChange}
 *   onSubmit={handleSubmit}
 * />
 * ```
 */
export function ICEInputForm({
  value = {},
  onChange,
  onSubmit,
  isLoading = false,
  showSubmitButton = true,
  className = '',
}: ICEInputFormProps): React.JSX.Element {
  const [localValue, setLocalValue] = useState<Partial<ICEInput>>(value);

  const currentValue = { ...value, ...localValue };

  const handleChange = useCallback(
    (field: keyof ICEInput) => (val: number) => {
      const updated = { ...currentValue, [field]: val };
      setLocalValue(updated);
      onChange?.(updated as ICEInput);
    },
    [currentValue, onChange]
  );

  const handleSubmit = useCallback(
    (e: React.FormEvent) => {
      e.preventDefault();
      if (currentValue.impact && currentValue.confidence && currentValue.ease) {
        onSubmit?.({
          impact: currentValue.impact,
          confidence: currentValue.confidence,
          ease: currentValue.ease,
        });
      }
    },
    [currentValue, onSubmit]
  );

  // Calculate preview score
  const previewScore = currentValue.impact && currentValue.confidence && currentValue.ease
    ? (currentValue.impact * currentValue.confidence * currentValue.ease).toFixed(1)
    : null;

  return (
    <Card
      title="ICE Scoring"
      variant="default"
      className={`ice-input-form ${className}`}
    >
      <form onSubmit={handleSubmit} className="framework-form">
        <div className="ice-intro">
          <p className="ice-description">
            ICE is a simplified prioritization framework that balances
            impact, confidence, and ease of implementation.
          </p>
        </div>

        {/* Impact Slider */}
        <Slider
          label="Impact"
          value={currentValue.impact ?? 5}
          onChange={handleChange('impact')}
          min={1}
          max={10}
          step={1}
          showValue
          helperText="How much will this initiative help achieve your goals?"
          marks={[
            { value: 1, label: 'Low' },
            { value: 5, label: 'Medium' },
            { value: 10, label: 'High' },
          ]}
        />

        {/* Confidence Slider */}
        <Slider
          label="Confidence"
          value={currentValue.confidence ?? 5}
          onChange={handleChange('confidence')}
          min={1}
          max={10}
          step={1}
          showValue
          helperText="How confident are you in your impact estimate?"
          marks={[
            { value: 1, label: 'Uncertain' },
            { value: 5, label: 'Moderate' },
            { value: 10, label: 'Certain' },
          ]}
        />

        {/* Ease Slider */}
        <Slider
          label="Ease"
          value={currentValue.ease ?? 5}
          onChange={handleChange('ease')}
          min={1}
          max={10}
          step={1}
          showValue
          helperText="How easy is this to implement? (Higher = easier)"
          marks={[
            { value: 1, label: 'Hard' },
            { value: 5, label: 'Moderate' },
            { value: 10, label: 'Easy' },
          ]}
        />

        {/* Score Preview */}
        {previewScore && (
          <div className="score-preview">
            <span className="score-preview-label">Estimated ICE Score:</span>
            <span className="score-preview-value">{previewScore}</span>
            <span className="score-preview-formula">
              ({currentValue.impact} x {currentValue.confidence} x {currentValue.ease})
            </span>
          </div>
        )}

        {showSubmitButton && (
          <div className="form-actions">
            <Button
              type="submit"
              variant="primary"
              isLoading={isLoading}
              disabled={!currentValue.impact || !currentValue.confidence || !currentValue.ease}
            >
              Calculate ICE Score
            </Button>
          </div>
        )}
      </form>
    </Card>
  );
}

export default ICEInputForm;
