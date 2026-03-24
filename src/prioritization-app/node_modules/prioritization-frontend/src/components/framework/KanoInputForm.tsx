/**
 * Kano Input Form component for the Prioritization Frameworks application.
 * Provides input fields for Kano model analysis with functional/dysfunctional questions.
 *
 * @module components/framework/KanoInputForm
 */

import React, { useState, useCallback } from 'react';
import { KanoInput } from '../../services/prioritization/types';
import { Card, Slider, Button } from '../common';

/**
 * Props for the KanoInputForm component.
 */
export interface KanoInputFormProps {
  /** Current input values */
  value?: Partial<KanoInput>;
  /** Input change handler */
  onChange?: (input: KanoInput) => void;
  /** Form submission handler */
  onSubmit?: (input: KanoInput) => void;
  /** Whether the form is in loading state */
  isLoading?: boolean;
  /** Whether to show the submit button */
  showSubmitButton?: boolean;
  /** Custom class name */
  className?: string;
}

/**
 * Rating labels for the Kano questions.
 */
const ratingLabels = ['Dislike', "Don't Like", 'Neutral', 'Like', 'Like Very Much'];

/**
 * Kano Input Form component.
 *
 * This component provides a complete form for collecting Kano model inputs:
 * - Functional score: How much do you LIKE having this feature?
 * - Dysfunctional score: How much do you LIKE NOT having this feature?
 * - Importance rating
 * - Satisfaction if present / Dissatisfaction if absent
 *
 * @param props - KanoInputForm props
 * @returns Rendered Kano input form
 *
 * @example
 * ```tsx
 * <KanoInputForm
 *   value={{ functionalScore: 4, dysfunctionalScore: 2, importance: 5 }}
 *   onChange={handleKanoChange}
 *   onSubmit={handleSubmit}
 * />
 * ```
 */
export function KanoInputForm({
  value = {},
  onChange,
  onSubmit,
  isLoading = false,
  showSubmitButton = true,
  className = '',
}: KanoInputFormProps): React.JSX.Element {
  const [localValue, setLocalValue] = useState<Partial<KanoInput>>(value);

  const currentValue = { ...value, ...localValue };

  const handleChange = useCallback(
    (field: keyof KanoInput) =>
      (val: number | React.ChangeEvent<HTMLInputElement>) => {
        const newValue = typeof val === 'object' ? Number(val.target.value) : val;
        const updated = { ...currentValue, [field]: newValue };
        setLocalValue(updated);
        onChange?.(updated as KanoInput);
      },
    [currentValue, onChange]
  );

  const handleSubmit = useCallback(
    (e: React.FormEvent) => {
      e.preventDefault();
      if (currentValue.functionalScore && currentValue.dysfunctionalScore) {
        onSubmit?.({
          functionalScore: currentValue.functionalScore,
          dysfunctionalScore: currentValue.dysfunctionalScore,
          importance: currentValue.importance,
          satisfactionIfPresent: currentValue.satisfactionIfPresent,
          dissatisfactionIfAbsent: currentValue.dissatisfactionIfAbsent,
        });
      }
    },
    [currentValue, onSubmit]
  );

  return (
    <Card
      title="Kano Model Analysis"
      variant="default"
      className={`kano-input-form ${className}`}
    >
      <form onSubmit={handleSubmit} className="framework-form">
        <div className="kano-intro">
          <p className="kano-description">
            The Kano model evaluates features from the customer perspective.
            Answer the questions below to classify this feature.
          </p>
        </div>

        {/* Functional Question */}
        <div className="kano-question">
          <h4 className="kano-question-title">Functional Question</h4>
          <p className="kano-question-text">
            How do you feel about <strong>having</strong> this feature?
          </p>
          <Slider
            label="Rating"
            value={currentValue.functionalScore ?? 3}
            onChange={handleChange('functionalScore')}
            min={1}
            max={5}
            step={1}
            showValue
            marks={ratingLabels.map((label, index) => ({
              value: index + 1,
              label,
            }))}
          />
        </div>

        {/* Dysfunctional Question */}
        <div className="kano-question">
          <h4 className="kano-question-title">Dysfunctional Question</h4>
          <p className="kano-question-text">
            How do you feel about <strong>not having</strong> this feature?
          </p>
          <Slider
            label="Rating"
            value={currentValue.dysfunctionalScore ?? 3}
            onChange={handleChange('dysfunctionalScore')}
            min={1}
            max={5}
            step={1}
            showValue
            marks={ratingLabels.map((label, index) => ({
              value: index + 1,
              label,
            }))}
          />
        </div>

        {/* Additional Optional Fields */}
        <div className="form-grid form-grid-3">
          <Slider
            label="Importance (Optional)"
            value={currentValue.importance ?? 3}
            onChange={handleChange('importance')}
            min={1}
            max={5}
            step={1}
            showValue
          />

          <Slider
            label="Satisfaction if Present (Optional)"
            value={currentValue.satisfactionIfPresent ?? 3}
            onChange={handleChange('satisfactionIfPresent')}
            min={1}
            max={5}
            step={1}
            showValue
          />

          <Slider
            label="Dissatisfaction if Absent (Optional)"
            value={currentValue.dissatisfactionIfAbsent ?? 3}
            onChange={handleChange('dissatisfactionIfAbsent')}
            min={1}
            max={5}
            step={1}
            showValue
          />
        </div>

        {showSubmitButton && (
          <div className="form-actions">
            <Button
              type="submit"
              variant="primary"
              isLoading={isLoading}
              disabled={!currentValue.functionalScore || !currentValue.dysfunctionalScore}
            >
              Analyze with Kano Model
            </Button>
          </div>
        )}
      </form>
    </Card>
  );
}

export default KanoInputForm;
