/**
 * Eisenhower Input Form component for the Prioritization Frameworks application.
 * Provides input toggles for Eisenhower Matrix (Urgent vs Important).
 *
 * @module components/framework/EisenhowerInputForm
 */

import React, { useState, useCallback } from 'react';
import { EisenhowerInput } from '../../services/prioritization/types';
import { Card, Slider, Button } from '../common';

/**
 * Props for the EisenhowerInputForm component.
 */
export interface EisenhowerInputFormProps {
  /** Current input values */
  value?: Partial<EisenhowerInput>;
  /** Input change handler */
  onChange?: (input: EisenhowerInput) => void;
  /** Form submission handler */
  onSubmit?: (input: EisenhowerInput) => void;
  /** Whether the form is in loading state */
  isLoading?: boolean;
  /** Whether to show the submit button */
  showSubmitButton?: boolean;
  /** Custom class name */
  className?: string;
}

/**
 * Toggle Switch component for binary inputs.
 */
function ToggleSwitch({
  checked,
  onChange,
  label,
  description,
}: {
  checked: boolean;
  onChange: (checked: boolean) => void;
  label: string;
  description?: string;
}): React.JSX.Element {
  return (
    <label className="toggle-switch">
      <input
        type="checkbox"
        checked={checked}
        onChange={(e) => onChange(e.target.checked)}
        className="toggle-input"
        aria-label={label}
      />
      <span className="toggle-slider"></span>
      <div className="toggle-content">
        <span className="toggle-label">{label}</span>
        {description && (
          <span className="toggle-description">{description}</span>
        )}
      </div>
    </label>
  );
}

/**
 * Eisenhower Input Form component.
 *
 * This component provides a complete form for collecting Eisenhower Matrix inputs:
 * - Urgent: Is this time-sensitive?
 * - Important: Does this have high impact?
 * - Urgency Level: Optional fine-grained urgency (1-10)
 * - Importance Level: Optional fine-grained importance (1-10)
 *
 * The combination determines the quadrant:
 * - Urgent + Important = Do First
 * - Not Urgent + Important = Schedule
 * - Urgent + Not Important = Delegate
 * - Not Urgent + Not Important = Eliminate
 *
 * @param props - EisenhowerInputForm props
 * @returns Rendered Eisenhower input form
 *
 * @example
 * ```tsx
 * <EisenhowerInputForm
 *   value={{ urgent: true, important: true, urgencyLevel: 8, importanceLevel: 9 }}
 *   onChange={handleEisenhowerChange}
 *   onSubmit={handleSubmit}
 * />
 * ```
 */
export function EisenhowerInputForm({
  value = {},
  onChange,
  onSubmit,
  isLoading = false,
  showSubmitButton = true,
  className = '',
}: EisenhowerInputFormProps): React.JSX.Element {
  const [localValue, setLocalValue] = useState<Partial<EisenhowerInput>>(value);

  const currentValue = { ...value, ...localValue };

  const handleToggleChange = useCallback(
    (field: 'urgent' | 'important') => (val: boolean) => {
      const updated = { ...currentValue, [field]: val };
      setLocalValue(updated);
      onChange?.(updated as EisenhowerInput);
    },
    [currentValue, onChange]
  );

  const handleLevelChange = useCallback(
    (field: 'urgencyLevel' | 'importanceLevel') => (val: number) => {
      const updated = { ...currentValue, [field]: val };
      setLocalValue(updated);
      onChange?.(updated as EisenhowerInput);
    },
    [currentValue, onChange]
  );

  const handleSubmit = useCallback(
    (e: React.FormEvent) => {
      e.preventDefault();
      if (currentValue.urgent !== undefined && currentValue.important !== undefined) {
        onSubmit?.({
          urgent: currentValue.urgent,
          important: currentValue.important,
          urgencyLevel: currentValue.urgencyLevel,
          importanceLevel: currentValue.importanceLevel,
        });
      }
    },
    [currentValue, onSubmit]
  );

  // Determine preview quadrant
  const getPreviewQuadrant = (): string | null => {
    if (currentValue.urgent === undefined || currentValue.important === undefined) {
      return null;
    }
    if (currentValue.urgent && currentValue.important) return 'Do First';
    if (!currentValue.urgent && currentValue.important) return 'Schedule';
    if (currentValue.urgent && !currentValue.important) return 'Delegate';
    return 'Eliminate';
  };

  const previewQuadrant = getPreviewQuadrant();

  return (
    <Card
      title="Eisenhower Matrix"
      variant="default"
      className={`eisenhower-input-form ${className}`}
    >
      <form onSubmit={handleSubmit} className="framework-form">
        <div className="eisenhower-intro">
          <p className="eisenhower-description">
            The Eisenhower Matrix helps prioritize tasks by urgency and importance.
            Categorize each task to determine the best action.
          </p>
        </div>

        {/* Toggle Switches */}
        <div className="toggle-group">
          <ToggleSwitch
            checked={currentValue.urgent ?? false}
            onChange={handleToggleChange('urgent')}
            label="Urgent"
            description="This task is time-sensitive and requires immediate attention"
          />

          <ToggleSwitch
            checked={currentValue.important ?? false}
            onChange={handleToggleChange('important')}
            label="Important"
            description="This task has significant impact on goals and objectives"
          />
        </div>

        {/* Optional Level Sliders */}
        <div className="form-grid form-grid-2">
          <Slider
            label="Urgency Level (Optional)"
            value={currentValue.urgencyLevel ?? 5}
            onChange={handleLevelChange('urgencyLevel')}
            min={1}
            max={10}
            step={1}
            showValue
            helperText="Fine-tune urgency within category"
          />

          <Slider
            label="Importance Level (Optional)"
            value={currentValue.importanceLevel ?? 5}
            onChange={handleLevelChange('importanceLevel')}
            min={1}
            max={10}
            step={1}
            showValue
            helperText="Fine-tune importance within category"
          />
        </div>

        {/* Quadrant Preview */}
        {previewQuadrant && (
          <div className="quadrant-preview">
            <span className="quadrant-label">Quadrant:</span>
            <span className={`quadrant-badge quadrant-${previewQuadrant.toLowerCase().replace(' ', '-')}`}>
              {previewQuadrant}
            </span>
          </div>
        )}

        {showSubmitButton && (
          <div className="form-actions">
            <Button
              type="submit"
              variant="primary"
              isLoading={isLoading}
              disabled={currentValue.urgent === undefined || currentValue.important === undefined}
            >
              Categorize Task
            </Button>
          </div>
        )}
      </form>
    </Card>
  );
}

export default EisenhowerInputForm;
