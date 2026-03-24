/**
 * P0-P4 Input Form component for the Prioritization Frameworks application.
 * Provides input sliders for P0-P4 priority classification.
 *
 * @module components/framework/P0P4InputForm
 */

import React, { useState, useCallback } from 'react';
import { P0P4Input, SeverityFactors } from '../../services/prioritization/types';
import { Card, Slider, Select, Input, Button } from '../common';

/**
 * Props for the P0P4InputForm component.
 */
export interface P0P4InputFormProps {
  /** Current input values */
  value?: Partial<P0P4Input>;
  /** Input change handler */
  onChange?: (input: P0P4Input) => void;
  /** Form submission handler */
  onSubmit?: (input: P0P4Input) => void;
  /** Whether the form is in loading state */
  isLoading?: boolean;
  /** Whether to show the submit button */
  showSubmitButton?: boolean;
  /** Custom class name */
  className?: string;
}

/**
 * Severity factor options.
 */
const severityOptions = [
  { value: 'critical', label: 'Critical' },
  { value: 'high', label: 'High' },
  { value: 'medium', label: 'Medium' },
  { value: 'low', label: 'Low' },
  { value: 'none', label: 'None' },
];

const usersAffectedOptions = [
  { value: 'all', label: 'All users' },
  { value: 'many', label: 'Many users' },
  { value: 'some', label: 'Some users' },
  { value: 'few', label: 'Few users' },
  { value: 'none', label: 'No users' },
];

/**
 * P0-P4 Input Form component.
 *
 * This component provides a complete form for collecting P0-P4 priority inputs:
 * - Base Severity: Overall severity level (1-5)
 * - Severity Factors: Detailed breakdown of impact areas
 * - Open Issues Count: Number of related issues
 * - Days Open: How long the issue has been reported
 *
 * @param props - P0P4InputForm props
 * @returns Rendered P0-P4 input form
 *
 * @example
 * ```tsx
 * <P0P4InputForm
 *   value={{
 *     baseSeverity: 4,
 *     severityFactors: {
 *       usersAffected: 'many',
 *       coreFunctionalityImpact: 'high',
 *       securityRisk: 'medium',
 *       reputationalRisk: 'high',
 *       revenueImpact: 'high',
 *     },
 *     openIssuesCount: 5,
 *     daysOpen: 3,
 *   }}
 *   onChange={handleP0P4Change}
 *   onSubmit={handleSubmit}
 * />
 * ```
 */
export function P0P4InputForm({
  value = {},
  onChange,
  onSubmit,
  isLoading = false,
  showSubmitButton = true,
  className = '',
}: P0P4InputFormProps): React.JSX.Element {
  const [localValue, setLocalValue] = useState<Partial<P0P4Input>>(value);

  const currentValue = { ...value, ...localValue };
  const currentSeverityFactors = currentValue.severityFactors || {
    usersAffected: 'none',
    coreFunctionalityImpact: 'none',
    securityRisk: 'none',
    reputationalRisk: 'none',
    revenueImpact: 'none',
  };

  const handleSeverityChange = useCallback(
    (field: keyof SeverityFactors) => (val: string) => {
      const updatedFactors = { ...currentSeverityFactors, [field]: val };
      const updated = { ...currentValue, severityFactors: updatedFactors };
      setLocalValue(updated);
      onChange?.(updated as P0P4Input);
    },
    [currentSeverityFactors, currentValue, onChange]
  );

  const handleBaseSeverityChange = useCallback(
    (val: number) => {
      const updated = { ...currentValue, baseSeverity: val };
      setLocalValue(updated);
      onChange?.(updated as P0P4Input);
    },
    [currentValue, onChange]
  );

  const handleNumericChange = useCallback(
    (field: 'openIssuesCount' | 'daysOpen') =>
      (val: number | React.ChangeEvent<HTMLInputElement>) => {
        const newValue = typeof val === 'object' ? Number(val.target.value) : val;
        const updated = { ...currentValue, [field]: newValue };
        setLocalValue(updated);
        onChange?.(updated as P0P4Input);
      },
    [currentValue, onChange]
  );

  const handleSubmit = useCallback(
    (e: React.FormEvent) => {
      e.preventDefault();
      if (currentValue.baseSeverity && currentValue.severityFactors) {
        onSubmit?.({
          baseSeverity: currentValue.baseSeverity,
          severityFactors: {
            usersAffected: currentValue.severityFactors.usersAffected || 'none',
            coreFunctionalityImpact: currentValue.severityFactors.coreFunctionalityImpact || 'none',
            securityRisk: currentValue.severityFactors.securityRisk || 'none',
            reputationalRisk: currentValue.severityFactors.reputationalRisk || 'none',
            revenueImpact: currentValue.severityFactors.revenueImpact || 'none',
          },
          openIssuesCount: currentValue.openIssuesCount,
          daysOpen: currentValue.daysOpen,
        });
      }
    },
    [currentValue, onSubmit]
  );

  // Calculate preview priority
  const getPreviewPriority = (): string | null => {
    if (!currentValue.baseSeverity) return null;
    const severity = currentValue.baseSeverity;
    if (severity >= 5) return 'P0 - Critical';
    if (severity >= 4) return 'P1 - High';
    if (severity >= 3) return 'P2 - Medium';
    if (severity >= 2) return 'P3 - Low';
    return 'P4 - Lowest';
  };

  const previewPriority = getPreviewPriority();

  return (
    <Card
      title="P0-P4 Priority Classification"
      variant="default"
      className={`p0p4-input-form ${className}`}
    >
      <form onSubmit={handleSubmit} className="framework-form">
        <div className="p0p4-intro">
          <p className="p0p4-description">
            Classify items by priority level from P0 (critical) to P4 (lowest).
            Higher severity scores indicate more urgent priorities.
          </p>
        </div>

        {/* Base Severity Slider */}
        <Slider
          label="Base Severity"
          value={currentValue.baseSeverity ?? 3}
          onChange={handleBaseSeverityChange}
          min={1}
          max={5}
          step={1}
          showValue
          helperText="Overall severity rating"
          marks={[
            { value: 1, label: 'P4' },
            { value: 2, label: 'P3' },
            { value: 3, label: 'P2' },
            { value: 4, label: 'P1' },
            { value: 5, label: 'P0' },
          ]}
        />

        {/* Severity Factors */}
        <div className="severity-factors-section">
          <h4 className="section-title">Severity Factors</h4>
          <div className="form-grid form-grid-2">
            <Select
              label="Users Affected"
              options={usersAffectedOptions}
              value={currentSeverityFactors.usersAffected || 'none'}
              onChange={(e) => handleSeverityChange('usersAffected')(e.target.value)}
              placeholder="Select level"
            />

            <Select
              label="Core Functionality Impact"
              options={severityOptions}
              value={currentSeverityFactors.coreFunctionalityImpact || 'none'}
              onChange={(e) => handleSeverityChange('coreFunctionalityImpact')(e.target.value)}
              placeholder="Select level"
            />

            <Select
              label="Security Risk"
              options={severityOptions}
              value={currentSeverityFactors.securityRisk || 'none'}
              onChange={(e) => handleSeverityChange('securityRisk')(e.target.value)}
              placeholder="Select level"
            />

            <Select
              label="Reputational Risk"
              options={severityOptions}
              value={currentSeverityFactors.reputationalRisk || 'none'}
              onChange={(e) => handleSeverityChange('reputationalRisk')(e.target.value)}
              placeholder="Select level"
            />

            <Select
              label="Revenue Impact"
              options={severityOptions}
              value={currentSeverityFactors.revenueImpact || 'none'}
              onChange={(e) => handleSeverityChange('revenueImpact')(e.target.value)}
              placeholder="Select level"
            />
          </div>
        </div>

        {/* Additional Fields */}
        <div className="form-grid form-grid-2">
          <Input
            label="Open Issues Count"
            type="number"
            value={currentValue.openIssuesCount ?? ''}
            onChange={handleNumericChange('openIssuesCount')}
            placeholder="e.g., 5"
            min={0}
          />

          <Input
            label="Days Open"
            type="number"
            value={currentValue.daysOpen ?? ''}
            onChange={handleNumericChange('daysOpen')}
            placeholder="e.g., 3"
            min={0}
          />
        </div>

        {/* Priority Preview */}
        {previewPriority && (
          <div className="quadrant-preview">
            <span className="quadrant-label">Estimated Priority:</span>
            <span className="quadrant-badge quadrant-priority">
              {previewPriority}
            </span>
          </div>
        )}

        {showSubmitButton && (
          <div className="form-actions">
            <Button
              type="submit"
              variant="primary"
              isLoading={isLoading}
              disabled={!currentValue.baseSeverity}
            >
              Classify Priority
            </Button>
          </div>
        )}
      </form>
    </Card>
  );
}

export default P0P4InputForm;
