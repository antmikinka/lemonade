/**
 * MoSCoW Input Form component for the Prioritization Frameworks application.
 * Provides input fields for MoSCoW categorization (Must, Should, Could, Won't have).
 *
 * @module components/framework/MoSCoWInputForm
 */

import React, { useState, useCallback } from 'react';
import { MoSCoWInput, BusinessValueLevel, RiskLevel } from '../../services/prioritization/types';
import { Card, Select, Button } from '../common';

/**
 * Props for the MoSCoWInputForm component.
 */
export interface MoSCoWInputFormProps {
  /** Current input values */
  value?: Partial<MoSCoWInput>;
  /** Input change handler */
  onChange?: (input: MoSCoWInput) => void;
  /** Form submission handler */
  onSubmit?: (input: MoSCoWInput) => void;
  /** Whether the form is in loading state */
  isLoading?: boolean;
  /** Whether to show the submit button */
  showSubmitButton?: boolean;
  /** Custom class name */
  className?: string;
}

/**
 * Business value level options.
 */
const businessValueOptions = [
  { value: 'critical', label: 'Critical - Core business requirement' },
  { value: 'high', label: 'High - Significant business value' },
  { value: 'medium', label: 'Medium - Moderate business value' },
  { value: 'low', label: 'Low - Minimal business value' },
];

/**
 * Risk level options.
 */
const riskLevelOptions = [
  { value: 'critical', label: 'Critical - Severe consequences' },
  { value: 'high', label: 'High - Major impact' },
  { value: 'medium', label: 'Medium - Moderate impact' },
  { value: 'low', label: 'Low - Minimal impact' },
];

/**
 * MoSCoW Input Form component.
 *
 * This component provides a complete form for collecting MoSCoW categorization inputs:
 * - Business Value: Criticality to business
 * - Legal Requirement: Compliance necessity
 * - Customer Request: Customer-driven requirement
 * - Risk if Not Delivered: Impact of omission
 *
 * @param props - MoSCoWInputForm props
 * @returns Rendered MoSCoW input form
 *
 * @example
 * ```tsx
 * <MoSCoWInputForm
 *   value={{ businessValue: 'high', legalRequirement: false, customerRequest: true, riskIfNotDelivered: 'medium' }}
 *   onChange={handleMoSCoWChange}
 *   onSubmit={handleSubmit}
 * />
 * ```
 */
export function MoSCoWInputForm({
  value = {},
  onChange,
  onSubmit,
  isLoading = false,
  showSubmitButton = true,
  className = '',
}: MoSCoWInputFormProps): React.JSX.Element {
  const [localValue, setLocalValue] = useState<Partial<MoSCoWInput>>(value);

  const currentValue = { ...value, ...localValue };

  const handleChange = useCallback(
    (field: keyof MoSCoWInput) =>
      (val: BusinessValueLevel | RiskLevel | boolean | React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
        const newValue = typeof val === 'object' ? val.target.type === 'checkbox' ? val.target.checked : val.target.value : val;
        const updated = { ...currentValue, [field]: newValue };
        setLocalValue(updated);
        onChange?.(updated as MoSCoWInput);
      },
    [currentValue, onChange]
  );

  const handleSubmit = useCallback(
    (e: React.FormEvent) => {
      e.preventDefault();
      if (currentValue.businessValue && currentValue.riskIfNotDelivered !== undefined) {
        onSubmit?.({
          businessValue: currentValue.businessValue,
          legalRequirement: currentValue.legalRequirement ?? false,
          customerRequest: currentValue.customerRequest ?? false,
          riskIfNotDelivered: currentValue.riskIfNotDelivered,
        });
      }
    },
    [currentValue, onSubmit]
  );

  return (
    <Card
      title="MoSCoW Categorization"
      variant="default"
      className={`moscow-input-form ${className}`}
    >
      <form onSubmit={handleSubmit} className="framework-form">
        <div className="form-grid form-grid-2">
          {/* Business Value Select */}
          <Select
            label="Business Value"
            options={businessValueOptions}
            value={currentValue.businessValue ?? ''}
            onChange={(e) => handleChange('businessValue')(e.target.value as BusinessValueLevel)}
            placeholder="Select business value level"
            helperText="How critical is this to business success?"
          />

          {/* Risk if Not Delivered Select */}
          <Select
            label="Risk if Not Delivered"
            options={riskLevelOptions}
            value={currentValue.riskIfNotDelivered ?? ''}
            onChange={(e) => handleChange('riskIfNotDelivered')(e.target.value as RiskLevel)}
            placeholder="Select risk level"
            helperText="What is the impact of not delivering this?"
          />
        </div>

        {/* Checkboxes section */}
        <div className="checkbox-group">
          <label className="checkbox-label">
            <input
              type="checkbox"
              checked={currentValue.legalRequirement ?? false}
              onChange={(e) => handleChange('legalRequirement')(e.target.checked)}
              className="checkbox-input"
            />
            <span className="checkbox-text">
              <strong>Legal/Compliance Requirement</strong>
              <span className="checkbox-description">
                This item is required for legal or regulatory compliance
              </span>
            </span>
          </label>

          <label className="checkbox-label">
            <input
              type="checkbox"
              checked={currentValue.customerRequest ?? false}
              onChange={(e) => handleChange('customerRequest')(e.target.checked)}
              className="checkbox-input"
            />
            <span className="checkbox-text">
              <strong>Customer Request</strong>
              <span className="checkbox-description">
                This item was specifically requested by customers
              </span>
            </span>
          </label>
        </div>

        {showSubmitButton && (
          <div className="form-actions">
            <Button
              type="submit"
              variant="primary"
              isLoading={isLoading}
              disabled={!currentValue.businessValue || currentValue.riskIfNotDelivered === undefined}
            >
              Categorize with MoSCoW
            </Button>
          </div>
        )}
      </form>
    </Card>
  );
}

export default MoSCoWInputForm;
