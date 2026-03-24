/**
 * Unit tests for MoSCoWCalculator
 *
 * Tests cover:
 * - Category calculation with various inputs
 * - Input validation (valid and invalid cases)
 * - Normalization across multiple results
 * - Auto-fill suggestions
 * - Edge cases and boundary conditions
 *
 * @module prioritization/calculators/tests/MoSCoWCalculator.test
 */

import { describe, it, expect, beforeEach } from 'vitest';
import {
  MoSCoWCalculator,
  createMoSCoWCalculator,
  BUSINESS_VALUE_SCORES,
  RISK_SCORES,
  MOSCOW_CATEGORY_WEIGHTS,
} from './MoSCoWCalculator';
import type {
  MoSCoWInput,
  MoSCoWResult,
  BusinessValueLevel,
  RiskLevel,
  PrioritizationItem,
} from '../types';

describe('MoSCoWCalculator', () => {
  let calculator: MoSCoWCalculator;

  beforeEach(() => {
    calculator = new MoSCoWCalculator();
  });

  describe('getFrameworkType', () => {
    it('should return "MoSCoW"', () => {
      expect(calculator.getFrameworkType()).toBe('MoSCoW');
    });
  });

  describe('calculate', () => {
    it('should categorize legal requirements as Must have', () => {
      const input: MoSCoWInput = {
        businessValue: 'medium',
        legalRequirement: true,
        customerRequest: false,
        riskIfNotDelivered: 'medium',
      };

      const result = calculator.calculate(input);

      expect(result.category).toBe('Must have');
      expect(result.legalRequirement).toBe(true);
    });

    it('should categorize critical business value as Must have', () => {
      const input: MoSCoWInput = {
        businessValue: 'critical',
        legalRequirement: false,
        customerRequest: false,
        riskIfNotDelivered: 'medium',
      };

      const result = calculator.calculate(input);

      expect(result.category).toBe('Must have');
    });

    it('should categorize critical risk as Must have', () => {
      const input: MoSCoWInput = {
        businessValue: 'medium',
        legalRequirement: false,
        customerRequest: false,
        riskIfNotDelivered: 'critical',
      };

      const result = calculator.calculate(input);

      expect(result.category).toBe('Must have');
    });

    it('should categorize high business value AND high risk as Should have', () => {
      const input: MoSCoWInput = {
        businessValue: 'high',
        legalRequirement: false,
        customerRequest: false,
        riskIfNotDelivered: 'high',
      };

      const result = calculator.calculate(input);

      expect(result.category).toBe('Should have');
    });

    it('should categorize high business value OR high risk as Should have', () => {
      const input1: MoSCoWInput = {
        businessValue: 'high',
        legalRequirement: false,
        customerRequest: false,
        riskIfNotDelivered: 'medium',
      };

      const input2: MoSCoWInput = {
        businessValue: 'medium',
        legalRequirement: false,
        customerRequest: false,
        riskIfNotDelivered: 'high',
      };

      const result1 = calculator.calculate(input1);
      const result2 = calculator.calculate(input2);

      expect(result1.category).toBe('Should have');
      expect(result2.category).toBe('Should have');
    });

    it('should categorize medium business value as Could have', () => {
      const input: MoSCoWInput = {
        businessValue: 'medium',
        legalRequirement: false,
        customerRequest: false,
        riskIfNotDelivered: 'medium',
      };

      const result = calculator.calculate(input);

      expect(result.category).toBe('Could have');
    });

    it('should categorize low business value as Won\'t have', () => {
      const input: MoSCoWInput = {
        businessValue: 'low',
        legalRequirement: false,
        customerRequest: false,
        riskIfNotDelivered: 'low',
      };

      const result = calculator.calculate(input);

      expect(result.category).toBe("Won't have");
    });

    it('should include business value score in details', () => {
      const input: MoSCoWInput = {
        businessValue: 'critical',
        legalRequirement: false,
        customerRequest: false,
        riskIfNotDelivered: 'high',
      };

      const result = calculator.calculate(input);

      expect(result.details.businessValueScore).toBe(
        BUSINESS_VALUE_SCORES['critical']
      );
    });

    it('should include risk score in details', () => {
      const input: MoSCoWInput = {
        businessValue: 'high',
        legalRequirement: false,
        customerRequest: false,
        riskIfNotDelivered: 'critical',
      };

      const result = calculator.calculate(input);

      expect(result.details.riskScore).toBe(RISK_SCORES['critical']);
    });

    it('should include category weight in details', () => {
      const input: MoSCoWInput = {
        businessValue: 'critical',
        legalRequirement: false,
        customerRequest: false,
        riskIfNotDelivered: 'high',
      };

      const result = calculator.calculate(input);

      expect(result.details.categoryWeight).toBe(
        MOSCOW_CATEGORY_WEIGHTS['Must have']
      );
    });

    it('should throw error for invalid input', () => {
      const invalidInput: Partial<MoSCoWInput> = {
        businessValue: 'invalid' as BusinessValueLevel,
      };

      expect(() => calculator.calculate(invalidInput as MoSCoWInput)).toThrow(
        /Invalid MoSCoW input/
      );
    });
  });

  describe('validate', () => {
    it('should return valid for correct input', () => {
      const input: MoSCoWInput = {
        businessValue: 'high',
        legalRequirement: false,
        customerRequest: true,
        riskIfNotDelivered: 'medium',
      };

      const validation = calculator.validate(input);

      expect(validation.isValid).toBe(true);
      expect(validation.errors).toHaveLength(0);
    });

    it('should detect missing businessValue', () => {
      const input: Partial<MoSCoWInput> = {
        legalRequirement: false,
        customerRequest: true,
        riskIfNotDelivered: 'medium',
      };

      const validation = calculator.validate(input);

      expect(validation.isValid).toBe(false);
      expect(validation.errors).toContainEqual(
        expect.stringContaining('businessValue')
      );
    });

    it('should detect invalid businessValue', () => {
      const input: Partial<MoSCoWInput> = {
        businessValue: 'super' as BusinessValueLevel,
        legalRequirement: false,
        customerRequest: true,
        riskIfNotDelivered: 'medium',
      };

      const validation = calculator.validate(input);

      expect(validation.isValid).toBe(false);
      expect(validation.errors).toContainEqual(
        expect.stringContaining('critical')
      );
    });

    it('should detect missing legalRequirement', () => {
      const input: Partial<MoSCoWInput> = {
        businessValue: 'high',
        customerRequest: true,
        riskIfNotDelivered: 'medium',
      };

      const validation = calculator.validate(input);

      expect(validation.isValid).toBe(false);
      expect(validation.errors).toContainEqual(
        expect.stringContaining('legalRequirement')
      );
    });

    it('should detect non-boolean legalRequirement', () => {
      const input: Partial<MoSCoWInput> = {
        businessValue: 'high',
        legalRequirement: 'yes' as unknown as boolean,
        customerRequest: true,
        riskIfNotDelivered: 'medium',
      };

      const validation = calculator.validate(input);

      expect(validation.isValid).toBe(false);
      expect(validation.errors).toContainEqual(
        expect.stringContaining('boolean')
      );
    });

    it('should detect missing customerRequest', () => {
      const input: Partial<MoSCoWInput> = {
        businessValue: 'high',
        legalRequirement: false,
        riskIfNotDelivered: 'medium',
      };

      const validation = calculator.validate(input);

      expect(validation.isValid).toBe(false);
      expect(validation.errors).toContainEqual(
        expect.stringContaining('customerRequest')
      );
    });

    it('should detect non-boolean customerRequest', () => {
      const input: Partial<MoSCoWInput> = {
        businessValue: 'high',
        legalRequirement: false,
        customerRequest: 1 as unknown as boolean,
        riskIfNotDelivered: 'medium',
      };

      const validation = calculator.validate(input);

      expect(validation.isValid).toBe(false);
      expect(validation.errors).toContainEqual(
        expect.stringContaining('boolean')
      );
    });

    it('should detect missing riskIfNotDelivered', () => {
      const input: Partial<MoSCoWInput> = {
        businessValue: 'high',
        legalRequirement: false,
        customerRequest: true,
      };

      const validation = calculator.validate(input);

      expect(validation.isValid).toBe(false);
      expect(validation.errors).toContainEqual(
        expect.stringContaining('riskIfNotDelivered')
      );
    });

    it('should detect invalid riskIfNotDelivered', () => {
      const input: Partial<MoSCoWInput> = {
        businessValue: 'high',
        legalRequirement: false,
        customerRequest: true,
        riskIfNotDelivered: 'extreme' as RiskLevel,
      };

      const validation = calculator.validate(input);

      expect(validation.isValid).toBe(false);
      expect(validation.errors).toContainEqual(
        expect.stringContaining('critical')
      );
    });

    it('should warn about legal requirement with low business value', () => {
      const input: MoSCoWInput = {
        businessValue: 'low',
        legalRequirement: true,
        customerRequest: false,
        riskIfNotDelivered: 'medium',
      };

      const validation = calculator.validate(input);

      expect(validation.warnings).toContainEqual(
        expect.stringContaining('Legal requirements')
      );
    });

    it('should warn about critical risk with low business value', () => {
      const input: MoSCoWInput = {
        businessValue: 'low',
        legalRequirement: false,
        customerRequest: false,
        riskIfNotDelivered: 'critical',
      };

      const validation = calculator.validate(input);

      expect(validation.warnings).toContainEqual(
        expect.stringContaining('contradictory')
      );
    });

    it('should return valid for partial input with required fields', () => {
      const input: Partial<MoSCoWInput> = { businessValue: 'high' };

      const validation = calculator.validate(input);

      // Only businessValue is provided, other required fields missing
      expect(validation.errors.length).toBeGreaterThan(0);
    });
  });

  describe('normalize', () => {
    let results: MoSCoWResult[];

    beforeEach(() => {
      results = [
        calculator.calculate({
          businessValue: 'critical',
          legalRequirement: false,
          customerRequest: false,
          riskIfNotDelivered: 'high',
        }),
        calculator.calculate({
          businessValue: 'medium',
          legalRequirement: false,
          customerRequest: true,
          riskIfNotDelivered: 'medium',
        }),
        calculator.calculate({
          businessValue: 'low',
          legalRequirement: false,
          customerRequest: false,
          riskIfNotDelivered: 'low',
        }),
        calculator.calculate({
          businessValue: 'high',
          legalRequirement: true,
          customerRequest: false,
          riskIfNotDelivered: 'high',
        }),
      ];
      // Categories: Must have, Could have, Won't have, Must have
    });

    it('should assign correct rank based on category and scores', () => {
      // results[0] has critical business value + high risk
      // Composite: 4*0.4 + 4*0.3 + 3*0.3 = 3.7 (highest)
      const normalized = calculator.normalize(results[0], results);
      expect(normalized.rank).toBe(1);
    });

    it('should assign lowest rank to Won\'t have category', () => {
      const normalized = calculator.normalize(results[2], results);
      expect(normalized.rank).toBe(4); // Lowest
    });

    it('should calculate correct percentile for highest priority item', () => {
      const normalized = calculator.normalize(results[0], results);
      // Highest score should be at 100th percentile
      expect(normalized.percentile).toBe(100);
    });

    it('should normalize scores to 0-100 scale', () => {
      const highest = calculator.normalize(results[0], results);
      const lowest = calculator.normalize(results[2], results);

      expect(highest.normalizedScore).toBe(100);
      expect(lowest.normalizedScore).toBe(0);
    });

    it('should handle empty results array', () => {
      const normalized = calculator.normalize(results[0], []);

      expect(normalized.rank).toBe(0);
      expect(normalized.percentile).toBe(0);
      expect(normalized.normalizedScore).toBe(0);
    });

    it('should handle single item array', () => {
      const singleResult = [results[0]];
      const normalized = calculator.normalize(results[0], singleResult);

      expect(normalized.rank).toBe(1);
      expect(normalized.percentile).toBe(100);
    });

    it('should handle identical results', () => {
      const sameInput: MoSCoWInput = {
        businessValue: 'medium',
        legalRequirement: false,
        customerRequest: false,
        riskIfNotDelivered: 'medium',
      };
      const sameResults = [
        calculator.calculate(sameInput),
        calculator.calculate(sameInput),
        calculator.calculate(sameInput),
      ];

      const normalized = calculator.normalize(sameResults[0], sameResults);

      expect(normalized.normalizedScore).toBe(50);
    });
  });

  describe('getAutoFillSuggestions', () => {
    const createItem = (
      title: string,
      description?: string,
      category?: string
    ): PrioritizationItem => ({
      id: 'test-1',
      title,
      description,
      category,
      createdAt: new Date(),
    });

    it('should detect legal requirements from GDPR keyword', () => {
      const item = createItem(
        'GDPR data compliance',
        'Implement data encryption'
      );

      const suggestions = calculator.getAutoFillSuggestions(item);

      expect(suggestions.legalRequirement).toBe(true);
      expect(suggestions.legalRequirementConfidence).toBeGreaterThan(0.8);
    });

    it('should detect legal requirements from compliance category', () => {
      const item = createItem(
        'Audit logging',
        'Track all user actions',
        'Compliance'
      );

      const suggestions = calculator.getAutoFillSuggestions(item);

      expect(suggestions.legalRequirement).toBe(true);
    });

    it('should detect non-legal items', () => {
      const item = createItem(
        'New button design',
        'Visual improvements'
      );

      const suggestions = calculator.getAutoFillSuggestions(item);

      expect(suggestions.legalRequirement).toBe(false);
    });

    it('should suggest critical business value for critical keywords', () => {
      const item = createItem(
        'Fix critical crash on startup',
        'Core functionality blocker'
      );

      const suggestions = calculator.getAutoFillSuggestions(item);

      expect(suggestions.businessValue).toBe('critical');
    });

    it('should suggest high business value for important keywords', () => {
      const item = createItem(
        'Important revenue feature',
        'Key customer request'
      );

      const suggestions = calculator.getAutoFillSuggestions(item);

      expect(suggestions.businessValue).toBe('high');
    });

    it('should suggest low business value for nice-to-have keywords', () => {
      const item = createItem(
        'Nice to have cosmetic update',
        'Optional enhancement'
      );

      const suggestions = calculator.getAutoFillSuggestions(item);

      expect(suggestions.businessValue).toBe('low');
    });

    it('should detect customer requested items', () => {
      const item = createItem(
        'Feature requested by customers',
        'Top user vote item'
      );

      const suggestions = calculator.getAutoFillSuggestions(item);

      expect(suggestions.customerRequest).toBe(true);
      expect(suggestions.customerRequestConfidence).toBeGreaterThan(0.5);
    });

    it('should detect customer request from metadata', () => {
      const item: PrioritizationItem = {
        id: 'test-1',
        title: 'Some feature',
        description: 'Description here',
        category: 'Feature',
        createdAt: new Date(),
        metadata: { customerRequested: true },
      };

      const suggestions = calculator.getAutoFillSuggestions(item);

      expect(suggestions.customerRequest).toBe(true);
      expect(suggestions.customerRequestConfidence).toBeGreaterThan(0.9);
    });

    it('should suggest critical risk for compliance items', () => {
      const item = createItem(
        'Security compliance update',
        'Prevent data breach'
      );

      const suggestions = calculator.getAutoFillSuggestions(item);

      expect(suggestions.riskIfNotDelivered).toBe('critical');
    });

    it('should suggest medium risk by default', () => {
      const item = createItem('Generic feature');

      const suggestions = calculator.getAutoFillSuggestions(item);

      expect(suggestions.riskIfNotDelivered).toBe('medium');
    });

    it('should provide all suggestion fields', () => {
      const item = createItem(
        'Feature with details',
        'Detailed description of the feature'
      );

      const suggestions = calculator.getAutoFillSuggestions(item);

      expect(suggestions).toHaveProperty('businessValue');
      expect(suggestions).toHaveProperty('legalRequirement');
      expect(suggestions).toHaveProperty('customerRequest');
      expect(suggestions).toHaveProperty('riskIfNotDelivered');
      expect(suggestions).toHaveProperty('businessValueConfidence');
      expect(suggestions).toHaveProperty('legalRequirementConfidence');
      expect(suggestions).toHaveProperty('customerRequestConfidence');
      expect(suggestions).toHaveProperty('riskConfidence');
    });
  });

  describe('BUSINESS_VALUE_SCORES', () => {
    it('should have correct values', () => {
      expect(BUSINESS_VALUE_SCORES.critical).toBe(4);
      expect(BUSINESS_VALUE_SCORES.high).toBe(3);
      expect(BUSINESS_VALUE_SCORES.medium).toBe(2);
      expect(BUSINESS_VALUE_SCORES.low).toBe(1);
    });
  });

  describe('RISK_SCORES', () => {
    it('should have correct values', () => {
      expect(RISK_SCORES.critical).toBe(4);
      expect(RISK_SCORES.high).toBe(3);
      expect(RISK_SCORES.medium).toBe(2);
      expect(RISK_SCORES.low).toBe(1);
    });
  });

  describe('MOSCOW_CATEGORY_WEIGHTS', () => {
    it('should have correct values', () => {
      expect(MOSCOW_CATEGORY_WEIGHTS['Must have']).toBe(4);
      expect(MOSCOW_CATEGORY_WEIGHTS['Should have']).toBe(3);
      expect(MOSCOW_CATEGORY_WEIGHTS['Could have']).toBe(2);
      expect(MOSCOW_CATEGORY_WEIGHTS["Won't have"]).toBe(1);
    });
  });

  describe('createMoSCoWCalculator factory', () => {
    it('should create calculator with default config', () => {
      const calc = createMoSCoWCalculator();

      expect(calc).toBeInstanceOf(MoSCoWCalculator);
      expect(calc.getFrameworkType()).toBe('MoSCoW');
    });

    it('should create calculator with custom config', () => {
      const calc = createMoSCoWCalculator({ decimalPlaces: 3 });

      // Config is used internally, just verify it's created
      expect(calc).toBeInstanceOf(MoSCoWCalculator);
    });
  });

  describe('Category Decision Edge Cases', () => {
    it('should prioritize legal requirement over all other factors', () => {
      const input: MoSCoWInput = {
        businessValue: 'low',
        legalRequirement: true,
        customerRequest: false,
        riskIfNotDelivered: 'low',
      };

      const result = calculator.calculate(input);

      // Even with low business value and low risk, legal requirement = Must have
      expect(result.category).toBe('Must have');
    });

    it('should categorize customer request with medium value as Should have', () => {
      const input: MoSCoWInput = {
        businessValue: 'medium',
        legalRequirement: false,
        customerRequest: true,
        riskIfNotDelivered: 'medium',
      };

      const result = calculator.calculate(input);

      expect(result.category).toBe('Should have');
    });

    it('should handle combination of high value and critical risk', () => {
      const input: MoSCoWInput = {
        businessValue: 'high',
        legalRequirement: false,
        customerRequest: true,
        riskIfNotDelivered: 'critical',
      };

      const result = calculator.calculate(input);

      expect(result.category).toBe('Must have'); // Critical risk
    });

    it('should categorize medium value with low risk as Could have', () => {
      const input: MoSCoWInput = {
        businessValue: 'medium',
        legalRequirement: false,
        customerRequest: false,
        riskIfNotDelivered: 'low',
      };

      const result = calculator.calculate(input);

      expect(result.category).toBe('Could have');
    });
  });

  describe('Priority Calculation', () => {
    it('should assign lower priority number to higher value items within category', () => {
      const criticalInput: MoSCoWInput = {
        businessValue: 'critical',
        legalRequirement: false,
        customerRequest: false,
        riskIfNotDelivered: 'high',
      };

      const mediumInput: MoSCoWInput = {
        businessValue: 'medium',
        legalRequirement: false,
        customerRequest: false,
        riskIfNotDelivered: 'medium',
      };

      const criticalResult = calculator.calculate(criticalInput);
      const mediumResult = calculator.calculate(mediumInput);

      // Critical should have lower priority number (higher importance)
      expect(criticalResult.priority).toBeLessThan(mediumResult.priority);
    });

    it('should assign lower priority number for customer requested items', () => {
      const withCustomerRequest: MoSCoWInput = {
        businessValue: 'high',
        legalRequirement: false,
        customerRequest: true,
        riskIfNotDelivered: 'high',
      };

      const withoutCustomerRequest: MoSCoWInput = {
        businessValue: 'high',
        legalRequirement: false,
        customerRequest: false,
        riskIfNotDelivered: 'high',
      };

      const withResult = calculator.calculate(withCustomerRequest);
      const withoutResult = calculator.calculate(withoutCustomerRequest);

      // Customer requested should have lower priority number
      expect(withResult.priority).toBeLessThanOrEqual(withoutResult.priority);
    });
  });
});
