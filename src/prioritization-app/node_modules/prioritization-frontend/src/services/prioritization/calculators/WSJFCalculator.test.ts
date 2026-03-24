/**
 * Unit tests for WSJFCalculator
 *
 * Tests cover:
 * - WSJF score calculation with various inputs
 * - Input validation (valid and invalid cases)
 * - Normalization across multiple results
 * - Auto-fill suggestions based on item metadata
 * - Edge cases and boundary conditions
 * - Fibonacci scale support
 *
 * @module prioritization/calculators/tests/WSJFCalculator.test
 */

import { describe, it, expect, beforeEach } from 'vitest';
import {
  WSJFCalculator,
  createWSJFCalculator,
  FIBONACCI_SCALE,
} from './WSJFCalculator';
import type { WSJFInput, WSJFResult, PrioritizationItem } from '../types';

describe('WSJFCalculator', () => {
  let calculator: WSJFCalculator;

  beforeEach(() => {
    calculator = new WSJFCalculator();
  });

  describe('getFrameworkType', () => {
    it('should return "WSJF"', () => {
      expect(calculator.getFrameworkType()).toBe('WSJF');
    });
  });

  describe('calculate', () => {
    it('should calculate correct WSJF score with basic input', () => {
      const input: WSJFInput = {
        userBusinessValue: 8,
        timeCriticality: 6,
        riskReductionOpportunity: 5,
        jobSize: 3,
      };

      const result = calculator.calculate(input);

      // Cost of Delay = 8 + 6 + 5 = 19
      // WSJF = 19 / 3 = 6.33
      expect(result.costOfDelay).toBe(19);
      expect(result.score).toBe(6.33);
      expect(result.framework).toBe('WSJF');
    });

    it('should calculate correct Cost of Delay components', () => {
      const input: WSJFInput = {
        userBusinessValue: 13,
        timeCriticality: 8,
        riskReductionOpportunity: 5,
        jobSize: 5,
      };

      const result = calculator.calculate(input);

      expect(result.userBusinessValue).toBe(13);
      expect(result.timeCriticality).toBe(8);
      expect(result.riskReductionOpportunity).toBe(5);
      expect(result.costOfDelay).toBe(26); // 13 + 8 + 5
    });

    it('should handle equal component values', () => {
      const input: WSJFInput = {
        userBusinessValue: 5,
        timeCriticality: 5,
        riskReductionOpportunity: 5,
        jobSize: 5,
      };

      const result = calculator.calculate(input);

      // CoD = 15, WSJF = 15/5 = 3
      expect(result.costOfDelay).toBe(15);
      expect(result.score).toBe(3);
    });

    it('should handle high CoD with small job size', () => {
      const input: WSJFInput = {
        userBusinessValue: 20,
        timeCriticality: 20,
        riskReductionOpportunity: 20,
        jobSize: 2,
      };

      const result = calculator.calculate(input);

      // CoD = 60, WSJF = 60/2 = 30
      expect(result.costOfDelay).toBe(60);
      expect(result.score).toBe(30);
    });

    it('should handle low CoD with large job size', () => {
      const input: WSJFInput = {
        userBusinessValue: 1,
        timeCriticality: 1,
        riskReductionOpportunity: 1,
        jobSize: 20,
      };

      const result = calculator.calculate(input);

      // CoD = 3, WSJF = 3/20 = 0.15
      expect(result.costOfDelay).toBe(3);
      expect(result.score).toBe(0.15);
    });

    it('should round score to configured decimal places', () => {
      const input: WSJFInput = {
        userBusinessValue: 10,
        timeCriticality: 7,
        riskReductionOpportunity: 5,
        jobSize: 3,
      };

      const result = calculator.calculate(input);

      // CoD = 22, WSJF = 22/3 = 7.333...
      expect(result.score).toBe(7.33);
    });

    it('should include WSJF interpretation in details', () => {
      const input: WSJFInput = {
        userBusinessValue: 13,
        timeCriticality: 8,
        riskReductionOpportunity: 8,
        jobSize: 3,
      };

      const result = calculator.calculate(input);

      expect(result.details).toHaveProperty('wsjfInterpretation');
      expect(result.details.wsjfInterpretation).toContain('High priority');
    });

    it('should use minimum job size when input is very small', () => {
      const calculatorWithMinJobSize = new WSJFCalculator({ minJobSize: 0.5 });
      const input: WSJFInput = {
        userBusinessValue: 5,
        timeCriticality: 5,
        riskReductionOpportunity: 5,
        jobSize: 0.1, // This is below minJobSize, validation will catch it
      };

      // Validation should reject jobSize below minJobSize
      const validation = calculatorWithMinJobSize.validate(input);
      expect(validation.isValid).toBe(false);
      expect(validation.errors).toContainEqual(
        expect.stringContaining('Job size must be at least 0.5')
      );
    });

    it('should throw error for invalid input', () => {
      const invalidInput: Partial<WSJFInput> = {
        userBusinessValue: -5,
        timeCriticality: 8,
        riskReductionOpportunity: 5,
        jobSize: 3,
      };

      expect(() => calculator.calculate(invalidInput as WSJFInput)).toThrow(
        /Invalid WSJF input/
      );
    });

    it('should throw error for zero job size', () => {
      const invalidInput: Partial<WSJFInput> = {
        userBusinessValue: 8,
        timeCriticality: 6,
        riskReductionOpportunity: 5,
        jobSize: 0,
      };

      expect(() => calculator.calculate(invalidInput as WSJFInput)).toThrow(
        /job size must be at least/i
      );
    });
  });

  describe('validate', () => {
    it('should return valid for correct input', () => {
      const input: WSJFInput = {
        userBusinessValue: 8,
        timeCriticality: 6,
        riskReductionOpportunity: 5,
        jobSize: 3,
      };

      const validation = calculator.validate(input);

      expect(validation.isValid).toBe(true);
      expect(validation.errors).toHaveLength(0);
    });

    it('should detect negative user-business value', () => {
      const input: Partial<WSJFInput> = { userBusinessValue: -5 };

      const validation = calculator.validate(input);

      expect(validation.isValid).toBe(false);
      expect(validation.errors).toContainEqual(
        expect.stringContaining('User-business value')
      );
    });

    it('should detect negative time criticality', () => {
      const input: Partial<WSJFInput> = { timeCriticality: -3 };

      const validation = calculator.validate(input);

      expect(validation.isValid).toBe(false);
      expect(validation.errors).toContainEqual(
        expect.stringContaining('Time criticality')
      );
    });

    it('should detect negative risk reduction', () => {
      const input: Partial<WSJFInput> = { riskReductionOpportunity: -2 };

      const validation = calculator.validate(input);

      expect(validation.isValid).toBe(false);
      expect(validation.errors).toContainEqual(
        expect.stringContaining('Risk reduction')
      );
    });

    it('should detect zero job size', () => {
      const input: Partial<WSJFInput> = { jobSize: 0 };

      const validation = calculator.validate(input);

      expect(validation.isValid).toBe(false);
      expect(validation.errors).toContainEqual(
        expect.stringContaining('Job size')
      );
    });

    it('should detect negative job size', () => {
      const input: Partial<WSJFInput> = { jobSize: -5 };

      const validation = calculator.validate(input);

      expect(validation.isValid).toBe(false);
      expect(validation.errors).toContainEqual(
        expect.stringContaining('Job size')
      );
    });

    it('should detect values exceeding maximum', () => {
      const input: Partial<WSJFInput> = { userBusinessValue: 150 };

      const validation = calculator.validate(input);

      expect(validation.isValid).toBe(false);
      expect(validation.errors).toContainEqual(
        expect.stringContaining('must be at most')
      );
    });

    it('should warn about low business value', () => {
      const input: Partial<WSJFInput> = { userBusinessValue: 2 };

      const validation = calculator.validate(input);

      expect(validation.warnings).toContainEqual(
        expect.stringContaining('Low business value')
      );
    });

    it('should warn about low time criticality', () => {
      const input: Partial<WSJFInput> = { timeCriticality: 1 };

      const validation = calculator.validate(input);

      expect(validation.warnings).toContainEqual(
        expect.stringContaining('Low time criticality')
      );
    });

    it('should warn about large job size', () => {
      const input: Partial<WSJFInput> = { jobSize: 50 };

      const validation = calculator.validate(input);

      expect(validation.warnings).toContainEqual(
        expect.stringContaining('Large job size')
      );
    });

    it('should warn about non-Fibonacci values when using Fibonacci scale', () => {
      const fibCalculator = new WSJFCalculator({ useFibonacciScale: true });
      const input: Partial<WSJFInput> = { userBusinessValue: 7 };

      const validation = fibCalculator.validate(input);

      expect(validation.warnings).toContainEqual(
        expect.stringContaining('not a Fibonacci number')
      );
    });

    it('should warn about very high CoD with very small job size', () => {
      const input: Partial<WSJFInput> = {
        userBusinessValue: 25,
        timeCriticality: 25,
        riskReductionOpportunity: 20,
        jobSize: 2,
      };

      const validation = calculator.validate(input);

      expect(validation.warnings).toContainEqual(
        expect.stringContaining('Very high CoD')
      );
    });

    it('should warn about low potential WSJF score', () => {
      const input: Partial<WSJFInput> = {
        userBusinessValue: 1,
        timeCriticality: 1,
        riskReductionOpportunity: 1,
        jobSize: 10,
      };

      const validation = calculator.validate(input);

      expect(validation.warnings).toContainEqual(
        expect.stringContaining('Low WSJF score')
      );
    });

    it('should detect NaN values', () => {
      const input: Partial<WSJFInput> = { userBusinessValue: NaN };

      const validation = calculator.validate(input);

      expect(validation.isValid).toBe(false);
      expect(validation.errors).toContainEqual(
        expect.stringContaining('valid number')
      );
    });

    it('should return valid for partial input with missing fields', () => {
      const input: Partial<WSJFInput> = { userBusinessValue: 8 };

      const validation = calculator.validate(input);

      expect(validation.errors).toHaveLength(0);
    });
  });

  describe('normalize', () => {
    let results: WSJFResult[];

    beforeEach(() => {
      results = [
        calculator.calculate({ userBusinessValue: 8, timeCriticality: 6, riskReductionOpportunity: 5, jobSize: 3 }), // WSJF = 6.33
        calculator.calculate({ userBusinessValue: 13, timeCriticality: 8, riskReductionOpportunity: 5, jobSize: 5 }), // WSJF = 5.2
        calculator.calculate({ userBusinessValue: 5, timeCriticality: 3, riskReductionOpportunity: 3, jobSize: 8 }), // WSJF = 1.38
        calculator.calculate({ userBusinessValue: 20, timeCriticality: 13, riskReductionOpportunity: 8, jobSize: 3 }), // WSJF = 13.67
      ];
      // Scores: 6.33, 5.2, 1.38, 13.67
    });

    it('should assign correct rank based on score', () => {
      const normalized = calculator.normalize(results[3], results);
      expect(normalized.rank).toBe(1); // Highest score (13.67)
    });

    it('should assign rank 4 to lowest score', () => {
      const normalized = calculator.normalize(results[2], results);
      expect(normalized.rank).toBe(4); // Lowest score (1.38)
    });

    it('should assign correct percentile to highest score', () => {
      const normalized = calculator.normalize(results[3], results);
      expect(normalized.percentile).toBe(100);
    });

    it('should normalize scores to 0-100 scale', () => {
      const highest = calculator.normalize(results[3], results);
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

    it('should handle identical scores', () => {
      const sameInput = { userBusinessValue: 8, timeCriticality: 6, riskReductionOpportunity: 5, jobSize: 3 };
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
      id: 'test-wsjf-1',
      title,
      description,
      category,
      createdAt: new Date(),
    });

    it('should suggest high business value for revenue items', () => {
      const item = createItem(
        'Revenue optimization for checkout flow',
        'Increase conversion rate'
      );

      const suggestions = calculator.getAutoFillSuggestions(item);

      expect(suggestions.userBusinessValue).toBeGreaterThanOrEqual(8);
      expect(suggestions.userBusinessValueConfidence).toBeGreaterThan(0.5);
    });

    it('should suggest low business value for cosmetic items', () => {
      const item = createItem(
        'Minor cosmetic improvements',
        'Nice to have UI tweaks'
      );

      const suggestions = calculator.getAutoFillSuggestions(item);

      expect(suggestions.userBusinessValue).toBeLessThanOrEqual(3);
    });

    it('should suggest high time criticality for deadline items', () => {
      const item = createItem(
        'Compliance feature for regulatory deadline',
        'Required for audit next month'
      );

      const suggestions = calculator.getAutoFillSuggestions(item);

      expect(suggestions.timeCriticality).toBeGreaterThanOrEqual(8);
      expect(suggestions.timeCriticalityConfidence).toBeGreaterThan(0.7);
    });

    it('should suggest high time criticality for urgent items', () => {
      const item = createItem(
        'ASAP fix for production issue',
        'Critical system down'
      );

      const suggestions = calculator.getAutoFillSuggestions(item);

      expect(suggestions.timeCriticality).toBeGreaterThanOrEqual(8);
    });

    it('should suggest moderate time criticality for generic items', () => {
      const item = createItem('Feature implementation');

      const suggestions = calculator.getAutoFillSuggestions(item);

      expect(suggestions.timeCriticality).toBe(5);
    });

    it('should suggest high risk reduction for security items', () => {
      const item = createItem(
        'Security vulnerability patch',
        'Fix critical breach risk'
      );

      const suggestions = calculator.getAutoFillSuggestions(item);

      expect(suggestions.riskReductionOpportunity).toBeGreaterThanOrEqual(7);
      expect(suggestions.riskReductionOpportunityConfidence).toBeGreaterThan(0.7);
    });

    it('should suggest high risk reduction for compliance items', () => {
      const item = createItem(
        'GDPR compliance implementation',
        'Legal requirement'
      );

      const suggestions = calculator.getAutoFillSuggestions(item);

      expect(suggestions.riskReductionOpportunity).toBeGreaterThanOrEqual(7);
    });

    it('should suggest high job size for migration items', () => {
      const item = createItem(
        'Database migration to new system',
        'Infrastructure overhaul'
      );

      const suggestions = calculator.getAutoFillSuggestions(item);

      expect(suggestions.jobSize).toBeGreaterThanOrEqual(8);
    });

    it('should suggest low job size for fix items', () => {
      const item = createItem(
        'Quick bug fix for button',
        'Simple CSS adjustment'
      );

      const suggestions = calculator.getAutoFillSuggestions(item);

      expect(suggestions.jobSize).toBeLessThanOrEqual(3);
    });

    it('should suggest medium job size for feature items', () => {
      const item = createItem(
        'New reporting feature',
        'Implement dashboard component'
      );

      const suggestions = calculator.getAutoFillSuggestions(item);

      expect(suggestions.jobSize).toBe(5);
    });

    it('should provide all suggestions with confidence scores', () => {
      const item = createItem(
        'Critical security patch for compliance',
        'Urgent fix needed'
      );

      const suggestions = calculator.getAutoFillSuggestions(item);

      expect(suggestions.userBusinessValue).toBeDefined();
      expect(suggestions.userBusinessValueConfidence).toBeDefined();
      expect(suggestions.timeCriticality).toBeDefined();
      expect(suggestions.timeCriticalityConfidence).toBeDefined();
      expect(suggestions.riskReductionOpportunity).toBeDefined();
      expect(suggestions.riskReductionOpportunityConfidence).toBeDefined();
      expect(suggestions.jobSize).toBeDefined();
      expect(suggestions.jobSizeConfidence).toBeDefined();
    });

    it('should provide default suggestions for generic items', () => {
      const item = createItem('Some task');

      const suggestions = calculator.getAutoFillSuggestions(item);

      expect(suggestions.userBusinessValue).toBe(5);
      expect(suggestions.timeCriticality).toBe(5);
      expect(suggestions.riskReductionOpportunity).toBe(3);
      expect(suggestions.jobSize).toBe(5);
    });
  });

  describe('Fibonacci Scale', () => {
    it('should have correct Fibonacci scale values', () => {
      expect(FIBONACCI_SCALE).toEqual([1, 2, 3, 5, 8, 13, 20, 40, 100]);
    });

    it('should validate Fibonacci values correctly when enabled', () => {
      const fibCalculator = new WSJFCalculator({ useFibonacciScale: true });

      const validInput: WSJFInput = {
        userBusinessValue: 8,
        timeCriticality: 13,
        riskReductionOpportunity: 5,
        jobSize: 3,
      };

      const validation = fibCalculator.validate(validInput);

      expect(validation.isValid).toBe(true);
      expect(validation.warnings).toHaveLength(0);
    });

    it('should warn about non-Fibonacci values', () => {
      const fibCalculator = new WSJFCalculator({ useFibonacciScale: true });

      const invalidInput: Partial<WSJFInput> = {
        userBusinessValue: 7,
        timeCriticality: 13,
        riskReductionOpportunity: 5,
        jobSize: 3,
      };

      const validation = fibCalculator.validate(invalidInput);

      expect(validation.warnings).toContainEqual(
        expect.stringContaining('not a Fibonacci number')
      );
    });

    it('should calculate correctly with Fibonacci scale', () => {
      const fibCalculator = new WSJFCalculator({ useFibonacciScale: true });

      const input: WSJFInput = {
        userBusinessValue: 13,
        timeCriticality: 8,
        riskReductionOpportunity: 5,
        jobSize: 3,
      };

      const result = fibCalculator.calculate(input);

      expect(result.costOfDelay).toBe(26);
      expect(result.score).toBe(8.67);
    });
  });

  describe('createWSJFCalculator factory', () => {
    it('should create calculator with default config', () => {
      const calc = createWSJFCalculator();

      expect(calc).toBeInstanceOf(WSJFCalculator);
      expect(calc.getFrameworkType()).toBe('WSJF');
    });

    it('should create calculator with custom config', () => {
      const calc = createWSJFCalculator({
        decimalPlaces: 3,
        useFibonacciScale: true,
      });

      // Use input that produces a non-integer result
      // (13 + 8 + 5) / 3 = 8.666...
      const result = calc.calculate({
        userBusinessValue: 13,
        timeCriticality: 8,
        riskReductionOpportunity: 5,
        jobSize: 3,
      });

      // Score should be 8.667 (3 decimal places)
      expect(result.score).toBe(8.667);
    });

    it('should create calculator with custom minJobSize', () => {
      const calc = createWSJFCalculator({ minJobSize: 0.5 });

      // Test with jobSize at the minimum allowed
      const result = calc.calculate({
        userBusinessValue: 5,
        timeCriticality: 5,
        riskReductionOpportunity: 5,
        jobSize: 0.5, // At minimum
      });

      // CoD = 15, WSJF = 15/0.5 = 30
      expect(result.score).toBe(30);
      expect(result.details.normalizedJobSize).toBe(0.5);
    });
  });

  describe('Edge Cases', () => {
    it('should handle minimum values', () => {
      const input: WSJFInput = {
        userBusinessValue: 1,
        timeCriticality: 1,
        riskReductionOpportunity: 1,
        jobSize: 1,
      };

      const result = calculator.calculate(input);

      // CoD = 3, WSJF = 3/1 = 3
      expect(result.costOfDelay).toBe(3);
      expect(result.score).toBe(3);
    });

    it('should handle maximum values', () => {
      const input: WSJFInput = {
        userBusinessValue: 100,
        timeCriticality: 100,
        riskReductionOpportunity: 100,
        jobSize: 100,
      };

      const result = calculator.calculate(input);

      // CoD = 300, WSJF = 300/100 = 3
      expect(result.costOfDelay).toBe(300);
      expect(result.score).toBe(3);
    });

    it('should handle very small job size', () => {
      const input: WSJFInput = {
        userBusinessValue: 10,
        timeCriticality: 10,
        riskReductionOpportunity: 10,
        jobSize: 0.5,
      };

      const result = calculator.calculate(input);

      // CoD = 30, WSJF = 30/0.5 = 60
      expect(result.score).toBe(60);
    });

    it('should return appropriate interpretation for high score', () => {
      const input: WSJFInput = {
        userBusinessValue: 20,
        timeCriticality: 20,
        riskReductionOpportunity: 20,
        jobSize: 2,
      };

      const result = calculator.calculate(input);

      expect(result.details.wsjfInterpretation).toContain('Exceptional');
    });

    it('should return appropriate interpretation for low score', () => {
      const input: WSJFInput = {
        userBusinessValue: 1,
        timeCriticality: 1,
        riskReductionOpportunity: 1,
        jobSize: 20,
      };

      const result = calculator.calculate(input);

      expect(result.details.wsjfInterpretation).toContain('Lowest priority');
    });

    it('should handle zero confidence components (all minimum)', () => {
      const input: WSJFInput = {
        userBusinessValue: 1,
        timeCriticality: 1,
        riskReductionOpportunity: 1,
        jobSize: 1,
      };

      const result = calculator.calculate(input);

      expect(result.score).toBeGreaterThan(0);
    });
  });
});
