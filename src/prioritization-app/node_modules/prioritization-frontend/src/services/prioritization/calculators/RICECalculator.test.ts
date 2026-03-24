/**
 * Unit tests for RICECalculator
 *
 * Tests cover:
 * - Score calculation with various inputs
 * - Input validation (valid and invalid cases)
 * - Normalization across multiple results
 * - Auto-fill suggestions
 * - Edge cases and boundary conditions
 *
 * @module prioritization/calculators/tests/RICECalculator.test
 */

import { describe, it, expect, beforeEach } from 'vitest';
import {
  RICECalculator,
  createRICECalculator,
  IMPACT_LEVELS,
} from './RICECalculator';
import type { RICEInput, RICEResult, PrioritizationItem } from '../types';

describe('RICECalculator', () => {
  let calculator: RICECalculator;

  beforeEach(() => {
    calculator = new RICECalculator();
  });

  describe('getFrameworkType', () => {
    it('should return "RICE"', () => {
      expect(calculator.getFrameworkType()).toBe('RICE');
    });
  });

  describe('calculate', () => {
    it('should calculate correct RICE score with basic input', () => {
      const input: RICEInput = {
        reach: 100,
        impact: 2,
        confidence: 80,
        effort: 2,
      };

      const result = calculator.calculate(input);

      // Expected: (100 * 2 * 0.8) / 2 = 80
      expect(result.score).toBe(80);
      expect(result.framework).toBe('RICE');
      expect(result.reach).toBe(100);
      expect(result.impact).toBe(2);
      expect(result.confidence).toBe(0.8);
      expect(result.effort).toBe(2);
    });

    it('should handle confidence as decimal (0-1)', () => {
      const input: RICEInput = {
        reach: 100,
        impact: 2,
        confidence: 0.75,
        effort: 2,
      };

      const result = calculator.calculate(input);

      // Expected: (100 * 2 * 0.75) / 2 = 75
      expect(result.score).toBe(75);
      expect(result.confidence).toBe(0.75);
    });

    it('should handle confidence as percentage (0-100)', () => {
      const input: RICEInput = {
        reach: 200,
        impact: 3,
        confidence: 90,
        effort: 3,
      };

      const result = calculator.calculate(input);

      // Expected: (200 * 3 * 0.9) / 3 = 180
      expect(result.score).toBe(180);
      expect(result.confidence).toBe(0.9);
    });

    it('should calculate with minimal impact (0.25)', () => {
      const input: RICEInput = {
        reach: 1000,
        impact: 0.25,
        confidence: 100,
        effort: 1,
      };

      const result = calculator.calculate(input);

      // Expected: (1000 * 0.25 * 1) / 1 = 250
      expect(result.score).toBe(250);
    });

    it('should calculate with massive impact (3)', () => {
      const input: RICEInput = {
        reach: 100,
        impact: 3,
        confidence: 100,
        effort: 1,
      };

      const result = calculator.calculate(input);

      // Expected: (100 * 3 * 1) / 1 = 300
      expect(result.score).toBe(300);
    });

    it('should round score to configured decimal places', () => {
      const input: RICEInput = {
        reach: 100,
        impact: 2,
        confidence: 75,
        effort: 3,
      };

      const result = calculator.calculate(input);

      // Expected: (100 * 2 * 0.75) / 3 = 50
      expect(result.score).toBe(50);
    });

    it('should include impact level in details', () => {
      const input: RICEInput = {
        reach: 100,
        impact: IMPACT_LEVELS.high,
        confidence: 80,
        effort: 2,
      };

      const result = calculator.calculate(input);

      expect(result.details).toHaveProperty('impactLevel', 'high');
      expect(result.details).toHaveProperty('confidencePercentage', 80);
    });

    it('should throw error for invalid input', () => {
      const invalidInput: Partial<RICEInput> = {
        reach: -10,
        impact: 2,
        confidence: 80,
        effort: 2,
      };

      expect(() => calculator.calculate(invalidInput as RICEInput)).toThrow(
        /Invalid RICE input/
      );
    });

    it('should throw error for zero effort', () => {
      const invalidInput: Partial<RICEInput> = {
        reach: 100,
        impact: 2,
        confidence: 80,
        effort: 0,
      };

      expect(() => calculator.calculate(invalidInput as RICEInput)).toThrow(
        /effort must be at least/i
      );
    });
  });

  describe('validate', () => {
    it('should return valid for correct input', () => {
      const input: RICEInput = {
        reach: 100,
        impact: 2,
        confidence: 80,
        effort: 2,
      };

      const validation = calculator.validate(input);

      expect(validation.isValid).toBe(true);
      expect(validation.errors).toHaveLength(0);
    });

    it('should detect negative reach', () => {
      const input: Partial<RICEInput> = { reach: -10 };

      const validation = calculator.validate(input);

      expect(validation.isValid).toBe(false);
      expect(validation.errors).toContainEqual(
        expect.stringContaining('Reach')
      );
    });

    it('should detect impact below minimum', () => {
      const input: Partial<RICEInput> = { impact: 0.1 };

      const validation = calculator.validate(input);

      expect(validation.isValid).toBe(false);
      expect(validation.errors).toContainEqual(
        expect.stringContaining('Impact')
      );
    });

    it('should detect impact above maximum', () => {
      const input: Partial<RICEInput> = { impact: 5 };

      const validation = calculator.validate(input);

      expect(validation.isValid).toBe(false);
      expect(validation.errors).toContainEqual(
        expect.stringContaining('Impact')
      );
    });

    it('should detect confidence below minimum', () => {
      const input: Partial<RICEInput> = { confidence: -10 };

      const validation = calculator.validate(input);

      expect(validation.isValid).toBe(false);
      expect(validation.errors).toContainEqual(
        expect.stringContaining('Confidence')
      );
    });

    it('should detect confidence above maximum', () => {
      const input: Partial<RICEInput> = { confidence: 150 };

      const validation = calculator.validate(input);

      expect(validation.isValid).toBe(false);
      expect(validation.errors).toContainEqual(
        expect.stringContaining('Confidence')
      );
    });

    it('should detect zero effort', () => {
      const input: Partial<RICEInput> = { effort: 0 };

      const validation = calculator.validate(input);

      expect(validation.isValid).toBe(false);
      expect(validation.errors).toContainEqual(
        expect.stringContaining('Effort')
      );
    });

    it('should detect negative effort', () => {
      const input: Partial<RICEInput> = { effort: -5 };

      const validation = calculator.validate(input);

      expect(validation.isValid).toBe(false);
      expect(validation.errors).toContainEqual(
        expect.stringContaining('Effort')
      );
    });

    it('should warn about zero reach', () => {
      const input: Partial<RICEInput> = { reach: 0 };

      const validation = calculator.validate(input);

      expect(validation.warnings).toContainEqual(
        expect.stringContaining('Reach of 0')
      );
    });

    it('should warn about low confidence', () => {
      const input: Partial<RICEInput> = { confidence: 30 };

      const validation = calculator.validate(input);

      expect(validation.warnings).toContainEqual(
        expect.stringContaining('Low confidence')
      );
    });

    it('should warn about non-standard impact values', () => {
      const input: Partial<RICEInput> = { impact: 1.5 };

      const validation = calculator.validate(input);

      expect(validation.warnings).toContainEqual(
        expect.stringContaining('standard scale')
      );
    });

    it('should warn about high effort values', () => {
      const input: Partial<RICEInput> = { effort: 15 };

      const validation = calculator.validate(input);

      expect(validation.warnings).toContainEqual(
        expect.stringContaining('High effort')
      );
    });

    it('should detect NaN values', () => {
      const input: Partial<RICEInput> = { reach: NaN };

      const validation = calculator.validate(input);

      expect(validation.isValid).toBe(false);
      expect(validation.errors).toContainEqual(
        expect.stringContaining('valid number')
      );
    });

    it('should detect non-numeric values', () => {
      const input = { reach: 'not a number' } as unknown as Partial<RICEInput>;

      const validation = calculator.validate(input);

      expect(validation.isValid).toBe(false);
    });

    it('should return valid for partial input with missing fields', () => {
      const input: Partial<RICEInput> = { reach: 100 };

      const validation = calculator.validate(input);

      // Only reach is provided and it's valid
      expect(validation.errors).toHaveLength(0);
    });
  });

  describe('normalize', () => {
    let results: RICEResult[];

    beforeEach(() => {
      results = [
        calculator.calculate({ reach: 100, impact: 2, confidence: 80, effort: 2 }),
        calculator.calculate({ reach: 200, impact: 3, confidence: 90, effort: 3 }),
        calculator.calculate({ reach: 50, impact: 1, confidence: 70, effort: 1 }),
        calculator.calculate({ reach: 300, impact: 2, confidence: 50, effort: 4 }),
      ];
      // Scores: 80, 180, 35, 75
    });

    it('should assign correct rank based on score', () => {
      const normalized = calculator.normalize(results[1], results);
      expect(normalized.rank).toBe(1); // Highest score
    });

    it('should assign rank 4 to lowest score', () => {
      const normalized = calculator.normalize(results[2], results);
      expect(normalized.rank).toBe(4); // Lowest score
    });

    it('should calculate correct percentile', () => {
      const normalized = calculator.normalize(results[1], results);
      // Highest score should be at 100th percentile
      expect(normalized.percentile).toBe(100);
    });

    it('should normalize scores to 0-100 scale', () => {
      const highest = calculator.normalize(results[1], results);
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
      const sameInput = { reach: 100, impact: 2, confidence: 80, effort: 2 };
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

    it('should suggest high reach for "all users" items', () => {
      const item = createItem(
        'Add export feature for all users',
        'High-demand feature'
      );

      const suggestions = calculator.getAutoFillSuggestions(item);

      expect(suggestions.reach).toBe(1000);
      expect(suggestions.reachConfidence).toBeGreaterThan(0.5);
    });

    it('should suggest moderate reach for enterprise items', () => {
      const item = createItem(
        'Enterprise reporting dashboard',
        'For business customers'
      );

      const suggestions = calculator.getAutoFillSuggestions(item);

      expect(suggestions.reach).toBe(100);
    });

    it('should suggest low reach for admin items', () => {
      const item = createItem(
        'Admin panel improvements',
        'Internal tool updates'
      );

      const suggestions = calculator.getAutoFillSuggestions(item);

      expect(suggestions.reach).toBe(20);
    });

    it('should suggest high impact for critical items', () => {
      const item = createItem(
        'Fix critical security vulnerability',
        'Core system protection'
      );

      const suggestions = calculator.getAutoFillSuggestions(item);

      expect(suggestions.impact).toBe(IMPACT_LEVELS.high);
    });

    it('should suggest low impact for cosmetic items', () => {
      const item = createItem(
        'Minor cosmetic improvements',
        'Nice to have UI tweaks'
      );

      const suggestions = calculator.getAutoFillSuggestions(item);

      expect(suggestions.impact).toBe(IMPACT_LEVELS.low);
    });

    it('should suggest high confidence for detailed descriptions', () => {
      const item = createItem(
        'Feature implementation',
        'This is a comprehensive feature with detailed requirements. ' +
          'It includes multiple components and thorough specifications. ' +
          'The implementation plan is well-documented and clear.'
      );

      const suggestions = calculator.getAutoFillSuggestions(item);

      expect(suggestions.confidence).toBeGreaterThan(50);
      expect(suggestions.confidenceConfidence).toBeGreaterThan(0.5);
    });

    it('should suggest low confidence for sparse descriptions', () => {
      const item = createItem('Quick fix');

      const suggestions = calculator.getAutoFillSuggestions(item);

      expect(suggestions.confidence).toBe(40);
    });

    it('should suggest high effort for migration items', () => {
      const item = createItem(
        'Database migration',
        'Infrastructure refactoring'
      );

      const suggestions = calculator.getAutoFillSuggestions(item);

      expect(suggestions.effort).toBeGreaterThan(2);
    });

    it('should suggest low effort for fix items', () => {
      const item = createItem(
        'Quick fix for button alignment',
        'Minor CSS adjustment'
      );

      const suggestions = calculator.getAutoFillSuggestions(item);

      expect(suggestions.effort).toBeLessThan(1);
    });

    it('should provide default suggestions for generic items', () => {
      const item = createItem('Some feature');

      const suggestions = calculator.getAutoFillSuggestions(item);

      expect(suggestions).toHaveProperty('reach');
      expect(suggestions).toHaveProperty('impact');
      expect(suggestions).toHaveProperty('confidence');
      expect(suggestions).toHaveProperty('effort');
    });

    it('should include confidence scores for all suggestions', () => {
      const item = createItem(
        'Feature with description',
        'Detailed description here'
      );

      const suggestions = calculator.getAutoFillSuggestions(item);

      expect(suggestions.reachConfidence).toBeDefined();
      expect(suggestions.impactConfidence).toBeDefined();
      expect(suggestions.confidenceConfidence).toBeDefined();
      expect(suggestions.effortConfidence).toBeDefined();
    });
  });

  describe('IMPACT_LEVELS', () => {
    it('should have correct values', () => {
      expect(IMPACT_LEVELS.massive).toBe(3);
      expect(IMPACT_LEVELS.high).toBe(2);
      expect(IMPACT_LEVELS.medium).toBe(1);
      expect(IMPACT_LEVELS.low).toBe(0.5);
      expect(IMPACT_LEVELS.minimal).toBe(0.25);
    });
  });

  describe('createRICECalculator factory', () => {
    it('should create calculator with default config', () => {
      const calc = createRICECalculator();

      expect(calc).toBeInstanceOf(RICECalculator);
      expect(calc.getFrameworkType()).toBe('RICE');
    });

    it('should create calculator with custom config', () => {
      const calc = createRICECalculator({ decimalPlaces: 3 });

      // Use input that produces a non-integer result
      // (100 * 2 * 0.83) / 3 = 55.333...
      const result = calc.calculate({
        reach: 100,
        impact: 2,
        confidence: 83,
        effort: 3,
      });

      // Score should be 55.333 (3 decimal places)
      expect(result.score).toBe(55.333);
      expect(result.score.toString()).toContain('.');
    });
  });

  describe('Edge Cases', () => {
    it('should handle very large reach values', () => {
      const input: RICEInput = {
        reach: 1000000,
        impact: 2,
        confidence: 80,
        effort: 10,
      };

      const result = calculator.calculate(input);

      // Expected: (1000000 * 2 * 0.8) / 10 = 160000
      expect(result.score).toBe(160000);
    });

    it('should handle very small effort values', () => {
      const input: RICEInput = {
        reach: 100,
        impact: 2,
        confidence: 80,
        effort: 0.1,
      };

      const result = calculator.calculate(input);

      // Expected: (100 * 2 * 0.8) / 0.1 = 1600
      expect(result.score).toBe(1600);
    });

    it('should handle zero confidence', () => {
      const input: RICEInput = {
        reach: 100,
        impact: 2,
        confidence: 0,
        effort: 2,
      };

      const result = calculator.calculate(input);

      expect(result.score).toBe(0);
    });

    it('should handle 100% confidence', () => {
      const input: RICEInput = {
        reach: 100,
        impact: 2,
        confidence: 100,
        effort: 2,
      };

      const result = calculator.calculate(input);

      // Expected: (100 * 2 * 1) / 2 = 100
      expect(result.score).toBe(100);
    });
  });
});
