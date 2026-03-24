/**
 * Unit tests for KanoCalculator
 *
 * Tests cover:
 * - Kano category determination with various response combinations
 * - Input validation (valid and invalid cases)
 * - Normalization across multiple results
 * - Auto-fill suggestions based on item metadata
 * - Edge cases and boundary conditions
 * - Coefficient calculations
 * - All Kano categories (MustBe, OneDimensional, Attractive, Indifferent, Reverse, Questionable)
 *
 * @module prioritization/calculators/tests/KanoCalculator.test
 */

import { describe, it, expect, beforeEach } from 'vitest';
import {
  KanoCalculator,
  createKanoCalculator,
  KANO_RESPONSE_SCALE,
  KANO_EVALUATION_MATRIX,
  KANO_CATEGORY_SCORES,
} from './KanoCalculator';
import type { KanoInput, KanoResult, PrioritizationItem } from '../types';

describe('KanoCalculator', () => {
  let calculator: KanoCalculator;

  beforeEach(() => {
    calculator = new KanoCalculator();
  });

  describe('getFrameworkType', () => {
    it('should return "Kano"', () => {
      expect(calculator.getFrameworkType()).toBe('Kano');
    });
  });

  describe('calculate - Category Determination', () => {
    it('should classify as OneDimensional when functional=Like, dysfunctional=Dislike', () => {
      const input: KanoInput = {
        functionalScore: 5,  // Like
        dysfunctionalScore: 1, // Dislike
      };

      const result = calculator.calculate(input);

      expect(result.category).toBe('OneDimensional');
      expect(result.functionalScore).toBe(5);
      expect(result.dysfunctionalScore).toBe(1);
    });

    it('should classify as MustBe when functional=Dislike, dysfunctional=Expect', () => {
      const input: KanoInput = {
        functionalScore: 1,  // Dislike
        dysfunctionalScore: 4, // Expect
      };

      const result = calculator.calculate(input);

      expect(result.category).toBe('MustBe');
    });

    it('should classify as Attractive when functional=Like, dysfunctional=Neutral', () => {
      const input: KanoInput = {
        functionalScore: 5,  // Like
        dysfunctionalScore: 3, // Neutral
      };

      const result = calculator.calculate(input);

      expect(result.category).toBe('Attractive');
    });

    it('should classify as Indifferent when functional=Neutral, dysfunctional=Neutral', () => {
      const input: KanoInput = {
        functionalScore: 3,  // Neutral
        dysfunctionalScore: 3, // Neutral
      };

      const result = calculator.calculate(input);

      expect(result.category).toBe('Indifferent');
    });

    it('should classify as Reverse when functional=Dislike, dysfunctional=Like', () => {
      const input: KanoInput = {
        functionalScore: 1,  // Dislike
        dysfunctionalScore: 5, // Like
      };

      const result = calculator.calculate(input);

      expect(result.category).toBe('Reverse');
    });

    it('should classify as Questionable when functional=Like, dysfunctional=Like', () => {
      const input: KanoInput = {
        functionalScore: 5,  // Like
        dysfunctionalScore: 5, // Like
      };

      const result = calculator.calculate(input);

      expect(result.category).toBe('Questionable');
    });

    it('should classify as Questionable when functional=Dislike, dysfunctional=Dislike', () => {
      const input: KanoInput = {
        functionalScore: 1,  // Dislike
        dysfunctionalScore: 1, // Dislike
      };

      const result = calculator.calculate(input);

      expect(result.category).toBe('Questionable');
    });

    it('should calculate satisfaction coefficient correctly', () => {
      const input: KanoInput = {
        functionalScore: 5,
        dysfunctionalScore: 1,
      };

      const result = calculator.calculate(input);

      // Satisfaction coefficient: (5 - 3) / 2 = 1.0
      expect(result.satisfactionCoefficient).toBe(1);
    });

    it('should calculate dissatisfaction coefficient correctly', () => {
      const input: KanoInput = {
        functionalScore: 5,
        dysfunctionalScore: 1,
      };

      const result = calculator.calculate(input);

      // Dissatisfaction coefficient: (3 - 1) / 2 = 1.0
      expect(result.dissatisfactionCoefficient).toBe(1);
    });

    it('should calculate negative coefficients for low scores', () => {
      const input: KanoInput = {
        functionalScore: 1,
        dysfunctionalScore: 5,
      };

      const result = calculator.calculate(input);

      // Satisfaction: (1 - 3) / 2 = -1.0
      // Dissatisfaction: (3 - 5) / 2 = -1.0
      expect(result.satisfactionCoefficient).toBe(-1);
      expect(result.dissatisfactionCoefficient).toBe(-1);
    });

    it('should include category interpretation in details', () => {
      const input: KanoInput = {
        functionalScore: 5,
        dysfunctionalScore: 1,
      };

      const result = calculator.calculate(input);

      expect(result.details).toHaveProperty('interpretation');
      expect(result.details.interpretation).toContain('Performance');
    });

    it('should include category recommendation in details', () => {
      const input: KanoInput = {
        functionalScore: 5,
        dysfunctionalScore: 1,
      };

      const result = calculator.calculate(input);

      expect(result.details).toHaveProperty('recommendation');
      expect(result.details.recommendation).toContain('Competitive priority');
    });

    it('should include importance in result when provided', () => {
      const input: KanoInput = {
        functionalScore: 5,
        dysfunctionalScore: 1,
        importance: 4,
      };

      const result = calculator.calculate(input);

      expect(result.importance).toBe(4);
    });

    it('should throw error for invalid input', () => {
      const invalidInput: Partial<KanoInput> = {
        functionalScore: 6,
        dysfunctionalScore: 1,
      };

      expect(() => calculator.calculate(invalidInput as KanoInput)).toThrow(
        /Invalid Kano input/
      );
    });
  });

  describe('validate', () => {
    it('should return valid for correct input', () => {
      const input: KanoInput = {
        functionalScore: 5,
        dysfunctionalScore: 1,
      };

      const validation = calculator.validate(input);

      expect(validation.isValid).toBe(true);
      expect(validation.errors).toHaveLength(0);
    });

    it('should detect functional score below minimum', () => {
      const input: Partial<KanoInput> = { functionalScore: 0 };

      const validation = calculator.validate(input);

      expect(validation.isValid).toBe(false);
      expect(validation.errors).toContainEqual(
        expect.stringContaining('Functional score')
      );
    });

    it('should detect functional score above maximum', () => {
      const input: Partial<KanoInput> = { functionalScore: 6 };

      const validation = calculator.validate(input);

      expect(validation.isValid).toBe(false);
      expect(validation.errors).toContainEqual(
        expect.stringContaining('Functional score')
      );
    });

    it('should detect dysfunctional score below minimum', () => {
      const input: Partial<KanoInput> = { dysfunctionalScore: 0 };

      const validation = calculator.validate(input);

      expect(validation.isValid).toBe(false);
      expect(validation.errors).toContainEqual(
        expect.stringContaining('Dysfunctional score')
      );
    });

    it('should detect dysfunctional score above maximum', () => {
      const input: Partial<KanoInput> = { dysfunctionalScore: 6 };

      const validation = calculator.validate(input);

      expect(validation.isValid).toBe(false);
      expect(validation.errors).toContainEqual(
        expect.stringContaining('Dysfunctional score')
      );
    });

    it('should detect invalid importance value', () => {
      const input: Partial<KanoInput> = {
        functionalScore: 4,
        dysfunctionalScore: 2,
        importance: 6,
      };

      const validation = calculator.validate(input);

      expect(validation.isValid).toBe(false);
      expect(validation.errors).toContainEqual(
        expect.stringContaining('Importance')
      );
    });

    it('should detect NaN values', () => {
      const input: Partial<KanoInput> = { functionalScore: NaN };

      const validation = calculator.validate(input);

      expect(validation.isValid).toBe(false);
      expect(validation.errors).toContainEqual(
        expect.stringContaining('valid number')
      );
    });

    it('should warn about contradictory responses (Like/Like)', () => {
      const input: Partial<KanoInput> = {
        functionalScore: 5,
        dysfunctionalScore: 5,
      };

      const validation = calculator.validate(input);

      expect(validation.warnings).toContainEqual(
        expect.stringContaining('Contradictory')
      );
    });

    it('should warn about contradictory responses (Dislike/Dislike)', () => {
      const input: Partial<KanoInput> = {
        functionalScore: 1,
        dysfunctionalScore: 1,
      };

      const validation = calculator.validate(input);

      expect(validation.warnings).toContainEqual(
        expect.stringContaining('Contradictory')
      );
    });

    it('should warn about reverse feature preference', () => {
      const input: Partial<KanoInput> = {
        functionalScore: 2,
        dysfunctionalScore: 5,
      };

      const validation = calculator.validate(input);

      expect(validation.warnings).toContainEqual(
        expect.stringContaining('prefers NOT having')
      );
    });

    it('should warn about extreme functional scores', () => {
      const input: Partial<KanoInput> = { functionalScore: 5 };

      const validation = calculator.validate(input);

      expect(validation.warnings).toContainEqual(
        expect.stringContaining('Maximum functional score')
      );
    });

    it('should warn about extreme dysfunctional scores', () => {
      const input: Partial<KanoInput> = {
        functionalScore: 3,
        dysfunctionalScore: 5,
      };

      const validation = calculator.validate(input);

      expect(validation.warnings).toContainEqual(
        expect.stringContaining('Maximum dysfunctional score')
      );
    });

    it('should return valid for partial input with missing fields', () => {
      const input: Partial<KanoInput> = { functionalScore: 4 };

      const validation = calculator.validate(input);

      expect(validation.errors).toHaveLength(0);
    });
  });

  describe('normalize', () => {
    let results: KanoResult[];

    beforeEach(() => {
      results = [
        calculator.calculate({ functionalScore: 5, dysfunctionalScore: 1, importance: 4 }), // Attractive
        calculator.calculate({ functionalScore: 1, dysfunctionalScore: 4, importance: 5 }), // MustBe
        calculator.calculate({ functionalScore: 3, dysfunctionalScore: 3, importance: 2 }), // Indifferent
        calculator.calculate({ functionalScore: 5, dysfunctionalScore: 5, importance: 3 }), // Questionable
      ];
    });

    it('should assign correct rank based on category and importance', () => {
      const normalized = calculator.normalize(results[0], results);
      // Attractive with importance 4 should rank high
      expect(normalized.rank).toBeLessThanOrEqual(2);
    });

    it('should assign correct rank to MustBe category', () => {
      const normalized = calculator.normalize(results[1], results);
      // MustBe with high importance (5) should rank high
      expect(normalized.rank).toBeLessThanOrEqual(2);
    });

    it('should assign lowest rank to Questionable category', () => {
      const normalized = calculator.normalize(results[3], results);
      // Questionable should have lowest rank
      expect(normalized.rank).toBe(4);
    });

    it('should calculate correct percentile', () => {
      const normalized = calculator.normalize(results[0], results);
      expect(normalized.percentile).toBeGreaterThan(50);
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

    it('should handle identical categories', () => {
      const sameInput = { functionalScore: 3, dysfunctionalScore: 3, importance: 3 };
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
      id: 'test-kano-1',
      title,
      description,
      category,
      createdAt: new Date(),
    });

    it('should suggest MustBe scores for basic/expected features', () => {
      const item = createItem(
        'User login functionality',
        'Core authentication system'
      );

      const suggestions = calculator.getAutoFillSuggestions(item);

      expect(suggestions.functionalScore).toBe(4);
      expect(suggestions.dysfunctionalScore).toBe(1);
      expect(suggestions.importance).toBe(5);
    });

    it('should suggest MustBe scores for security features', () => {
      const item = createItem(
        'Password encryption',
        'Security compliance requirement'
      );

      const suggestions = calculator.getAutoFillSuggestions(item);

      expect(suggestions.functionalScore).toBe(4);
      expect(suggestions.dysfunctionalScore).toBe(1);
    });

    it('should suggest OneDimensional scores for performance features', () => {
      const item = createItem(
        'Faster page load optimization',
        'Improve performance and speed'
      );

      const suggestions = calculator.getAutoFillSuggestions(item);

      expect(suggestions.functionalScore).toBe(5);
      expect(suggestions.dysfunctionalScore).toBe(2);
    });

    it('should suggest Attractive scores for innovative features', () => {
      const item = createItem(
        'AI-powered smart recommendations',
        'Innovative personalized suggestions'
      );

      const suggestions = calculator.getAutoFillSuggestions(item);

      expect(suggestions.functionalScore).toBe(5);
      expect(suggestions.dysfunctionalScore).toBe(3);
    });

    it('should suggest Attractive scores for delighter features', () => {
      const item = createItem(
        'Voice-activated controls',
        'Unique wow feature for app'
      );

      const suggestions = calculator.getAutoFillSuggestions(item);

      expect(suggestions.functionalScore).toBe(5);
    });

    it('should provide default suggestions for generic items', () => {
      const item = createItem('Some feature');

      const suggestions = calculator.getAutoFillSuggestions(item);

      expect(suggestions.functionalScore).toBe(4);
      expect(suggestions.dysfunctionalScore).toBe(3);
      expect(suggestions.importance).toBe(3);
    });

    it('should include confidence scores for all suggestions', () => {
      const item = createItem(
        'Security feature',
        'Required compliance'
      );

      const suggestions = calculator.getAutoFillSuggestions(item);

      expect(suggestions.functionalScoreConfidence).toBeDefined();
      expect(suggestions.dysfunctionalScoreConfidence).toBeDefined();
      expect(suggestions.importanceConfidence).toBeDefined();
    });

    it('should provide higher confidence for keyword-matched features', () => {
      const specificItem = createItem(
        'Automated AI insights',
        'Innovative smart predictions'
      );
      const genericItem = createItem('Some task');

      const specificSuggestions = calculator.getAutoFillSuggestions(specificItem);
      const genericSuggestions = calculator.getAutoFillSuggestions(genericItem);

      expect(specificSuggestions.functionalScoreConfidence).toBeGreaterThan(
        genericSuggestions.functionalScoreConfidence ?? 0
      );
    });
  });

  describe('KANO_RESPONSE_SCALE', () => {
    it('should have correct scale values', () => {
      expect(KANO_RESPONSE_SCALE.LIKE).toBe(5);
      expect(KANO_RESPONSE_SCALE.EXPECT).toBe(4);
      expect(KANO_RESPONSE_SCALE.NEUTRAL).toBe(3);
      expect(KANO_RESPONSE_SCALE.LIVE_WITH).toBe(2);
      expect(KANO_RESPONSE_SCALE.DISLIKE).toBe(1);
    });
  });

  describe('KANO_CATEGORY_SCORES', () => {
    it('should have correct category scores', () => {
      expect(KANO_CATEGORY_SCORES.MustBe).toBe(1);
      expect(KANO_CATEGORY_SCORES.OneDimensional).toBe(2);
      expect(KANO_CATEGORY_SCORES.Attractive).toBe(3);
      expect(KANO_CATEGORY_SCORES.Indifferent).toBe(0);
      expect(KANO_CATEGORY_SCORES.Reverse).toBe(-1);
      expect(KANO_CATEGORY_SCORES.Questionable).toBe(-2);
    });
  });

  describe('KANO_EVALUATION_MATRIX', () => {
    it('should have all 25 response combinations', () => {
      // 5x5 matrix = 25 combinations
      expect(Object.keys(KANO_EVALUATION_MATRIX).length).toBe(25);
    });

    it('should map (5,1) to OneDimensional', () => {
      expect(KANO_EVALUATION_MATRIX['5,1']).toBe('OneDimensional');
    });

    it('should map (1,4) to MustBe', () => {
      expect(KANO_EVALUATION_MATRIX['1,4']).toBe('MustBe');
    });

    it('should map (5,3) to Attractive', () => {
      expect(KANO_EVALUATION_MATRIX['5,3']).toBe('Attractive');
    });

    it('should map (3,3) to Indifferent', () => {
      expect(KANO_EVALUATION_MATRIX['3,3']).toBe('Indifferent');
    });

    it('should map (1,5) to Reverse', () => {
      expect(KANO_EVALUATION_MATRIX['1,5']).toBe('Reverse');
    });

    it('should map (5,5) to Questionable', () => {
      expect(KANO_EVALUATION_MATRIX['5,5']).toBe('Questionable');
    });
  });

  describe('createKanoCalculator factory', () => {
    it('should create calculator with default config', () => {
      const calc = createKanoCalculator();

      expect(calc).toBeInstanceOf(KanoCalculator);
      expect(calc.getFrameworkType()).toBe('Kano');
    });

    it('should create calculator with custom config', () => {
      const calc = createKanoCalculator({
        decimalPlaces: 3,
        includeCoefficients: false,
      });

      const result = calc.calculate({
        functionalScore: 5,
        dysfunctionalScore: 1,
      });

      expect(result.satisfactionCoefficient).toBeUndefined();
      expect(result.dissatisfactionCoefficient).toBeUndefined();
    });

    it('should calculate with custom decimal places', () => {
      const calc = createKanoCalculator({ decimalPlaces: 3 });

      const result = calc.calculate({
        functionalScore: 4,
        dysfunctionalScore: 2,
      });

      // (4-3)/2 = 0.5 -> 0.500
      expect(result.satisfactionCoefficient).toBe(0.5);
    });
  });

  describe('Edge Cases', () => {
    it('should handle minimum scores', () => {
      const input: KanoInput = {
        functionalScore: 1,
        dysfunctionalScore: 1,
      };

      const result = calculator.calculate(input);

      expect(result.category).toBe('Questionable');
      expect(result.satisfactionCoefficient).toBe(-1);
    });

    it('should handle maximum scores', () => {
      const input: KanoInput = {
        functionalScore: 5,
        dysfunctionalScore: 5,
      };

      const result = calculator.calculate(input);

      expect(result.category).toBe('Questionable');
      expect(result.satisfactionCoefficient).toBe(1);
    });

    it('should handle neutral responses', () => {
      const input: KanoInput = {
        functionalScore: 3,
        dysfunctionalScore: 3,
      };

      const result = calculator.calculate(input);

      expect(result.category).toBe('Indifferent');
      expect(result.satisfactionCoefficient).toBe(0);
      expect(result.dissatisfactionCoefficient).toBe(0);
    });

    it('should handle boundary scores (2 and 4)', () => {
      const input: KanoInput = {
        functionalScore: 4,
        dysfunctionalScore: 2,
      };

      const result = calculator.calculate(input);

      // (4,2) -> Indifferent per Kano evaluation matrix
      expect(result.category).toBe('Indifferent');
      expect(result.satisfactionCoefficient).toBe(0.5);
      expect(result.dissatisfactionCoefficient).toBe(0.5);
    });

    it('should handle all Live With responses', () => {
      const input: KanoInput = {
        functionalScore: 2,
        dysfunctionalScore: 2,
      };

      const result = calculator.calculate(input);

      expect(result.category).toBe('Indifferent');
    });

    it('should handle Reverse classification correctly', () => {
      const input: KanoInput = {
        functionalScore: 2,
        dysfunctionalScore: 5,
      };

      const result = calculator.calculate(input);

      expect(result.category).toBe('Reverse');
      expect(result.details.recommendation).toContain('Reconsider');
    });

    it('should include satisfactionIfPresent and dissatisfactionIfAbsent in details', () => {
      const input: KanoInput = {
        functionalScore: 5,
        dysfunctionalScore: 1,
        satisfactionIfPresent: 5,
        dissatisfactionIfAbsent: 2,
      };

      const result = calculator.calculate(input);

      expect(result.details.satisfactionIfPresent).toBe(5);
      expect(result.details.dissatisfactionIfAbsent).toBe(2);
    });
  });

  describe('Category Interpretations', () => {
    it('should provide correct interpretation for MustBe', () => {
      const result = calculator.calculate({ functionalScore: 1, dysfunctionalScore: 4 });
      expect(result.details.interpretation).toContain('Basic expectation');
    });

    it('should provide correct interpretation for OneDimensional', () => {
      const result = calculator.calculate({ functionalScore: 5, dysfunctionalScore: 1 });
      expect(result.details.interpretation).toContain('Performance');
    });

    it('should provide correct interpretation for Attractive', () => {
      const result = calculator.calculate({ functionalScore: 5, dysfunctionalScore: 3 });
      expect(result.details.interpretation).toContain('Delighter');
    });

    it('should provide correct interpretation for Indifferent', () => {
      const result = calculator.calculate({ functionalScore: 3, dysfunctionalScore: 3 });
      expect(result.details.interpretation).toContain('Neutral');
    });

    it('should provide correct interpretation for Reverse', () => {
      const result = calculator.calculate({ functionalScore: 1, dysfunctionalScore: 5 });
      expect(result.details.interpretation).toContain('Negative');
    });

    it('should provide correct interpretation for Questionable', () => {
      const result = calculator.calculate({ functionalScore: 5, dysfunctionalScore: 5 });
      expect(result.details.interpretation).toContain('Unclear');
    });
  });

  describe('Category Recommendations', () => {
    it('should provide correct recommendation for MustBe', () => {
      const result = calculator.calculate({ functionalScore: 1, dysfunctionalScore: 4 });
      expect(result.details.recommendation).toContain('Must implement');
    });

    it('should provide correct recommendation for OneDimensional', () => {
      const result = calculator.calculate({ functionalScore: 5, dysfunctionalScore: 1 });
      expect(result.details.recommendation).toContain('Competitive priority');
    });

    it('should provide correct recommendation for Attractive', () => {
      const result = calculator.calculate({ functionalScore: 5, dysfunctionalScore: 3 });
      expect(result.details.recommendation).toContain('Differentiator');
    });

    it('should provide correct recommendation for Indifferent', () => {
      const result = calculator.calculate({ functionalScore: 3, dysfunctionalScore: 3 });
      expect(result.details.recommendation).toContain('Evaluate cost-benefit');
    });

    it('should provide correct recommendation for Reverse', () => {
      const result = calculator.calculate({ functionalScore: 1, dysfunctionalScore: 5 });
      expect(result.details.recommendation).toContain('Reconsider');
    });

    it('should provide correct recommendation for Questionable', () => {
      const result = calculator.calculate({ functionalScore: 5, dysfunctionalScore: 5 });
      expect(result.details.recommendation).toContain('Gather more feedback');
    });
  });
});
