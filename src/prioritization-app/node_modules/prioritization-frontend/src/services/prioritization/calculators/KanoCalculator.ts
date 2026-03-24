/**
 * Kano Model Prioritization Framework Calculator
 *
 * Implements the Kano model for product development and customer satisfaction analysis.
 * The Kano model classifies features based on customer perceptions into categories:
 *
 * - Must-be (Basic): Expected features that cause dissatisfaction when absent
 * - One-dimensional (Performance): More is better, linear satisfaction relationship
 * - Attractive (Delighters): Unexpected features that cause satisfaction when present
 * - Indifferent: Features customers don't care about either way
 * - Reverse: Features that cause dissatisfaction when present
 * - Questionable: Results from unclear or conflicting responses
 *
 * The model uses paired questions (functional and dysfunctional) to determine
 * how customers feel about feature presence and absence.
 *
 * @see {@link https://kano.method.sh/}
 * @see {@link {https://www.sciencedirect.com/science/article/abs/pii/S0166497204000584}}
 * @module prioritization/calculators/KanoCalculator
 */

import {
  IFrameworkCalculator,
  PrioritizationItem,
  KanoInput,
  KanoResult,
  KanoCategory,
  ValidationResult,
  NormalizedResult,
  FrameworkType,
} from '../types';

/**
 * Response options for Kano questionnaire.
 * Scale: 1-5 mapping to customer feelings
 */
export const KANO_RESPONSE_SCALE = {
  LIKE: 5,
  EXPECT: 4,
  NEUTRAL: 3,
  LIVE_WITH: 2,
  DISLIKE: 1,
} as const;

/**
 * Kano evaluation matrix for determining category from responses.
 * Keys: [functionalScore, dysfunctionalScore]
 * Values: KanoCategory
 */
export const KANO_EVALUATION_MATRIX: Record<string, KanoCategory> = {
  // Functional: Like (5)
  '5,5': 'Questionable',
  '5,4': 'Attractive',
  '5,3': 'Attractive',
  '5,2': 'Attractive',
  '5,1': 'OneDimensional',
  // Functional: Expect (4)
  '4,5': 'Reverse',
  '4,4': 'Indifferent',
  '4,3': 'Indifferent',
  '4,2': 'Indifferent',
  '4,1': 'OneDimensional',
  // Functional: Neutral (3)
  '3,5': 'Reverse',
  '3,4': 'Indifferent',
  '3,3': 'Indifferent',
  '3,2': 'Indifferent',
  '3,1': 'OneDimensional',
  // Functional: Live with (2)
  '2,5': 'Reverse',
  '2,4': 'Indifferent',
  '2,3': 'Indifferent',
  '2,2': 'Indifferent',
  '2,1': 'OneDimensional',
  // Functional: Dislike (1)
  '1,5': 'Reverse',
  '1,4': 'MustBe',
  '1,3': 'MustBe',
  '1,2': 'MustBe',
  '1,1': 'Questionable',
} as const;

/**
 * Numeric scores for each Kano category (for scoring calculations).
 */
export const KANO_CATEGORY_SCORES: Record<KanoCategory, number> = {
  MustBe: 1,
  OneDimensional: 2,
  Attractive: 3,
  Indifferent: 0,
  Reverse: -1,
  Questionable: -2,
} as const;

/**
 * Default configuration for Kano calculator.
 */
const DEFAULT_CONFIG = {
  /** Minimum score value (1 = Dislike) */
  minScore: 1,
  /** Maximum score value (5 = Like) */
  maxScore: 5,
  /** Decimal places for rounding */
  decimalPlaces: 2,
  /** Whether to include coefficient calculations */
  includeCoefficients: true,
};

/**
 * Kano Model Framework Calculator implementation.
 *
 * Determines feature category based on functional and dysfunctional scores
 * using the Kano evaluation matrix.
 *
 * @example
 * ```typescript
 * const calculator = new KanoCalculator();
 *
 * const input = {
 *   functionalScore: 5,      // Customer likes having the feature
 *   dysfunctionalScore: 1,   // Customer dislikes not having it
 *   importance: 4
 * };
 *
 * const result = calculator.calculate(input);
 * // result.category = 'OneDimensional'
 * ```
 */
export class KanoCalculator implements IFrameworkCalculator<KanoInput, KanoResult> {
  private readonly minScore: number;
  private readonly maxScore: number;
  private readonly decimalPlaces: number;
  private readonly includeCoefficients: boolean;

  /**
   * Creates a new Kano calculator instance.
   * @param config - Optional configuration to override defaults
   */
  constructor(config: Partial<typeof DEFAULT_CONFIG> = DEFAULT_CONFIG) {
    this.minScore = config.minScore ?? DEFAULT_CONFIG.minScore;
    this.maxScore = config.maxScore ?? DEFAULT_CONFIG.maxScore;
    this.decimalPlaces = config.decimalPlaces ?? DEFAULT_CONFIG.decimalPlaces;
    this.includeCoefficients = config.includeCoefficients ?? DEFAULT_CONFIG.includeCoefficients;
  }

  /**
   * Returns the framework type identifier.
   *
   * @returns 'Kano' as the framework type
   */
  getFrameworkType(): FrameworkType {
    return 'Kano';
  }

  /**
   * Calculates the Kano category for given input parameters.
   *
   * Uses the Kano evaluation matrix to determine the category based on
   * functional (how do you feel if feature is present?) and
   * dysfunctional (how do you feel if feature is absent?) scores.
   *
   * @param input - The Kano input parameters
   * @returns KanoResult containing the assigned category and coefficients
   * @throws {Error} If input validation fails
   *
   * @example
   * ```typescript
   * const result = calculator.calculate({
   *   functionalScore: 5,
   *   dysfunctionalScore: 1
   * });
   * // result.category = 'OneDimensional'
   * // result.satisfactionCoefficient = 1.0
   * // result.dissatisfactionCoefficient = -1.0
   * ```
   */
  calculate(input: KanoInput): KanoResult {
    // Validate input first
    const validation = this.validate(input);
    if (!validation.isValid) {
      throw new Error(`Invalid Kano input: ${validation.errors.join(', ')}`);
    }

    // Determine category using evaluation matrix
    const category = this.determineCategory(input.functionalScore, input.dysfunctionalScore);

    // Calculate satisfaction and dissatisfaction coefficients
    const satisfactionCoefficient = this.includeCoefficients
      ? this.calculateSatisfactionCoefficient(input.functionalScore)
      : undefined;
    const dissatisfactionCoefficient = this.includeCoefficients
      ? this.calculateDissatisfactionCoefficient(input.dysfunctionalScore)
      : undefined;

    return {
      framework: 'Kano',
      functionalScore: input.functionalScore,
      dysfunctionalScore: input.dysfunctionalScore,
      category,
      satisfactionCoefficient,
      dissatisfactionCoefficient,
      importance: input.importance,
      details: {
        categoryScore: KANO_CATEGORY_SCORES[category],
        interpretation: this.getCategoryInterpretation(category),
        recommendation: this.getCategoryRecommendation(category),
        satisfactionIfPresent: input.satisfactionIfPresent,
        dissatisfactionIfAbsent: input.dissatisfactionIfAbsent,
      },
    };
  }

  /**
   * Validates Kano input parameters.
   *
   * Checks:
   * - Functional score must be between 1-5
   * - Dysfunctional score must be between 1-5
   * - Optional fields (importance, satisfactionIfPresent, dissatisfactionIfAbsent)
   *   must be within valid range if provided
   *
   * @param input - Partial Kano input parameters to validate
   * @returns ValidationResult with errors and warnings
   *
   * @example
   * ```typescript
   * const validation = calculator.validate({
   *   functionalScore: 6,      // Error: exceeds max
   *   dysfunctionalScore: 0    // Error: below min
   * });
   * // validation.isValid = false
   * ```
   */
  validate(input: Partial<KanoInput>): ValidationResult {
    const errors: string[] = [];
    const warnings: string[] = [];

    // Validate functionalScore
    if (input.functionalScore !== undefined) {
      if (typeof input.functionalScore !== 'number' || isNaN(input.functionalScore)) {
        errors.push('Functional score must be a valid number');
      } else if (input.functionalScore < this.minScore) {
        errors.push(`Functional score must be at least ${this.minScore} (Dislike)`);
      } else if (input.functionalScore > this.maxScore) {
        errors.push(`Functional score must be at most ${this.maxScore} (Like)`);
      } else if (input.functionalScore === KANO_RESPONSE_SCALE.LIKE &&
                 this.isExtremeResponse(input, 'functional')) {
        warnings.push('Maximum functional score - ensure this reflects true customer sentiment');
      }
    }

    // Validate dysfunctionalScore
    if (input.dysfunctionalScore !== undefined) {
      if (typeof input.dysfunctionalScore !== 'number' || isNaN(input.dysfunctionalScore)) {
        errors.push('Dysfunctional score must be a valid number');
      } else if (input.dysfunctionalScore < this.minScore) {
        errors.push(`Dysfunctional score must be at least ${this.minScore} (Dislike)`);
      } else if (input.dysfunctionalScore > this.maxScore) {
        errors.push(`Dysfunctional score must be at most ${this.maxScore} (Like)`);
      } else if (input.dysfunctionalScore === KANO_RESPONSE_SCALE.LIKE &&
                 this.isExtremeResponse(input, 'dysfunctional')) {
        warnings.push('Maximum dysfunctional score - customer likes NOT having the feature');
      }
    }

    // Validate importance if provided
    if (input.importance !== undefined) {
      if (typeof input.importance !== 'number' || isNaN(input.importance)) {
        errors.push('Importance must be a valid number');
      } else if (input.importance < this.minScore) {
        errors.push(`Importance must be at least ${this.minScore}`);
      } else if (input.importance > this.maxScore) {
        errors.push(`Importance must be at most ${this.maxScore}`);
      }
    }

    // Validate satisfactionIfPresent if provided
    if (input.satisfactionIfPresent !== undefined) {
      if (typeof input.satisfactionIfPresent !== 'number' || isNaN(input.satisfactionIfPresent)) {
        errors.push('Satisfaction if present must be a valid number');
      } else if (input.satisfactionIfPresent < this.minScore) {
        errors.push(`Satisfaction if present must be at least ${this.minScore}`);
      } else if (input.satisfactionIfPresent > this.maxScore) {
        errors.push(`Satisfaction if present must be at most ${this.maxScore}`);
      }
    }

    // Validate dissatisfactionIfAbsent if provided
    if (input.dissatisfactionIfAbsent !== undefined) {
      if (typeof input.dissatisfactionIfAbsent !== 'number' || isNaN(input.dissatisfactionIfAbsent)) {
        errors.push('Dissatisfaction if absent must be a valid number');
      } else if (input.dissatisfactionIfAbsent < this.minScore) {
        errors.push(`Dissatisfaction if absent must be at least ${this.minScore}`);
      } else if (input.dissatisfactionIfAbsent > this.maxScore) {
        errors.push(`Dissatisfaction if absent must be at most ${this.maxScore}`);
      }
    }

    // Warning: Questionable responses (same extreme scores)
    if (
      input.functionalScore !== undefined &&
      input.dysfunctionalScore !== undefined
    ) {
      if (
        (input.functionalScore === 5 && input.dysfunctionalScore === 5) ||
        (input.functionalScore === 1 && input.dysfunctionalScore === 1)
      ) {
        warnings.push('Contradictory responses may indicate survey confusion - review for Questionable classification');
      }

      // Warning: Reverse features (like not having it more than having it)
      if (input.dysfunctionalScore > input.functionalScore + 1) {
        warnings.push('Customer prefers NOT having the feature - may be classified as Reverse');
      }
    }

    return {
      isValid: errors.length === 0,
      errors,
      warnings,
    };
  }

  /**
   * Normalizes Kano results across a dataset.
   *
   * Normalization considers:
   * 1. Category score (Attractive > OneDimensional > MustBe > Indifferent > Reverse > Questionable)
   * 2. Importance rating (if provided)
   * 3. Satisfaction coefficients
   *
   * @param result - The Kano result to normalize
   * @param allResults - All Kano results in the dataset
   * @returns NormalizedResult with rank, percentile, and normalized score
   *
   * @example
   * ```typescript
   * const results = [
   *   calculator.calculate({ functionalScore: 5, dysfunctionalScore: 1, importance: 4 }),
   *   calculator.calculate({ functionalScore: 1, dysfunctionalScore: 4, importance: 3 }),
   *   calculator.calculate({ functionalScore: 3, dysfunctionalScore: 3, importance: 2 }),
   * ];
   *
   * const normalized = calculator.normalize(results[0], results);
   * // normalized.rank = 1 (Attractive with high importance)
   * // normalized.percentile = 100
   * // normalized.normalizedScore = 100
   * ```
   */
  normalize(result: KanoResult, allResults: KanoResult[]): NormalizedResult {
    if (allResults.length === 0) {
      return {
        normalizedScore: 0,
        rank: 0,
        percentile: 0,
      };
    }

    // Calculate composite score for sorting
    const getCompositeScore = (r: KanoResult): number => {
      const categoryScore = KANO_CATEGORY_SCORES[r.category];
      const importanceBonus = (r.importance ?? 3) / 5; // Normalize to 0-1

      // Coefficient contribution (if available)
      const satCoeff = r.satisfactionCoefficient ?? 0;
      const dissatCoeff = r.dissatisfactionCoefficient ?? 0;
      const coefficientBonus = (satCoeff - dissatCoeff) / 2;

      // Composite: category (50%) + importance (30%) + coefficients (20%)
      return (categoryScore * 0.5) + (importanceBonus * 0.3) + (coefficientBonus * 0.2);
    };

    // Sort by composite score (descending)
    const sortedResults = [...allResults].sort((a, b) => {
      return getCompositeScore(b) - getCompositeScore(a);
    });

    // Find the rank of this result (1-indexed)
    const rank = sortedResults.findIndex(
      (r) => r.category === result.category && r.functionalScore === result.functionalScore
    ) + 1;

    // Calculate percentile
    const itemsBelow = sortedResults.filter((r) => {
      return getCompositeScore(r) < getCompositeScore(result);
    }).length;

    const percentile =
      allResults.length > 1
        ? parseFloat(((itemsBelow / (allResults.length - 1)) * 100).toFixed(this.decimalPlaces))
        : 100;

    // Calculate normalized score on 0-100 scale
    const scores = allResults.map(getCompositeScore);
    const minScore = Math.min(...scores);
    const maxScore = Math.max(...scores);

    let normalizedScore: number;
    if (maxScore === minScore) {
      normalizedScore = 50;
    } else {
      normalizedScore = parseFloat(
        (((KANO_CATEGORY_SCORES[result.category] - minScore) / (maxScore - minScore)) * 100).toFixed(this.decimalPlaces)
      );
    }

    return {
      normalizedScore,
      rank,
      percentile,
    };
  }

  /**
   * Generates auto-fill suggestions based on item metadata.
   *
   * Analyzes the item's title, description, and category to suggest
   * reasonable Kano input values using keyword matching and heuristics.
   *
   * @param item - The prioritization item to analyze
   * @returns Partial<KanoInput> with suggested values and confidence scores
   *
   * @example
   * ```typescript
   * const item = {
   *   id: 'feat-1',
   *   title: 'Login with fingerprint authentication',
   *   description: 'Biometric login for mobile app',
   *   category: 'Feature',
   *   createdAt: new Date(),
   * };
   *
   * const suggestions = calculator.getAutoFillSuggestions(item);
   * // {
   * //   functionalScore: 5,
   * //   functionalScoreConfidence: 0.75,
   * //   dysfunctionalScore: 2,
   * //   dysfunctionalScoreConfidence: 0.7,
   * //   importance: 4,
   * //   importanceConfidence: 0.6
   * // }
   * ```
   */
  getAutoFillSuggestions(item: PrioritizationItem): Partial<KanoInput> & {
    functionalScoreConfidence?: number;
    dysfunctionalScoreConfidence?: number;
    importanceConfidence?: number;
    satisfactionIfPresentConfidence?: number;
    dissatisfactionIfAbsentConfidence?: number;
  } {
    const text = `${item.title} ${item.description || ''} ${item.category || ''}`.toLowerCase();
    const suggestions: Partial<KanoInput> & {
      functionalScoreConfidence?: number;
      dysfunctionalScoreConfidence?: number;
      importanceConfidence?: number;
      satisfactionIfPresentConfidence?: number;
      dissatisfactionIfAbsentConfidence?: number;
    } = {};

    // Must-be (Basic) indicators - expected features
    const mustBeKeywords = [
      'basic',
      'standard',
      'expected',
      'required',
      'fundamental',
      'essential',
      'core',
      'login',
      'security',
      'password',
      'must have',
      'compliance',
    ];

    // One-dimensional (Performance) indicators - more is better
    const oneDimensionalKeywords = [
      'faster',
      'performance',
      'speed',
      'efficiency',
      'optimize',
      'improve',
      'better',
      'enhance',
      'increase',
      'reduce',
      'scale',
      'capacity',
    ];

    // Attractive (Delighter) indicators - unexpected features
    const attractiveKeywords = [
      'innovative',
      'delight',
      'surprise',
      'wow',
      'new',
      'first',
      'unique',
      'ai',
      'automation',
      'smart',
      'personalized',
      'predictive',
      'recommendation',
      'voice',
      'gesture',
    ];

    // Determine likely category based on keywords
    let likelyCategory: KanoCategory = 'Indifferent';

    if (mustBeKeywords.some((k) => text.includes(k))) {
      likelyCategory = 'MustBe';
    } else if (attractiveKeywords.some((k) => text.includes(k))) {
      likelyCategory = 'Attractive';
    } else if (oneDimensionalKeywords.some((k) => text.includes(k))) {
      likelyCategory = 'OneDimensional';
    }

    // Set scores based on likely category
    switch (likelyCategory) {
      case 'MustBe':
        suggestions.functionalScore = 4; // Expect if present
        suggestions.functionalScoreConfidence = 0.7;
        suggestions.dysfunctionalScore = 1; // Dislike if absent
        suggestions.dysfunctionalScoreConfidence = 0.8;
        suggestions.importance = 5;
        suggestions.importanceConfidence = 0.75;
        break;
      case 'OneDimensional':
        suggestions.functionalScore = 5; // Like if present
        suggestions.functionalScoreConfidence = 0.75;
        suggestions.dysfunctionalScore = 2; // Live with if absent
        suggestions.dysfunctionalScoreConfidence = 0.7;
        suggestions.importance = 4;
        suggestions.importanceConfidence = 0.65;
        break;
      case 'Attractive':
        suggestions.functionalScore = 5; // Like if present
        suggestions.functionalScoreConfidence = 0.8;
        suggestions.dysfunctionalScore = 3; // Neutral if absent
        suggestions.dysfunctionalScoreConfidence = 0.7;
        suggestions.importance = 3;
        suggestions.importanceConfidence = 0.6;
        break;
      default:
        suggestions.functionalScore = 4;
        suggestions.functionalScoreConfidence = 0.5;
        suggestions.dysfunctionalScore = 3;
        suggestions.dysfunctionalScoreConfidence = 0.5;
        suggestions.importance = 3;
        suggestions.importanceConfidence = 0.4;
    }

    // Additional satisfaction/dissatisfaction scores
    suggestions.satisfactionIfPresent = suggestions.functionalScore;
    suggestions.satisfactionIfPresentConfidence = suggestions.functionalScoreConfidence;
    suggestions.dissatisfactionIfAbsent = suggestions.dysfunctionalScore;
    suggestions.dissatisfactionIfAbsentConfidence = suggestions.dysfunctionalScoreConfidence;

    return suggestions;
  }

  /**
   * Determines the Kano category from functional and dysfunctional scores.
   * @param functionalScore - How the customer feels if feature is present
   * @param dysfunctionalScore - How the customer feels if feature is absent
   * @returns The assigned Kano category
   * @private
   */
  private determineCategory(functionalScore: number, dysfunctionalScore: number): KanoCategory {
    const key = `${functionalScore},${dysfunctionalScore}`;
    return KANO_EVALUATION_MATRIX[key] || 'Indifferent';
  }

  /**
   * Calculates the satisfaction coefficient.
   * Formula: (Like + Expect) - (Dislike) / Total responses
   * Simplified: functionalScore / maxScore - neutral point
   * @param functionalScore - The functional score
   * @returns Satisfaction coefficient (-1 to 1)
   * @private
   */
  private calculateSatisfactionCoefficient(functionalScore: number): number {
    // Normalize to -1 to 1 scale
    // 5 (Like) -> 1.0, 3 (Neutral) -> 0, 1 (Dislike) -> -1.0
    return parseFloat(((functionalScore - 3) / 2).toFixed(this.decimalPlaces));
  }

  /**
   * Calculates the dissatisfaction coefficient.
   * Formula: (Dislike + Live with) - (Like) / Total responses
   * Simplified: inverted dysfunctionalScore
   * @param dysfunctionalScore - The dysfunctional score
   * @returns Dissatisfaction coefficient (-1 to 1)
   * @private
   */
  private calculateDissatisfactionCoefficient(dysfunctionalScore: number): number {
    // Invert: if they Like not having it, dissatisfaction is high (negative)
    // 5 (Like not having) -> -1.0, 3 (Neutral) -> 0, 1 (Dislike not having) -> 1.0
    return parseFloat(((3 - dysfunctionalScore) / 2).toFixed(this.decimalPlaces));
  }

  /**
   * Returns an interpretation for the given category.
   * @param category - The Kano category
   * @returns A descriptive interpretation
   * @private
   */
  private getCategoryInterpretation(category: KanoCategory): string {
    const interpretations: Record<KanoCategory, string> = {
      MustBe: 'Basic expectation - absence causes significant dissatisfaction',
      OneDimensional: 'Performance attribute - satisfaction increases with quality',
      Attractive: 'Delighter - creates positive surprise and differentiation',
      Indifferent: 'Neutral - customers do not strongly care either way',
      Reverse: 'Negative feature - may cause dissatisfaction when present',
      Questionable: 'Unclear - responses were contradictory or confusing',
    };
    return interpretations[category];
  }

  /**
   * Returns a recommendation for the given category.
   * @param category - The Kano category
   * @returns A product recommendation
   * @private
   */
  private getCategoryRecommendation(category: KanoCategory): string {
    const recommendations: Record<KanoCategory, string> = {
      MustBe: 'Must implement - failure to deliver will cause customer dissatisfaction',
      OneDimensional: 'Competitive priority - more investment yields higher satisfaction',
      Attractive: 'Differentiator - implement selectively to create wow moments',
      Indifferent: 'Evaluate cost-benefit - low impact on satisfaction',
      Reverse: 'Reconsider - may need redesign or removal',
      Questionable: 'Gather more feedback - conduct additional customer research',
    };
    return recommendations[category];
  }

  /**
   * Checks if a response is at extreme (1 or 5).
   * @param input - The Kano input
   * @param type - 'functional' or 'dysfunctional'
   * @returns True if the response is extreme
   * @private
   */
  private isExtremeResponse(input: Partial<KanoInput>, type: 'functional' | 'dysfunctional'): boolean {
    const score = type === 'functional' ? input.functionalScore : input.dysfunctionalScore;
    return score === 1 || score === 5;
  }
}

/**
 * Factory function to create a new Kano calculator instance.
 *
 * @param config - Optional configuration to override defaults
 * @returns A new KanoCalculator instance
 *
 * @example
 * ```typescript
 * const calculator = createKanoCalculator({
 *   includeCoefficients: false,
 *   decimalPlaces: 3
 * });
 * ```
 */
export function createKanoCalculator(config: Partial<typeof DEFAULT_CONFIG> = {}): KanoCalculator {
  return new KanoCalculator({ ...DEFAULT_CONFIG, ...config });
}

export default KanoCalculator;
