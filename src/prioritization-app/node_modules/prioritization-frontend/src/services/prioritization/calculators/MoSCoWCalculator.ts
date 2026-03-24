/**
 * MoSCoW Prioritization Framework Calculator
 *
 * Implements the MoSCoW method for prioritizing requirements and features.
 * MoSCoW stands for: Must have, Should have, Could have, Won't have
 *
 * This framework categorizes items based on:
 * - Business value criticality
 * - Legal or compliance requirements
 * - Customer requests
 * - Risk if not delivered
 *
 * @see {@link https://www.agilebusiness.org/page/MoSCoW}
 * @module prioritization/calculators/MoSCoWCalculator
 */

import {
  IFrameworkCalculator,
  PrioritizationItem,
  MoSCoWInput,
  MoSCoWResult,
  MoSCoWCategory,
  BusinessValueLevel,
  RiskLevel,
  ValidationResult,
  NormalizedResult,
  FrameworkType,
} from '../types';

/**
 * Numeric values for business value levels.
 * Used for internal calculations and comparisons.
 */
export const BUSINESS_VALUE_SCORES: Record<BusinessValueLevel, number> = {
  critical: 4,
  high: 3,
  medium: 2,
  low: 1,
} as const;

/**
 * Numeric values for risk levels.
 * Used for internal calculations and comparisons.
 */
export const RISK_SCORES: Record<RiskLevel, number> = {
  critical: 4,
  high: 3,
  medium: 2,
  low: 1,
} as const;

/**
 * Priority weights for each MoSCoW category.
 * Higher values indicate higher priority.
 */
export const MOSCOW_CATEGORY_WEIGHTS: Record<MoSCoWCategory, number> = {
  'Must have': 4,
  'Should have': 3,
  'Could have': 2,
  "Won't have": 1,
} as const;

/**
 * Default configuration for MoSCoW calculator.
 */
const DEFAULT_CONFIG = {
  /** Minimum items per category for balanced distribution */
  minMustHavePercentage: 10,
  /** Maximum items that should be marked as Must Have */
  maxMustHavePercentage: 60,
  /** Decimal places for normalized scores */
  decimalPlaces: 2,
};

/**
 * MoSCoW Framework Calculator implementation.
 *
 * Categorizes requirements using the MoSCoW method based on:
 * 1. Business value criticality
 * 2. Legal/compliance requirements (automatic Must have)
 * 3. Customer requests (influences category)
 * 4. Risk if not delivered
 *
 * Category Assignment Rules:
 * - Must have: Critical business value OR legal requirement OR critical risk
 * - Should have: High business value AND high risk
 * - Could have: Medium business value AND medium/low risk
 * - Won't have: Low priority items
 *
 * @example
 * ```typescript
 * const calculator = new MoSCoWCalculator();
 *
 * const input = {
 *   businessValue: 'critical',
 *   legalRequirement: false,
 *   customerRequest: true,
 *   riskIfNotDelivered: 'high'
 * };
 *
 * const result = calculator.calculate(input);
 * // result.category = 'Must have'
 * ```
 */
export class MoSCoWCalculator implements IFrameworkCalculator<MoSCoWInput, MoSCoWResult> {
  private readonly decimalPlaces: number;

  /**
   * Creates a new MoSCoW calculator instance.
   * @param config - Optional configuration to override defaults
   */
  constructor(config = DEFAULT_CONFIG) {
    this.decimalPlaces = config.decimalPlaces ?? DEFAULT_CONFIG.decimalPlaces;
  }

  /**
   * Returns the framework type identifier.
   *
   * @returns 'MoSCoW' as the framework type
   */
  getFrameworkType(): FrameworkType {
    return 'MoSCoW';
  }

  /**
   * Calculates the MoSCoW category for a given set of input parameters.
   *
   * Uses a decision tree based on:
   * 1. Legal requirements (automatic Must have)
   * 2. Business value level
   * 3. Risk if not delivered
   * 4. Customer request status (tiebreaker)
   *
   * @param input - The MoSCoW input parameters
   * @returns MoSCoWResult containing the assigned category and priority
   * @throws {Error} If input validation fails
   *
   * @example
   * ```typescript
   * const result = calculator.calculate({
   *   businessValue: 'high',
   *   legalRequirement: false,
   *   customerRequest: true,
   *   riskIfNotDelivered: 'high'
   * });
   * // result.category = 'Should have'
   * // result.priority = 1
   * ```
   */
  calculate(input: MoSCoWInput): MoSCoWResult {
    // Validate input first
    const validation = this.validate(input);
    if (!validation.isValid) {
      throw new Error(`Invalid MoSCoW input: ${validation.errors.join(', ')}`);
    }

    // Determine category using decision logic
    const category = this.determineCategory(input);

    // Calculate priority within category (will be refined when comparing with other items)
    const priority = this.calculateInitialPriority(input, category);

    return {
      framework: 'MoSCoW',
      category,
      priority,
      businessValue: input.businessValue,
      legalRequirement: input.legalRequirement,
      customerRequest: input.customerRequest,
      riskIfNotDelivered: input.riskIfNotDelivered,
      details: {
        businessValueScore: BUSINESS_VALUE_SCORES[input.businessValue],
        riskScore: RISK_SCORES[input.riskIfNotDelivered],
        categoryWeight: MOSCOW_CATEGORY_WEIGHTS[category],
        decisionFactors: this.getDecisionFactors(input),
      },
    };
  }

  /**
   * Validates MoSCoW input parameters.
   *
   * Checks:
   * - All required fields are present
   * - Business value is a valid level
   * - Risk level is valid
   * - Boolean fields are actually boolean
   *
   * @param input - Partial MoSCoW input parameters to validate
   * @returns ValidationResult with errors and warnings
   *
   * @example
   * ```typescript
   * const validation = calculator.validate({
   *   businessValue: 'invalid',  // Error
   *   legalRequirement: 'yes',   // Error: not boolean
   *   customerRequest: true,     // Valid
   *   // riskIfNotDelivered missing - Error
   * });
   * // validation.isValid = false
   * ```
   */
  validate(input: Partial<MoSCoWInput>): ValidationResult {
    const errors: string[] = [];
    const warnings: string[] = [];

    const validBusinessValues: BusinessValueLevel[] = [
      'critical',
      'high',
      'medium',
      'low',
    ];
    const validRiskLevels: RiskLevel[] = ['critical', 'high', 'medium', 'low'];

    // Validate businessValue
    if (input.businessValue === undefined) {
      errors.push('businessValue is required');
    } else if (!validBusinessValues.includes(input.businessValue)) {
      errors.push(
        `businessValue must be one of: ${validBusinessValues.join(', ')}`
      );
    }

    // Validate legalRequirement
    if (input.legalRequirement === undefined) {
      errors.push('legalRequirement is required');
    } else if (typeof input.legalRequirement !== 'boolean') {
      errors.push('legalRequirement must be a boolean (true/false)');
    }

    // Validate customerRequest
    if (input.customerRequest === undefined) {
      errors.push('customerRequest is required');
    } else if (typeof input.customerRequest !== 'boolean') {
      errors.push('customerRequest must be a boolean (true/false)');
    }

    // Validate riskIfNotDelivered
    if (input.riskIfNotDelivered === undefined) {
      errors.push('riskIfNotDelivered is required');
    } else if (!validRiskLevels.includes(input.riskIfNotDelivered)) {
      errors.push(
        `riskIfNotDelivered must be one of: ${validRiskLevels.join(', ')}`
      );
    }

    // Warning: Legal requirements should typically have critical/high business value
    if (input.legalRequirement === true && input.businessValue === 'low') {
      warnings.push(
        'Legal requirements typically have higher business value - review classification'
      );
    }

    // Warning: Critical risk with low business value is unusual
    if (
      input.riskIfNotDelivered === 'critical' &&
      input.businessValue === 'low'
    ) {
      warnings.push(
        'Critical risk with low business value is contradictory - review classification'
      );
    }

    return {
      isValid: errors.length === 0,
      errors,
      warnings,
    };
  }

  /**
   * Normalizes MoSCoW results across a dataset of results.
   *
   * For MoSCoW, normalization considers:
   * 1. Category weight (Must have > Should have > Could have > Won't have)
   * 2. Priority within category
   * 3. Business value and risk scores as tiebreakers
   *
   * @param result - The MoSCoW result to normalize
   * @param allResults - All MoSCoW results in the dataset
   * @returns NormalizedResult with rank, percentile, and normalized score
   *
   * @example
   * ```typescript
   * const results = [
   *   calculator.calculate({ businessValue: 'critical', legalRequirement: true, customerRequest: false, riskIfNotDelivered: 'critical' }),
   *   calculator.calculate({ businessValue: 'medium', legalRequirement: false, customerRequest: true, riskIfNotDelivered: 'low' }),
   * ];
   *
   * const normalized = calculator.normalize(results[0], results);
   * // normalized.rank = 1
   * // normalized.percentile = 100
   * // normalized.normalizedScore = 100
   * ```
   */
  normalize(result: MoSCoWResult, allResults: MoSCoWResult[]): NormalizedResult {
    if (allResults.length === 0) {
      return {
        normalizedScore: 0,
        rank: 0,
        percentile: 0,
      };
    }

    // Calculate composite score for sorting
    // Category weight (40%) + business value score (30%) + risk score (30%)
    const getCompositeScore = (r: MoSCoWResult): number => {
      const categoryWeight = MOSCOW_CATEGORY_WEIGHTS[r.category];
      const businessScore = BUSINESS_VALUE_SCORES[r.businessValue];
      const riskScore = RISK_SCORES[r.riskIfNotDelivered];

      // Weighted composite (max score = 4)
      return categoryWeight * 0.4 + businessScore * 0.3 + riskScore * 0.3;
    };

    // Sort by composite score (descending), then by priority (ascending)
    const sortedResults = [...allResults].sort((a, b) => {
      const scoreA = getCompositeScore(a);
      const scoreB = getCompositeScore(b);
      if (scoreB !== scoreA) {
        return scoreB - scoreA;
      }
      return a.priority - b.priority;
    });

    // Find the rank of this result (1-indexed)
    const rank =
      sortedResults.findIndex(
        (r) =>
          r.category === result.category &&
          r.priority === result.priority &&
          r.businessValue === result.businessValue
      ) + 1;

    // Calculate percentile
    const itemsBelow = sortedResults.filter((r) => {
      const scoreR = getCompositeScore(r);
      const scoreResult = getCompositeScore(result);
      return scoreR < scoreResult;
    }).length;

    const percentile =
      allResults.length > 1
        ? parseFloat(((itemsBelow / (allResults.length - 1)) * 100).toFixed(this.decimalPlaces))
        : 100;

    // Calculate normalized score on 0-100 scale
    const minScore = Math.min(...allResults.map(getCompositeScore));
    const maxScore = Math.max(...allResults.map(getCompositeScore));
    const resultScore = getCompositeScore(result);

    let normalizedScore: number;
    if (maxScore === minScore) {
      normalizedScore = 50;
    } else {
      normalizedScore = parseFloat(
        (((resultScore - minScore) / (maxScore - minScore)) * 100).toFixed(this.decimalPlaces)
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
   * appropriate MoSCoW input values using keyword matching and heuristics.
   *
   * @param item - The prioritization item to analyze
   * @returns Partial<MoSCoWInput> with suggested values and confidence scores
   *
   * @example
   * ```typescript
   * const item = {
   *   id: 'req-1',
   *   title: 'GDPR compliance for user data',
   *   description: 'Implement data encryption and consent management',
   *   category: 'Compliance',
   *   createdAt: new Date(),
   * };
   *
   * const suggestions = calculator.getAutoFillSuggestions(item);
   * // {
   * //   businessValue: 'critical',
   * //   businessValueConfidence: 0.85,
   * //   legalRequirement: true,
   * //   legalRequirementConfidence: 0.95,
   * //   customerRequest: false,
   * //   customerRequestConfidence: 0.7,
   * //   riskIfNotDelivered: 'critical',
   * //   riskConfidence: 0.9
   * // }
   * ```
   */
  getAutoFillSuggestions(
    item: PrioritizationItem
  ): Partial<MoSCoWInput> & {
    businessValueConfidence?: number;
    legalRequirementConfidence?: number;
    customerRequestConfidence?: number;
    riskConfidence?: number;
  } {
    const text = `${item.title} ${item.description || ''} ${item.category || ''}`.toLowerCase();
    const suggestions: Partial<MoSCoWInput> & {
      businessValueConfidence?: number;
      legalRequirementConfidence?: number;
      customerRequestConfidence?: number;
      riskConfidence?: number;
    } = {};

    // Legal requirement detection
    const legalKeywords = [
      'gdpr',
      'compliance',
      'legal',
      'regulatory',
      'regulation',
      'law',
      'pci',
      'hipaa',
      'soc2',
      'iso',
      'accessibility',
      'wcag',
      'tax',
      'audit',
    ];

    const hasLegalKeywords = legalKeywords.some((k) => text.includes(k));
    if (hasLegalKeywords) {
      suggestions.legalRequirement = true;
      suggestions.legalRequirementConfidence = 0.9;
    } else if (item.category?.toLowerCase().includes('compliance')) {
      suggestions.legalRequirement = true;
      suggestions.legalRequirementConfidence = 0.75;
    } else {
      suggestions.legalRequirement = false;
      suggestions.legalRequirementConfidence = 0.6;
    }

    // Business value detection
    const criticalKeywords = [
      'critical',
      'blocker',
      'must',
      'required',
      'mandatory',
      'essential',
      'core',
      'fundamental',
      'break',
      'crash',
      'security',
    ];
    const highKeywords = [
      'important',
      'high priority',
      'key',
      'major',
      'significant',
      'revenue',
      'customer',
    ];
    const lowKeywords = [
      'nice to have',
      'optional',
      'low priority',
      'minor',
      'cosmetic',
      'enhancement',
    ];

    if (criticalKeywords.some((k) => text.includes(k))) {
      suggestions.businessValue = 'critical';
      suggestions.businessValueConfidence = 0.8;
    } else if (highKeywords.some((k) => text.includes(k))) {
      suggestions.businessValue = 'high';
      suggestions.businessValueConfidence = 0.7;
    } else if (lowKeywords.some((k) => text.includes(k))) {
      suggestions.businessValue = 'low';
      suggestions.businessValueConfidence = 0.65;
    } else {
      suggestions.businessValue = 'medium';
      suggestions.businessValueConfidence = 0.4;
    }

    // Customer request detection
    const customerKeywords = [
      'customer requested',
      'client request',
      'user request',
      'requested by',
      'vote',
      'feedback',
      'demand',
      'asked',
    ];

    if (customerKeywords.some((k) => text.includes(k))) {
      suggestions.customerRequest = true;
      suggestions.customerRequestConfidence = 0.75;
    } else if (item.metadata?.['customerRequested'] === true) {
      suggestions.customerRequest = true;
      suggestions.customerRequestConfidence = 0.95;
    } else {
      suggestions.customerRequest = false;
      suggestions.customerRequestConfidence = 0.5;
    }

    // Risk detection
    const criticalRiskKeywords = [
      'security breach',
      'data loss',
      'compliance',
      'legal',
      'reputation',
      'revenue loss',
    ];
    const highRiskKeywords = [
      'customer churn',
      'competitive disadvantage',
      'significant',
      'major impact',
    ];
    const lowRiskKeywords = ['minor inconvenience', 'workaround exists', 'low impact'];

    if (criticalRiskKeywords.some((k) => text.includes(k))) {
      suggestions.riskIfNotDelivered = 'critical';
      suggestions.riskConfidence = 0.8;
    } else if (highRiskKeywords.some((k) => text.includes(k))) {
      suggestions.riskIfNotDelivered = 'high';
      suggestions.riskConfidence = 0.7;
    } else if (lowRiskKeywords.some((k) => text.includes(k))) {
      suggestions.riskIfNotDelivered = 'low';
      suggestions.riskConfidence = 0.65;
    } else {
      // Default risk based on business value
      if (suggestions.businessValue === 'critical') {
        suggestions.riskIfNotDelivered = 'high';
      } else if (suggestions.businessValue === 'high') {
        suggestions.riskIfNotDelivered = 'medium';
      } else {
        suggestions.riskIfNotDelivered = 'medium';
      }
      suggestions.riskConfidence = 0.4;
    }

    return suggestions;
  }

  /**
   * Determines the MoSCoW category based on input parameters.
   *
   * Decision tree:
   * 1. Legal requirement -> Must have
   * 2. Critical business value -> Must have
   * 3. Critical risk -> Must have
   * 4. High business value AND high risk -> Should have
   * 5. High business value OR high risk -> Should have
   * 6. Medium business value -> Could have
   * 7. Low business value -> Won't have
   *
   * @param input - The validated MoSCoW input
   * @returns The assigned MoSCoW category
   * @private
   */
  private determineCategory(input: MoSCoWInput): MoSCoWCategory {
    // Rule 1: Legal requirements are always Must have
    if (input.legalRequirement) {
      return 'Must have';
    }

    // Rule 2: Critical business value -> Must have
    if (input.businessValue === 'critical') {
      return 'Must have';
    }

    // Rule 3: Critical risk -> Must have
    if (input.riskIfNotDelivered === 'critical') {
      return 'Must have';
    }

    // Rule 4: High business value AND high risk -> Should have
    if (
      input.businessValue === 'high' &&
      input.riskIfNotDelivered === 'high'
    ) {
      return 'Should have';
    }

    // Rule 5: High business value OR high risk (but not both) -> Should have
    if (
      input.businessValue === 'high' ||
      input.riskIfNotDelivered === 'high'
    ) {
      return 'Should have';
    }

    // Rule 6: Customer request with medium business value -> Could have (or Should have)
    if (
      input.customerRequest &&
      input.businessValue === 'medium' &&
      input.riskIfNotDelivered === 'medium'
    ) {
      return 'Should have';
    }

    // Rule 7: Medium business value -> Could have
    if (input.businessValue === 'medium') {
      return 'Could have';
    }

    // Rule 8: Low business value -> Won't have
    return "Won't have";
  }

  /**
   * Calculates initial priority within a category.
   *
   * Priority is based on a combination of business value score and risk score.
   * Lower numbers indicate higher priority within the category.
   *
   * @param input - The MoSCoW input parameters
   * @param _category - The assigned category (used for future extensions)
   * @returns Initial priority value (1 = highest within category)
   * @private
   */
  private calculateInitialPriority(
    input: MoSCoWInput,
    _category: MoSCoWCategory
  ): number {
    const businessScore = BUSINESS_VALUE_SCORES[input.businessValue];
    const riskScore = RISK_SCORES[input.riskIfNotDelivered];

    // Combined score (higher = more important within category)
    const combinedScore = businessScore + riskScore;

    // Add bonus for customer requests
    const customerBonus = input.customerRequest ? 1 : 0;

    // Priority: lower number = higher priority
    // Max combined score is 8 (4+4), so we invert
    return 9 - combinedScore - customerBonus;
  }

  /**
   * Returns the factors that influenced the category decision.
   *
   * @param input - The MoSCoW input parameters
   * @returns Array of decision factor descriptions
   * @private
   */
  private getDecisionFactors(input: MoSCoWInput): string[] {
    const factors: string[] = [];

    if (input.legalRequirement) {
      factors.push('Legal/compliance requirement');
    }

    factors.push(`Business value: ${input.businessValue}`);

    if (input.customerRequest) {
      factors.push('Customer requested');
    }

    factors.push(`Risk if not delivered: ${input.riskIfNotDelivered}`);

    return factors;
  }
}

/**
 * Factory function to create a new MoSCoW calculator instance.
 *
 * @param config - Optional configuration to override defaults
 * @returns A new MoSCoWCalculator instance
 *
 * @example
 * ```typescript
 * const calculator = createMoSCoWCalculator({
 *   maxMustHavePercentage: 50,
 *   decimalPlaces: 3
 * });
 * ```
 */
export function createMoSCoWCalculator(
  config: Partial<typeof DEFAULT_CONFIG> = {}
): MoSCoWCalculator {
  return new MoSCoWCalculator({ ...DEFAULT_CONFIG, ...config });
}

export default MoSCoWCalculator;
