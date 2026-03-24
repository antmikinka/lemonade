/**
 * RICE Prioritization Framework Calculator
 *
 * Implements the RICE scoring model for prioritizing features and initiatives.
 * RICE stands for: Reach, Impact, Confidence, Effort
 *
 * Formula: RICE Score = (Reach × Impact × Confidence) / Effort
 *
 * @see {@link https://www.intercom.com/blog/intercom-how-to-build-prioritization-process/}
 * @module prioritization/calculators/RICECalculator
 */
import { IFrameworkCalculator, PrioritizationItem, RICEInput, RICEResult, ValidationResult, NormalizedResult, FrameworkType } from '../types';
/**
 * Standard impact scale values for RICE scoring.
 */
export declare const RICE_IMPACT_SCALE: {
    readonly MASSIVE: 3;
    readonly HIGH: 2;
    readonly MEDIUM: 1;
    readonly LOW: 0.5;
    readonly MINIMAL: 0.25;
};
/**
 * Impact level labels mapped to their numeric values.
 */
export declare const IMPACT_LEVELS: {
    readonly massive: 3;
    readonly high: 2;
    readonly medium: 1;
    readonly low: 0.5;
    readonly minimal: 0.25;
};
/**
 * Default configuration for RICE calculator.
 */
declare const DEFAULT_CONFIG: {
    minReach: number;
    maxReach: number;
    minImpact: number;
    maxImpact: number;
    minConfidence: number;
    maxConfidence: number;
    minEffort: number;
    maxEffort: number;
    decimalPlaces: number;
};
/**
 * RICE Framework Calculator implementation.
 *
 * Calculates RICE scores using the formula:
 * RICE = (Reach × Impact × Confidence) / Effort
 *
 * @example
 * ```typescript
 * const calculator = new RICECalculator();
 *
 * const input = {
 *   reach: 500,        // 500 users affected
 *   impact: 2,         // High impact
 *   confidence: 80,    // 80% confidence
 *   effort: 3          // 3 person-months
 * };
 *
 * const result = calculator.calculate(input);
 * // result.score = (500 × 2 × 0.8) / 3 = 266.67
 * ```
 */
export declare class RICECalculator implements IFrameworkCalculator<RICEInput, RICEResult> {
    private readonly minReach;
    private readonly maxReach;
    private readonly minImpact;
    private readonly maxImpact;
    private readonly minConfidence;
    private readonly maxConfidence;
    private readonly minEffort;
    private readonly maxEffort;
    private readonly decimalPlaces;
    /**
     * Creates a new RICE calculator instance.
     * @param config - Optional configuration to override defaults
     */
    constructor(config?: {
        minReach: number;
        maxReach: number;
        minImpact: number;
        maxImpact: number;
        minConfidence: number;
        maxConfidence: number;
        minEffort: number;
        maxEffort: number;
        decimalPlaces: number;
    });
    /**
     * Returns the framework type identifier.
     *
     * @returns 'RICE' as the framework type
     */
    getFrameworkType(): FrameworkType;
    /**
     * Calculates the RICE score for a given set of input parameters.
     *
     * Formula: RICE Score = (Reach × Impact × Confidence) / Effort
     *
     * Confidence is automatically normalized from percentage (0-100) to decimal (0-1)
     * if the value exceeds 1.
     *
     * @param input - The RICE input parameters
     * @returns RICEResult containing the calculated score and all input values
     * @throws {Error} If input validation fails
     *
     * @example
     * ```typescript
     * const result = calculator.calculate({
     *   reach: 1000,
     *   impact: 2,
     *   confidence: 75,
     *   effort: 4
     * });
     * // result.score = 375
     * ```
     */
    calculate(input: RICEInput): RICEResult;
    /**
     * Validates RICE input parameters against configured constraints.
     *
     * Checks:
     * - Reach must be a non-negative number
     * - Impact must be between 0.25 and 3 (standard scale)
     * - Confidence must be between 0 and 100 (percentage) or 0 and 1 (decimal)
     * - Effort must be a positive number (to avoid division by zero)
     *
     * @param input - Partial RICE input parameters to validate
     * @returns ValidationResult with errors and warnings
     *
     * @example
     * ```typescript
     * const validation = calculator.validate({
     *   reach: -5,      // Error: negative
     *   impact: 5,      // Error: exceeds max
     *   confidence: 50, // Valid
     *   effort: 0       // Error: must be positive
     * });
     * // validation.isValid = false
     * // validation.errors = ['Reach must be non-negative', ...]
     * ```
     */
    validate(input: Partial<RICEInput>): ValidationResult;
    /**
     * Normalizes a RICE result across a dataset of results.
     *
     * Normalization converts raw RICE scores to:
     * - normalizedScore: 0-100 scale based on position in dataset
     * - rank: Position when sorted by score (1 = highest)
     * - percentile: Percentage of items with lower scores
     *
     * @param result - The RICE result to normalize
     * @param allResults - All RICE results in the dataset
     * @returns NormalizedResult with rank, percentile, and normalized score
     *
     * @example
     * ```typescript
     * const results = [
     *   calculator.calculate({ reach: 100, impact: 2, confidence: 80, effort: 2 }),
     *   calculator.calculate({ reach: 200, impact: 3, confidence: 90, effort: 3 }),
     *   calculator.calculate({ reach: 50, impact: 1, confidence: 70, effort: 1 }),
     * ];
     *
     * const normalized = calculator.normalize(results[0], results);
     * // normalized.rank = 2
     * // normalized.percentile = 66.67
     * // normalized.normalizedScore = 66.67
     * ```
     */
    normalize(result: RICEResult, allResults: RICEResult[]): NormalizedResult;
    /**
     * Generates auto-fill suggestions based on item metadata.
     *
     * Analyzes the item's title, description, and category to suggest
     * reasonable RICE input values. Uses keyword matching and heuristics.
     *
     * Suggestions include confidence scores indicating the reliability
     * of each suggestion based on available information.
     *
     * @param item - The prioritization item to analyze
     * @returns Partial<RICEInput> with suggested values and confidence scores
     *
     * @example
     * ```typescript
     * const item = {
     *   id: 'feat-1',
     *   title: 'Add export to CSV for all users',
     *   description: 'High-demand feature requested by enterprise customers',
     *   category: 'Feature',
     *   createdAt: new Date(),
     * };
     *
     * const suggestions = calculator.getAutoFillSuggestions(item);
     * // {
     * //   reach: 1000,
     * //   reachConfidence: 0.7,
     * //   impact: 2,
     * //   impactConfidence: 0.8,
     * //   confidence: 70,
     * //   effort: 2,
     * //   effortConfidence: 0.5
     * // }
     * ```
     */
    getAutoFillSuggestions(item: PrioritizationItem): Partial<RICEInput> & {
        reachConfidence?: number;
        impactConfidence?: number;
        confidenceConfidence?: number;
        effortConfidence?: number;
    };
    /**
     * Normalizes confidence value to 0-1 range.
     *
     * If the value is greater than 1, it's treated as a percentage and divided by 100.
     *
     * @param confidence - The confidence value (either 0-1 or 0-100)
     * @returns Confidence as a decimal between 0 and 1
     * @private
     */
    private normalizeConfidence;
    /**
     * Returns the impact level label for a given impact value.
     *
     * @param impact - The numeric impact value
     * @returns The impact level label (e.g., 'high', 'medium')
     * @private
     */
    private getImpactLevel;
}
/**
 * Factory function to create a new RICE calculator instance.
 *
 * @param config - Optional configuration to override defaults
 * @returns A new RICECalculator instance
 *
 * @example
 * ```typescript
 * const calculator = createRICECalculator({
 *   decimalPlaces: 3,
 *   strictMode: true
 * });
 * ```
 */
export declare function createRICECalculator(config?: Partial<typeof DEFAULT_CONFIG>): RICECalculator;
export default RICECalculator;
//# sourceMappingURL=RICECalculator.d.ts.map