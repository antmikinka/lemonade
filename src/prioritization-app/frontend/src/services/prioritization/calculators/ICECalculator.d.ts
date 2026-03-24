/**
 * ICE Prioritization Framework Calculator
 *
 * Implements the ICE scoring model for prioritizing features and initiatives.
 * ICE stands for: Impact, Confidence, Ease
 *
 * Formula: ICE Score = Impact × Confidence × Ease
 *
 * ICE is a simplified version of RICE, focusing on:
 * - Impact: How much will this help?
 * - Confidence: How confident are we in our estimates?
 * - Ease: How easy is this to implement? (higher = easier)
 *
 * @see {@link https://growthmarketingmentor.com/ice-scoring-prioritization-framework/}
 * @module prioritization/calculators/ICECalculator
 */
import { IFrameworkCalculator, PrioritizationItem, ICEInput, ICEResult, ValidationResult, NormalizedResult, FrameworkType } from '../types';
/**
 * Default configuration for ICE calculator.
 */
declare const DEFAULT_CONFIG: {
    /** Minimum score value */
    readonly minScore: 1;
    /** Maximum score value */
    readonly maxScore: 10;
    /** Decimal places for rounding */
    readonly decimalPlaces: 2;
    /** Whether confidence is 0-1 (false) or 0-100 (true) */
    readonly confidenceAsPercentage: false;
};
/**
 * ICE Framework Calculator implementation.
 *
 * Calculates ICE scores using the formula:
 * ICE = Impact × Confidence × Ease
 *
 * @example
 * ```typescript
 * const calculator = new ICECalculator();
 *
 * const input = {
 *   impact: 8,       // High impact
 *   confidence: 0.7, // 70% confidence
 *   ease: 6          // Moderately easy
 * };
 *
 * const result = calculator.calculate(input);
 * // result.score = 8 × 0.7 × 6 = 33.6
 * ```
 */
export declare class ICECalculator implements IFrameworkCalculator<ICEInput, ICEResult> {
    private readonly minScore;
    private readonly maxScore;
    private readonly decimalPlaces;
    private readonly confidenceAsPercentage;
    /**
     * Creates a new ICE calculator instance.
     * @param config - Optional configuration to override defaults
     */
    constructor(config?: {
        /** Minimum score value */
        readonly minScore: 1;
        /** Maximum score value */
        readonly maxScore: 10;
        /** Decimal places for rounding */
        readonly decimalPlaces: 2;
        /** Whether confidence is 0-1 (false) or 0-100 (true) */
        readonly confidenceAsPercentage: false;
    });
    /**
     * Returns the framework type identifier.
     *
     * @returns 'ICE' as the framework type
     */
    getFrameworkType(): FrameworkType;
    /**
     * Calculates the ICE score for given input parameters.
     *
     * Formula: ICE Score = Impact × Confidence × Ease
     *
     * @param input - The ICE input parameters
     * @returns ICEResult containing the calculated score and details
     * @throws {Error} If input validation fails
     *
     * @example
     * ```typescript
     * const result = calculator.calculate({
     *   impact: 9,
     *   confidence: 80,
     *   ease: 7
     * });
     * // result.score = 504 (if confidence as percentage)
     * ```
     */
    calculate(input: ICEInput): ICEResult;
    /**
     * Validates ICE input parameters.
     *
     * Checks:
     * - Impact must be a number within range
     * - Confidence must be between 0-1 or 0-100
     * - Ease must be a positive number within range
     *
     * @param input - Partial ICE input parameters to validate
     * @returns ValidationResult with errors and warnings
     */
    validate(input: Partial<ICEInput>): ValidationResult;
    /**
     * Normalizes ICE results across a dataset.
     *
     * @param result - The ICE result to normalize
     * @param allResults - All ICE results in the dataset
     * @returns NormalizedResult with rank, percentile, and normalized score
     */
    normalize(result: ICEResult, allResults: ICEResult[]): NormalizedResult;
    /**
     * Generates auto-fill suggestions based on item metadata.
     *
     * @param item - The prioritization item to analyze
     * @returns Partial<ICEInput> with suggested values and confidence scores
     */
    getAutoFillSuggestions(item: PrioritizationItem): Partial<ICEInput> & {
        impactConfidence?: number;
        confidenceConfidence?: number;
        easeConfidence?: number;
    };
    /**
     * Normalizes confidence value to 0-1 range.
     * @private
     */
    private normalizeConfidence;
    /**
     * Returns the impact level label for a given impact value.
     * @private
     */
    private getImpactLevel;
    /**
     * Returns the ease level label for a given ease value.
     * @private
     */
    private getEaseLevel;
}
/**
 * Factory function to create a new ICE calculator instance.
 *
 * @param config - Optional configuration to override defaults
 * @returns A new ICECalculator instance
 */
export declare function createICECalculator(config?: Partial<typeof DEFAULT_CONFIG>): ICECalculator;
export default ICECalculator;
//# sourceMappingURL=ICECalculator.d.ts.map