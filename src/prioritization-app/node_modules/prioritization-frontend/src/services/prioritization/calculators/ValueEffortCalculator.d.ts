/**
 * Value vs Effort Prioritization Framework Calculator
 *
 * Implements the Value vs Effort matrix for prioritizing features and initiatives.
 * This framework plots items on a 2x2 matrix based on their business value and implementation effort.
 *
 * Quadrants:
 * - QuickWin: High value, low effort - do these first
 * - MajorProject: High value, high effort - plan carefully
 * - FillIn: Low value, low effort - do when time permits
 * - Avoid: Low value, high effort - question these items
 *
 * @see {@link https://www.productplan.com/glossary/value-vs-effort-prioritization/}
 * @module prioritization/calculators/ValueEffortCalculator
 */
import { IFrameworkCalculator, PrioritizationItem, ValueEffortInput, ValueEffortResult, ValueEffortQuadrant, ValidationResult, NormalizedResult, FrameworkType } from '../types';
/**
 * Default configuration for Value vs Effort calculator.
 */
declare const DEFAULT_CONFIG: {
    /** Threshold for high value (scores >= this are "high") */
    readonly highValueThreshold: 6;
    /** Threshold for high effort (scores >= this are "high") */
    readonly highEffortThreshold: 6;
    /** Minimum score value */
    readonly minScore: 1;
    /** Maximum score value */
    readonly maxScore: 10;
    /** Decimal places for rounding */
    readonly decimalPlaces: 2;
};
/**
 * Quadrant weights for scoring and ranking.
 * Higher values indicate higher priority.
 */
export declare const QUADRANT_WEIGHTS: Record<ValueEffortQuadrant, number>;
/**
 * Value vs Effort Framework Calculator implementation.
 *
 * Calculates ROI (Return on Investment) as value/effort ratio
 * and assigns items to quadrants based on value and effort thresholds.
 *
 * @example
 * ```typescript
 * const calculator = new ValueEffortCalculator();
 *
 * const input = {
 *   value: 8,    // High business value
 *   effort: 3    // Low effort
 * };
 *
 * const result = calculator.calculate(input);
 * // result.quadrant = 'QuickWin'
 * // result.roi = 2.67
 * ```
 */
export declare class ValueEffortCalculator implements IFrameworkCalculator<ValueEffortInput, ValueEffortResult> {
    private readonly highValueThreshold;
    private readonly highEffortThreshold;
    private readonly minScore;
    private readonly maxScore;
    private readonly decimalPlaces;
    /**
     * Creates a new Value vs Effort calculator instance.
     * @param config - Optional configuration to override defaults
     */
    constructor(config?: {
        /** Threshold for high value (scores >= this are "high") */
        readonly highValueThreshold: 6;
        /** Threshold for high effort (scores >= this are "high") */
        readonly highEffortThreshold: 6;
        /** Minimum score value */
        readonly minScore: 1;
        /** Maximum score value */
        readonly maxScore: 10;
        /** Decimal places for rounding */
        readonly decimalPlaces: 2;
    });
    /**
     * Returns the framework type identifier.
     *
     * @returns 'ValueEffort' as the framework type
     */
    getFrameworkType(): FrameworkType;
    /**
     * Calculates the Value vs Effort analysis for given input parameters.
     *
     * Computes ROI as value/effort ratio and assigns quadrant based on thresholds.
     *
     * @param input - The Value vs Effort input parameters
     * @returns ValueEffortResult containing quadrant, ROI, and details
     * @throws {Error} If input validation fails
     *
     * @example
     * ```typescript
     * const result = calculator.calculate({
     *   value: 9,
     *   effort: 2
     * });
     * // result.quadrant = 'QuickWin'
     * // result.roi = 4.5
     * ```
     */
    calculate(input: ValueEffortInput): ValueEffortResult;
    /**
     * Validates Value vs Effort input parameters.
     *
     * Checks:
     * - Value must be a number within min/max range
     * - Effort must be a positive number within range
     *
     * @param input - Partial Value vs Effort input parameters to validate
     * @returns ValidationResult with errors and warnings
     *
     * @example
     * ```typescript
     * const validation = calculator.validate({
     *   value: 15,    // Error: exceeds max
     *   effort: 0     // Error: must be positive
     * });
     * // validation.isValid = false
     * ```
     */
    validate(input: Partial<ValueEffortInput>): ValidationResult;
    /**
     * Normalizes Value vs Effort results across a dataset.
     *
     * Normalization considers:
     * 1. ROI ratio (primary factor)
     * 2. Quadrant weight (secondary factor)
     * 3. Absolute value score (tiebreaker)
     *
     * @param result - The Value vs Effort result to normalize
     * @param allResults - All Value vs Effort results in the dataset
     * @returns NormalizedResult with rank, percentile, and normalized score
     *
     * @example
     * ```typescript
     * const results = [
     *   calculator.calculate({ value: 8, effort: 2 }),  // ROI = 4, QuickWin
     *   calculator.calculate({ value: 9, effort: 8 }),  // ROI = 1.125, MajorProject
     *   calculator.calculate({ value: 3, effort: 3 }),  // ROI = 1, FillIn
     * ];
     *
     * const normalized = calculator.normalize(results[0], results);
     * // normalized.rank = 1
     * // normalized.percentile = 100
     * // normalized.normalizedScore = 100
     * ```
     */
    normalize(result: ValueEffortResult, allResults: ValueEffortResult[]): NormalizedResult;
    /**
     * Generates auto-fill suggestions based on item metadata.
     *
     * Analyzes the item's title, description, and category to suggest
     * reasonable value and effort estimates using keyword matching.
     *
     * @param item - The prioritization item to analyze
     * @returns Partial<ValueEffortInput> with suggested values and confidence scores
     *
     * @example
     * ```typescript
     * const item = {
     *   id: 'feat-1',
     *   title: 'Add one-click export for enterprise customers',
     *   description: 'High-value feature requested by multiple enterprise clients',
     *   category: 'Feature',
     *   createdAt: new Date(),
     * };
     *
     * const suggestions = calculator.getAutoFillSuggestions(item);
     * // {
     * //   value: 8,
     * //   valueConfidence: 0.75,
     * //   effort: 3,
     * //   effortConfidence: 0.6
     * // }
     * ```
     */
    getAutoFillSuggestions(item: PrioritizationItem): Partial<ValueEffortInput> & {
        valueConfidence?: number;
        effortConfidence?: number;
    };
    /**
     * Determines the quadrant based on value and effort scores.
     *
     * @param value - The business value score
     * @param effort - The effort score
     * @returns The assigned quadrant
     * @private
     */
    private determineQuadrant;
}
/**
 * Factory function to create a new Value vs Effort calculator instance.
 *
 * @param config - Optional configuration to override defaults
 * @returns A new ValueEffortCalculator instance
 *
 * @example
 * ```typescript
 * const calculator = createValueEffortCalculator({
 *   highValueThreshold: 7,
 *   highEffortThreshold: 7,
 *   decimalPlaces: 3
 * });
 * ```
 */
export declare function createValueEffortCalculator(config?: Partial<typeof DEFAULT_CONFIG>): ValueEffortCalculator;
export default ValueEffortCalculator;
//# sourceMappingURL=ValueEffortCalculator.d.ts.map