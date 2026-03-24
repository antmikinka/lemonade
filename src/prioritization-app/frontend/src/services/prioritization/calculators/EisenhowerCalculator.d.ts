/**
 * Eisenhower Matrix Prioritization Framework Calculator
 *
 * Implements the Eisenhower Matrix (Urgent-Important Matrix) for task prioritization.
 * This framework categorizes tasks based on two dimensions:
 * - Urgency: How time-sensitive is this task?
 * - Importance: How much impact does this task have?
 *
 * Quadrants:
 * - DoFirst: Urgent and important - do immediately
 * - Schedule: Not urgent but important - plan to do
 * - Delegate: Urgent but not important - delegate if possible
 * - Eliminate: Not urgent and not important - consider eliminating
 *
 * @see {@link https://www.eisenhower.me/eisenhower-matrix/}
 * @module prioritization/calculators/EisenhowerCalculator
 */
import { IFrameworkCalculator, PrioritizationItem, EisenhowerInput, EisenhowerResult, EisenhowerQuadrant, ValidationResult, NormalizedResult, FrameworkType } from '../types';
/**
 * Default configuration for Eisenhower Matrix calculator.
 */
declare const DEFAULT_CONFIG: {
    /** Threshold for considering something urgent (when using urgencyLevel) */
    readonly urgencyThreshold: 6;
    /** Threshold for considering something important (when using importanceLevel) */
    readonly importanceThreshold: 6;
    /** Minimum level value */
    readonly minLevel: 1;
    /** Maximum level value */
    readonly maxLevel: 10;
    /** Decimal places for rounding */
    readonly decimalPlaces: 2;
};
/**
 * Quadrant weights for scoring and ranking.
 * Higher values indicate higher priority.
 */
export declare const EISENHOWER_QUADRANT_WEIGHTS: Record<EisenhowerQuadrant, number>;
/**
 * Eisenhower Matrix Framework Calculator implementation.
 *
 * Categorizes tasks into four quadrants based on urgency and importance.
 *
 * @example
 * ```typescript
 * const calculator = new EisenhowerCalculator();
 *
 * const input = {
 *   urgent: true,
 *   important: true
 * };
 *
 * const result = calculator.calculate(input);
 * // result.quadrant = 'DoFirst'
 * ```
 */
export declare class EisenhowerCalculator implements IFrameworkCalculator<EisenhowerInput, EisenhowerResult> {
    private readonly urgencyThreshold;
    private readonly importanceThreshold;
    private readonly minLevel;
    private readonly maxLevel;
    private readonly decimalPlaces;
    /**
     * Creates a new Eisenhower Matrix calculator instance.
     * @param config - Optional configuration to override defaults
     */
    constructor(config?: {
        /** Threshold for considering something urgent (when using urgencyLevel) */
        readonly urgencyThreshold: 6;
        /** Threshold for considering something important (when using importanceLevel) */
        readonly importanceThreshold: 6;
        /** Minimum level value */
        readonly minLevel: 1;
        /** Maximum level value */
        readonly maxLevel: 10;
        /** Decimal places for rounding */
        readonly decimalPlaces: 2;
    });
    /**
     * Returns the framework type identifier.
     *
     * @returns 'Eisenhower' as the framework type
     */
    getFrameworkType(): FrameworkType;
    /**
     * Calculates the Eisenhower Matrix quadrant for given input parameters.
     *
     * @param input - The Eisenhower input parameters
     * @returns EisenhowerResult containing the assigned quadrant
     * @throws {Error} If input validation fails
     *
     * @example
     * ```typescript
     * const result = calculator.calculate({
     *   urgent: true,
     *   important: true,
     *   urgencyLevel: 8,
     *   importanceLevel: 9
     * });
     * // result.quadrant = 'DoFirst'
     * ```
     */
    calculate(input: EisenhowerInput): EisenhowerResult;
    /**
     * Validates Eisenhower input parameters.
     *
     * @param input - Partial Eisenhower input parameters to validate
     * @returns ValidationResult with errors and warnings
     */
    validate(input: Partial<EisenhowerInput>): ValidationResult;
    /**
     * Normalizes Eisenhower results across a dataset.
     *
     * @param result - The Eisenhower result to normalize
     * @param allResults - All Eisenhower results in the dataset
     * @returns NormalizedResult with rank, percentile, and normalized score
     */
    normalize(result: EisenhowerResult, allResults: EisenhowerResult[]): NormalizedResult;
    /**
     * Generates auto-fill suggestions based on item metadata.
     *
     * @param item - The prioritization item to analyze
     * @returns Partial<EisenhowerInput> with suggested values and confidence scores
     */
    getAutoFillSuggestions(item: PrioritizationItem): Partial<EisenhowerInput> & {
        urgentConfidence?: number;
        importantConfidence?: number;
        urgencyLevelConfidence?: number;
        importanceLevelConfidence?: number;
    };
    /**
     * Determines the quadrant based on urgency and importance.
     * @private
     */
    private determineQuadrant;
    /**
     * Calculates a priority score based on urgency and importance levels.
     * @private
     */
    private calculatePriorityScore;
    /**
     * Returns the action verb for a given quadrant.
     * @private
     */
    private getActionVerb;
}
/**
 * Factory function to create a new Eisenhower Matrix calculator instance.
 *
 * @param config - Optional configuration to override defaults
 * @returns A new EisenhowerCalculator instance
 */
export declare function createEisenhowerCalculator(config?: Partial<typeof DEFAULT_CONFIG>): EisenhowerCalculator;
export default EisenhowerCalculator;
//# sourceMappingURL=EisenhowerCalculator.d.ts.map