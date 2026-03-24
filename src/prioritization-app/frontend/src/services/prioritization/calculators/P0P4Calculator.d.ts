/**
 * P0-P4 Priority Framework Calculator
 *
 * Implements the P0-P4 priority classification system commonly used in tech companies.
 * This framework classifies items based on severity and impact:
 *
 * Priority Levels:
 * - P0: Critical - must be fixed immediately (system down, data loss, security breach)
 * - P1: High - should be fixed soon (major functionality broken, significant user impact)
 * - P2: Medium - normal priority (standard bugs and features)
 * - P3: Low - can be deferred (minor issues, nice-to-have features)
 * - P4: Lowest - backlog or won't fix (trivial issues, future consideration)
 *
 * @see {@link https://en.wikipedia.org/wiki/Severity_(software_bug)}
 * @module prioritization/calculators/P0P4Calculator
 */
import { IFrameworkCalculator, PrioritizationItem, P0P4Input, P0P4Result, P0P4Level, ValidationResult, NormalizedResult, FrameworkType } from '../types';
/**
 * Default configuration for P0-P4 calculator.
 */
declare const DEFAULT_CONFIG: {
    /** P0 threshold (severity score >= this is P0) */
    readonly p0Threshold: 90;
    /** P1 threshold (severity score >= this is P1) */
    readonly p1Threshold: 70;
    /** P2 threshold (severity score >= this is P2) */
    readonly p2Threshold: 50;
    /** P3 threshold (severity score >= this is P3) */
    readonly p3Threshold: 30;
    /** Decimal places for rounding */
    readonly decimalPlaces: 2;
};
/**
 * Priority level weights for scoring and ranking.
 * Higher values indicate higher priority.
 */
export declare const PRIORITY_WEIGHTS: Record<P0P4Level, number>;
/**
 * Severity factor scores for each level.
 */
export declare const SEVERITY_FACTOR_SCORES: {
    readonly usersAffected: {
        readonly all: 5;
        readonly many: 4;
        readonly some: 3;
        readonly few: 2;
        readonly none: 1;
    };
    readonly coreFunctionalityImpact: {
        readonly critical: 5;
        readonly high: 4;
        readonly medium: 3;
        readonly low: 2;
        readonly none: 1;
    };
    readonly securityRisk: {
        readonly critical: 5;
        readonly high: 4;
        readonly medium: 3;
        readonly low: 2;
        readonly none: 1;
    };
    readonly reputationalRisk: {
        readonly critical: 5;
        readonly high: 4;
        readonly medium: 3;
        readonly low: 2;
        readonly none: 1;
    };
    readonly revenueImpact: {
        readonly critical: 5;
        readonly high: 4;
        readonly medium: 3;
        readonly low: 2;
        readonly none: 1;
    };
};
/**
 * Recommended timeframes for each priority level.
 */
export declare const PRIORITY_TIMEFRAMES: Record<P0P4Level, string>;
/**
 * P0-P4 Priority Framework Calculator implementation.
 *
 * Calculates priority based on:
 * 1. Base severity score (1-5 scale)
 * 2. Severity factors (users affected, functionality impact, security, etc.)
 * 3. Time-based modifiers (days open, open issues count)
 *
 * @example
 * ```typescript
 * const calculator = new P0P4Calculator();
 *
 * const input = {
 *   baseSeverity: 4,
 *   severityFactors: {
 *     usersAffected: 'many',
 *     coreFunctionalityImpact: 'high',
 *     securityRisk: 'medium',
 *     reputationalRisk: 'low',
 *     revenueImpact: 'high'
 *   },
 *   daysOpen: 5
 * };
 *
 * const result = calculator.calculate(input);
 * // result.priority = 'P1'
 * ```
 */
export declare class P0P4Calculator implements IFrameworkCalculator<P0P4Input, P0P4Result> {
    private readonly p0Threshold;
    private readonly p1Threshold;
    private readonly p2Threshold;
    private readonly p3Threshold;
    private readonly decimalPlaces;
    /**
     * Creates a new P0-P4 calculator instance.
     * @param config - Optional configuration to override defaults
     */
    constructor(config?: {
        /** P0 threshold (severity score >= this is P0) */
        readonly p0Threshold: 90;
        /** P1 threshold (severity score >= this is P1) */
        readonly p1Threshold: 70;
        /** P2 threshold (severity score >= this is P2) */
        readonly p2Threshold: 50;
        /** P3 threshold (severity score >= this is P3) */
        readonly p3Threshold: 30;
        /** Decimal places for rounding */
        readonly decimalPlaces: 2;
    });
    /**
     * Returns the framework type identifier.
     *
     * @returns 'P0P4' as the framework type
     */
    getFrameworkType(): FrameworkType;
    /**
     * Calculates the P0-P4 priority for given input parameters.
     *
     * @param input - The P0-P4 input parameters
     * @returns P0P4Result containing the assigned priority level
     * @throws {Error} If input validation fails
     *
     * @example
     * ```typescript
     * const result = calculator.calculate({
     *   baseSeverity: 5,
     *   severityFactors: {
     *     usersAffected: 'all',
     *     coreFunctionalityImpact: 'critical',
     *     securityRisk: 'critical',
     *     reputationalRisk: 'high',
     *     revenueImpact: 'critical'
     *   },
     *   openIssuesCount: 10,
     *   daysOpen: 3
     * });
     * // result.priority = 'P0'
     * ```
     */
    calculate(input: P0P4Input): P0P4Result;
    /**
     * Validates P0-P4 input parameters.
     *
     * @param input - Partial P0-P4 input parameters to validate
     * @returns ValidationResult with errors and warnings
     */
    validate(input: Partial<P0P4Input>): ValidationResult;
    /**
     * Normalizes P0-P4 results across a dataset.
     *
     * @param result - The P0-P4 result to normalize
     * @param allResults - All P0-P4 results in the dataset
     * @returns NormalizedResult with rank, percentile, and normalized score
     */
    normalize(result: P0P4Result, allResults: P0P4Result[]): NormalizedResult;
    /**
     * Generates auto-fill suggestions based on item metadata.
     *
     * @param item - The prioritization item to analyze
     * @returns Partial<P0P4Input> with suggested values and confidence scores
     */
    getAutoFillSuggestions(item: PrioritizationItem): Partial<P0P4Input> & {
        baseSeverityConfidence?: number;
        factorsConfidence?: number;
    };
    /**
     * Calculates the total severity score.
     * @private
     */
    private calculateSeverityScore;
    /**
     * Calculates the score from severity factors.
     * @private
     */
    private calculateFactorsScore;
    /**
     * Calculates time-based modifier.
     * @private
     */
    private calculateTimeModifier;
    /**
     * Determines priority level from severity score.
     * @private
     */
    private determinePriority;
}
/**
 * Factory function to create a new P0-P4 calculator instance.
 *
 * @param config - Optional configuration to override defaults
 * @returns A new P0P4Calculator instance
 */
export declare function createP0P4Calculator(config?: Partial<typeof DEFAULT_CONFIG>): P0P4Calculator;
export default P0P4Calculator;
//# sourceMappingURL=P0P4Calculator.d.ts.map