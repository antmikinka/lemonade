/**
 * WSJF (Weighted Shortest Job First) Prioritization Framework Calculator
 *
 * Implements the WSJF scoring model from SAFe (Scaled Agile Framework) for prioritizing
 * features, capabilities, and epics based on economic principles.
 *
 * WSJF = Cost of Delay / Job Size
 *
 * Cost of Delay is composed of:
 * - User-business value: The value to the customer and business
 * - Time criticality: How urgent is the delivery
 * - Risk reduction and opportunity enablement: Risk mitigation and future opportunities
 *
 * Higher WSJF scores indicate higher priority - these items deliver maximum value
 * in minimum time.
 *
 * @see {@link https://www.scaledagileframework.com/wsjf/}
 * @module prioritization/calculators/WSJFCalculator
 */
import { IFrameworkCalculator, PrioritizationItem, WSJFInput, WSJFResult, ValidationResult, NormalizedResult, FrameworkType } from '../types';
/**
 * Default configuration for WSJF calculator.
 */
declare const DEFAULT_CONFIG: {
    /** Minimum score value (typically 1 in Fibonacci-like scales) */
    minScore: number;
    /** Maximum score value (typically 20 or 100 depending on scale) */
    maxScore: number;
    /** Decimal places for rounding */
    decimalPlaces: number;
    /** Minimum job size to prevent division by zero */
    minJobSize: number;
    /** Whether to use Fibonacci scale (true) or linear scale (false) */
    useFibonacciScale: boolean;
};
/**
 * Fibonacci sequence values for WSJF estimation.
 */
export declare const FIBONACCI_SCALE: readonly [1, 2, 3, 5, 8, 13, 20, 40, 100];
/**
 * WSJF Framework Calculator implementation.
 *
 * Calculates WSJF scores using the formula:
 * WSJF = (User-Business Value + Time Criticality + Risk Reduction/Opportunity) / Job Size
 *
 * @example
 * ```typescript
 * const calculator = new WSJFCalculator();
 *
 * const input = {
 *   userBusinessValue: 8,      // High business value
 *   timeCriticality: 6,        // Moderately time-sensitive
 *   riskReductionOpportunity: 5, // Some risk reduction
 *   jobSize: 3                 // Relatively small job
 * };
 *
 * const result = calculator.calculate(input);
 * // result.costOfDelay = 19
 * // result.score = 19 / 3 = 6.33
 * ```
 */
export declare class WSJFCalculator implements IFrameworkCalculator<WSJFInput, WSJFResult> {
    private readonly minScore;
    private readonly maxScore;
    private readonly decimalPlaces;
    private readonly minJobSize;
    private readonly useFibonacciScale;
    /**
     * Creates a new WSJF calculator instance.
     * @param config - Optional configuration to override defaults
     */
    constructor(config?: Partial<typeof DEFAULT_CONFIG>);
    /**
     * Returns the framework type identifier.
     *
     * @returns 'WSJF' as the framework type
     */
    getFrameworkType(): FrameworkType;
    /**
     * Calculates the WSJF score for given input parameters.
     *
     * Formula: WSJF = Cost of Delay / Job Size
     * Where: Cost of Delay = User-Business Value + Time Criticality + Risk Reduction/Opportunity
     *
     * @param input - The WSJF input parameters
     * @returns WSJFResult containing the calculated score and details
     * @throws {Error} If input validation fails
     *
     * @example
     * ```typescript
     * const result = calculator.calculate({
     *   userBusinessValue: 13,
     *   timeCriticality: 8,
     *   riskReductionOpportunity: 5,
     *   jobSize: 5
     * });
     * // result.costOfDelay = 26
     * // result.score = 5.2
     * ```
     */
    calculate(input: WSJFInput): WSJFResult;
    /**
     * Validates WSJF input parameters.
     *
     * Checks:
     * - All CoD components must be positive numbers within range
     * - Job size must be a positive number (to avoid division by zero)
     * - If using Fibonacci scale, values should match Fibonacci sequence
     *
     * @param input - Partial WSJF input parameters to validate
     * @returns ValidationResult with errors and warnings
     *
     * @example
     * ```typescript
     * const validation = calculator.validate({
     *   userBusinessValue: -5,    // Error: negative
     *   timeCriticality: 8,       // Valid
     *   riskReductionOpportunity: 5, // Valid
     *   jobSize: 0                // Error: must be positive
     * });
     * // validation.isValid = false
     * ```
     */
    validate(input: Partial<WSJFInput>): ValidationResult;
    /**
     * Normalizes WSJF results across a dataset.
     *
     * Normalization converts raw WSJF scores to:
     * - normalizedScore: 0-100 scale based on position in dataset
     * - rank: Position when sorted by score (1 = highest)
     * - percentile: Percentage of items with lower scores
     *
     * @param result - The WSJF result to normalize
     * @param allResults - All WSJF results in the dataset
     * @returns NormalizedResult with rank, percentile, and normalized score
     *
     * @example
     * ```typescript
     * const results = [
     *   calculator.calculate({ userBusinessValue: 8, timeCriticality: 6, riskReductionOpportunity: 5, jobSize: 3 }),
     *   calculator.calculate({ userBusinessValue: 13, timeCriticality: 8, riskReductionOpportunity: 5, jobSize: 5 }),
     *   calculator.calculate({ userBusinessValue: 5, timeCriticality: 3, riskReductionOpportunity: 3, jobSize: 8 }),
     * ];
     *
     * const normalized = calculator.normalize(results[0], results);
     * // normalized.rank = 1 or 2 (depending on scores)
     * // normalized.percentile = calculated percentile
     * // normalized.normalizedScore = 0-100 scale
     * ```
     */
    normalize(result: WSJFResult, allResults: WSJFResult[]): NormalizedResult;
    /**
     * Generates auto-fill suggestions based on item metadata.
     *
     * Analyzes the item's title, description, and category to suggest
     * reasonable WSJF input values using keyword matching and heuristics.
     *
     * Suggestions include confidence scores indicating the reliability
     * of each suggestion based on available information.
     *
     * @param item - The prioritization item to analyze
     * @returns Partial<WSJFInput> with suggested values and confidence scores
     *
     * @example
     * ```typescript
     * const item = {
     *   id: 'feat-1',
     *   title: 'Implement critical security patch for compliance deadline',
     *   description: 'Required for SOC2 compliance audit next month',
     *   category: 'Security',
     *   createdAt: new Date(),
     * };
     *
     * const suggestions = calculator.getAutoFillSuggestions(item);
     * // {
     * //   userBusinessValue: 13,
     * //   userBusinessValueConfidence: 0.85,
     * //   timeCriticality: 13,
     * //   timeCriticalityConfidence: 0.9,
     * //   riskReductionOpportunity: 8,
     * //   riskReductionOpportunityConfidence: 0.8,
     * //   jobSize: 3,
     * //   jobSizeConfidence: 0.5
     * // }
     * ```
     */
    getAutoFillSuggestions(item: PrioritizationItem): Partial<WSJFInput> & {
        userBusinessValueConfidence?: number;
        timeCriticalityConfidence?: number;
        riskReductionOpportunityConfidence?: number;
        jobSizeConfidence?: number;
    };
    /**
     * Returns an interpretation of the WSJF score.
     * @param score - The WSJF score to interpret
     * @returns A descriptive interpretation
     * @private
     */
    private getWSJFInterpretation;
    /**
     * Checks if a value is in the Fibonacci scale.
     * @param value - The value to check
     * @returns True if the value is a Fibonacci number
     * @private
     */
    private isValidFibonacci;
}
/**
 * Factory function to create a new WSJF calculator instance.
 *
 * @param config - Optional configuration to override defaults
 * @returns A new WSJFCalculator instance
 *
 * @example
 * ```typescript
 * const calculator = createWSJFCalculator({
 *   useFibonacciScale: true,
 *   decimalPlaces: 3
 * });
 * ```
 */
export declare function createWSJFCalculator(config?: Partial<typeof DEFAULT_CONFIG>): WSJFCalculator;
export default WSJFCalculator;
//# sourceMappingURL=WSJFCalculator.d.ts.map