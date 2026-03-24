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
import { IFrameworkCalculator, PrioritizationItem, KanoInput, KanoResult, KanoCategory, ValidationResult, NormalizedResult, FrameworkType } from '../types';
/**
 * Response options for Kano questionnaire.
 * Scale: 1-5 mapping to customer feelings
 */
export declare const KANO_RESPONSE_SCALE: {
    readonly LIKE: 5;
    readonly EXPECT: 4;
    readonly NEUTRAL: 3;
    readonly LIVE_WITH: 2;
    readonly DISLIKE: 1;
};
/**
 * Kano evaluation matrix for determining category from responses.
 * Keys: [functionalScore, dysfunctionalScore]
 * Values: KanoCategory
 */
export declare const KANO_EVALUATION_MATRIX: Record<string, KanoCategory>;
/**
 * Numeric scores for each Kano category (for scoring calculations).
 */
export declare const KANO_CATEGORY_SCORES: Record<KanoCategory, number>;
/**
 * Default configuration for Kano calculator.
 */
declare const DEFAULT_CONFIG: {
    /** Minimum score value (1 = Dislike) */
    minScore: number;
    /** Maximum score value (5 = Like) */
    maxScore: number;
    /** Decimal places for rounding */
    decimalPlaces: number;
    /** Whether to include coefficient calculations */
    includeCoefficients: boolean;
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
export declare class KanoCalculator implements IFrameworkCalculator<KanoInput, KanoResult> {
    private readonly minScore;
    private readonly maxScore;
    private readonly decimalPlaces;
    private readonly includeCoefficients;
    /**
     * Creates a new Kano calculator instance.
     * @param config - Optional configuration to override defaults
     */
    constructor(config?: Partial<typeof DEFAULT_CONFIG>);
    /**
     * Returns the framework type identifier.
     *
     * @returns 'Kano' as the framework type
     */
    getFrameworkType(): FrameworkType;
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
    calculate(input: KanoInput): KanoResult;
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
    validate(input: Partial<KanoInput>): ValidationResult;
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
    normalize(result: KanoResult, allResults: KanoResult[]): NormalizedResult;
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
    };
    /**
     * Determines the Kano category from functional and dysfunctional scores.
     * @param functionalScore - How the customer feels if feature is present
     * @param dysfunctionalScore - How the customer feels if feature is absent
     * @returns The assigned Kano category
     * @private
     */
    private determineCategory;
    /**
     * Calculates the satisfaction coefficient.
     * Formula: (Like + Expect) - (Dislike) / Total responses
     * Simplified: functionalScore / maxScore - neutral point
     * @param functionalScore - The functional score
     * @returns Satisfaction coefficient (-1 to 1)
     * @private
     */
    private calculateSatisfactionCoefficient;
    /**
     * Calculates the dissatisfaction coefficient.
     * Formula: (Dislike + Live with) - (Like) / Total responses
     * Simplified: inverted dysfunctionalScore
     * @param dysfunctionalScore - The dysfunctional score
     * @returns Dissatisfaction coefficient (-1 to 1)
     * @private
     */
    private calculateDissatisfactionCoefficient;
    /**
     * Returns an interpretation for the given category.
     * @param category - The Kano category
     * @returns A descriptive interpretation
     * @private
     */
    private getCategoryInterpretation;
    /**
     * Returns a recommendation for the given category.
     * @param category - The Kano category
     * @returns A product recommendation
     * @private
     */
    private getCategoryRecommendation;
    /**
     * Checks if a response is at extreme (1 or 5).
     * @param input - The Kano input
     * @param type - 'functional' or 'dysfunctional'
     * @returns True if the response is extreme
     * @private
     */
    private isExtremeResponse;
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
export declare function createKanoCalculator(config?: Partial<typeof DEFAULT_CONFIG>): KanoCalculator;
export default KanoCalculator;
//# sourceMappingURL=KanoCalculator.d.ts.map