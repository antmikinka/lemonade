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
import { IFrameworkCalculator, PrioritizationItem, MoSCoWInput, MoSCoWResult, MoSCoWCategory, BusinessValueLevel, RiskLevel, ValidationResult, NormalizedResult, FrameworkType } from '../types';
/**
 * Numeric values for business value levels.
 * Used for internal calculations and comparisons.
 */
export declare const BUSINESS_VALUE_SCORES: Record<BusinessValueLevel, number>;
/**
 * Numeric values for risk levels.
 * Used for internal calculations and comparisons.
 */
export declare const RISK_SCORES: Record<RiskLevel, number>;
/**
 * Priority weights for each MoSCoW category.
 * Higher values indicate higher priority.
 */
export declare const MOSCOW_CATEGORY_WEIGHTS: Record<MoSCoWCategory, number>;
/**
 * Default configuration for MoSCoW calculator.
 */
declare const DEFAULT_CONFIG: {
    /** Minimum items per category for balanced distribution */
    minMustHavePercentage: number;
    /** Maximum items that should be marked as Must Have */
    maxMustHavePercentage: number;
    /** Decimal places for normalized scores */
    decimalPlaces: number;
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
export declare class MoSCoWCalculator implements IFrameworkCalculator<MoSCoWInput, MoSCoWResult> {
    private readonly decimalPlaces;
    /**
     * Creates a new MoSCoW calculator instance.
     * @param config - Optional configuration to override defaults
     */
    constructor(config?: {
        /** Minimum items per category for balanced distribution */
        minMustHavePercentage: number;
        /** Maximum items that should be marked as Must Have */
        maxMustHavePercentage: number;
        /** Decimal places for normalized scores */
        decimalPlaces: number;
    });
    /**
     * Returns the framework type identifier.
     *
     * @returns 'MoSCoW' as the framework type
     */
    getFrameworkType(): FrameworkType;
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
    calculate(input: MoSCoWInput): MoSCoWResult;
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
    validate(input: Partial<MoSCoWInput>): ValidationResult;
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
    normalize(result: MoSCoWResult, allResults: MoSCoWResult[]): NormalizedResult;
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
    getAutoFillSuggestions(item: PrioritizationItem): Partial<MoSCoWInput> & {
        businessValueConfidence?: number;
        legalRequirementConfidence?: number;
        customerRequestConfidence?: number;
        riskConfidence?: number;
    };
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
    private determineCategory;
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
    private calculateInitialPriority;
    /**
     * Returns the factors that influenced the category decision.
     *
     * @param input - The MoSCoW input parameters
     * @returns Array of decision factor descriptions
     * @private
     */
    private getDecisionFactors;
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
export declare function createMoSCoWCalculator(config?: Partial<typeof DEFAULT_CONFIG>): MoSCoWCalculator;
export default MoSCoWCalculator;
//# sourceMappingURL=MoSCoWCalculator.d.ts.map