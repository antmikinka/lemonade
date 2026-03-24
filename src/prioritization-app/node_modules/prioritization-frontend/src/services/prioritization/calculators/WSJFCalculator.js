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
/**
 * Default configuration for WSJF calculator.
 */
const DEFAULT_CONFIG = {
    /** Minimum score value (typically 1 in Fibonacci-like scales) */
    minScore: 1,
    /** Maximum score value (typically 20 or 100 depending on scale) */
    maxScore: 100,
    /** Decimal places for rounding */
    decimalPlaces: 2,
    /** Minimum job size to prevent division by zero */
    minJobSize: 0.1,
    /** Whether to use Fibonacci scale (true) or linear scale (false) */
    useFibonacciScale: false,
};
/**
 * Fibonacci sequence values for WSJF estimation.
 */
export const FIBONACCI_SCALE = [1, 2, 3, 5, 8, 13, 20, 40, 100];
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
export class WSJFCalculator {
    /**
     * Creates a new WSJF calculator instance.
     * @param config - Optional configuration to override defaults
     */
    constructor(config = DEFAULT_CONFIG) {
        Object.defineProperty(this, "minScore", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: void 0
        });
        Object.defineProperty(this, "maxScore", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: void 0
        });
        Object.defineProperty(this, "decimalPlaces", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: void 0
        });
        Object.defineProperty(this, "minJobSize", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: void 0
        });
        Object.defineProperty(this, "useFibonacciScale", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: void 0
        });
        this.minScore = config.minScore ?? DEFAULT_CONFIG.minScore;
        this.maxScore = config.maxScore ?? DEFAULT_CONFIG.maxScore;
        this.decimalPlaces = config.decimalPlaces ?? DEFAULT_CONFIG.decimalPlaces;
        this.minJobSize = config.minJobSize ?? DEFAULT_CONFIG.minJobSize;
        this.useFibonacciScale = config.useFibonacciScale ?? DEFAULT_CONFIG.useFibonacciScale;
    }
    /**
     * Returns the framework type identifier.
     *
     * @returns 'WSJF' as the framework type
     */
    getFrameworkType() {
        return 'WSJF';
    }
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
    calculate(input) {
        // Validate input first
        const validation = this.validate(input);
        if (!validation.isValid) {
            throw new Error(`Invalid WSJF input: ${validation.errors.join(', ')}`);
        }
        // Ensure job size is at least the minimum to prevent division issues
        const effectiveJobSize = Math.max(input.jobSize, this.minJobSize);
        // Calculate Cost of Delay (CoD)
        const costOfDelay = input.userBusinessValue + input.timeCriticality + input.riskReductionOpportunity;
        // Calculate WSJF score
        const rawScore = costOfDelay / effectiveJobSize;
        const score = parseFloat(rawScore.toFixed(this.decimalPlaces));
        return {
            framework: 'WSJF',
            costOfDelay,
            jobSize: input.jobSize,
            score,
            userBusinessValue: input.userBusinessValue,
            timeCriticality: input.timeCriticality,
            riskReductionOpportunity: input.riskReductionOpportunity,
            details: {
                codComponents: {
                    userBusinessValue: input.userBusinessValue,
                    timeCriticality: input.timeCriticality,
                    riskReductionOpportunity: input.riskReductionOpportunity,
                },
                normalizedJobSize: effectiveJobSize,
                wsjfInterpretation: this.getWSJFInterpretation(score),
            },
        };
    }
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
    validate(input) {
        const errors = [];
        const warnings = [];
        // Validate userBusinessValue
        if (input.userBusinessValue !== undefined) {
            if (typeof input.userBusinessValue !== 'number' || isNaN(input.userBusinessValue)) {
                errors.push('User-business value must be a valid number');
            }
            else if (input.userBusinessValue < this.minScore) {
                errors.push(`User-business value must be at least ${this.minScore}`);
            }
            else if (input.userBusinessValue > this.maxScore) {
                errors.push(`User-business value must be at most ${this.maxScore}`);
            }
            else if (this.useFibonacciScale && !this.isValidFibonacci(input.userBusinessValue)) {
                warnings.push(`User-business value ${input.userBusinessValue} is not a Fibonacci number. Recommended: ${FIBONACCI_SCALE.join(', ')}`);
            }
            else if (input.userBusinessValue < 3) {
                warnings.push('Low business value - consider if this item should be prioritized');
            }
        }
        // Validate timeCriticality
        if (input.timeCriticality !== undefined) {
            if (typeof input.timeCriticality !== 'number' || isNaN(input.timeCriticality)) {
                errors.push('Time criticality must be a valid number');
            }
            else if (input.timeCriticality < this.minScore) {
                errors.push(`Time criticality must be at least ${this.minScore}`);
            }
            else if (input.timeCriticality > this.maxScore) {
                errors.push(`Time criticality must be at most ${this.maxScore}`);
            }
            else if (this.useFibonacciScale && !this.isValidFibonacci(input.timeCriticality)) {
                warnings.push(`Time criticality ${input.timeCriticality} is not a Fibonacci number. Recommended: ${FIBONACCI_SCALE.join(', ')}`);
            }
            else if (input.timeCriticality < 3) {
                warnings.push('Low time criticality - ensure deadline is realistic');
            }
        }
        // Validate riskReductionOpportunity
        if (input.riskReductionOpportunity !== undefined) {
            if (typeof input.riskReductionOpportunity !== 'number' || isNaN(input.riskReductionOpportunity)) {
                errors.push('Risk reduction/opportunity enablement must be a valid number');
            }
            else if (input.riskReductionOpportunity < this.minScore) {
                errors.push(`Risk reduction/opportunity must be at least ${this.minScore}`);
            }
            else if (input.riskReductionOpportunity > this.maxScore) {
                errors.push(`Risk reduction/opportunity must be at most ${this.maxScore}`);
            }
            else if (this.useFibonacciScale && !this.isValidFibonacci(input.riskReductionOpportunity)) {
                warnings.push(`Risk reduction ${input.riskReductionOpportunity} is not a Fibonacci number. Recommended: ${FIBONACCI_SCALE.join(', ')}`);
            }
            else if (input.riskReductionOpportunity < 3) {
                warnings.push('Low risk reduction - ensure risks are properly assessed');
            }
        }
        // Validate jobSize
        if (input.jobSize !== undefined) {
            if (typeof input.jobSize !== 'number' || isNaN(input.jobSize)) {
                errors.push('Job size must be a valid number');
            }
            else if (input.jobSize < this.minJobSize) {
                errors.push(`Job size must be at least ${this.minJobSize} (to avoid division by zero)`);
            }
            else if (input.jobSize > this.maxScore) {
                errors.push(`Job size must be at most ${this.maxScore}`);
            }
            else if (this.useFibonacciScale && !this.isValidFibonacci(input.jobSize)) {
                warnings.push(`Job size ${input.jobSize} is not a Fibonacci number. Recommended: ${FIBONACCI_SCALE.join(', ')}`);
            }
            else if (input.jobSize < 2) {
                warnings.push('Very small job size - ensure estimation is realistic');
            }
            else if (input.jobSize > 40) {
                warnings.push('Large job size - consider breaking this into smaller stories');
            }
        }
        // Warning: Very high CoD with very small job size might indicate estimation issues
        if (input.userBusinessValue !== undefined &&
            input.timeCriticality !== undefined &&
            input.riskReductionOpportunity !== undefined &&
            input.jobSize !== undefined) {
            const totalCoD = input.userBusinessValue + input.timeCriticality + input.riskReductionOpportunity;
            if (totalCoD >= 60 && input.jobSize <= 3) {
                warnings.push('Very high CoD with very small job size - review estimates for accuracy');
            }
        }
        // Warning: Low WSJF ratio indication
        if (input.userBusinessValue !== undefined &&
            input.timeCriticality !== undefined &&
            input.riskReductionOpportunity !== undefined &&
            input.jobSize !== undefined) {
            const potentialWSJF = (input.userBusinessValue + input.timeCriticality + input.riskReductionOpportunity) / Math.max(input.jobSize, this.minJobSize);
            if (potentialWSJF < 1) {
                warnings.push(`Low WSJF score (${potentialWSJF.toFixed(2)}) - this item may have lower priority`);
            }
        }
        return {
            isValid: errors.length === 0,
            errors,
            warnings,
        };
    }
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
    normalize(result, allResults) {
        if (allResults.length === 0) {
            return {
                normalizedScore: 0,
                rank: 0,
                percentile: 0,
            };
        }
        // Sort results by WSJF score in descending order (highest first)
        const sortedResults = [...allResults].sort((a, b) => b.score - a.score);
        // Find the rank of this result (1-indexed)
        const rank = sortedResults.findIndex((r) => r.score === result.score) + 1;
        // Calculate percentile: percentage of items with lower scores
        const itemsBelow = sortedResults.filter((r) => r.score < result.score).length;
        const percentile = allResults.length > 1
            ? parseFloat(((itemsBelow / (allResults.length - 1)) * 100).toFixed(this.decimalPlaces))
            : 100;
        // Calculate normalized score on 0-100 scale
        const minScore = Math.min(...allResults.map((r) => r.score));
        const maxScore = Math.max(...allResults.map((r) => r.score));
        let normalizedScore;
        if (maxScore === minScore) {
            // All scores are the same
            normalizedScore = 50;
        }
        else {
            // Linear interpolation to 0-100 scale
            normalizedScore = parseFloat((((result.score - minScore) / (maxScore - minScore)) * 100).toFixed(this.decimalPlaces));
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
    getAutoFillSuggestions(item) {
        const text = `${item.title} ${item.description || ''} ${item.category || ''}`.toLowerCase();
        const suggestions = {};
        // User-Business Value suggestions
        const highValueKeywords = [
            'revenue',
            'profit',
            'conversion',
            'retention',
            'strategic',
            'core',
            'competitive advantage',
            'market',
            'customer acquisition',
            'enterprise',
            'key feature',
        ];
        const lowValueKeywords = [
            'nice to have',
            'optional',
            'cosmetic',
            'minor',
            'polish',
            'internal tool',
            'quality of life',
        ];
        if (highValueKeywords.some((k) => text.includes(k))) {
            suggestions.userBusinessValue = this.useFibonacciScale ? 8 : 8;
            suggestions.userBusinessValueConfidence = 0.75;
        }
        else if (lowValueKeywords.some((k) => text.includes(k))) {
            suggestions.userBusinessValue = this.useFibonacciScale ? 3 : 3;
            suggestions.userBusinessValueConfidence = 0.7;
        }
        else {
            suggestions.userBusinessValue = this.useFibonacciScale ? 5 : 5;
            suggestions.userBusinessValueConfidence = 0.5;
        }
        // Time Criticality suggestions
        const urgentKeywords = [
            'deadline',
            'asap',
            'urgent',
            'critical',
            'rush',
            'immediate',
            'this sprint',
            'this week',
            'this month',
            'quarter end',
            'fiscal year',
            'compliance',
            'regulatory',
            'audit',
            'contract',
            'sla',
        ];
        const timeSensitiveKeywords = [
            'time-sensitive',
            'window of opportunity',
            'first to market',
            'seasonal',
            'launch',
            'release',
            'milestone',
        ];
        if (urgentKeywords.some((k) => text.includes(k))) {
            suggestions.timeCriticality = this.useFibonacciScale ? 13 : 9;
            suggestions.timeCriticalityConfidence = 0.85;
        }
        else if (timeSensitiveKeywords.some((k) => text.includes(k))) {
            suggestions.timeCriticality = this.useFibonacciScale ? 8 : 7;
            suggestions.timeCriticalityConfidence = 0.75;
        }
        else {
            suggestions.timeCriticality = this.useFibonacciScale ? 5 : 5;
            suggestions.timeCriticalityConfidence = 0.5;
        }
        // Risk Reduction / Opportunity Enablement suggestions
        const highRiskKeywords = [
            'security',
            'risk',
            'compliance',
            'legal',
            'gdpr',
            'hipaa',
            'soc2',
            'pci',
            'vulnerability',
            'breach',
            'enable',
            'foundation',
            'platform',
            'infrastructure',
            'unblock',
            'dependency',
        ];
        const mediumRiskKeywords = [
            'refactor',
            'technical debt',
            'stability',
            'reliability',
            'scalability',
            'maintenance',
            'future-proof',
        ];
        if (highRiskKeywords.some((k) => text.includes(k))) {
            suggestions.riskReductionOpportunity = this.useFibonacciScale ? 8 : 7;
            suggestions.riskReductionOpportunityConfidence = 0.8;
        }
        else if (mediumRiskKeywords.some((k) => text.includes(k))) {
            suggestions.riskReductionOpportunity = this.useFibonacciScale ? 5 : 5;
            suggestions.riskReductionOpportunityConfidence = 0.7;
        }
        else {
            suggestions.riskReductionOpportunity = this.useFibonacciScale ? 3 : 3;
            suggestions.riskReductionOpportunityConfidence = 0.5;
        }
        // Job Size suggestions based on complexity keywords
        const largeJobKeywords = [
            'migration',
            'rewrite',
            'refactor',
            'platform',
            'infrastructure',
            'system',
            'architecture',
            'major',
            'overhaul',
            'complex',
            'enterprise',
            'integration',
        ];
        const smallJobKeywords = [
            'fix',
            'update',
            'tweak',
            'adjust',
            'minor',
            'quick',
            'simple',
            'small',
            'config',
            'bug',
            'patch',
            'hotfix',
        ];
        if (largeJobKeywords.some((k) => text.includes(k))) {
            suggestions.jobSize = this.useFibonacciScale ? 13 : 8;
            suggestions.jobSizeConfidence = 0.6;
        }
        else if (smallJobKeywords.some((k) => text.includes(k))) {
            suggestions.jobSize = this.useFibonacciScale ? 3 : 2;
            suggestions.jobSizeConfidence = 0.65;
        }
        else {
            suggestions.jobSize = this.useFibonacciScale ? 5 : 5;
            suggestions.jobSizeConfidence = 0.45;
        }
        return suggestions;
    }
    /**
     * Returns an interpretation of the WSJF score.
     * @param score - The WSJF score to interpret
     * @returns A descriptive interpretation
     * @private
     */
    getWSJFInterpretation(score) {
        if (score >= 10) {
            return 'Exceptional priority - deliver as soon as possible';
        }
        else if (score >= 5) {
            return 'High priority - schedule for next available capacity';
        }
        else if (score >= 2) {
            return 'Medium priority - include in regular planning';
        }
        else if (score >= 1) {
            return 'Lower priority - consider for backlog refinement';
        }
        else {
            return 'Lowest priority - evaluate for deferral or rejection';
        }
    }
    /**
     * Checks if a value is in the Fibonacci scale.
     * @param value - The value to check
     * @returns True if the value is a Fibonacci number
     * @private
     */
    isValidFibonacci(value) {
        return FIBONACCI_SCALE.includes(value);
    }
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
export function createWSJFCalculator(config = {}) {
    return new WSJFCalculator({ ...DEFAULT_CONFIG, ...config });
}
export default WSJFCalculator;
