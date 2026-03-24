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
/**
 * Standard impact scale values for RICE scoring.
 */
export const RICE_IMPACT_SCALE = {
    MASSIVE: 3,
    HIGH: 2,
    MEDIUM: 1,
    LOW: 0.5,
    MINIMAL: 0.25,
};
/**
 * Impact level labels mapped to their numeric values.
 */
export const IMPACT_LEVELS = {
    massive: 3,
    high: 2,
    medium: 1,
    low: 0.5,
    minimal: 0.25,
};
/**
 * Default configuration for RICE calculator.
 */
const DEFAULT_CONFIG = {
    minReach: 0,
    maxReach: Infinity,
    minImpact: 0.25,
    maxImpact: 3,
    minConfidence: 0,
    maxConfidence: 100,
    minEffort: 0.01, // Prevent division by zero
    maxEffort: Infinity,
    decimalPlaces: 2,
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
export class RICECalculator {
    /**
     * Creates a new RICE calculator instance.
     * @param config - Optional configuration to override defaults
     */
    constructor(config = DEFAULT_CONFIG) {
        Object.defineProperty(this, "minReach", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: void 0
        });
        Object.defineProperty(this, "maxReach", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: void 0
        });
        Object.defineProperty(this, "minImpact", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: void 0
        });
        Object.defineProperty(this, "maxImpact", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: void 0
        });
        Object.defineProperty(this, "minConfidence", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: void 0
        });
        Object.defineProperty(this, "maxConfidence", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: void 0
        });
        Object.defineProperty(this, "minEffort", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: void 0
        });
        Object.defineProperty(this, "maxEffort", {
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
        this.minReach = config.minReach ?? DEFAULT_CONFIG.minReach;
        this.maxReach = config.maxReach ?? DEFAULT_CONFIG.maxReach;
        this.minImpact = config.minImpact ?? DEFAULT_CONFIG.minImpact;
        this.maxImpact = config.maxImpact ?? DEFAULT_CONFIG.maxImpact;
        this.minConfidence = config.minConfidence ?? DEFAULT_CONFIG.minConfidence;
        this.maxConfidence = config.maxConfidence ?? DEFAULT_CONFIG.maxConfidence;
        this.minEffort = config.minEffort ?? DEFAULT_CONFIG.minEffort;
        this.maxEffort = config.maxEffort ?? DEFAULT_CONFIG.maxEffort;
        this.decimalPlaces = config.decimalPlaces ?? DEFAULT_CONFIG.decimalPlaces;
    }
    /**
     * Returns the framework type identifier.
     *
     * @returns 'RICE' as the framework type
     */
    getFrameworkType() {
        return 'RICE';
    }
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
    calculate(input) {
        // Validate input first
        const validation = this.validate(input);
        if (!validation.isValid) {
            throw new Error(`Invalid RICE input: ${validation.errors.join(', ')}`);
        }
        // Normalize confidence to 0-1 range if it's a percentage
        const normalizedConfidence = this.normalizeConfidence(input.confidence);
        // Calculate RICE score
        // Formula: (Reach × Impact × Confidence) / Effort
        const rawScore = (input.reach * input.impact * normalizedConfidence) / input.effort;
        // Round to configured decimal places
        const score = parseFloat(rawScore.toFixed(this.decimalPlaces));
        return {
            framework: 'RICE',
            reach: input.reach,
            impact: input.impact,
            confidence: normalizedConfidence,
            effort: input.effort,
            score,
            details: {
                rawScore,
                impactLevel: this.getImpactLevel(input.impact),
                confidencePercentage: normalizedConfidence * 100,
            },
        };
    }
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
    validate(input) {
        const errors = [];
        const warnings = [];
        // Validate reach
        if (input.reach !== undefined) {
            if (typeof input.reach !== 'number' || isNaN(input.reach)) {
                errors.push('Reach must be a valid number');
            }
            else if (input.reach < this.minReach) {
                errors.push(`Reach must be at least ${this.minReach}`);
            }
            else if (input.reach > this.maxReach) {
                errors.push(`Reach exceeds maximum of ${this.maxReach}`);
            }
            else if (input.reach === 0) {
                warnings.push('Reach of 0 means no users affected - consider if this item should be prioritized');
            }
        }
        // Validate impact
        if (input.impact !== undefined) {
            if (typeof input.impact !== 'number' || isNaN(input.impact)) {
                errors.push('Impact must be a valid number');
            }
            else if (input.impact < this.minImpact) {
                errors.push(`Impact must be at least ${this.minImpact} (minimal impact level)`);
            }
            else if (input.impact > this.maxImpact) {
                errors.push(`Impact must be at most ${this.maxImpact} (massive impact level)`);
            }
            else {
                // Check if impact matches standard scale values
                const standardValues = Object.values(IMPACT_LEVELS);
                if (!standardValues.includes(input.impact)) {
                    warnings.push(`Impact ${input.impact} doesn't match standard scale. Recommended values: ${standardValues.join(', ')}`);
                }
            }
        }
        // Validate confidence
        if (input.confidence !== undefined) {
            if (typeof input.confidence !== 'number' || isNaN(input.confidence)) {
                errors.push('Confidence must be a valid number');
            }
            else {
                if (input.confidence < this.minConfidence) {
                    errors.push(`Confidence must be at least ${this.minConfidence}`);
                }
                else if (input.confidence > this.maxConfidence) {
                    errors.push(`Confidence must be at most ${this.maxConfidence}`);
                }
                else if (input.confidence === 0) {
                    warnings.push('Zero confidence means you have no certainty in your estimates');
                }
                else if (input.confidence < 50 && input.confidence > 1) {
                    warnings.push(`Low confidence (${input.confidence}%) - consider gathering more data before prioritizing`);
                }
            }
        }
        // Validate effort
        if (input.effort !== undefined) {
            if (typeof input.effort !== 'number' || isNaN(input.effort)) {
                errors.push('Effort must be a valid number');
            }
            else if (input.effort < this.minEffort) {
                errors.push(`Effort must be at least ${this.minEffort} (to avoid division by zero)`);
            }
            else if (input.effort > this.maxEffort) {
                errors.push(`Effort exceeds maximum of ${this.maxEffort}`);
            }
            else if (input.effort < 0.1) {
                warnings.push('Very low effort value - ensure this is measured in the correct units (person-months)');
            }
            else if (input.effort > 12) {
                warnings.push('High effort value (>12 person-months) - consider breaking this into smaller initiatives');
            }
        }
        return {
            isValid: errors.length === 0,
            errors,
            warnings,
        };
    }
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
    normalize(result, allResults) {
        if (allResults.length === 0) {
            return {
                normalizedScore: 0,
                rank: 0,
                percentile: 0,
            };
        }
        // Sort results by score in descending order (highest first)
        const sortedResults = [...allResults].sort((a, b) => b.score - a.score);
        // Find the rank of this result (1-indexed)
        const rank = sortedResults.findIndex((r) => r.score === result.score) + 1;
        // Calculate percentile: percentage of items with lower scores
        const itemsBelow = sortedResults.filter((r) => r.score < result.score).length;
        const percentile = allResults.length > 1
            ? parseFloat(((itemsBelow / (allResults.length - 1)) * 100).toFixed(2))
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
            normalizedScore = parseFloat((((result.score - minScore) / (maxScore - minScore)) * 100).toFixed(2));
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
    getAutoFillSuggestions(item) {
        const text = `${item.title} ${item.description || ''} ${item.category || ''}`.toLowerCase();
        const suggestions = {};
        // Reach suggestions based on audience keywords
        if (text.includes('all users') || text.includes('everyone')) {
            suggestions.reach = 1000;
            suggestions.reachConfidence = 0.8;
        }
        else if (text.includes('enterprise') || text.includes('business')) {
            suggestions.reach = 100;
            suggestions.reachConfidence = 0.7;
        }
        else if (text.includes('admin') || text.includes('internal')) {
            suggestions.reach = 20;
            suggestions.reachConfidence = 0.75;
        }
        else if (text.includes('some users') || text.includes('select')) {
            suggestions.reach = 200;
            suggestions.reachConfidence = 0.6;
        }
        else {
            // Default moderate reach estimate
            suggestions.reach = 500;
            suggestions.reachConfidence = 0.4;
        }
        // Impact suggestions based on value keywords
        const highImpactKeywords = [
            'critical',
            'blocker',
            'revenue',
            'conversion',
            'retention',
            'core',
            'security',
            'compliance',
        ];
        const mediumImpactKeywords = [
            'important',
            'improve',
            'enhance',
            'optimize',
            'performance',
        ];
        const lowImpactKeywords = [
            'nice to have',
            'minor',
            'cosmetic',
            'optional',
            'quality of life',
        ];
        if (highImpactKeywords.some((k) => text.includes(k))) {
            suggestions.impact = IMPACT_LEVELS.high;
            suggestions.impactConfidence = 0.75;
        }
        else if (lowImpactKeywords.some((k) => text.includes(k))) {
            suggestions.impact = IMPACT_LEVELS.low;
            suggestions.impactConfidence = 0.7;
        }
        else if (mediumImpactKeywords.some((k) => text.includes(k))) {
            suggestions.impact = IMPACT_LEVELS.medium;
            suggestions.impactConfidence = 0.7;
        }
        else {
            suggestions.impact = IMPACT_LEVELS.medium;
            suggestions.impactConfidence = 0.4;
        }
        // Confidence suggestions based on specificity of description
        if (item.description && item.description.length > 100) {
            suggestions.confidence = 70;
            suggestions.confidenceConfidence = 0.7;
        }
        else if (item.description && item.description.length > 50) {
            suggestions.confidence = 50;
            suggestions.confidenceConfidence = 0.6;
        }
        else if (item.metadata && Object.keys(item.metadata).length > 0) {
            suggestions.confidence = 60;
            suggestions.confidenceConfidence = 0.65;
        }
        else {
            suggestions.confidence = 40;
            suggestions.confidenceConfidence = 0.5;
        }
        // Effort suggestions based on complexity keywords
        const highEffortKeywords = [
            'migration',
            'refactor',
            'infrastructure',
            'platform',
            'integration',
            'rewrite',
        ];
        const mediumEffortKeywords = [
            'feature',
            'implement',
            'develop',
            'module',
            'component',
        ];
        const lowEffortKeywords = [
            'fix',
            'update',
            'tweak',
            'adjust',
            'minor',
            'quick',
        ];
        if (highEffortKeywords.some((k) => text.includes(k))) {
            suggestions.effort = 4;
            suggestions.effortConfidence = 0.6;
        }
        else if (lowEffortKeywords.some((k) => text.includes(k))) {
            suggestions.effort = 0.5;
            suggestions.effortConfidence = 0.65;
        }
        else if (mediumEffortKeywords.some((k) => text.includes(k))) {
            suggestions.effort = 2;
            suggestions.effortConfidence = 0.55;
        }
        else {
            suggestions.effort = 1.5;
            suggestions.effortConfidence = 0.35;
        }
        return suggestions;
    }
    /**
     * Normalizes confidence value to 0-1 range.
     *
     * If the value is greater than 1, it's treated as a percentage and divided by 100.
     *
     * @param confidence - The confidence value (either 0-1 or 0-100)
     * @returns Confidence as a decimal between 0 and 1
     * @private
     */
    normalizeConfidence(confidence) {
        // If confidence > 1, assume it's a percentage
        if (confidence > 1) {
            return confidence / 100;
        }
        return confidence;
    }
    /**
     * Returns the impact level label for a given impact value.
     *
     * @param impact - The numeric impact value
     * @returns The impact level label (e.g., 'high', 'medium')
     * @private
     */
    getImpactLevel(impact) {
        for (const [level, value] of Object.entries(IMPACT_LEVELS)) {
            if (value === impact) {
                return level;
            }
        }
        // Find closest standard value
        const entries = Object.entries(IMPACT_LEVELS);
        const closest = entries.reduce((prev, curr) => Math.abs(curr[1] - impact) < Math.abs(prev[1] - impact) ? curr : prev);
        return closest[0];
    }
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
export function createRICECalculator(config = {}) {
    return new RICECalculator({ ...DEFAULT_CONFIG, ...config });
}
export default RICECalculator;
