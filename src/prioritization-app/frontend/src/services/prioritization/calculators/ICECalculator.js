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
/**
 * Default configuration for ICE calculator.
 */
const DEFAULT_CONFIG = {
    /** Minimum score value */
    minScore: 1,
    /** Maximum score value */
    maxScore: 10,
    /** Decimal places for rounding */
    decimalPlaces: 2,
    /** Whether confidence is 0-1 (false) or 0-100 (true) */
    confidenceAsPercentage: false,
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
export class ICECalculator {
    /**
     * Creates a new ICE calculator instance.
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
        Object.defineProperty(this, "confidenceAsPercentage", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: void 0
        });
        this.minScore = config.minScore ?? DEFAULT_CONFIG.minScore;
        this.maxScore = config.maxScore ?? DEFAULT_CONFIG.maxScore;
        this.decimalPlaces = config.decimalPlaces ?? DEFAULT_CONFIG.decimalPlaces;
        this.confidenceAsPercentage = config.confidenceAsPercentage ?? DEFAULT_CONFIG.confidenceAsPercentage;
    }
    /**
     * Returns the framework type identifier.
     *
     * @returns 'ICE' as the framework type
     */
    getFrameworkType() {
        return 'ICE';
    }
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
    calculate(input) {
        // Validate input first
        const validation = this.validate(input);
        if (!validation.isValid) {
            throw new Error(`Invalid ICE input: ${validation.errors.join(', ')}`);
        }
        // Normalize confidence to 0-1 range if needed
        const normalizedConfidence = this.normalizeConfidence(input.confidence);
        // Calculate ICE score
        const rawScore = input.impact * normalizedConfidence * input.ease;
        const score = parseFloat(rawScore.toFixed(this.decimalPlaces));
        return {
            framework: 'ICE',
            impact: input.impact,
            confidence: normalizedConfidence,
            ease: input.ease,
            score,
            details: {
                impactLevel: this.getImpactLevel(input.impact),
                confidencePercentage: normalizedConfidence * 100,
                easeLevel: this.getEaseLevel(input.ease),
            },
        };
    }
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
    validate(input) {
        const errors = [];
        const warnings = [];
        // Validate impact
        if (input.impact !== undefined) {
            if (typeof input.impact !== 'number' || isNaN(input.impact)) {
                errors.push('Impact must be a valid number');
            }
            else if (input.impact < this.minScore) {
                errors.push(`Impact must be at least ${this.minScore}`);
            }
            else if (input.impact > this.maxScore) {
                errors.push(`Impact must be at most ${this.maxScore}`);
            }
            else if (input.impact < 3) {
                warnings.push('Low impact - consider if this item deserves prioritization');
            }
        }
        // Validate confidence
        if (input.confidence !== undefined) {
            if (typeof input.confidence !== 'number' || isNaN(input.confidence)) {
                errors.push('Confidence must be a valid number');
            }
            else {
                const maxConf = this.confidenceAsPercentage ? 100 : 1;
                if (input.confidence < 0) {
                    errors.push('Confidence cannot be negative');
                }
                else if (input.confidence > maxConf) {
                    errors.push(`Confidence must be at most ${maxConf}`);
                }
                else if (input.confidence === 0) {
                    warnings.push('Zero confidence - you have no certainty in your estimates');
                }
                else if (input.confidence < 0.5 && !this.confidenceAsPercentage) {
                    warnings.push('Low confidence - consider gathering more data');
                }
                else if (input.confidence < 50 && this.confidenceAsPercentage) {
                    warnings.push('Low confidence - consider gathering more data');
                }
            }
        }
        // Validate ease
        if (input.ease !== undefined) {
            if (typeof input.ease !== 'number' || isNaN(input.ease)) {
                errors.push('Ease must be a valid number');
            }
            else if (input.ease < this.minScore) {
                errors.push(`Ease must be at least ${this.minScore}`);
            }
            else if (input.ease > this.maxScore) {
                errors.push(`Ease must be at most ${this.maxScore}`);
            }
            else if (input.ease < 3) {
                warnings.push('Low ease (hard to implement) - factor this into prioritization');
            }
            else if (input.ease >= 9) {
                warnings.push('Very high ease - ensure this is realistic');
            }
        }
        return {
            isValid: errors.length === 0,
            errors,
            warnings,
        };
    }
    /**
     * Normalizes ICE results across a dataset.
     *
     * @param result - The ICE result to normalize
     * @param allResults - All ICE results in the dataset
     * @returns NormalizedResult with rank, percentile, and normalized score
     */
    normalize(result, allResults) {
        if (allResults.length === 0) {
            return {
                normalizedScore: 0,
                rank: 0,
                percentile: 0,
            };
        }
        // Sort results by score in descending order
        const sortedResults = [...allResults].sort((a, b) => b.score - a.score);
        // Find the rank of this result (1-indexed)
        const rank = sortedResults.findIndex((r) => r.score === result.score) + 1;
        // Calculate percentile
        const itemsBelow = sortedResults.filter((r) => r.score < result.score).length;
        const percentile = allResults.length > 1
            ? parseFloat(((itemsBelow / (allResults.length - 1)) * 100).toFixed(this.decimalPlaces))
            : 100;
        // Calculate normalized score on 0-100 scale
        const minScore = Math.min(...allResults.map((r) => r.score));
        const maxScore = Math.max(...allResults.map((r) => r.score));
        let normalizedScore;
        if (maxScore === minScore) {
            normalizedScore = 50;
        }
        else {
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
     * @param item - The prioritization item to analyze
     * @returns Partial<ICEInput> with suggested values and confidence scores
     */
    getAutoFillSuggestions(item) {
        const text = `${item.title} ${item.description || ''} ${item.category || ''}`.toLowerCase();
        const suggestions = {};
        // Impact suggestions
        const highImpactKeywords = [
            'critical',
            'revenue',
            'conversion',
            'retention',
            'strategic',
            'blocker',
            'core',
        ];
        const lowImpactKeywords = ['nice to have', 'minor', 'cosmetic', 'optional', 'polish'];
        if (highImpactKeywords.some((k) => text.includes(k))) {
            suggestions.impact = 8;
            suggestions.impactConfidence = 0.75;
        }
        else if (lowImpactKeywords.some((k) => text.includes(k))) {
            suggestions.impact = 3;
            suggestions.impactConfidence = 0.7;
        }
        else {
            suggestions.impact = 5;
            suggestions.impactConfidence = 0.5;
        }
        // Confidence suggestions based on description quality
        if (item.description && item.description.length > 150) {
            suggestions.confidence = this.confidenceAsPercentage ? 70 : 0.7;
            suggestions.confidenceConfidence = 0.7;
        }
        else if (item.description && item.description.length > 80) {
            suggestions.confidence = this.confidenceAsPercentage ? 50 : 0.5;
            suggestions.confidenceConfidence = 0.6;
        }
        else {
            suggestions.confidence = this.confidenceAsPercentage ? 40 : 0.4;
            suggestions.confidenceConfidence = 0.5;
        }
        // Ease suggestions
        const highEaseKeywords = ['fix', 'update', 'quick', 'simple', 'small', 'config', 'tweak'];
        const lowEaseKeywords = ['migration', 'refactor', 'infrastructure', 'complex', 'major', 'rewrite'];
        if (highEaseKeywords.some((k) => text.includes(k))) {
            suggestions.ease = 8;
            suggestions.easeConfidence = 0.65;
        }
        else if (lowEaseKeywords.some((k) => text.includes(k))) {
            suggestions.ease = 3;
            suggestions.easeConfidence = 0.65;
        }
        else {
            suggestions.ease = 5;
            suggestions.easeConfidence = 0.45;
        }
        return suggestions;
    }
    /**
     * Normalizes confidence value to 0-1 range.
     * @private
     */
    normalizeConfidence(confidence) {
        if (confidence > 1) {
            return confidence / 100;
        }
        return confidence;
    }
    /**
     * Returns the impact level label for a given impact value.
     * @private
     */
    getImpactLevel(impact) {
        if (impact >= 8)
            return 'very high';
        if (impact >= 6)
            return 'high';
        if (impact >= 4)
            return 'medium';
        if (impact >= 2)
            return 'low';
        return 'very low';
    }
    /**
     * Returns the ease level label for a given ease value.
     * @private
     */
    getEaseLevel(ease) {
        if (ease >= 8)
            return 'very easy';
        if (ease >= 6)
            return 'easy';
        if (ease >= 4)
            return 'moderate';
        if (ease >= 2)
            return 'hard';
        return 'very hard';
    }
}
/**
 * Factory function to create a new ICE calculator instance.
 *
 * @param config - Optional configuration to override defaults
 * @returns A new ICECalculator instance
 */
export function createICECalculator(config = {}) {
    return new ICECalculator({ ...DEFAULT_CONFIG, ...config });
}
export default ICECalculator;
