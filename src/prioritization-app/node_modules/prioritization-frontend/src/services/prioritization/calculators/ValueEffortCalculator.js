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
/**
 * Default configuration for Value vs Effort calculator.
 */
const DEFAULT_CONFIG = {
    /** Threshold for high value (scores >= this are "high") */
    highValueThreshold: 6,
    /** Threshold for high effort (scores >= this are "high") */
    highEffortThreshold: 6,
    /** Minimum score value */
    minScore: 1,
    /** Maximum score value */
    maxScore: 10,
    /** Decimal places for rounding */
    decimalPlaces: 2,
};
/**
 * Quadrant weights for scoring and ranking.
 * Higher values indicate higher priority.
 */
export const QUADRANT_WEIGHTS = {
    QuickWin: 4,
    MajorProject: 3,
    FillIn: 2,
    Avoid: 1,
};
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
export class ValueEffortCalculator {
    /**
     * Creates a new Value vs Effort calculator instance.
     * @param config - Optional configuration to override defaults
     */
    constructor(config = DEFAULT_CONFIG) {
        Object.defineProperty(this, "highValueThreshold", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: void 0
        });
        Object.defineProperty(this, "highEffortThreshold", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: void 0
        });
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
        this.highValueThreshold = config.highValueThreshold ?? DEFAULT_CONFIG.highValueThreshold;
        this.highEffortThreshold = config.highEffortThreshold ?? DEFAULT_CONFIG.highEffortThreshold;
        this.minScore = config.minScore ?? DEFAULT_CONFIG.minScore;
        this.maxScore = config.maxScore ?? DEFAULT_CONFIG.maxScore;
        this.decimalPlaces = config.decimalPlaces ?? DEFAULT_CONFIG.decimalPlaces;
    }
    /**
     * Returns the framework type identifier.
     *
     * @returns 'ValueEffort' as the framework type
     */
    getFrameworkType() {
        return 'ValueEffort';
    }
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
    calculate(input) {
        // Validate input first
        const validation = this.validate(input);
        if (!validation.isValid) {
            throw new Error(`Invalid Value vs Effort input: ${validation.errors.join(', ')}`);
        }
        // Calculate ROI (Return on Investment)
        const roi = parseFloat((input.value / input.effort).toFixed(this.decimalPlaces));
        // Determine quadrant
        const quadrant = this.determineQuadrant(input.value, input.effort);
        return {
            framework: 'ValueEffort',
            value: input.value,
            effort: input.effort,
            quadrant,
            roi,
            details: {
                quadrantWeight: QUADRANT_WEIGHTS[quadrant],
                valueLevel: input.value >= this.highValueThreshold ? 'high' : 'low',
                effortLevel: input.effort >= this.highEffortThreshold ? 'high' : 'low',
            },
        };
    }
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
    validate(input) {
        const errors = [];
        const warnings = [];
        // Validate value
        if (input.value !== undefined) {
            if (typeof input.value !== 'number' || isNaN(input.value)) {
                errors.push('Value must be a valid number');
            }
            else if (input.value < this.minScore) {
                errors.push(`Value must be at least ${this.minScore}`);
            }
            else if (input.value > this.maxScore) {
                errors.push(`Value must be at most ${this.maxScore}`);
            }
            else if (input.value < 3) {
                warnings.push('Very low value - consider if this item should be pursued');
            }
        }
        // Validate effort
        if (input.effort !== undefined) {
            if (typeof input.effort !== 'number' || isNaN(input.effort)) {
                errors.push('Effort must be a valid number');
            }
            else if (input.effort < this.minScore) {
                errors.push(`Effort must be at least ${this.minScore}`);
            }
            else if (input.effort > this.maxScore) {
                errors.push(`Effort must be at most ${this.maxScore}`);
            }
            else if (input.effort < 2) {
                warnings.push('Very low effort - ensure this is realistic');
            }
            else if (input.effort >= this.maxScore) {
                warnings.push('Maximum effort - consider breaking this into smaller initiatives');
            }
        }
        // Warning: Low value with high effort (Avoid quadrant)
        if (input.value !== undefined &&
            input.effort !== undefined &&
            input.value < this.highValueThreshold &&
            input.effort >= this.highEffortThreshold) {
            warnings.push('Low value, high effort item - carefully evaluate if this should be done');
        }
        return {
            isValid: errors.length === 0,
            errors,
            warnings,
        };
    }
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
    normalize(result, allResults) {
        if (allResults.length === 0) {
            return {
                normalizedScore: 0,
                rank: 0,
                percentile: 0,
            };
        }
        // Calculate composite score for sorting
        // ROI (60%) + Quadrant weight (30%) + Value (10%)
        const getCompositeScore = (r) => {
            const maxRoi = Math.max(...allResults.map((x) => x.roi));
            const normalizedRoi = maxRoi > 0 ? r.roi / maxRoi : 0;
            const quadrantScore = QUADRANT_WEIGHTS[r.quadrant] / 4; // Normalize to 0-1
            const normalizedValue = r.value / this.maxScore;
            return normalizedRoi * 0.6 + quadrantScore * 0.3 + normalizedValue * 0.1;
        };
        // Sort by composite score (descending)
        const sortedResults = [...allResults].sort((a, b) => {
            return getCompositeScore(b) - getCompositeScore(a);
        });
        // Find the rank of this result (1-indexed)
        const rank = sortedResults.findIndex((r) => r.quadrant === result.quadrant && r.roi === result.roi && r.value === result.value) + 1;
        // Calculate percentile
        const itemsBelow = sortedResults.filter((r) => {
            return getCompositeScore(r) < getCompositeScore(result);
        }).length;
        const percentile = allResults.length > 1
            ? parseFloat(((itemsBelow / (allResults.length - 1)) * 100).toFixed(this.decimalPlaces))
            : 100;
        // Calculate normalized score on 0-100 scale
        const scores = allResults.map(getCompositeScore);
        const minScore = Math.min(...scores);
        const maxScore = Math.max(...scores);
        const resultScore = getCompositeScore(result);
        let normalizedScore;
        if (maxScore === minScore) {
            normalizedScore = 50;
        }
        else {
            normalizedScore = parseFloat((((resultScore - minScore) / (maxScore - minScore)) * 100).toFixed(this.decimalPlaces));
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
    getAutoFillSuggestions(item) {
        const text = `${item.title} ${item.description || ''} ${item.category || ''}`.toLowerCase();
        const suggestions = {};
        // Value suggestions based on value-related keywords
        const highValueKeywords = [
            'revenue',
            'profit',
            'conversion',
            'retention',
            'critical',
            'strategic',
            'competitive',
            'enterprise',
            'customer request',
            'high demand',
            'core',
            'key feature',
        ];
        const mediumValueKeywords = [
            'improve',
            'enhance',
            'optimize',
            'better',
            'important',
            'useful',
            'helpful',
        ];
        const lowValueKeywords = [
            'nice to have',
            'optional',
            'cosmetic',
            'minor',
            'quality of life',
            'polish',
        ];
        if (highValueKeywords.some((k) => text.includes(k))) {
            suggestions.value = 8;
            suggestions.valueConfidence = 0.75;
        }
        else if (lowValueKeywords.some((k) => text.includes(k))) {
            suggestions.value = 3;
            suggestions.valueConfidence = 0.7;
        }
        else if (mediumValueKeywords.some((k) => text.includes(k))) {
            suggestions.value = 5;
            suggestions.valueConfidence = 0.65;
        }
        else {
            suggestions.value = 5;
            suggestions.valueConfidence = 0.4;
        }
        // Effort suggestions based on complexity keywords
        const highEffortKeywords = [
            'migration',
            'refactor',
            'infrastructure',
            'platform',
            'integration',
            'rewrite',
            'complex',
            'major',
            'overhaul',
            'rebuild',
            'architecture',
        ];
        const mediumEffortKeywords = [
            'feature',
            'implement',
            'develop',
            'module',
            'component',
            'system',
            'new',
        ];
        const lowEffortKeywords = [
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
        ];
        if (highEffortKeywords.some((k) => text.includes(k))) {
            suggestions.effort = 8;
            suggestions.effortConfidence = 0.65;
        }
        else if (lowEffortKeywords.some((k) => text.includes(k))) {
            suggestions.effort = 2;
            suggestions.effortConfidence = 0.7;
        }
        else if (mediumEffortKeywords.some((k) => text.includes(k))) {
            suggestions.effort = 5;
            suggestions.effortConfidence = 0.55;
        }
        else {
            suggestions.effort = 5;
            suggestions.effortConfidence = 0.35;
        }
        return suggestions;
    }
    /**
     * Determines the quadrant based on value and effort scores.
     *
     * @param value - The business value score
     * @param effort - The effort score
     * @returns The assigned quadrant
     * @private
     */
    determineQuadrant(value, effort) {
        const isHighValue = value >= this.highValueThreshold;
        const isHighEffort = effort >= this.highEffortThreshold;
        if (isHighValue && !isHighEffort) {
            return 'QuickWin';
        }
        else if (isHighValue && isHighEffort) {
            return 'MajorProject';
        }
        else if (!isHighValue && !isHighEffort) {
            return 'FillIn';
        }
        else {
            return 'Avoid';
        }
    }
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
export function createValueEffortCalculator(config = {}) {
    return new ValueEffortCalculator({ ...DEFAULT_CONFIG, ...config });
}
export default ValueEffortCalculator;
