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
/**
 * Default configuration for Eisenhower Matrix calculator.
 */
const DEFAULT_CONFIG = {
    /** Threshold for considering something urgent (when using urgencyLevel) */
    urgencyThreshold: 6,
    /** Threshold for considering something important (when using importanceLevel) */
    importanceThreshold: 6,
    /** Minimum level value */
    minLevel: 1,
    /** Maximum level value */
    maxLevel: 10,
    /** Decimal places for rounding */
    decimalPlaces: 2,
};
/**
 * Quadrant weights for scoring and ranking.
 * Higher values indicate higher priority.
 */
export const EISENHOWER_QUADRANT_WEIGHTS = {
    DoFirst: 4,
    Schedule: 3,
    Delegate: 2,
    Eliminate: 1,
};
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
export class EisenhowerCalculator {
    /**
     * Creates a new Eisenhower Matrix calculator instance.
     * @param config - Optional configuration to override defaults
     */
    constructor(config = DEFAULT_CONFIG) {
        Object.defineProperty(this, "urgencyThreshold", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: void 0
        });
        Object.defineProperty(this, "importanceThreshold", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: void 0
        });
        Object.defineProperty(this, "minLevel", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: void 0
        });
        Object.defineProperty(this, "maxLevel", {
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
        this.urgencyThreshold = config.urgencyThreshold ?? DEFAULT_CONFIG.urgencyThreshold;
        this.importanceThreshold = config.importanceThreshold ?? DEFAULT_CONFIG.importanceThreshold;
        this.minLevel = config.minLevel ?? DEFAULT_CONFIG.minLevel;
        this.maxLevel = config.maxLevel ?? DEFAULT_CONFIG.maxLevel;
        this.decimalPlaces = config.decimalPlaces ?? DEFAULT_CONFIG.decimalPlaces;
    }
    /**
     * Returns the framework type identifier.
     *
     * @returns 'Eisenhower' as the framework type
     */
    getFrameworkType() {
        return 'Eisenhower';
    }
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
    calculate(input) {
        // Validate input first
        const validation = this.validate(input);
        if (!validation.isValid) {
            throw new Error(`Invalid Eisenhower input: ${validation.errors.join(', ')}`);
        }
        // Determine quadrant based on urgent/important booleans
        const quadrant = this.determineQuadrant(input.urgent, input.important);
        // Calculate priority score if levels are provided
        const priorityScore = this.calculatePriorityScore(input);
        return {
            framework: 'Eisenhower',
            urgent: input.urgent,
            important: input.important,
            quadrant,
            urgencyLevel: input.urgencyLevel,
            importanceLevel: input.importanceLevel,
            details: {
                quadrantWeight: EISENHOWER_QUADRANT_WEIGHTS[quadrant],
                priorityScore,
                actionVerb: this.getActionVerb(quadrant),
            },
        };
    }
    /**
     * Validates Eisenhower input parameters.
     *
     * @param input - Partial Eisenhower input parameters to validate
     * @returns ValidationResult with errors and warnings
     */
    validate(input) {
        const errors = [];
        const warnings = [];
        // Validate urgent boolean
        if (input.urgent === undefined) {
            errors.push('urgent is required (true/false)');
        }
        else if (typeof input.urgent !== 'boolean') {
            errors.push('urgent must be a boolean (true/false)');
        }
        // Validate important boolean
        if (input.important === undefined) {
            errors.push('important is required (true/false)');
        }
        else if (typeof input.important !== 'boolean') {
            errors.push('important must be a boolean (true/false)');
        }
        // Validate urgencyLevel if provided
        if (input.urgencyLevel !== undefined) {
            if (typeof input.urgencyLevel !== 'number' || isNaN(input.urgencyLevel)) {
                errors.push('urgencyLevel must be a valid number');
            }
            else if (input.urgencyLevel < this.minLevel) {
                errors.push(`urgencyLevel must be at least ${this.minLevel}`);
            }
            else if (input.urgencyLevel > this.maxLevel) {
                errors.push(`urgencyLevel must be at most ${this.maxLevel}`);
            }
            else {
                // Check consistency with urgent boolean
                const shouldBeUrgent = input.urgencyLevel >= this.urgencyThreshold;
                if (shouldBeUrgent && input.urgent === false) {
                    warnings.push(`urgencyLevel ${input.urgencyLevel} suggests urgent, but urgent is false`);
                }
                else if (!shouldBeUrgent && input.urgent === true) {
                    warnings.push(`urgencyLevel ${input.urgencyLevel} suggests not urgent, but urgent is true`);
                }
            }
        }
        // Validate importanceLevel if provided
        if (input.importanceLevel !== undefined) {
            if (typeof input.importanceLevel !== 'number' || isNaN(input.importanceLevel)) {
                errors.push('importanceLevel must be a valid number');
            }
            else if (input.importanceLevel < this.minLevel) {
                errors.push(`importanceLevel must be at least ${this.minLevel}`);
            }
            else if (input.importanceLevel > this.maxLevel) {
                errors.push(`importanceLevel must be at most ${this.maxLevel}`);
            }
            else {
                // Check consistency with important boolean
                const shouldBeImportant = input.importanceLevel >= this.importanceThreshold;
                if (shouldBeImportant && input.important === false) {
                    warnings.push(`importanceLevel ${input.importanceLevel} suggests important, but important is false`);
                }
                else if (!shouldBeImportant && input.important === true) {
                    warnings.push(`importanceLevel ${input.importanceLevel} suggests not important, but important is true`);
                }
            }
        }
        // Warning: Everything being urgent and important suggests poor prioritization
        if (input.urgent === true && input.important === true) {
            warnings.push('Marking everything as urgent and important may indicate prioritization challenges');
        }
        return {
            isValid: errors.length === 0,
            errors,
            warnings,
        };
    }
    /**
     * Normalizes Eisenhower results across a dataset.
     *
     * @param result - The Eisenhower result to normalize
     * @param allResults - All Eisenhower results in the dataset
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
        // Calculate composite score for sorting
        const getCompositeScore = (r) => {
            const quadrantWeight = EISENHOWER_QUADRANT_WEIGHTS[r.quadrant];
            // Add priority score if available (for finer granularity)
            const priorityBonus = r.details.priorityScore ? r.details.priorityScore / 100 : 0;
            return quadrantWeight + priorityBonus;
        };
        // Sort by composite score (descending)
        const sortedResults = [...allResults].sort((a, b) => {
            return getCompositeScore(b) - getCompositeScore(a);
        });
        // Find the rank
        const rank = sortedResults.findIndex((r) => r.quadrant === result.quadrant && r.details.priorityScore === result.details.priorityScore) + 1;
        // Calculate percentile
        const itemsBelow = sortedResults.filter((r) => {
            return getCompositeScore(r) < getCompositeScore(result);
        }).length;
        const percentile = allResults.length > 1
            ? parseFloat(((itemsBelow / (allResults.length - 1)) * 100).toFixed(this.decimalPlaces))
            : 100;
        // Calculate normalized score
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
     * @param item - The prioritization item to analyze
     * @returns Partial<EisenhowerInput> with suggested values and confidence scores
     */
    getAutoFillSuggestions(item) {
        const text = `${item.title} ${item.description || ''} ${item.category || ''}`.toLowerCase();
        const suggestions = {};
        // Urgency detection
        const urgentKeywords = [
            'urgent',
            'asap',
            'deadline',
            'critical',
            'emergency',
            'rush',
            'immediate',
            'today',
            'this week',
            'blocking',
            'blocker',
            'production issue',
            'outage',
            'down',
        ];
        if (urgentKeywords.some((k) => text.includes(k))) {
            suggestions.urgent = true;
            suggestions.urgentConfidence = 0.85;
            suggestions.urgencyLevel = 8;
            suggestions.urgencyLevelConfidence = 0.7;
        }
        else {
            suggestions.urgent = false;
            suggestions.urgentConfidence = 0.6;
            suggestions.urgencyLevel = 3;
            suggestions.urgencyLevelConfidence = 0.5;
        }
        // Importance detection
        const importantKeywords = [
            'critical',
            'strategic',
            'revenue',
            'customer',
            'core',
            'key',
            'major',
            'high priority',
            'business critical',
            'security',
            'compliance',
            'legal',
        ];
        if (importantKeywords.some((k) => text.includes(k))) {
            suggestions.important = true;
            suggestions.importantConfidence = 0.8;
            suggestions.importanceLevel = 8;
            suggestions.importanceLevelConfidence = 0.7;
        }
        else {
            suggestions.important = false;
            suggestions.importantConfidence = 0.5;
            suggestions.importanceLevel = 4;
            suggestions.importanceLevelConfidence = 0.4;
        }
        return suggestions;
    }
    /**
     * Determines the quadrant based on urgency and importance.
     * @private
     */
    determineQuadrant(urgent, important) {
        if (urgent && important) {
            return 'DoFirst';
        }
        else if (!urgent && important) {
            return 'Schedule';
        }
        else if (urgent && !important) {
            return 'Delegate';
        }
        else {
            return 'Eliminate';
        }
    }
    /**
     * Calculates a priority score based on urgency and importance levels.
     * @private
     */
    calculatePriorityScore(input) {
        const urgency = input.urgencyLevel ?? (input.urgent ? 7 : 3);
        const importance = input.importanceLevel ?? (input.important ? 7 : 3);
        // Weight importance slightly higher than urgency
        return parseFloat(((urgency * 0.4 + importance * 0.6) * 10).toFixed(this.decimalPlaces));
    }
    /**
     * Returns the action verb for a given quadrant.
     * @private
     */
    getActionVerb(quadrant) {
        switch (quadrant) {
            case 'DoFirst':
                return 'Do immediately';
            case 'Schedule':
                return 'Plan and schedule';
            case 'Delegate':
                return 'Delegate or automate';
            case 'Eliminate':
                return 'Eliminate or defer';
            default:
                return 'Review';
        }
    }
}
/**
 * Factory function to create a new Eisenhower Matrix calculator instance.
 *
 * @param config - Optional configuration to override defaults
 * @returns A new EisenhowerCalculator instance
 */
export function createEisenhowerCalculator(config = {}) {
    return new EisenhowerCalculator({ ...DEFAULT_CONFIG, ...config });
}
export default EisenhowerCalculator;
