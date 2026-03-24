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
/**
 * Default configuration for P0-P4 calculator.
 */
const DEFAULT_CONFIG = {
    /** P0 threshold (severity score >= this is P0) */
    p0Threshold: 90,
    /** P1 threshold (severity score >= this is P1) */
    p1Threshold: 70,
    /** P2 threshold (severity score >= this is P2) */
    p2Threshold: 50,
    /** P3 threshold (severity score >= this is P3) */
    p3Threshold: 30,
    /** Decimal places for rounding */
    decimalPlaces: 2,
};
/**
 * Priority level weights for scoring and ranking.
 * Higher values indicate higher priority.
 */
export const PRIORITY_WEIGHTS = {
    P0: 5,
    P1: 4,
    P2: 3,
    P3: 2,
    P4: 1,
};
/**
 * Severity factor scores for each level.
 */
export const SEVERITY_FACTOR_SCORES = {
    usersAffected: {
        all: 5,
        many: 4,
        some: 3,
        few: 2,
        none: 1,
    },
    coreFunctionalityImpact: {
        critical: 5,
        high: 4,
        medium: 3,
        low: 2,
        none: 1,
    },
    securityRisk: {
        critical: 5,
        high: 4,
        medium: 3,
        low: 2,
        none: 1,
    },
    reputationalRisk: {
        critical: 5,
        high: 4,
        medium: 3,
        low: 2,
        none: 1,
    },
    revenueImpact: {
        critical: 5,
        high: 4,
        medium: 3,
        low: 2,
        none: 1,
    },
};
/**
 * Recommended timeframes for each priority level.
 */
export const PRIORITY_TIMEFRAMES = {
    P0: 'Immediate (within 24 hours)',
    P1: 'This sprint (1-2 weeks)',
    P2: 'Next sprint (2-4 weeks)',
    P3: 'Next quarter (1-3 months)',
    P4: 'Backlog / Future consideration',
};
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
export class P0P4Calculator {
    /**
     * Creates a new P0-P4 calculator instance.
     * @param config - Optional configuration to override defaults
     */
    constructor(config = DEFAULT_CONFIG) {
        Object.defineProperty(this, "p0Threshold", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: void 0
        });
        Object.defineProperty(this, "p1Threshold", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: void 0
        });
        Object.defineProperty(this, "p2Threshold", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: void 0
        });
        Object.defineProperty(this, "p3Threshold", {
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
        this.p0Threshold = config.p0Threshold ?? DEFAULT_CONFIG.p0Threshold;
        this.p1Threshold = config.p1Threshold ?? DEFAULT_CONFIG.p1Threshold;
        this.p2Threshold = config.p2Threshold ?? DEFAULT_CONFIG.p2Threshold;
        this.p3Threshold = config.p3Threshold ?? DEFAULT_CONFIG.p3Threshold;
        this.decimalPlaces = config.decimalPlaces ?? DEFAULT_CONFIG.decimalPlaces;
    }
    /**
     * Returns the framework type identifier.
     *
     * @returns 'P0P4' as the framework type
     */
    getFrameworkType() {
        return 'P0P4';
    }
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
    calculate(input) {
        // Validate input first
        const validation = this.validate(input);
        if (!validation.isValid) {
            throw new Error(`Invalid P0-P4 input: ${validation.errors.join(', ')}`);
        }
        // Calculate severity score
        const severityScore = this.calculateSeverityScore(input);
        // Determine priority level
        const priority = this.determinePriority(severityScore);
        // Get recommended timeframe
        const recommendedTimeframe = PRIORITY_TIMEFRAMES[priority];
        return {
            framework: 'P0P4',
            priority,
            severityScore,
            severityFactors: input.severityFactors,
            recommendedTimeframe,
            details: {
                baseSeverity: input.baseSeverity,
                factorsScore: this.calculateFactorsScore(input.severityFactors),
                timeModifier: this.calculateTimeModifier(input),
                priorityWeight: PRIORITY_WEIGHTS[priority],
            },
        };
    }
    /**
     * Validates P0-P4 input parameters.
     *
     * @param input - Partial P0-P4 input parameters to validate
     * @returns ValidationResult with errors and warnings
     */
    validate(input) {
        const errors = [];
        const warnings = [];
        // Validate baseSeverity
        if (input.baseSeverity !== undefined) {
            if (typeof input.baseSeverity !== 'number' || isNaN(input.baseSeverity)) {
                errors.push('baseSeverity must be a valid number');
            }
            else if (input.baseSeverity < 1) {
                errors.push('baseSeverity must be at least 1');
            }
            else if (input.baseSeverity > 5) {
                errors.push('baseSeverity must be at most 5');
            }
        }
        // Validate severityFactors
        if (input.severityFactors !== undefined) {
            const factors = input.severityFactors;
            // Validate usersAffected
            if (factors.usersAffected !== undefined) {
                const validUsers = ['all', 'many', 'some', 'few', 'none'];
                if (!validUsers.includes(factors.usersAffected)) {
                    errors.push(`usersAffected must be one of: ${validUsers.join(', ')}`);
                }
            }
            // Validate coreFunctionalityImpact
            if (factors.coreFunctionalityImpact !== undefined) {
                const validImpacts = ['critical', 'high', 'medium', 'low', 'none'];
                if (!validImpacts.includes(factors.coreFunctionalityImpact)) {
                    errors.push(`coreFunctionalityImpact must be one of: ${validImpacts.join(', ')}`);
                }
            }
            // Validate securityRisk
            if (factors.securityRisk !== undefined) {
                const validRisks = ['critical', 'high', 'medium', 'low', 'none'];
                if (!validRisks.includes(factors.securityRisk)) {
                    errors.push(`securityRisk must be one of: ${validRisks.join(', ')}`);
                }
            }
            // Validate reputationalRisk
            if (factors.reputationalRisk !== undefined) {
                const validRisks = ['critical', 'high', 'medium', 'low', 'none'];
                if (!validRisks.includes(factors.reputationalRisk)) {
                    errors.push(`reputationalRisk must be one of: ${validRisks.join(', ')}`);
                }
            }
            // Validate revenueImpact
            if (factors.revenueImpact !== undefined) {
                const validImpacts = ['critical', 'high', 'medium', 'low', 'none'];
                if (!validImpacts.includes(factors.revenueImpact)) {
                    errors.push(`revenueImpact must be one of: ${validImpacts.join(', ')}`);
                }
            }
        }
        // Validate openIssuesCount
        if (input.openIssuesCount !== undefined) {
            if (typeof input.openIssuesCount !== 'number' || isNaN(input.openIssuesCount)) {
                errors.push('openIssuesCount must be a valid number');
            }
            else if (input.openIssuesCount < 0) {
                errors.push('openIssuesCount cannot be negative');
            }
            else if (input.openIssuesCount > 100) {
                warnings.push('Very high openIssuesCount - consider creating a meta-issue');
            }
        }
        // Validate daysOpen
        if (input.daysOpen !== undefined) {
            if (typeof input.daysOpen !== 'number' || isNaN(input.daysOpen)) {
                errors.push('daysOpen must be a valid number');
            }
            else if (input.daysOpen < 0) {
                errors.push('daysOpen cannot be negative');
            }
            else if (input.daysOpen > 365) {
                warnings.push('Issue has been open for over a year - review relevance');
            }
        }
        // Warning: Critical severity factors should have high base severity
        if (input.baseSeverity !== undefined &&
            input.severityFactors !== undefined &&
            input.baseSeverity < 4 &&
            (input.severityFactors.securityRisk === 'critical' ||
                input.severityFactors.coreFunctionalityImpact === 'critical')) {
            warnings.push('Critical security/functionality risk with low base severity - review classification');
        }
        return {
            isValid: errors.length === 0,
            errors,
            warnings,
        };
    }
    /**
     * Normalizes P0-P4 results across a dataset.
     *
     * @param result - The P0-P4 result to normalize
     * @param allResults - All P0-P4 results in the dataset
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
        // Sort by severity score (descending)
        const sortedResults = [...allResults].sort((a, b) => b.severityScore - a.severityScore);
        // Find the rank
        const rank = sortedResults.findIndex((r) => r.severityScore === result.severityScore) + 1;
        // Calculate percentile
        const itemsBelow = sortedResults.filter((r) => r.severityScore < result.severityScore).length;
        const percentile = allResults.length > 1
            ? parseFloat(((itemsBelow / (allResults.length - 1)) * 100).toFixed(this.decimalPlaces))
            : 100;
        // Calculate normalized score on 0-100 scale
        const minScore = Math.min(...allResults.map((r) => r.severityScore));
        const maxScore = Math.max(...allResults.map((r) => r.severityScore));
        let normalizedScore;
        if (maxScore === minScore) {
            normalizedScore = 50;
        }
        else {
            normalizedScore = parseFloat((((result.severityScore - minScore) / (maxScore - minScore)) * 100).toFixed(this.decimalPlaces));
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
     * @returns Partial<P0P4Input> with suggested values and confidence scores
     */
    getAutoFillSuggestions(item) {
        const text = `${item.title} ${item.description || ''} ${item.category || ''}`.toLowerCase();
        const suggestions = {};
        // P0 indicators (critical issues)
        const p0Keywords = [
            'outage',
            'down',
            'critical',
            'security breach',
            'data loss',
            'production down',
            'service unavailable',
            'p0',
        ];
        // P1 indicators (high priority)
        const p1Keywords = [
            'broken',
            'not working',
            'major issue',
            'high priority',
            'blocking',
            'blocker',
            'severe',
            'p1',
        ];
        // P3/P4 indicators (low priority)
        const lowPriorityKeywords = [
            'nice to have',
            'minor',
            'cosmetic',
            'polish',
            'enhancement',
            'quality of life',
            'p3',
            'p4',
        ];
        // Determine base severity
        if (p0Keywords.some((k) => text.includes(k))) {
            suggestions.baseSeverity = 5;
            suggestions.baseSeverityConfidence = 0.9;
            suggestions.severityFactors = {
                usersAffected: 'all',
                coreFunctionalityImpact: 'critical',
                securityRisk: 'critical',
                reputationalRisk: 'high',
                revenueImpact: 'critical',
            };
            suggestions.factorsConfidence = 0.8;
        }
        else if (p1Keywords.some((k) => text.includes(k))) {
            suggestions.baseSeverity = 4;
            suggestions.baseSeverityConfidence = 0.8;
            suggestions.severityFactors = {
                usersAffected: 'many',
                coreFunctionalityImpact: 'high',
                securityRisk: 'medium',
                reputationalRisk: 'medium',
                revenueImpact: 'high',
            };
            suggestions.factorsConfidence = 0.7;
        }
        else if (lowPriorityKeywords.some((k) => text.includes(k))) {
            suggestions.baseSeverity = 2;
            suggestions.baseSeverityConfidence = 0.7;
            suggestions.severityFactors = {
                usersAffected: 'few',
                coreFunctionalityImpact: 'low',
                securityRisk: 'none',
                reputationalRisk: 'low',
                revenueImpact: 'low',
            };
            suggestions.factorsConfidence = 0.6;
        }
        else {
            // Default to P2 (medium)
            suggestions.baseSeverity = 3;
            suggestions.baseSeverityConfidence = 0.5;
            suggestions.severityFactors = {
                usersAffected: 'some',
                coreFunctionalityImpact: 'medium',
                securityRisk: 'low',
                reputationalRisk: 'low',
                revenueImpact: 'medium',
            };
            suggestions.factorsConfidence = 0.4;
        }
        return suggestions;
    }
    /**
     * Calculates the total severity score.
     * @private
     */
    calculateSeverityScore(input) {
        // Base severity contribution (max 50 points)
        const baseContribution = (input.baseSeverity / 5) * 50;
        // Factors contribution (max 50 points)
        const factorsScore = this.calculateFactorsScore(input.severityFactors);
        const factorsContribution = (factorsScore / 25) * 50; // Max factor score is 25
        // Time-based modifier
        const timeModifier = this.calculateTimeModifier(input);
        // Total score (capped at 100)
        const totalScore = baseContribution + factorsContribution + timeModifier;
        return parseFloat(Math.min(100, totalScore).toFixed(this.decimalPlaces));
    }
    /**
     * Calculates the score from severity factors.
     * @private
     */
    calculateFactorsScore(factors) {
        let score = 0;
        score += SEVERITY_FACTOR_SCORES.usersAffected[factors.usersAffected];
        score += SEVERITY_FACTOR_SCORES.coreFunctionalityImpact[factors.coreFunctionalityImpact];
        score += SEVERITY_FACTOR_SCORES.securityRisk[factors.securityRisk];
        score += SEVERITY_FACTOR_SCORES.reputationalRisk[factors.reputationalRisk];
        score += SEVERITY_FACTOR_SCORES.revenueImpact[factors.revenueImpact];
        return score;
    }
    /**
     * Calculates time-based modifier.
     * @private
     */
    calculateTimeModifier(input) {
        let modifier = 0;
        // Days open modifier (up to 10 points for issues open > 30 days)
        if (input.daysOpen !== undefined) {
            if (input.daysOpen > 30) {
                modifier += Math.min(10, (input.daysOpen - 30) / 3);
            }
        }
        // Open issues count modifier (up to 10 points for > 5 related issues)
        if (input.openIssuesCount !== undefined) {
            if (input.openIssuesCount > 5) {
                modifier += Math.min(10, (input.openIssuesCount - 5) / 2);
            }
        }
        return modifier;
    }
    /**
     * Determines priority level from severity score.
     * @private
     */
    determinePriority(severityScore) {
        if (severityScore >= this.p0Threshold) {
            return 'P0';
        }
        else if (severityScore >= this.p1Threshold) {
            return 'P1';
        }
        else if (severityScore >= this.p2Threshold) {
            return 'P2';
        }
        else if (severityScore >= this.p3Threshold) {
            return 'P3';
        }
        else {
            return 'P4';
        }
    }
}
/**
 * Factory function to create a new P0-P4 calculator instance.
 *
 * @param config - Optional configuration to override defaults
 * @returns A new P0P4Calculator instance
 */
export function createP0P4Calculator(config = {}) {
    return new P0P4Calculator({ ...DEFAULT_CONFIG, ...config });
}
export default P0P4Calculator;
