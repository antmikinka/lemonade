/**
 * Metrics Extractor for Prioritization Frameworks.
 *
 * This module extracts numerical estimates and metrics from item labels,
 * titles, and descriptions. Uses pattern matching for common phrases
 * and numerical expressions.
 *
 * @module prioritization/auto-fill/MetricsExtractor
 */

/**
 * Scale detection result.
 */
export type DetectedScale = 'small' | 'medium' | 'large';

/**
 * Extracted metrics from text analysis.
 */
export interface ExtractedMetrics {
  /** Reach estimate (number of users/events affected) */
  reach?: number;
  /** Impact estimate (multiplier) */
  impact?: number;
  /** Confidence estimate (percentage 0-100) */
  confidence?: number;
  /** Effort estimate (person-months or story points) */
  effort?: number;
  /** Value estimate (1-10 scale) */
  value?: number;
  /** Ease estimate (1-10 scale) */
  ease?: number;
  /** Urgency level (1-10 scale) */
  urgency?: number;
  /** Importance level (1-10 scale) */
  importance?: number;
  /** Confidence in extraction (0-1) */
  extractionConfidence: number;
}

/**
 * Patterns for detecting numerical values and scales.
 */
const NUMBER_PATTERNS = {
  /** Pattern for explicit numbers with optional units */
  EXPLICIT_NUMBER: /(\d+(?:\.\d+)?)\s*(%|percent|users|weeks|months|days|hours|people|customers|events|page views)?/gi,
  /** Pattern for range expressions */
  RANGE: /(\d+(?:\.\d+)?)\s*[-–to]+\s*(\d+(?:\.\d+)?)/i,
  /** Pattern for approximate numbers */
  APPROXIMATE: /(about|around|approximately|roughly|~)\s*(\d+(?:\.\d+)?)/gi,
  /** Pattern for multipliers */
  MULTIPLIER: /(\d+(?:\.\d+)?)\s*(x|times|fold)/gi,
  /** Pattern for percentages */
  PERCENTAGE: /(\d+(?:\.\d+)?)\s*%/gi,
  /** Pattern for large numbers */
  LARGE_NUMBER: /(\d+(?:\.\d+)?)\s*(k|m|b|thousand|million|billion)/gi
};

/**
 * Scale keywords for detecting size categories.
 */
const SCALE_KEYWORDS = {
  REACH: {
    small: ['few', 'some', 'select', 'limited', 'niche', 'specific'],
    medium: ['many', 'several', 'moderate', 'significant'],
    large: ['all', 'every', 'mass', 'widespread', 'universal', 'enterprise-wide']
  },
  EFFORT: {
    small: ['quick', 'simple', 'easy', 'minor', 'trivial', 'small', 'tiny', 'day', 'hour'],
    medium: ['moderate', 'medium', 'standard', 'normal', 'weeks', 'sprint'],
    large: ['large', 'major', 'complex', 'massive', 'months', 'quarter', 'epic']
  },
  IMPACT: {
    small: ['minimal', 'minor', 'small', 'negligible', 'marginal', 'slight'],
    medium: ['moderate', 'medium', 'noticeable', 'significant', 'meaningful'],
    large: ['major', 'massive', 'huge', 'enormous', 'transformative', 'critical']
  },
  VALUE: {
    small: ['low', 'minimal', 'minor', 'marginal'],
    medium: ['medium', 'moderate', 'reasonable', 'decent'],
    large: ['high', 'major', 'significant', 'substantial', ' tremendous']
  },
  IMPORTANCE: {
    small: ['minor', 'optional', 'nice to have', 'low priority'],
    medium: ['moderate', 'normal', 'standard', 'medium priority'],
    large: ['critical', 'essential', 'important', 'high priority', 'strategic']
  }
};

/**
 * Default values for different metrics.
 */
const DEFAULT_VALUES = {
  REACH: { small: 50, medium: 500, large: 5000 },
  EFFORT: { small: 0.5, medium: 2, large: 6 },
  IMPACT: { small: 0.5, medium: 1, large: 2 },
  VALUE: { small: 3, medium: 5, large: 8 },
  CONFIDENCE: { small: 30, medium: 50, large: 70 },
  EASE: { small: 3, medium: 5, large: 8 },
  URGENCY: { small: 2, medium: 5, large: 8 },
  IMPORTANCE: { small: 3, medium: 5, large: 8 }
};

/**
 * MetricsExtractor class for extracting numerical estimates from text.
 *
 * Analyzes text to find explicit numbers, detect scale indicators,
 * and infer reasonable estimates for prioritization parameters.
 *
 * @example
 * ```typescript
 * const extractor = new MetricsExtractor();
 *
 * const reach = extractor.extractReach('Feature for 10,000 enterprise users');
 * // Returns 10000
 *
 * const effort = extractor.extractEffort('Quick 2-week fix');
 * // Returns 0.5 (person-months)
 *
 * const scale = extractor.detectScale('Major infrastructure overhaul');
 * // Returns 'large'
 * ```
 */
export class MetricsExtractor {
  /**
   * Extract reach estimate from text.
   *
   * @param label - The label/title to analyze
   * @param description - Optional description for additional context
   * @returns Reach estimate (number of users/events affected)
   *
   * @example
   * ```typescript
   * const reach = extractor.extractReach(
   *   'Dashboard for all users',
   *   'Expected to affect 50,000 monthly active users'
   * );
   * // Returns 50000
   * ```
   */
  extractReach(label: string, description?: string): number {
    const text = `${label} ${description || ''}`.toLowerCase();

    // First, try to find explicit numbers with user-related context
    const userPatterns = [
      /(\d+(?:,\d+)*)\s*(users|customers|visitors|signups|accounts)/i,
      /(\d+(?:,\d+)*)\s*(monthly|daily|weekly)\s*(active)?\s*(users)?/i,
      /(\d+(?:,\d+)*)\s*(page\s*views| impressions|traffic)/i,
      /(\d+(?:,\d+)?)\s*(k|m)\s*(users|customers)?/i
    ];

    for (const pattern of userPatterns) {
      const match = pattern.exec(text);
      if (match) {
        return this.parseNumberWithUnit(match[1], match[2]);
      }
    }

    // Try to find explicit numbers
    const numberMatch = this.extractNumber(text);
    if (numberMatch !== null) {
      // Adjust based on context
      if (text.includes('enterprise') || text.includes('business')) {
        return Math.max(numberMatch, 100);
      }
      if (text.includes('admin') || text.includes('internal')) {
        return Math.min(numberMatch, 50);
      }
      return numberMatch;
    }

    // Detect scale from keywords
    const scale = this.detectScale(text, SCALE_KEYWORDS.REACH);
    return DEFAULT_VALUES.REACH[scale];
  }

  /**
   * Extract effort estimate from text.
   *
   * @param label - The label/title to analyze
   * @param description - Optional description for additional context
   * @returns Effort estimate in person-months
   *
   * @example
   * ```typescript
   * const effort = extractor.extractEffort(
   *   'API integration',
   *   'Should take about 3 weeks of development'
   * );
   * // Returns 0.75 (3 weeks ≈ 0.75 person-months)
   * ```
   */
  extractEffort(label: string, description?: string): number {
    const text = `${label} ${description || ''}`.toLowerCase();

    // Look for time-related patterns
    const timePatterns = [
      { pattern: /(\d+(?:\.\d+)?)\s*(week|weeks)/i, toMonths: (w: number) => w / 4 },
      { pattern: /(\d+(?:\.\d+)?)\s*(month|months)/i, toMonths: (m: number) => m },
      { pattern: /(\d+(?:\.\d+)?)\s*(day|days)/i, toMonths: (d: number) => d / 20 },
      { pattern: /(\d+(?:\.\d+)?)\s*(hour|hours)/i, toMonths: (h: number) => h / 160 },
      { pattern: /(\d+(?:\.\d+)?)\s*(sprint|sprints)/i, toMonths: (s: number) => s * 0.5 },
      { pattern: /(\d+(?:\.\d+)?)\s*(quarter|quarters)/i, toMonths: (q: number) => q * 3 }
    ];

    for (const { pattern, toMonths } of timePatterns) {
      const match = pattern.exec(text);
      if (match) {
        const weeks = parseFloat(match[1]);
        return Math.max(0.1, toMonths(weeks));
      }
    }

    // Detect scale from keywords
    const scale = this.detectScale(text, SCALE_KEYWORDS.EFFORT);
    return DEFAULT_VALUES.EFFORT[scale];
  }

  /**
   * Extract impact estimate from text.
   *
   * @param label - The label/title to analyze
   * @param description - Optional description for additional context
   * @returns Impact estimate (multiplier on standard scale)
   *
   * @example
   * ```typescript
   * const impact = extractor.extractImpact(
   *   'Critical security patch',
   *   'Prevents data breach affecting all customers'
   * );
   * // Returns 3 (massive impact)
   * ```
   */
  extractImpact(label: string, description?: string): number {
    const text = `${label} ${description || ''}`.toLowerCase();

    // High impact indicators
    const highImpactPatterns = [
      'critical', 'blocker', 'security', 'revenue', 'conversion',
      'retention', 'core', 'essential', 'fundamental', 'transformative'
    ];

    // Low impact indicators
    const lowImpactPatterns = [
      'nice to have', 'minor', 'cosmetic', 'optional', 'quality of life',
      'polish', 'nice-to-have', 'low priority'
    ];

    // Check for high impact
    for (const pattern of highImpactPatterns) {
      if (text.includes(pattern)) {
        // Determine level based on additional context
        if (['critical', 'blocker', 'security'].includes(pattern)) {
          return 3; // Massive
        }
        if (['revenue', 'conversion', 'retention'].includes(pattern)) {
          return 2; // High
        }
        return 2; // Default high
      }
    }

    // Check for low impact
    for (const pattern of lowImpactPatterns) {
      if (text.includes(pattern)) {
        return 0.25; // Minimal
      }
    }

    // Detect scale from keywords
    const scale = this.detectScale(text, SCALE_KEYWORDS.IMPACT);

    // Map scale to RICE impact scale
    if (scale === 'large') return 2;
    if (scale === 'medium') return 1;
    return 0.5;
  }

  /**
   * Extract confidence estimate from text.
   *
   * @param label - The label/title to analyze
   * @param description - Optional description for additional context
   * @returns Confidence estimate (0-100)
   *
   * @example
   * ```typescript
   * const confidence = extractor.extractConfidence(
   *   'Performance optimization',
   *   'Based on user research with 500 participants'
   * );
   * // Returns 80 (high confidence due to research data)
   * ```
   */
  extractConfidence(label: string, description?: string): number {
    const text = `${label} ${description || ''}`.toLowerCase();

    // High confidence indicators
    const highConfidencePatterns = [
      'data shows', 'research validated', 'proven', 'tested',
      'evidence-based', 'metrics confirm', 'analytics show',
      'user study', 'a/b test', 'experiment results'
    ];

    // Low confidence indicators
    const lowConfidencePatterns = [
      'assumption', 'hypothesis', 'guess', 'estimate',
      'hopefully', 'might', 'could', 'possibly', 'maybe'
    ];

    // Check for high confidence
    let hasHighConfidence = false;
    for (const pattern of highConfidencePatterns) {
      if (text.includes(pattern)) {
        hasHighConfidence = true;
        break;
      }
    }

    if (hasHighConfidence) {
      // Check for data quantity
      const dataMatch = /(\d+)\s*(users|participants|samples|responses|data points)/i.exec(text);
      if (dataMatch) {
        const quantity = parseInt(dataMatch[1], 10);
        if (quantity >= 1000) return 90;
        if (quantity >= 100) return 80;
        if (quantity >= 10) return 70;
        return 60;
      }
      return 75;
    }

    // Check for low confidence
    for (const pattern of lowConfidencePatterns) {
      if (text.includes(pattern)) {
        return 30;
      }
    }

    // Default based on description length (more detail = more confidence)
    if (description && description.length > 100) {
      return 60;
    }
    if (description && description.length > 50) {
      return 50;
    }
    return 40;
  }

  /**
   * Extract value estimate from text.
   *
   * @param label - The label/title to analyze
   * @param description - Optional description for additional context
   * @returns Value estimate (1-10 scale)
   */
  extractValue(label: string, description?: string): number {
    const text = `${label} ${description || ''}`.toLowerCase();

    // High value indicators
    const highValuePatterns = [
      'revenue', 'profit', 'strategic', 'competitive', 'differentiator',
      'market leader', 'growth driver', 'key initiative', 'okr'
    ];

    // Low value indicators
    const lowValuePatterns = [
      'internal only', 'edge case', 'rare', 'technical debt',
      'refactoring', 'maintenance', 'cleanup'
    ];

    for (const pattern of highValuePatterns) {
      if (text.includes(pattern)) {
        return 8;
      }
    }

    for (const pattern of lowValuePatterns) {
      if (text.includes(pattern)) {
        return 4;
      }
    }

    const scale = this.detectScale(text, SCALE_KEYWORDS.VALUE);
    return DEFAULT_VALUES.VALUE[scale];
  }

  /**
   * Extract ease estimate from text.
   *
   * @param label - The label/title to analyze
   * @param description - Optional description for additional context
   * @returns Ease estimate (1-10 scale, higher = easier)
   */
  extractEase(label: string, description?: string): number {
    const text = `${label} ${description || ''}`.toLowerCase();

    // Easy indicators
    const easyPatterns = [
      'simple', 'easy', 'straightforward', 'quick win', 'trivial',
      'no-brainer', 'low hanging fruit', 'config change'
    ];

    // Hard indicators
    const hardPatterns = [
      'complex', 'difficult', 'challenging', 'requires expertise',
      'cross-team', 'dependency', 'legacy system', 'migration'
    ];

    for (const pattern of easyPatterns) {
      if (text.includes(pattern)) {
        return 8;
      }
    }

    for (const pattern of hardPatterns) {
      if (text.includes(pattern)) {
        return 3;
      }
    }

    // Inverse of effort
    const effort = this.extractEffort(label, description);
    if (effort <= 0.5) return 8;
    if (effort <= 1) return 6;
    if (effort <= 2) return 4;
    return 2;
  }

  /**
   * Extract urgency level from text.
   *
   * @param label - The label/title to analyze
   * @param description - Optional description for additional context
   * @returns Urgency level (1-10 scale)
   */
  extractUrgency(label: string, description?: string): number {
    const text = `${label} ${description || ''}`.toLowerCase();

    // High urgency indicators
    const highUrgencyPatterns = [
      'urgent', 'asap', 'emergency', 'critical', 'immediate',
      'today', 'this week', 'deadline', 'escalated', 'hotfix',
      'production issue', 'outage', 'incident', 'blocking'
    ];

    // Time-related patterns with specific values
    const timePatterns = [
      { pattern: /today|eod|end of day/i, value: 10 },
      { pattern: /tomorrow/i, value: 9 },
      { pattern: /this week/i, value: 8 },
      { pattern: /next week/i, value: 7 },
      { pattern: /this sprint/i, value: 7 },
      { pattern: /next sprint/i, value: 6 },
      { pattern: /this month/i, value: 5 },
      { pattern: /next month/i, value: 4 },
      { pattern: /this quarter/i, value: 3 },
      { pattern: /next quarter/i, value: 2 }
    ];

    for (const { pattern, value } of timePatterns) {
      if (pattern.test(text)) {
        return value;
      }
    }

    for (const pattern of highUrgencyPatterns) {
      if (text.includes(pattern)) {
        return 9;
      }
    }

    // Low urgency indicators
    const lowUrgencyPatterns = [
      'backlog', 'someday', 'icebox', 'when time permits',
      'future', 'later', 'eventually'
    ];

    for (const pattern of lowUrgencyPatterns) {
      if (text.includes(pattern)) {
        return 2;
      }
    }

    return 5; // Default medium urgency
  }

  /**
   * Extract importance level from text.
   *
   * @param label - The label/title to analyze
   * @param description - Optional description for additional context
   * @returns Importance level (1-10 scale)
   */
  extractImportance(label: string, description?: string): number {
    const text = `${label} ${description || ''}`.toLowerCase();

    // High importance indicators
    const highImportancePatterns = [
      'strategic', 'goals', 'objectives', 'roadmap', 'quarterly',
      'okrs', 'key results', 'initiative', 'priority', 'important',
      'business critical', 'mission critical', 'core'
    ];

    // Low importance indicators
    const lowImportancePatterns = [
      'nice to have', 'optional', 'enhancement', 'quality of life',
      'polish', 'cosmetic', 'minor'
    ];

    for (const pattern of highImportancePatterns) {
      if (text.includes(pattern)) {
        return 8;
      }
    }

    for (const pattern of lowImportancePatterns) {
      if (text.includes(pattern)) {
        return 3;
      }
    }

    const scale = this.detectScale(text, SCALE_KEYWORDS.IMPORTANCE);
    return DEFAULT_VALUES.IMPORTANCE[scale];
  }

  /**
   * Extract all metrics at once.
   *
   * @param label - The label/title to analyze
   * @param description - Optional description for additional context
   * @returns Extracted metrics object
   */
  extractAll(label: string, description?: string): ExtractedMetrics {
    return {
      reach: this.extractReach(label, description),
      impact: this.extractImpact(label, description),
      confidence: this.extractConfidence(label, description),
      effort: this.extractEffort(label, description),
      value: this.extractValue(label, description),
      ease: this.extractEase(label, description),
      urgency: this.extractUrgency(label, description),
      importance: this.extractImportance(label, description),
      extractionConfidence: this.calculateExtractionConfidence(label, description)
    };
  }

  /**
   * Extract a number from text.
   *
   * @param text - Text to extract number from
   * @returns Extracted number or null
   */
  private extractNumber(text: string): number | null {
    // Try explicit number pattern
    const match = NUMBER_PATTERNS.EXPLICIT_NUMBER.exec(text);
    if (match) {
      return this.parseNumber(match[1]);
    }

    // Try large number pattern
    const largeMatch = NUMBER_PATTERNS.LARGE_NUMBER.exec(text);
    if (largeMatch) {
      return this.parseNumberWithUnit(largeMatch[1], largeMatch[2]);
    }

    return null;
  }

  /**
   * Detect scale from text using keywords.
   *
   * @param text - Text to analyze
   * @param keywords - Keywords for each scale level
   * @returns Detected scale
   */
  detectScale(text: string, keywords: Record<DetectedScale, string[]>): DetectedScale {
    let largeMatches = 0;
    let mediumMatches = 0;
    let smallMatches = 0;

    for (const keyword of keywords.large) {
      if (text.includes(keyword)) largeMatches++;
    }

    for (const keyword of keywords.medium) {
      if (text.includes(keyword)) mediumMatches++;
    }

    for (const keyword of keywords.small) {
      if (text.includes(keyword)) smallMatches++;
    }

    if (largeMatches > mediumMatches && largeMatches > smallMatches) {
      return 'large';
    }
    if (smallMatches > mediumMatches) {
      return 'small';
    }
    return 'medium';
  }

  /**
   * Parse a number string to numeric value.
   */
  private parseNumber(numStr: string): number {
    const num = parseFloat(numStr.replace(/,/g, ''));
    return isNaN(num) ? 0 : num;
  }

  /**
   * Parse a number with unit to numeric value.
   */
  private parseNumberWithUnit(numStr: string, unit?: string): number {
    const num = this.parseNumber(numStr);
    const unitLower = (unit || '').toLowerCase();

    // Handle K/M/B suffixes
    if (unitLower === 'k' || unitLower === 'thousand') {
      return num * 1000;
    }
    if (unitLower === 'm' || unitLower === 'million') {
      return num * 1000000;
    }
    if (unitLower === 'b' || unitLower === 'billion') {
      return num * 1000000000;
    }

    return num;
  }

  /**
   * Calculate confidence in the extraction.
   */
  private calculateExtractionConfidence(label: string, description?: string): number {
    const text = `${label} ${description || ''}`.toLowerCase();

    // Higher confidence with more text
    let confidence = 0.3;
    if (text.length > 100) confidence = 0.5;
    if (text.length > 200) confidence = 0.7;

    // Higher confidence with explicit numbers
    if (NUMBER_PATTERNS.EXPLICIT_NUMBER.test(text)) {
      confidence = Math.min(confidence + 0.3, 1.0);
    }

    // Higher confidence with specific patterns
    if (/(users|customers|weeks|months|%)/i.test(text)) {
      confidence = Math.min(confidence + 0.2, 1.0);
    }

    return confidence;
  }
}

export default MetricsExtractor;
