/**
 * Label Analyzer for Prioritization Frameworks.
 *
 * This module analyzes item labels, titles, and descriptions to detect
 * patterns that suggest framework-specific parameter values.
 * Uses keyword matching and pattern recognition for auto-fill suggestions.
 *
 * @module prioritization/auto-fill/LabelAnalyzer
 */

import { FrameworkType } from '../types';

/**
 * Pattern match result from label analysis.
 */
export interface PatternMatch {
  /** Pattern category (e.g., 'REACH', 'IMPACT') */
  category: string;
  /** Matched keyword or phrase */
  keyword: string;
  /** Suggested value based on match */
  suggestedValue: number | string | boolean;
  /** Confidence score (0-1) */
  confidence: number;
}

/**
 * Urgency level detected from label analysis.
 */
export type UrgencyLevel = 'high' | 'medium' | 'low';

/**
 * Business value level detected from label analysis.
 */
export type BusinessValueLevel = 'critical' | 'high' | 'medium' | 'low';

/**
 * Keyword patterns for different frameworks and parameter types.
 */
export interface KeywordPatterns {
  /** RICE framework patterns */
  RICE: {
    /** Keywords suggesting reach (user impact scope) */
    REACH_KEYWORDS: string[];
    /** Keywords suggesting impact level */
    IMPACT_KEYWORDS: string[];
    /** Keywords suggesting confidence level */
    CONFIDENCE_KEYWORDS: string[];
    /** Keywords suggesting effort required */
    EFFORT_KEYWORDS: string[];
  };
  /** MoSCoW framework patterns */
  MoSCoW: {
    /** Must have indicators */
    MUST_KEYWORDS: string[];
    /** Should have indicators */
    SHOULD_KEYWORDS: string[];
    /** Could have indicators */
    COULD_KEYWORDS: string[];
    /** Won't have indicators */
    WONT_KEYWORDS: string[];
  };
  /** Value vs Effort patterns */
  ValueEffort: {
    /** High value indicators */
    HIGH_VALUE_KEYWORDS: string[];
    /** Low value indicators */
    LOW_VALUE_KEYWORDS: string[];
    /** High effort indicators */
    HIGH_EFFORT_KEYWORDS: string[];
    /** Low effort indicators */
    LOW_EFFORT_KEYWORDS: string[];
  };
  /** Eisenhower Matrix patterns */
  Eisenhower: {
    /** Urgent indicators */
    URGENT_KEYWORDS: string[];
    /** Important indicators */
    IMPORTANT_KEYWORDS: string[];
  };
  /** P0-P4 Priority patterns */
  P0P4: {
    /** Critical/P0 indicators */
    P0_KEYWORDS: string[];
    /** High/P1 indicators */
    P1_KEYWORDS: string[];
    /** Medium/P2 indicators */
    P2_KEYWORDS: string[];
    /** Low/P3-P4 indicators */
    LOW_PRIORITY_KEYWORDS: string[];
  };
  /** WSJF patterns */
  WSJF: {
    /** High business value indicators */
    BUSINESS_VALUE_KEYWORDS: string[];
    /** Time criticality indicators */
    TIME_CRITICAL_KEYWORDS: string[];
    /** Risk reduction indicators */
    RISK_REDUCTION_KEYWORDS: string[];
    /** Job size indicators */
    JOB_SIZE_KEYWORDS: string[];
  };
  /** Kano model patterns */
  Kano: {
    /** Must-be (basic expectation) indicators */
    MUST_BE_KEYWORDS: string[];
    /** One-dimensional (performance) indicators */
    ONE_DIMENSIONAL_KEYWORDS: string[];
    /** Attractive (delighter) indicators */
    ATTRACTIVE_KEYWORDS: string[];
  };
}

/**
 * Default keyword patterns for label analysis.
 */
const DEFAULT_PATTERNS: KeywordPatterns = {
  RICE: {
    REACH_KEYWORDS: [
      'all users',
      'every user',
      'users',
      'customers',
      'page views',
      'signups',
      'visitors',
      'audience',
      'traffic',
      'enterprise',
      'business customers',
      'admin',
      'internal',
      'team',
      'some users',
      'select users'
    ],
    IMPACT_KEYWORDS: [
      'conversion',
      'revenue',
      'retention',
      'engagement',
      'critical',
      'blocker',
      'security',
      'compliance',
      'core',
      'important',
      'improve',
      'enhance',
      'optimize',
      'performance',
      'nice to have',
      'minor',
      'cosmetic',
      'optional'
    ],
    CONFIDENCE_KEYWORDS: [
      'data',
      'research',
      'validated',
      'proven',
      'tested',
      'evidence',
      'metrics',
      'analytics',
      'study',
      'survey',
      'feedback',
      'requested',
      'assumption',
      'hypothesis',
      'guess'
    ],
    EFFORT_KEYWORDS: [
      'weeks',
      'months',
      'engineering',
      'development',
      'migration',
      'refactor',
      'infrastructure',
      'platform',
      'integration',
      'rewrite',
      'feature',
      'implement',
      'module',
      'component',
      'fix',
      'update',
      'tweak',
      'adjust',
      'quick',
      'simple'
    ]
  },
  MoSCoW: {
    MUST_KEYWORDS: [
      'compliance',
      'legal',
      'requirement',
      'critical',
      'mandatory',
      'must have',
      'required',
      'essential',
      'blocker',
      'showstopper',
      'regulatory',
      'security fix',
      'core functionality'
    ],
    SHOULD_KEYWORDS: [
      'important',
      'needed',
      'required',
      'high priority',
      'should have',
      'priority',
      'business critical',
      'customer requested'
    ],
    COULD_KEYWORDS: [
      'nice to have',
      'optional',
      'enhancement',
      'could have',
      'desirable',
      'if time permits',
      'bonus',
      'quality of life'
    ],
    WONT_KEYWORDS: [
      'won\'t have',
      'not now',
      'later',
      'future',
      'backlog',
      'deprioritized',
      'rejected'
    ]
  },
  ValueEffort: {
    HIGH_VALUE_KEYWORDS: [
      'revenue',
      'profit',
      'conversion',
      'retention',
      'acquisition',
      'strategic',
      'competitive',
      'differentiator',
      'market',
      'growth',
      'scale'
    ],
    LOW_VALUE_KEYWORDS: [
      'internal only',
      'edge case',
      'rare',
      'minor improvement',
      'cosmetic',
      'nice to have'
    ],
    HIGH_EFFORT_KEYWORDS: [
      'migration',
      'rewrite',
      'rebuild',
      'infrastructure',
      'platform',
      'complex',
      'major',
      'quarter-long',
      'multi-team'
    ],
    LOW_EFFORT_KEYWORDS: [
      'quick win',
      'simple',
      'easy',
      'minor',
      'tweak',
      'adjust',
      'config',
      'one-liner',
      'day'
    ]
  },
  Eisenhower: {
    URGENT_KEYWORDS: [
      'urgent',
      'asap',
      'emergency',
      'critical',
      'immediate',
      'today',
      'this week',
      'deadline',
      'escalated',
      'hotfix',
      'production issue',
      'outage',
      'incident'
    ],
    IMPORTANT_KEYWORDS: [
      'important',
      'strategic',
      'goals',
      'objectives',
      'roadmap',
      'quarterly',
      'okrs',
      'key results',
      'initiative',
      'priority'
    ]
  },
  P0P4: {
    P0_KEYWORDS: [
      'p0',
      'critical',
      'emergency',
      'outage',
      'down',
      'broken',
      'blocking',
      'showstopper',
      'production issue',
      'security vulnerability',
      'data loss'
    ],
    P1_KEYWORDS: [
      'p1',
      'high priority',
      'blocker',
      'major issue',
      'escalated',
      'vip',
      'enterprise blocker'
    ],
    P2_KEYWORDS: [
      'p2',
      'normal',
      'standard',
      'regular',
      'backlog',
      'planned'
    ],
    LOW_PRIORITY_KEYWORDS: [
      'p3',
      'p4',
      'low priority',
      'nice to have',
      'someday',
      'icebox',
      'backlog item'
    ]
  },
  WSJF: {
    BUSINESS_VALUE_KEYWORDS: [
      'revenue',
      'profit',
      'business value',
      'roi',
      'return on investment',
      'customer value',
      'user value',
      'market',
      'competitive advantage'
    ],
    TIME_CRITICAL_KEYWORDS: [
      'time critical',
      'deadline',
      'window',
      'seasonal',
      'event-driven',
      'regulatory deadline',
      'contract deadline',
      'first mover'
    ],
    RISK_REDUCTION_KEYWORDS: [
      'risk reduction',
      'mitigate risk',
      'security',
      'compliance',
      'technical debt',
      'stability',
      'reliability',
      'opportunity enablement'
    ],
    JOB_SIZE_KEYWORDS: [
      'large',
      'complex',
      'simple',
      'small',
      'epic',
      'story',
      'spike',
      'exploration'
    ]
  },
  Kano: {
    MUST_BE_KEYWORDS: [
      'basic',
      'expected',
      'standard',
      'minimum',
      'table stakes',
      'fundamental',
      'core',
      'prerequisite'
    ],
    ONE_DIMENSIONAL_KEYWORDS: [
      'faster',
      'better',
      'more',
      'improved',
      'enhanced',
      'optimized',
      'performance',
      'efficiency'
    ],
    ATTRACTIVE_KEYWORDS: [
      'delighter',
      'wow',
      'innovative',
      'surprise',
      'exciting',
      'novel',
      'first of kind',
      'unique'
    ]
  }
};

/**
 * LabelAnalyzer class for detecting patterns in prioritization item labels.
 *
 * Analyzes text to identify keywords that suggest framework-specific
 * parameter values, urgency levels, and business value assessments.
 *
 * @example
 * ```typescript
 * const analyzer = new LabelAnalyzer();
 *
 * const matches = analyzer.analyzeLabel('Critical security fix for all users');
 * // Returns pattern matches for urgency, impact, and reach
 *
 * const urgency = analyzer.detectUrgency('ASAP: Production outage');
 * // Returns 'high'
 *
 * const businessValue = analyzer.detectBusinessValue('Revenue-generating feature');
 * // Returns 'critical'
 * ```
 */
export class LabelAnalyzer {
  private readonly patterns: KeywordPatterns;

  /**
   * Creates a new LabelAnalyzer instance.
   * @param patterns - Optional custom patterns to override defaults
   */
  constructor(patterns: Partial<KeywordPatterns> = {}) {
    this.patterns = { ...DEFAULT_PATTERNS, ...patterns };
  }

  /**
   * Analyze a label for pattern matches across all frameworks.
   *
   * @param label - The label/title/description to analyze
   * @returns Array of pattern matches found
   *
   * @example
   * ```typescript
   * const matches = analyzer.analyzeLabel('Quick win: Add CSV export for enterprise users');
   * // Returns matches for reach (enterprise), effort (quick), etc.
   * ```
   */
  analyzeLabel(label: string): PatternMatch[] {
    const matches: PatternMatch[] = [];
    const lowerLabel = label.toLowerCase();

    // Analyze RICE patterns
    matches.push(...this.analyzeRICEPatterns(lowerLabel));

    // Analyze MoSCoW patterns
    matches.push(...this.analyzeMoSCoWPatterns(lowerLabel));

    // Analyze Value/Effort patterns
    matches.push(...this.analyzeValueEffortPatterns(lowerLabel));

    // Analyze Eisenhower patterns
    matches.push(...this.analyzeEisenhowerPatterns(lowerLabel));

    // Analyze P0P4 patterns
    matches.push(...this.analyzeP0P4Patterns(lowerLabel));

    // Analyze WSJF patterns
    matches.push(...this.analyzeWSJFPatterns(lowerLabel));

    // Analyze Kano patterns
    matches.push(...this.analyzeKanoPatterns(lowerLabel));

    return matches;
  }

  /**
   * Detect urgency level from a label.
   *
   * @param label - The label to analyze
   * @returns Urgency level: 'high', 'medium', or 'low'
   *
   * @example
   * ```typescript
   * const urgency = analyzer.detectUrgency('Critical: API down');
   * // Returns 'high'
   * ```
   */
  detectUrgency(label: string): UrgencyLevel {
    const lowerLabel = label.toLowerCase();
    const urgentMatches = this.countMatches(
      lowerLabel,
      this.patterns.Eisenhower.URGENT_KEYWORDS
    );
    const p0Matches = this.countMatches(
      lowerLabel,
      this.patterns.P0P4.P0_KEYWORDS
    );
    const p1Matches = this.countMatches(
      lowerLabel,
      this.patterns.P0P4.P1_KEYWORDS
    );

    const urgentScore = urgentMatches + p0Matches * 2 + p1Matches;

    if (urgentScore >= 2) {
      return 'high';
    } else if (urgentScore >= 1) {
      return 'medium';
    }
    return 'low';
  }

  /**
   * Detect business value level from a label.
   *
   * @param label - The label to analyze
   * @returns Business value level: 'critical', 'high', 'medium', or 'low'
   *
   * @example
   * ```typescript
   * const value = analyzer.detectBusinessValue('Revenue-critical feature for enterprise');
   * // Returns 'critical'
   * ```
   */
  detectBusinessValue(label: string): BusinessValueLevel {
    const lowerLabel = label.toLowerCase();
    const highValueMatches = this.countMatches(
      lowerLabel,
      this.patterns.ValueEffort.HIGH_VALUE_KEYWORDS
    );
    const lowValueMatches = this.countMatches(
      lowerLabel,
      this.patterns.ValueEffort.LOW_VALUE_KEYWORDS
    );
    const mustHaveMatches = this.countMatches(
      lowerLabel,
      this.patterns.MoSCoW.MUST_KEYWORDS
    );

    const valueScore = highValueMatches + mustHaveMatches * 2 - lowValueMatches;

    if (valueScore >= 3) {
      return 'critical';
    } else if (valueScore >= 1) {
      return 'high';
    } else if (valueScore >= -1) {
      return 'medium';
    }
    return 'low';
  }

  /**
   * Get pattern matches specific to a framework.
   *
   * @param label - The label to analyze
   * @param framework - The framework to analyze for
   * @returns Array of pattern matches for the specified framework
   */
  getFrameworkMatches(label: string, framework: FrameworkType): PatternMatch[] {
    const lowerLabel = label.toLowerCase();

    switch (framework) {
      case 'RICE':
        return this.analyzeRICEPatterns(lowerLabel);
      case 'MoSCoW':
        return this.analyzeMoSCoWPatterns(lowerLabel);
      case 'ValueEffort':
        return this.analyzeValueEffortPatterns(lowerLabel);
      case 'Eisenhower':
        return this.analyzeEisenhowerPatterns(lowerLabel);
      case 'P0P4':
        return this.analyzeP0P4Patterns(lowerLabel);
      case 'WSJF':
        return this.analyzeWSJFPatterns(lowerLabel);
      case 'Kano':
        return this.analyzeKanoPatterns(lowerLabel);
      case 'ICE':
        // ICE uses similar patterns to RICE
        return this.analyzeRICEPatterns(lowerLabel).filter(
          m => ['IMPACT', 'CONFIDENCE', 'EFFORT'].includes(m.category)
        );
      default:
        return [];
    }
  }

  /**
   * Count keyword matches in text.
   */
  private countMatches(text: string, keywords: string[]): number {
    return keywords.filter(keyword => text.includes(keyword)).length;
  }

  /**
   * Find the best matching keyword in text.
   */
  private findBestMatch(
    text: string,
    keywords: string[]
  ): { keyword: string; confidence: number } | null {
    for (const keyword of keywords) {
      if (text.includes(keyword)) {
        return {
          keyword,
          confidence: 0.8 // Base confidence for direct match
        };
      }
    }
    return null;
  }

  /**
   * Analyze RICE-specific patterns.
   */
  private analyzeRICEPatterns(label: string): PatternMatch[] {
    const matches: PatternMatch[] = [];

    // Reach patterns
    const reachMatch = this.findBestMatch(
      label,
      this.patterns.RICE.REACH_KEYWORDS
    );
    if (reachMatch) {
      let suggestedReach = 500; // Default
      if (['all users', 'every user', 'users'].includes(reachMatch.keyword)) {
        suggestedReach = 1000;
      } else if (
        ['enterprise', 'business customers'].includes(reachMatch.keyword)
      ) {
        suggestedReach = 100;
      } else if (['admin', 'internal'].includes(reachMatch.keyword)) {
        suggestedReach = 20;
      }
      matches.push({
        category: 'REACH',
        keyword: reachMatch.keyword,
        suggestedValue: suggestedReach,
        confidence: reachMatch.confidence
      });
    }

    // Impact patterns
    const impactMatch = this.findBestMatch(
      label,
      this.patterns.RICE.IMPACT_KEYWORDS
    );
    if (impactMatch) {
      let suggestedImpact = 1; // Default medium
      if (['critical', 'blocker', 'revenue', 'security'].includes(impactMatch.keyword)) {
        suggestedImpact = 3; // Massive
      } else if (['conversion', 'retention', 'core'].includes(impactMatch.keyword)) {
        suggestedImpact = 2; // High
      } else if (['nice to have', 'minor', 'cosmetic'].includes(impactMatch.keyword)) {
        suggestedImpact = 0.25; // Minimal
      }
      matches.push({
        category: 'IMPACT',
        keyword: impactMatch.keyword,
        suggestedValue: suggestedImpact,
        confidence: impactMatch.confidence
      });
    }

    // Confidence patterns
    const confidenceMatch = this.findBestMatch(
      label,
      this.patterns.RICE.CONFIDENCE_KEYWORDS
    );
    if (confidenceMatch) {
      let suggestedConfidence = 50; // Default
      if (['data', 'research', 'validated', 'proven', 'metrics'].includes(confidenceMatch.keyword)) {
        suggestedConfidence = 80;
      } else if (['tested', 'evidence'].includes(confidenceMatch.keyword)) {
        suggestedConfidence = 70;
      } else if (['assumption', 'hypothesis', 'guess'].includes(confidenceMatch.keyword)) {
        suggestedConfidence = 30;
      }
      matches.push({
        category: 'CONFIDENCE',
        keyword: confidenceMatch.keyword,
        suggestedValue: suggestedConfidence,
        confidence: confidenceMatch.confidence
      });
    }

    // Effort patterns
    const effortMatch = this.findBestMatch(
      label,
      this.patterns.RICE.EFFORT_KEYWORDS
    );
    if (effortMatch) {
      let suggestedEffort = 2; // Default
      if (['migration', 'refactor', 'infrastructure', 'rewrite'].includes(effortMatch.keyword)) {
        suggestedEffort = 4;
      } else if (['feature', 'implement', 'module'].includes(effortMatch.keyword)) {
        suggestedEffort = 2;
      } else if (['fix', 'update', 'quick', 'simple'].includes(effortMatch.keyword)) {
        suggestedEffort = 0.5;
      }
      matches.push({
        category: 'EFFORT',
        keyword: effortMatch.keyword,
        suggestedValue: suggestedEffort,
        confidence: effortMatch.confidence
      });
    }

    return matches;
  }

  /**
   * Analyze MoSCoW-specific patterns.
   */
  private analyzeMoSCoWPatterns(label: string): PatternMatch[] {
    const matches: PatternMatch[] = [];

    // Check each MoSCoW category
    const mustMatch = this.findBestMatch(
      label,
      this.patterns.MoSCoW.MUST_KEYWORDS
    );
    if (mustMatch) {
      matches.push({
        category: 'MOSCOW',
        keyword: mustMatch.keyword,
        suggestedValue: 'Must have',
        confidence: mustMatch.confidence
      });
    }

    const shouldMatch = this.findBestMatch(
      label,
      this.patterns.MoSCoW.SHOULD_KEYWORDS
    );
    if (shouldMatch && !mustMatch) {
      matches.push({
        category: 'MOSCOW',
        keyword: shouldMatch.keyword,
        suggestedValue: 'Should have',
        confidence: shouldMatch.confidence * 0.9
      });
    }

    const couldMatch = this.findBestMatch(
      label,
      this.patterns.MoSCoW.COULD_KEYWORDS
    );
    if (couldMatch && !mustMatch && !shouldMatch) {
      matches.push({
        category: 'MOSCOW',
        keyword: couldMatch.keyword,
        suggestedValue: 'Could have',
        confidence: couldMatch.confidence * 0.8
      });
    }

    return matches;
  }

  /**
   * Analyze Value/Effort-specific patterns.
   */
  private analyzeValueEffortPatterns(label: string): PatternMatch[] {
    const matches: PatternMatch[] = [];

    // Value patterns
    const highValueMatch = this.findBestMatch(
      label,
      this.patterns.ValueEffort.HIGH_VALUE_KEYWORDS
    );
    const lowValueMatch = this.findBestMatch(
      label,
      this.patterns.ValueEffort.LOW_VALUE_KEYWORDS
    );

    if (highValueMatch) {
      matches.push({
        category: 'VALUE',
        keyword: highValueMatch.keyword,
        suggestedValue: 8,
        confidence: highValueMatch.confidence
      });
    } else if (lowValueMatch) {
      matches.push({
        category: 'VALUE',
        keyword: lowValueMatch.keyword,
        suggestedValue: 3,
        confidence: lowValueMatch.confidence
      });
    }

    // Effort patterns
    const highEffortMatch = this.findBestMatch(
      label,
      this.patterns.ValueEffort.HIGH_EFFORT_KEYWORDS
    );
    const lowEffortMatch = this.findBestMatch(
      label,
      this.patterns.ValueEffort.LOW_EFFORT_KEYWORDS
    );

    if (highEffortMatch) {
      matches.push({
        category: 'EFFORT',
        keyword: highEffortMatch.keyword,
        suggestedValue: 8,
        confidence: highEffortMatch.confidence
      });
    } else if (lowEffortMatch) {
      matches.push({
        category: 'EFFORT',
        keyword: lowEffortMatch.keyword,
        suggestedValue: 2,
        confidence: lowEffortMatch.confidence
      });
    }

    return matches;
  }

  /**
   * Analyze Eisenhower-specific patterns.
   */
  private analyzeEisenhowerPatterns(label: string): PatternMatch[] {
    const matches: PatternMatch[] = [];

    // Urgent patterns
    const urgentMatch = this.findBestMatch(
      label,
      this.patterns.Eisenhower.URGENT_KEYWORDS
    );
    if (urgentMatch) {
      matches.push({
        category: 'EISENHOWER',
        keyword: urgentMatch.keyword,
        suggestedValue: true,
        confidence: urgentMatch.confidence
      });
    }

    // Important patterns
    const importantMatch = this.findBestMatch(
      label,
      this.patterns.Eisenhower.IMPORTANT_KEYWORDS
    );
    if (importantMatch) {
      matches.push({
        category: 'EISENHOWER',
        keyword: importantMatch.keyword,
        suggestedValue: true,
        confidence: importantMatch.confidence
      });
    }

    return matches;
  }

  /**
   * Analyze P0P4-specific patterns.
   */
  private analyzeP0P4Patterns(label: string): PatternMatch[] {
    const matches: PatternMatch[] = [];

    // Check priority levels in order
    const p0Match = this.findBestMatch(label, this.patterns.P0P4.P0_KEYWORDS);
    if (p0Match) {
      matches.push({
        category: 'P0P4',
        keyword: p0Match.keyword,
        suggestedValue: 'P0',
        confidence: p0Match.confidence
      });
      return matches;
    }

    const p1Match = this.findBestMatch(label, this.patterns.P0P4.P1_KEYWORDS);
    if (p1Match) {
      matches.push({
        category: 'P0P4',
        keyword: p1Match.keyword,
        suggestedValue: 'P1',
        confidence: p1Match.confidence
      });
      return matches;
    }

    const lowPriorityMatch = this.findBestMatch(
      label,
      this.patterns.P0P4.LOW_PRIORITY_KEYWORDS
    );
    if (lowPriorityMatch) {
      matches.push({
        category: 'P0P4',
        keyword: lowPriorityMatch.keyword,
        suggestedValue: 'P3',
        confidence: lowPriorityMatch.confidence
      });
    }

    return matches;
  }

  /**
   * Analyze WSJF-specific patterns.
   */
  private analyzeWSJFPatterns(label: string): PatternMatch[] {
    const matches: PatternMatch[] = [];

    // Business value
    const bvMatch = this.findBestMatch(
      label,
      this.patterns.WSJF.BUSINESS_VALUE_KEYWORDS
    );
    if (bvMatch) {
      matches.push({
        category: 'WSJF_BUSINESS_VALUE',
        keyword: bvMatch.keyword,
        suggestedValue: 8,
        confidence: bvMatch.confidence
      });
    }

    // Time criticality
    const tcMatch = this.findBestMatch(
      label,
      this.patterns.WSJF.TIME_CRITICAL_KEYWORDS
    );
    if (tcMatch) {
      matches.push({
        category: 'WSJF_TIME_CRITICAL',
        keyword: tcMatch.keyword,
        suggestedValue: 8,
        confidence: tcMatch.confidence
      });
    }

    // Risk reduction
    const rrMatch = this.findBestMatch(
      label,
      this.patterns.WSJF.RISK_REDUCTION_KEYWORDS
    );
    if (rrMatch) {
      matches.push({
        category: 'WSJF_RISK_REDUCTION',
        keyword: rrMatch.keyword,
        suggestedValue: 7,
        confidence: rrMatch.confidence
      });
    }

    return matches;
  }

  /**
   * Analyze Kano-specific patterns.
   */
  private analyzeKanoPatterns(label: string): PatternMatch[] {
    const matches: PatternMatch[] = [];

    // Must-be patterns
    const mustBeMatch = this.findBestMatch(
      label,
      this.patterns.Kano.MUST_BE_KEYWORDS
    );
    if (mustBeMatch) {
      matches.push({
        category: 'KANO',
        keyword: mustBeMatch.keyword,
        suggestedValue: 'MustBe',
        confidence: mustBeMatch.confidence
      });
      return matches;
    }

    // One-dimensional patterns
    const odMatch = this.findBestMatch(
      label,
      this.patterns.Kano.ONE_DIMENSIONAL_KEYWORDS
    );
    if (odMatch) {
      matches.push({
        category: 'KANO',
        keyword: odMatch.keyword,
        suggestedValue: 'OneDimensional',
        confidence: odMatch.confidence
      });
      return matches;
    }

    // Attractive patterns
    const attrMatch = this.findBestMatch(
      label,
      this.patterns.Kano.ATTRACTIVE_KEYWORDS
    );
    if (attrMatch) {
      matches.push({
        category: 'KANO',
        keyword: attrMatch.keyword,
        suggestedValue: 'Attractive',
        confidence: attrMatch.confidence
      });
    }

    return matches;
  }
}

export default LabelAnalyzer;
