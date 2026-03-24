/**
 * Auto-Fill Engine for Prioritization Frameworks.
 *
 * This module provides intelligent auto-fill suggestions for prioritization
 * parameters by analyzing item labels, descriptions, and historical data.
 * Combines label analysis and metrics extraction to provide confident suggestions.
 *
 * @module prioritization/auto-fill/AutoFillEngine
 */

import { FrameworkType, PrioritizationItem } from '../types';
import { LabelAnalyzer, PatternMatch } from './LabelAnalyzer';
import { MetricsExtractor, ExtractedMetrics } from './MetricsExtractor';

/**
 * Context for auto-fill analysis.
 */
export interface AutoFillContext {
  /** Project name for contextual analysis */
  projectName: string;
  /** Project description for additional context */
  projectDescription?: string;
  /** Team name for domain-specific suggestions */
  teamName?: string;
  /** Historical data for pattern learning */
  historicalData?: PrioritizationItem[];
  /** Industry context (e.g., 'fintech', 'healthcare', 'e-commerce') */
  industry?: string;
}

/**
 * Framework suggestion with confidence score.
 */
export interface FrameworkSuggestion {
  /** Field being suggested (e.g., 'reach', 'impact') */
  field: string;
  /** Suggested value */
  suggestedValue: number | string | boolean;
  /** Confidence score (0-1) */
  confidence: number;
  /** Source of suggestion ('pattern' | 'metrics' | 'historical') */
  source: 'pattern' | 'metrics' | 'historical';
  /** Reason/explanation for suggestion */
  reason: string;
}

/**
 * Collection of suggestions for a framework.
 */
export interface FrameworkSuggestions {
  /** Framework type */
  framework: FrameworkType;
  /** Array of individual suggestions */
  suggestions: FrameworkSuggestion[];
  /** Overall confidence across all suggestions (0-1) */
  overallConfidence: number;
  /** Item ID these suggestions are for */
  itemId?: string;
}

/**
 * Analysis result for a prioritization item.
 */
export interface AnalysisResult {
  /** Original item analyzed */
  item: PrioritizationItem;
  /** Generated suggestions */
  suggestions: FrameworkSuggestions;
  /** Time taken for analysis in milliseconds */
  analysisTimeMs: number;
}

/**
 * Default configuration for auto-fill engine.
 */
const DEFAULT_CONFIG = {
  /** Minimum confidence threshold for suggestions (0-1) */
  minConfidence: 0.4,
  /** Maximum time allowed for analysis in ms */
  maxAnalysisTimeMs: 100,
  /** Weight for pattern-based suggestions */
  patternWeight: 0.5,
  /** Weight for metrics-based suggestions */
  metricsWeight: 0.3,
  /** Weight for historical-based suggestions */
  historicalWeight: 0.2
};

/**
 * Type configuration for confidence adjustment.
 */
interface TypeConfig {
  reachMultiplier: number;
  impactMultiplier: number;
  confidenceBonus: number;
  effortBaseline: number;
}

/**
 * Industry-specific configurations.
 */
const INDUSTRY_CONFIGS: Record<string, TypeConfig> = {
  fintech: {
    reachMultiplier: 0.8,
    impactMultiplier: 1.2,
    confidenceBonus: 0.1,
    effortBaseline: 1.5
  },
  healthcare: {
    reachMultiplier: 0.7,
    impactMultiplier: 1.3,
    confidenceBonus: 0.15,
    effortBaseline: 1.8
  },
  ecommerce: {
    reachMultiplier: 1.2,
    impactMultiplier: 1.1,
    confidenceBonus: 0.05,
    effortBaseline: 1.2
  },
  enterprise: {
    reachMultiplier: 0.9,
    impactMultiplier: 1.0,
    confidenceBonus: 0.1,
    effortBaseline: 2.0
  },
  startup: {
    reachMultiplier: 1.1,
    impactMultiplier: 0.9,
    confidenceBonus: -0.1,
    effortBaseline: 0.8
  }
};

/**
 * AutoFillEngine class for generating prioritization suggestions.
 *
 * Analyzes items using label patterns, metrics extraction, and historical
 * data to provide intelligent auto-fill suggestions for framework parameters.
 *
 * @example
 * ```typescript
 * const engine = new AutoFillEngine();
 *
 * engine.setContext({
 *   projectName: 'E-commerce Platform',
 *   industry: 'ecommerce',
 *   teamName: 'Checkout Team'
 * });
 *
 * const suggestions = engine.analyzeItem({
 *   id: 'feat-1',
 *   title: 'Add one-click checkout for premium users',
 *   description: 'Streamline checkout process',
 *   createdAt: new Date()
 * });
 *
 * // Returns suggestions for RICE parameters with confidence scores
 * ```
 */
export class AutoFillEngine {
  private context: AutoFillContext | null = null;
  private readonly labelAnalyzer: LabelAnalyzer;
  private readonly metricsExtractor: MetricsExtractor;
  private readonly config: typeof DEFAULT_CONFIG;

  /**
   * Creates a new AutoFillEngine instance.
   * @param config - Optional configuration to override defaults
   */
  constructor(config: Partial<typeof DEFAULT_CONFIG> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
    this.labelAnalyzer = new LabelAnalyzer();
    this.metricsExtractor = new MetricsExtractor();
  }

  /**
   * Set the analysis context.
   *
   * @param context - Context object with project and team information
   *
   * @example
   * ```typescript
   * engine.setContext({
   *   projectName: 'Mobile App Redesign',
   *   projectDescription: 'Complete UI overhaul for iOS and Android',
   *   teamName: 'Mobile Platform',
   *   industry: 'fintech'
   * });
   * ```
   */
  setContext(context: AutoFillContext): void {
    this.context = context;
  }

  /**
   * Get the current analysis context.
   *
   * @returns Current context or null if not set
   */
  getContext(): AutoFillContext | null {
    return this.context;
  }

  /**
   * Analyze an item and generate framework suggestions.
   *
   * @param item - The prioritization item to analyze
   * @param framework - Optional specific framework to analyze for
   * @returns Framework suggestions object
   *
   * @example
   * ```typescript
   * const suggestions = engine.analyzeItem(item, 'RICE');
   * console.log(suggestions.suggestions);
   * // Array of { field, suggestedValue, confidence, reason }
   * ```
   */
  analyzeItem(item: PrioritizationItem, framework?: FrameworkType): FrameworkSuggestions {
    const startTime = performance.now();
    const text = `${item.title} ${item.description || ''}`;

    // Get pattern matches from label analyzer
    const patternMatches = framework
      ? this.labelAnalyzer.getFrameworkMatches(text, framework)
      : this.labelAnalyzer.analyzeLabel(text);

    // Get metrics from extractor
    const metrics = this.metricsExtractor.extractAll(item.title, item.description);

    // Generate suggestions based on framework
    let suggestions: FrameworkSuggestion[] = [];

    if (framework === 'RICE' || !framework) {
      suggestions = [
        ...this.generateRICESuggestions(item, patternMatches, metrics),
        ...suggestions
      ];
    }

    if (framework === 'MoSCoW' || !framework) {
      suggestions = [
        ...this.generateMoSCoWSuggestions(item, patternMatches),
        ...suggestions
      ];
    }

    if (framework === 'ValueEffort' || !framework) {
      suggestions = [
        ...this.generateValueEffortSuggestions(item, patternMatches, metrics),
        ...suggestions
      ];
    }

    if (framework === 'Eisenhower' || !framework) {
      suggestions = [
        ...this.generateEisenhowerSuggestions(item, patternMatches),
        ...suggestions
      ];
    }

    if (framework === 'P0P4' || !framework) {
      suggestions = [
        ...this.generateP0P4Suggestions(item, patternMatches),
        ...suggestions
      ];
    }

    // Filter by confidence threshold
    suggestions = suggestions.filter(s => s.confidence >= this.config.minConfidence);

    // Calculate overall confidence
    const overallConfidence = suggestions.length > 0
      ? suggestions.reduce((sum, s) => sum + s.confidence, 0) / suggestions.length
      : 0;

    const analysisTimeMs = performance.now() - startTime;

    // Warn if analysis took too long
    if (analysisTimeMs > this.config.maxAnalysisTimeMs) {
      console.warn(`Auto-fill analysis took ${analysisTimeMs.toFixed(2)}ms (target: ${this.config.maxAnalysisTimeMs}ms)`);
    }

    return {
      framework: framework || 'RICE',
      suggestions,
      overallConfidence,
      itemId: item.id
    };
  }

  /**
   * Get confidence score for a specific suggestion.
   *
   * @param suggestion - The suggestion to evaluate
   * @returns Confidence score (0-1)
   */
  getConfidenceScore(suggestion: FrameworkSuggestion): number {
    let confidence = suggestion.confidence;

    // Adjust based on source
    if (suggestion.source === 'historical' && this.context?.historicalData) {
      // Boost confidence if we have historical data to back it up
      confidence = Math.min(confidence + 0.1, 1.0);
    }

    // Adjust based on industry context
    if (this.context?.industry && INDUSTRY_CONFIGS[this.context.industry]) {
      const industryConfig = INDUSTRY_CONFIGS[this.context.industry];
      if (suggestion.field === 'confidence') {
        confidence = Math.min(confidence + industryConfig.confidenceBonus, 1.0);
      }
    }

    return confidence;
  }

  /**
   * Analyze multiple items in batch.
   *
   * @param items - Array of items to analyze
   * @param framework - Optional framework to analyze for
   * @returns Array of analysis results
   */
  analyzeBatch(
    items: PrioritizationItem[],
    framework?: FrameworkType
  ): AnalysisResult[] {
    return items.map(item => {
      const suggestions = this.analyzeItem(item, framework);
      return {
        item,
        suggestions,
        analysisTimeMs: 0 // Individual timing not tracked in batch
      };
    });
  }

  /**
   * Generate RICE framework suggestions.
   */
  private generateRICESuggestions(
    _item: PrioritizationItem,
    patternMatches: PatternMatch[],
    metrics: ExtractedMetrics
  ): FrameworkSuggestion[] {
    const suggestions: FrameworkSuggestion[] = [];

    // Reach suggestion
    const reachMatch = patternMatches.find(m => m.category === 'REACH');
    if (reachMatch) {
      suggestions.push({
        field: 'reach',
        suggestedValue: reachMatch.suggestedValue,
        confidence: reachMatch.confidence * this.config.patternWeight,
        source: 'pattern',
        reason: `Detected "${reachMatch.keyword}" suggesting ${reachMatch.suggestedValue} users affected`
      });
    } else {
      suggestions.push({
        field: 'reach',
        suggestedValue: metrics.reach || 500,
        confidence: metrics.extractionConfidence * this.config.metricsWeight,
        source: 'metrics',
        reason: 'Estimated based on item description analysis'
      });
    }

    // Impact suggestion
    const impactMatch = patternMatches.find(m => m.category === 'IMPACT');
    if (impactMatch) {
      suggestions.push({
        field: 'impact',
        suggestedValue: impactMatch.suggestedValue,
        confidence: impactMatch.confidence * this.config.patternWeight,
        source: 'pattern',
        reason: `Detected "${impactMatch.keyword}" suggesting ${impactMatch.suggestedValue}x impact`
      });
    } else {
      suggestions.push({
        field: 'impact',
        suggestedValue: metrics.impact || 1,
        confidence: metrics.extractionConfidence * this.config.metricsWeight,
        source: 'metrics',
        reason: 'Estimated based on value keywords in description'
      });
    }

    // Confidence suggestion
    const confidenceMatch = patternMatches.find(m => m.category === 'CONFIDENCE');
    if (confidenceMatch) {
      suggestions.push({
        field: 'confidence',
        suggestedValue: confidenceMatch.suggestedValue,
        confidence: confidenceMatch.confidence * this.config.patternWeight,
        source: 'pattern',
        reason: `Detected "${confidenceMatch.keyword}" suggesting ${(confidenceMatch.suggestedValue as number)}% confidence`
      });
    } else {
      suggestions.push({
        field: 'confidence',
        suggestedValue: metrics.confidence || 50,
        confidence: metrics.extractionConfidence * this.config.metricsWeight,
        source: 'metrics',
        reason: 'Estimated based on description completeness'
      });
    }

    // Effort suggestion
    const effortMatch = patternMatches.find(m => m.category === 'EFFORT');
    if (effortMatch) {
      suggestions.push({
        field: 'effort',
        suggestedValue: effortMatch.suggestedValue,
        confidence: effortMatch.confidence * this.config.patternWeight,
        source: 'pattern',
        reason: `Detected "${effortMatch.keyword}" suggesting ${effortMatch.suggestedValue} person-months`
      });
    } else {
      suggestions.push({
        field: 'effort',
        suggestedValue: metrics.effort || 1,
        confidence: metrics.extractionConfidence * this.config.metricsWeight,
        source: 'metrics',
        reason: 'Estimated based on complexity keywords'
      });
    }

    // Apply industry adjustments
    if (this.context?.industry && INDUSTRY_CONFIGS[this.context.industry]) {
      const config = INDUSTRY_CONFIGS[this.context.industry];
      const reachSuggestion = suggestions.find(s => s.field === 'reach');
      const effortSuggestion = suggestions.find(s => s.field === 'effort');

      if (reachSuggestion && typeof reachSuggestion.suggestedValue === 'number') {
        reachSuggestion.suggestedValue *= config.reachMultiplier;
      }
      if (effortSuggestion && typeof effortSuggestion.suggestedValue === 'number') {
        effortSuggestion.suggestedValue *= config.effortBaseline;
      }
    }

    return suggestions;
  }

  /**
   * Generate MoSCoW framework suggestions.
   */
  private generateMoSCoWSuggestions(
    item: PrioritizationItem,
    patternMatches: PatternMatch[]
  ): FrameworkSuggestion[] {
    const suggestions: FrameworkSuggestion[] = [];

    // Check for MoSCoW category match
    const moscowMatch = patternMatches.find(m => m.category === 'MOSCOW');
    if (moscowMatch) {
      suggestions.push({
        field: 'category',
        suggestedValue: moscowMatch.suggestedValue,
        confidence: moscowMatch.confidence,
        source: 'pattern',
        reason: `Detected "${moscowMatch.keyword}" suggesting ${moscowMatch.suggestedValue} priority`
      });
    }

    // Detect urgency for priority assessment
    const urgency = this.labelAnalyzer.detectUrgency(item.title);
    if (urgency === 'high' && !moscowMatch) {
      suggestions.push({
        field: 'category',
        suggestedValue: 'Should have',
        confidence: 0.6,
        source: 'pattern',
        reason: 'High urgency detected in item title'
      });
    }

    // Detect business value
    const businessValue = this.labelAnalyzer.detectBusinessValue(item.title);
    if (businessValue === 'critical' && !moscowMatch) {
      suggestions.push({
        field: 'category',
        suggestedValue: 'Must have',
        confidence: 0.7,
        source: 'pattern',
        reason: 'Critical business value detected'
      });
    }

    return suggestions;
  }

  /**
   * Generate Value vs Effort framework suggestions.
   */
  private generateValueEffortSuggestions(
    _item: PrioritizationItem,
    patternMatches: PatternMatch[],
    metrics: ExtractedMetrics
  ): FrameworkSuggestion[] {
    const suggestions: FrameworkSuggestion[] = [];

    // Value suggestion
    const valueMatch = patternMatches.find(m => m.category === 'VALUE');
    if (valueMatch) {
      suggestions.push({
        field: 'value',
        suggestedValue: valueMatch.suggestedValue,
        confidence: valueMatch.confidence * this.config.patternWeight,
        source: 'pattern',
        reason: `Detected "${valueMatch.keyword}" suggesting value score of ${valueMatch.suggestedValue}`
      });
    } else {
      suggestions.push({
        field: 'value',
        suggestedValue: metrics.value || 5,
        confidence: metrics.extractionConfidence * this.config.metricsWeight,
        source: 'metrics',
        reason: 'Estimated based on business value indicators'
      });
    }

    // Effort suggestion
    const effortMatch = patternMatches.find(m => m.category === 'EFFORT');
    if (effortMatch) {
      suggestions.push({
        field: 'effort',
        suggestedValue: typeof effortMatch.suggestedValue === 'number'
          ? Math.round(effortMatch.suggestedValue)
          : 5,
        confidence: effortMatch.confidence * this.config.patternWeight,
        source: 'pattern',
        reason: `Detected "${effortMatch.keyword}" suggesting effort score`
      });
    } else {
      suggestions.push({
        field: 'effort',
        suggestedValue: metrics.effort ? Math.round(metrics.effort * 2) : 5,
        confidence: metrics.extractionConfidence * this.config.metricsWeight,
        source: 'metrics',
        reason: 'Estimated based on complexity indicators'
      });
    }

    return suggestions;
  }

  /**
   * Generate Eisenhower Matrix suggestions.
   */
  private generateEisenhowerSuggestions(
    item: PrioritizationItem,
    patternMatches: PatternMatch[]
  ): FrameworkSuggestion[] {
    const suggestions: FrameworkSuggestion[] = [];

    // Check for urgency match
    const eisenhowerMatch = patternMatches.find(m => m.category === 'EISENHOWER');
    const urgency = this.labelAnalyzer.detectUrgency(item.title);

    if (eisenhowerMatch && typeof eisenhowerMatch.suggestedValue === 'object') {
      const value = eisenhowerMatch.suggestedValue as { urgent?: boolean; important?: boolean };
      if (value.urgent !== undefined) {
        suggestions.push({
          field: 'urgent',
          suggestedValue: value.urgent,
          confidence: eisenhowerMatch.confidence,
          source: 'pattern',
          reason: `Detected "${eisenhowerMatch.keyword}" suggesting urgency`
        });
      }
      if (value.important !== undefined) {
        suggestions.push({
          field: 'important',
          suggestedValue: value.important,
          confidence: eisenhowerMatch.confidence,
          source: 'pattern',
          reason: `Detected "${eisenhowerMatch.keyword}" suggesting importance`
        });
      }
    } else {
      // Default suggestions based on urgency detection
      suggestions.push({
        field: 'urgent',
        suggestedValue: urgency === 'high',
        confidence: urgency === 'high' ? 0.7 : 0.5,
        source: 'pattern',
        reason: `Detected ${urgency} urgency level in item`
      });

      const businessValue = this.labelAnalyzer.detectBusinessValue(item.title);
      suggestions.push({
        field: 'important',
        suggestedValue: businessValue === 'critical' || businessValue === 'high',
        confidence: businessValue === 'critical' ? 0.8 : 0.6,
        source: 'pattern',
        reason: `Detected ${businessValue} business value`
      });
    }

    return suggestions;
  }

  /**
   * Generate P0P4 priority suggestions.
   */
  private generateP0P4Suggestions(
    _item: PrioritizationItem,
    patternMatches: PatternMatch[]
  ): FrameworkSuggestion[] {
    const suggestions: FrameworkSuggestion[] = [];

    // Check for P0P4 match
    const p0p4Match = patternMatches.find(m => m.category === 'P0P4');
    if (p0p4Match) {
      suggestions.push({
        field: 'priority',
        suggestedValue: p0p4Match.suggestedValue,
        confidence: p0p4Match.confidence,
        source: 'pattern',
        reason: `Detected "${p0p4Match.keyword}" suggesting ${p0p4Match.suggestedValue} priority`
      });
    }

    return suggestions;
  }
}

export default AutoFillEngine;
