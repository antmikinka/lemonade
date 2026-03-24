/**
 * Auto-Fill Module for Prioritization Frameworks.
 *
 * This module provides intelligent auto-fill suggestions for prioritization
 * parameters using label analysis, metrics extraction, and pattern matching.
 *
 * @module prioritization/auto-fill
 */

export { AutoFillEngine } from './AutoFillEngine';
export type {
  AutoFillContext,
  FrameworkSuggestion,
  FrameworkSuggestions,
  AnalysisResult
} from './AutoFillEngine';

export { LabelAnalyzer } from './LabelAnalyzer';
export type {
  PatternMatch,
  UrgencyLevel,
  BusinessValueLevel,
  KeywordPatterns
} from './LabelAnalyzer';

export { MetricsExtractor } from './MetricsExtractor';
export type {
  ExtractedMetrics,
  DetectedScale
} from './MetricsExtractor';
