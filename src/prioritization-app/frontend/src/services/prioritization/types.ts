/**
 * Core type definitions for the Prioritization Frameworks application.
 *
 * This module defines the base interfaces and types used across all
 * prioritization frameworks, including framework-specific result types
 * and the calculator strategy interface.
 *
 * @module prioritization/types
 */

/**
 * Union type of all supported prioritization frameworks.
 * Each framework implements a different approach to task prioritization.
 */
export type FrameworkType =
  | 'RICE'           // Reach, Impact, Confidence, Effort
  | 'MoSCoW'         // Must have, Should have, Could have, Won't have
  | 'ValueEffort'    // Value vs Effort matrix
  | 'ICE'            // Impact, Confidence, Ease
  | 'Eisenhower'     // Urgent vs Important matrix
  | 'P0P4'           // Priority levels P0-P4
  | 'WSJF'           // Weighted Shortest Job First
  | 'Kano';          // Kano model for customer satisfaction

/**
 * Base interface for any item that can be prioritized.
 * All framework-specific inputs must extend this interface.
 */
export interface PrioritizationItem {
  /** Unique identifier for the item */
  id: string;
  /** Title or name of the item */
  title: string;
  /** Detailed description of the item */
  description?: string;
  /** Category or grouping for the item */
  category?: string;
  /** Creation timestamp */
  createdAt: Date;
  /** Last update timestamp */
  updatedAt?: Date;
  /** Optional metadata for framework-specific data */
  metadata?: Record<string, unknown>;
}

/**
 * Validation result for calculator inputs.
 * Used to report validation errors or warnings.
 */
export interface ValidationResult {
  /** Whether the input is valid */
  isValid: boolean;
  /** Array of error messages */
  errors: string[];
  /** Array of warning messages */
  warnings: string[];
}

/**
 * Base interface for normalized scores.
 * Normalization allows comparison across items.
 */
export interface NormalizedResult {
  /** Normalized score (0-100) */
  normalizedScore: number;
  /** Rank position (1 = highest priority) */
  rank: number;
  /** Percentile within the dataset */
  percentile: number;
}

/**
 * Base interface for all framework results.
 * Each framework-specific result extends this interface.
 */
export interface FrameworkResult {
  /** The framework that produced this result */
  framework: FrameworkType;
  /** Calculated score (if applicable) */
  score?: number;
  /** Normalized result data */
  normalized?: NormalizedResult;
  /** Additional framework-specific data */
  details: Record<string, unknown>;
}

// ============================================================================
// RICE Framework Types
// ============================================================================

/**
 * Input parameters for RICE scoring.
 *
 * RICE is a prioritization framework that scores items based on:
 * - Reach: How many users will this affect?
 * - Impact: How much will this impact each user?
 * - Confidence: How confident are we in our estimates?
 * - Effort: How much work will this require?
 */
export interface RICEInput {
  /** Number of users or events affected per period */
  reach: number;
  /**
   * Impact multiplier on a standard scale:
   * 3 = Massive impact
   * 2 = High impact
   * 1 = Medium impact
   * 0.5 = Low impact
   * 0.25 = Minimal impact
   */
  impact: number;
  /**
   * Confidence level as a decimal (0-1) or percentage (0-100).
   * Represents certainty in the estimates.
   */
  confidence: number;
  /** Effort required in person-months or story points */
  effort: number;
}

/**
 * Result of RICE score calculation.
 */
export interface RICEResult extends FrameworkResult {
  framework: 'RICE';
  /** Reach value used in calculation */
  reach: number;
  /** Impact value used in calculation */
  impact: number;
  /** Confidence value used in calculation (normalized to 0-1) */
  confidence: number;
  /** Effort value used in calculation */
  effort: number;
  /** Calculated RICE score */
  score: number;
}

// ============================================================================
// MoSCoW Framework Types
// ============================================================================

/**
 * MoSCoW priority categories.
 *
 * - Must have: Critical requirements that cannot be omitted
 * - Should have: Important but not critical requirements
 * - Could have: Desirable but not necessary requirements
 * - Won't have: Lowest priority, may be considered later
 */
export type MoSCoWCategory = 'Must have' | 'Should have' | 'Could have' | "Won't have";

/**
 * Business value levels for MoSCoW categorization.
 */
export type BusinessValueLevel = 'critical' | 'high' | 'medium' | 'low';

/**
 * Risk levels for MoSCoW categorization.
 */
export type RiskLevel = 'critical' | 'high' | 'medium' | 'low';

/**
 * Input parameters for MoSCoW categorization.
 *
 * MoSCoW categorizes requirements based on:
 * - Business value criticality
 * - Legal or compliance requirements
 * - Customer requests
 * - Risk if not delivered
 */
export interface MoSCoWInput {
  /** Business value level of the item */
  businessValue: BusinessValueLevel;
  /** Whether the item is a legal or compliance requirement */
  legalRequirement: boolean;
  /** Whether the item was specifically requested by customers */
  customerRequest: boolean;
  /** Risk level if the item is not delivered */
  riskIfNotDelivered: RiskLevel;
}

/**
 * Result of MoSCoW categorization.
 */
export interface MoSCoWResult extends FrameworkResult {
  framework: 'MoSCoW';
  /** Assigned MoSCoW category */
  category: MoSCoWCategory;
  /** Numeric priority within category (1 = highest) */
  priority: number;
  /** Business value used in calculation */
  businessValue: BusinessValueLevel;
  /** Whether item is a legal requirement */
  legalRequirement: boolean;
  /** Whether item was customer requested */
  customerRequest: boolean;
  /** Risk level if not delivered */
  riskIfNotDelivered: RiskLevel;
}

// ============================================================================
// Value vs Effort Framework Types
// ============================================================================

/**
 * Quadrants in the Value vs Effort matrix.
 *
 * - QuickWin: High value, low effort - do these first
 * - MajorProject: High value, high effort - plan carefully
 * - FillIn: Low value, low effort - do when time permits
 * - Avoid: Low value, high effort - question these items
 */
export type ValueEffortQuadrant = 'QuickWin' | 'MajorProject' | 'FillIn' | 'Avoid';

/**
 * Input parameters for Value vs Effort scoring.
 */
export interface ValueEffortInput {
  /** Business value score (typically 1-10) */
  value: number;
  /** Effort required (typically 1-10, higher = more effort) */
  effort: number;
}

/**
 * Result of Value vs Effort analysis.
 */
export interface ValueEffortResult extends FrameworkResult {
  framework: 'ValueEffort';
  /** Value score used in calculation */
  value: number;
  /** Effort score used in calculation */
  effort: number;
  /** Assigned quadrant */
  quadrant: ValueEffortQuadrant;
  /** Return on investment (value/effort ratio) */
  roi: number;
}

// ============================================================================
// ICE Framework Types
// ============================================================================

/**
 * Input parameters for ICE scoring.
 *
 * ICE is a simplified version of RICE:
 * - Impact: How much will this help?
 * - Confidence: How confident are we?
 * - Ease: How easy is this to implement?
 */
export interface ICEInput {
  /** Impact score (typically 1-10) */
  impact: number;
  /** Confidence score (typically 1-10 or 0-1) */
  confidence: number;
  /** Ease of implementation (typically 1-10, higher = easier) */
  ease: number;
}

/**
 * Result of ICE score calculation.
 */
export interface ICEResult extends FrameworkResult {
  framework: 'ICE';
  /** Impact value used in calculation */
  impact: number;
  /** Confidence value used in calculation */
  confidence: number;
  /** Ease value used in calculation */
  ease: number;
  /** Calculated ICE score */
  score: number;
}

// ============================================================================
// Eisenhower Matrix Framework Types
// ============================================================================

/**
 * Quadrants in the Eisenhower Matrix.
 *
 * - DoFirst: Urgent and important - do immediately
 * - Schedule: Not urgent but important - plan to do
 * - Delegate: Urgent but not important - delegate if possible
 * - Eliminate: Not urgent and not important - consider eliminating
 */
export type EisenhowerQuadrant = 'DoFirst' | 'Schedule' | 'Delegate' | 'Eliminate';

/**
 * Input parameters for Eisenhower Matrix categorization.
 */
export interface EisenhowerInput {
  /** Whether the item is urgent (time-sensitive) */
  urgent: boolean;
  /** Whether the item is important (high impact) */
  important: boolean;
  /** Optional urgency level for finer grading (1-10) */
  urgencyLevel?: number;
  /** Optional importance level for finer grading (1-10) */
  importanceLevel?: number;
}

/**
 * Result of Eisenhower Matrix analysis.
 */
export interface EisenhowerResult extends FrameworkResult {
  framework: 'Eisenhower';
  /** Whether item is urgent */
  urgent: boolean;
  /** Whether item is important */
  important: boolean;
  /** Assigned quadrant */
  quadrant: EisenhowerQuadrant;
  /** Optional urgency level */
  urgencyLevel?: number;
  /** Optional importance level */
  importanceLevel?: number;
}

// ============================================================================
// P0-P4 Priority Framework Types
// ============================================================================

/**
 * Priority levels in the P0-P4 system.
 *
 * - P0: Critical - must be fixed immediately
 * - P1: High - should be fixed soon
 * - P2: Medium - normal priority
 * - P3: Low - can be deferred
 * - P4: Lowest - backlog or won't fix
 */
export type P0P4Level = 'P0' | 'P1' | 'P2' | 'P3' | 'P4';

/**
 * Severity factors for P0-P4 classification.
 */
export interface SeverityFactors {
  /** Number of users affected */
  usersAffected: 'all' | 'many' | 'some' | 'few' | 'none';
  /** Impact on core functionality */
  coreFunctionalityImpact: 'critical' | 'high' | 'medium' | 'low' | 'none';
  /** Security implications */
  securityRisk: 'critical' | 'high' | 'medium' | 'low' | 'none';
  /** Reputational risk */
  reputationalRisk: 'critical' | 'high' | 'medium' | 'low' | 'none';
  /** Revenue impact */
  revenueImpact: 'critical' | 'high' | 'medium' | 'low' | 'none';
}

/**
 * Input parameters for P0-P4 prioritization.
 */
export interface P0P4Input {
  /** Base severity level (1-5, where 5 is most severe) */
  baseSeverity: number;
  /** Detailed severity factors */
  severityFactors: SeverityFactors;
  /** Number of open issues/bugs related to this item */
  openIssuesCount?: number;
  /** Days since the issue was first reported */
  daysOpen?: number;
}

/**
 * Result of P0-P4 prioritization.
 */
export interface P0P4Result extends FrameworkResult {
  framework: 'P0P4';
  /** Assigned priority level */
  priority: P0P4Level;
  /** Calculated severity score */
  severityScore: number;
  /** Severity factors used in calculation */
  severityFactors: SeverityFactors;
  /** Recommended action timeframe */
  recommendedTimeframe: string;
}

// ============================================================================
// WSJF (Weighted Shortest Job First) Framework Types
// ============================================================================

/**
 * Input parameters for WSJF scoring.
 *
 * WSJF = Cost of Delay / Job Size
 *
 * Cost of Delay is composed of:
 * - User-business value
 * - Time criticality
 * - Risk reduction and opportunity enablement
 */
export interface WSJFInput {
  /** User and business value (typically 1-10 or Fibonacci) */
  userBusinessValue: number;
  /** Time criticality score (typically 1-10 or Fibonacci) */
  timeCriticality: number;
  /** Risk reduction and opportunity enablement score */
  riskReductionOpportunity: number;
  /** Job size (typically 1-10 or Fibonacci, lower = smaller) */
  jobSize: number;
}

/**
 * Result of WSJF calculation.
 */
export interface WSJFResult extends FrameworkResult {
  framework: 'WSJF';
  /** Cost of Delay (sum of value, time criticality, and risk reduction) */
  costOfDelay: number;
  /** Job size used in calculation */
  jobSize: number;
  /** Calculated WSJF score */
  score: number;
  /** User-business value component */
  userBusinessValue: number;
  /** Time criticality component */
  timeCriticality: number;
  /** Risk reduction/opportunity enablement component */
  riskReductionOpportunity: number;
}

// ============================================================================
// Kano Model Framework Types
// ============================================================================

/**
 * Kano model categories for feature classification.
 *
 * - MustBe: Basic expectations, cause dissatisfaction if missing
 * - OneDimensional: More is better, linear satisfaction relationship
 * - Attractive: Delighters, cause satisfaction when present
 * - Indifferent: Customers don't care either way
 * - Reverse: Causes dissatisfaction when present
 * - Questionable: Unclear or conflicting responses
 */
export type KanoCategory =
  | 'MustBe'
  | 'OneDimensional'
  | 'Attractive'
  | 'Indifferent'
  | 'Reverse'
  | 'Questionable';

/**
 * Input parameters for Kano analysis.
 */
export interface KanoInput {
  /**
   * Functional score: How much do you LIKE having this feature?
   * Scale: 1 (Dislike) to 5 (Like)
   */
  functionalScore: number;
  /**
   * Dysfunctional score: How much do you LIKE NOT having this feature?
   * Scale: 1 (Dislike) to 5 (Like)
   */
  dysfunctionalScore: number;
  /** Importance rating (1-5) */
  importance?: number;
  /** Satisfaction rating if feature is present (1-5) */
  satisfactionIfPresent?: number;
  /** Dissatisfaction rating if feature is absent (1-5) */
  dissatisfactionIfAbsent?: number;
}

/**
 * Result of Kano analysis.
 */
export interface KanoResult extends FrameworkResult {
  framework: 'Kano';
  /** Functional score used in calculation */
  functionalScore: number;
  /** Dysfunctional score used in calculation */
  dysfunctionalScore: number;
  /** Calculated Kano category */
  category: KanoCategory;
  /** Satisfaction coefficient (-1 to 1) */
  satisfactionCoefficient?: number;
  /** Dissatisfaction coefficient (-1 to 1) */
  dissatisfactionCoefficient?: number;
  /** Importance rating if provided */
  importance?: number;
}

// ============================================================================
// Framework Calculator Interface (Strategy Pattern)
// ============================================================================

/**
 * Generic interface for all framework calculators.
 *
 * This interface implements the Strategy pattern, allowing different
 * prioritization frameworks to be used interchangeably while maintaining
 * a consistent API for calculation, validation, and normalization.
 *
 * @typeParam TInput - The input type specific to each framework
 * @typeParam TResult - The result type specific to each framework
 */
export interface IFrameworkCalculator<TInput, TResult extends FrameworkResult> {
  /**
   * Calculates the prioritization score or category for an item.
   *
   * @param input - The framework-specific input parameters
   * @returns The calculated result with score and details
   * @throws {Error} If input validation fails
   */
  calculate(input: TInput): TResult;

  /**
   * Validates the input parameters before calculation.
   *
   * @param input - Partial input parameters to validate
   * @returns Validation result with errors and warnings
   */
  validate(input: Partial<TInput>): ValidationResult;

  /**
   * Normalizes results across multiple items for comparison.
   *
   * @param result - A single result to normalize
   * @param allResults - All results in the dataset for context
   * @returns Normalized result with rank and percentile
   */
  normalize(result: TResult, allResults: TResult[]): NormalizedResult;

  /**
   * Generates auto-fill suggestions based on item metadata.
   *
   * This method uses available item information to suggest
   * reasonable default values for framework parameters,
   * reducing manual input burden.
   *
   * @param item - The prioritization item to analyze
   * @returns Partial input with suggested values
   */
  getAutoFillSuggestions(item: PrioritizationItem): Partial<TInput>;

  /**
   * Returns the framework type identifier.
   *
   * @returns The framework type string
   */
  getFrameworkType(): FrameworkType;
}

/**
 * Configuration options for calculators.
 */
export interface CalculatorConfig {
  /** Minimum allowed score */
  minScore?: number;
  /** Maximum allowed score */
  maxScore?: number;
  /** Number of decimal places for rounding */
  decimalPlaces?: number;
  /** Whether to enable strict validation */
  strictMode?: boolean;
}

/**
 * Combined result containing results from multiple frameworks.
 */
export interface MultiFrameworkResult {
  /** The prioritization item */
  item: PrioritizationItem;
  /** Results from each applied framework */
  results: {
    RICE?: RICEResult;
    MoSCoW?: MoSCoWResult;
    ValueEffort?: ValueEffortResult;
    ICE?: ICEResult;
    Eisenhower?: EisenhowerResult;
    P0P4?: P0P4Result;
    WSJF?: WSJFResult;
    Kano?: KanoResult;
  };
  /** Overall composite score (if calculated) */
  compositeScore?: number;
  /** Overall rank across frameworks */
  overallRank?: number;
}
