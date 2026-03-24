/**
 * Prioritization Service for calculating priority scores.
 *
 * This service implements all framework calculators and provides
 * a unified API for priority calculations.
 *
 * @module services/PrioritizationService
 */

import {
  FrameworkType,
  PrioritizationItem,
  PrioritizeRequestDTO,
  PrioritizeResponseDTO,
  BulkPrioritizeRequestDTO,
  FrameworkSuggestions,
  FrameworkSuggestion
} from '../types';

/**
 * Mapping of framework types to their required input fields.
 */
const frameworkInputFields: Record<FrameworkType, string[]> = {
  RICE: ['reach', 'impact', 'confidence', 'effort'],
  MoSCoW: ['businessValue', 'legalRequirement', 'customerRequest', 'riskIfNotDelivered'],
  ValueEffort: ['value', 'effort'],
  ICE: ['impact', 'confidence', 'ease'],
  Eisenhower: ['urgent', 'important'],
  P0P4: ['baseSeverity', 'severityFactors'],
  WSJF: ['userBusinessValue', 'timeCriticality', 'riskReductionOpportunity', 'jobSize'],
  Kano: ['functionalScore', 'dysfunctionalScore']
};

/**
 * Calculate RICE score.
 */
function calculateRICE(data: Record<string, unknown>): { score: number; details: Record<string, unknown> } {
  const reach = Number(data.reach) || 100;
  const impact = Number(data.impact) || 1;
  let confidence = Number(data.confidence) || 50;
  const effort = Number(data.effort) || 1;

  // Normalize confidence to 0-1 if it's a percentage
  if (confidence > 1) {
    confidence = confidence / 100;
  }

  const score = (reach * impact * confidence) / effort;

  return {
    score: Math.round(score * 100) / 100,
    details: {
      reach,
      impact,
      confidence: Math.round(confidence * 100),
      effort,
      rawScore: score
    }
  };
}

/**
 * Calculate MoSCoW priority score.
 */
function calculateMoSCoW(data: Record<string, unknown>): { score: number; details: Record<string, unknown> } {
  const businessValue = String(data.businessValue || 'medium');
  const legalRequirement = Boolean(data.legalRequirement);
  const customerRequest = Boolean(data.customerRequest);
  const riskIfNotDelivered = String(data.riskIfNotDelivered || 'medium');

  // Determine category
  let category: string = 'Could have';
  let score = 50;

  if (legalRequirement || businessValue === 'critical') {
    category = 'Must have';
    score = 100;
  } else if (businessValue === 'high' || customerRequest || riskIfNotDelivered === 'high') {
    category = 'Should have';
    score = 75;
  } else if (businessValue === 'medium') {
    category = 'Could have';
    score = 50;
  } else {
    category = "Won't have";
    score = 25;
  }

  return {
    score,
    details: {
      category,
      businessValue,
      legalRequirement,
      customerRequest,
      riskIfNotDelivered
    }
  };
}

/**
 * Calculate Value vs Effort score.
 */
function calculateValueEffort(data: Record<string, unknown>): { score: number; details: Record<string, unknown> } {
  const value = Number(data.value) || 5;
  const effort = Number(data.effort) || 5;

  const roi = effort > 0 ? value / effort : 0;

  // Determine quadrant
  let quadrant: string = 'FillIn';
  if (value >= 5 && effort < 5) {
    quadrant = 'QuickWin';
  } else if (value >= 5 && effort >= 5) {
    quadrant = 'MajorProject';
  } else if (value < 5 && effort < 5) {
    quadrant = 'FillIn';
  } else {
    quadrant = 'Avoid';
  }

  return {
    score: Math.round(roi * 100) / 100,
    details: {
      value,
      effort,
      roi: Math.round(roi * 100) / 100,
      quadrant
    }
  };
}

/**
 * Calculate ICE score.
 */
function calculateICE(data: Record<string, unknown>): { score: number; details: Record<string, unknown> } {
  const impact = Number(data.impact) || 5;
  let confidence = Number(data.confidence) || 50;
  const ease = Number(data.ease) || 5;

  // Normalize confidence to 0-1 if it's a percentage
  if (confidence > 1) {
    confidence = confidence / 100;
  }

  const score = impact * confidence * ease;

  return {
    score: Math.round(score * 100) / 100,
    details: {
      impact,
      confidence: Math.round(confidence * 100),
      ease,
      rawScore: score
    }
  };
}

/**
 * Calculate Eisenhower Matrix quadrant.
 */
function calculateEisenhower(data: Record<string, unknown>): { score: number; details: Record<string, unknown> } {
  const urgent = Boolean(data.urgent);
  const important = Boolean(data.important);
  const urgencyLevel = Number(data.urgencyLevel) || 5;
  const importanceLevel = Number(data.importanceLevel) || 5;

  // Determine quadrant
  let quadrant: string = 'Eliminate';
  let score = 25;

  if (urgent && important) {
    quadrant = 'DoFirst';
    score = 100;
  } else if (!urgent && important) {
    quadrant = 'Schedule';
    score = 75;
  } else if (urgent && !important) {
    quadrant = 'Delegate';
    score = 50;
  }

  return {
    score,
    details: {
      urgent,
      important,
      quadrant,
      urgencyLevel,
      importanceLevel
    }
  };
}

/**
 * Calculate P0P4 priority.
 */
function calculateP0P4(data: Record<string, unknown>): { score: number; details: Record<string, unknown> } {
  const baseSeverity = Number(data.baseSeverity) || 3;
  const severityFactors = data.severityFactors as Record<string, string> || {
    usersAffected: 'some',
    coreFunctionalityImpact: 'medium',
    securityRisk: 'low',
    reputationalRisk: 'low',
    revenueImpact: 'low'
  };

  // Calculate severity score
  let severityScore = baseSeverity;

  // Adjust based on factors
  const factorScores: Record<string, number> = {
    none: 0,
    low: 1,
    medium: 2,
    high: 3,
    critical: 4,
    few: 1,
    some: 2,
    many: 3,
    all: 4
  };

  if (severityFactors) {
    severityScore += factorScores[severityFactors.usersAffected] || 0;
    severityScore += factorScores[severityFactors.coreFunctionalityImpact] || 0;
    severityScore += factorScores[severityFactors.securityRisk] || 0;
    severityScore += factorScores[severityFactors.reputationalRisk] || 0;
    severityScore += factorScores[severityFactors.revenueImpact] || 0;
  }

  // Determine priority level
  let priority: string = 'P2';
  if (severityScore >= 15) {
    priority = 'P0';
  } else if (severityScore >= 12) {
    priority = 'P1';
  } else if (severityScore >= 8) {
    priority = 'P2';
  } else if (severityScore >= 5) {
    priority = 'P3';
  } else {
    priority = 'P4';
  }

  const scoreMap: Record<string, number> = {
    'P0': 100,
    'P1': 80,
    'P2': 60,
    'P3': 40,
    'P4': 20
  };

  return {
    score: scoreMap[priority],
    details: {
      priority,
      severityScore,
      severityFactors,
      baseSeverity
    }
  };
}

/**
 * Calculate WSJF score.
 */
function calculateWSJF(data: Record<string, unknown>): { score: number; details: Record<string, unknown> } {
  const userBusinessValue = Number(data.userBusinessValue) || 5;
  const timeCriticality = Number(data.timeCriticality) || 5;
  const riskReductionOpportunity = Number(data.riskReductionOpportunity) || 5;
  const jobSize = Number(data.jobSize) || 5;

  const costOfDelay = userBusinessValue + timeCriticality + riskReductionOpportunity;
  const score = jobSize > 0 ? costOfDelay / jobSize : 0;

  return {
    score: Math.round(score * 100) / 100,
    details: {
      userBusinessValue,
      timeCriticality,
      riskReductionOpportunity,
      jobSize,
      costOfDelay,
      rawScore: score
    }
  };
}

/**
 * Calculate Kano category.
 */
function calculateKano(data: Record<string, unknown>): { score: number; details: Record<string, unknown> } {
  const functionalScore = Number(data.functionalScore) || 3;
  const dysfunctionalScore = Number(data.dysfunctionalScore) || 3;

  // Determine category based on scores
  let category: string = 'Indifferent';
  let score = 50;

  if (functionalScore >= 4 && dysfunctionalScore <= 2) {
    category = 'OneDimensional';
    score = 90;
  } else if (functionalScore >= 4 && dysfunctionalScore >= 4) {
    category = 'MustBe';
    score = 80;
  } else if (functionalScore >= 4 && dysfunctionalScore <= 2) {
    category = 'Attractive';
    score = 100;
  } else if (functionalScore <= 2 && dysfunctionalScore >= 4) {
    category = 'Reverse';
    score = 10;
  } else if (functionalScore <= 2 && dysfunctionalScore <= 2) {
    category = 'Indifferent';
    score = 30;
  }

  return {
    score,
    details: {
      category,
      functionalScore,
      dysfunctionalScore
    }
  };
}

/**
 * Calculate framework score.
 */
function calculateFramework(
  framework: FrameworkType,
  data: Record<string, unknown>
): { score: number; details: Record<string, unknown> } {
  switch (framework) {
    case 'RICE':
      return calculateRICE(data);
    case 'MoSCoW':
      return calculateMoSCoW(data);
    case 'ValueEffort':
      return calculateValueEffort(data);
    case 'ICE':
      return calculateICE(data);
    case 'Eisenhower':
      return calculateEisenhower(data);
    case 'P0P4':
      return calculateP0P4(data);
    case 'WSJF':
      return calculateWSJF(data);
    case 'Kano':
      return calculateKano(data);
    default:
      throw new Error(`Unsupported framework: ${framework}`);
  }
}

/**
 * PrioritizationService class for calculating priority scores.
 *
 * Provides methods for single and bulk priority calculations
 * across all supported frameworks.
 */
export class PrioritizationService {
  /**
   * Calculate priority for a single item.
   *
   * @param request - Prioritization request with framework and item data
   * @returns Prioritization response with score and details
   * @throws {Error} If framework is not supported or input is invalid
   *
   * @example
   * ```typescript
   * const result = await PrioritizationService.prioritize({
   *   framework: 'RICE',
   *   item: {
   *     title: 'Add CSV export',
   *     description: 'Export feature for all users',
   *     frameworkData: { reach: 1000, impact: 2, confidence: 80, effort: 3 }
   *   }
   * });
   * ```
   */
  public static async prioritize(
    request: PrioritizeRequestDTO
  ): Promise<PrioritizeResponseDTO> {
    // Get framework data from item
    const frameworkData = request.item.frameworkData || {};

    // Check if we have the required fields
    const requiredFields = frameworkInputFields[request.framework];
    const missingFields = requiredFields.filter(
      field => !(field in frameworkData)
    );

    // If missing fields, generate auto-fill suggestions
    let suggestions: FrameworkSuggestions | undefined;
    if (missingFields.length > 0) {
      suggestions = this.generateSuggestions(request.framework, request.item);
    }

    // Calculate based on framework type
    const result = calculateFramework(request.framework, frameworkData);

    // Add rank if context is provided
    if (request.context && request.context.length > 1) {
      const allScores = request.context.map(item => {
        const itemData = item.frameworkData || {};
        const itemResult = calculateFramework(request.framework, itemData);
        return itemResult.score;
      });

      const sortedScores = [...allScores].sort((a, b) => b - a);
      result.details.rank = sortedScores.indexOf(result.score) + 1;
    }

    return {
      priorityScore: result.score,
      rank: result.details.rank as number | undefined,
      framework: request.framework,
      details: result.details,
      suggestions
    };
  }

  /**
   * Calculate priorities for multiple items at once.
   *
   * @param request - Bulk prioritization request
   * @returns Array of prioritization responses
   *
   * @example
   * ```typescript
   * const results = await PrioritizationService.prioritizeBulk({
   *   framework: 'RICE',
   *   items: [...]
   * });
   * ```
   */
  public static async prioritizeBulk(
    request: BulkPrioritizeRequestDTO
  ): Promise<PrioritizeResponseDTO[]> {
    const results: PrioritizeResponseDTO[] = [];

    // Calculate all items first
    for (const item of request.items) {
      try {
        const result = await this.prioritize({
          framework: request.framework,
          item
        });
        results.push(result);
      } catch (error) {
        // Add error result
        results.push({
          priorityScore: 0,
          framework: request.framework,
          details: {
            error: error instanceof Error ? error.message : 'Unknown error'
          }
        });
      }
    }

    // Normalize ranks based on scores
    const sortedIndices = results
      .map((r, i) => ({ score: r.priorityScore, index: i }))
      .sort((a, b) => b.score - a.score);

    sortedIndices.forEach((item, rank) => {
      results[item.index].rank = rank + 1;
    });

    return results;
  }

  /**
   * Generate auto-fill suggestions for an item.
   *
   * @param framework - Framework type
   * @param item - Item to analyze
   * @returns Framework suggestions object
   */
  private static generateSuggestions(
    framework: FrameworkType,
    item: Omit<PrioritizationItem, 'id' | 'createdAt' | 'updatedAt'>
  ): FrameworkSuggestions {
    const suggestions: FrameworkSuggestion[] = [];
    const text = `${item.title} ${item.description || ''}`.toLowerCase();

    // Generate suggestions based on framework
    if (framework === 'RICE') {
      // Reach suggestions based on audience keywords
      if (text.includes('all users') || text.includes('everyone')) {
        suggestions.push({
          field: 'reach',
          suggestedValue: 1000,
          confidence: 0.8,
          source: 'pattern',
          reason: 'Detected "all users" or "everyone" suggesting broad reach'
        });
      } else if (text.includes('enterprise') || text.includes('business')) {
        suggestions.push({
          field: 'reach',
          suggestedValue: 100,
          confidence: 0.7,
          source: 'pattern',
          reason: 'Detected enterprise/business context suggesting B2B reach'
        });
      } else if (text.includes('admin') || text.includes('internal')) {
        suggestions.push({
          field: 'reach',
          suggestedValue: 20,
          confidence: 0.75,
          source: 'pattern',
          reason: 'Detected admin/internal context suggesting limited user base'
        });
      } else {
        suggestions.push({
          field: 'reach',
          suggestedValue: 500,
          confidence: 0.4,
          source: 'metrics',
          reason: 'Default moderate reach estimate'
        });
      }

      // Impact suggestions based on value keywords
      const highImpactKeywords = ['critical', 'blocker', 'revenue', 'conversion', 'retention', 'security'];
      const lowImpactKeywords = ['nice to have', 'minor', 'cosmetic', 'optional'];

      if (highImpactKeywords.some(k => text.includes(k))) {
        suggestions.push({
          field: 'impact',
          suggestedValue: 2,
          confidence: 0.75,
          source: 'pattern',
          reason: 'Detected high-impact keywords'
        });
      } else if (lowImpactKeywords.some(k => text.includes(k))) {
        suggestions.push({
          field: 'impact',
          suggestedValue: 0.5,
          confidence: 0.7,
          source: 'pattern',
          reason: 'Detected low-impact keywords'
        });
      } else {
        suggestions.push({
          field: 'impact',
          suggestedValue: 1,
          confidence: 0.4,
          source: 'metrics',
          reason: 'Default medium impact estimate'
        });
      }

      // Confidence based on description length
      if (item.description && item.description.length > 100) {
        suggestions.push({
          field: 'confidence',
          suggestedValue: 70,
          confidence: 0.7,
          source: 'metrics',
          reason: 'Detailed description suggests well-thought-out item'
        });
      } else if (item.description && item.description.length > 50) {
        suggestions.push({
          field: 'confidence',
          suggestedValue: 50,
          confidence: 0.6,
          source: 'metrics',
          reason: 'Moderate description length'
        });
      } else {
        suggestions.push({
          field: 'confidence',
          suggestedValue: 40,
          confidence: 0.5,
          source: 'metrics',
          reason: 'Limited description suggests uncertainty'
        });
      }

      // Effort based on complexity keywords
      const highEffortKeywords = ['migration', 'refactor', 'infrastructure', 'integration', 'rewrite'];
      const lowEffortKeywords = ['fix', 'update', 'tweak', 'quick', 'simple'];

      if (highEffortKeywords.some(k => text.includes(k))) {
        suggestions.push({
          field: 'effort',
          suggestedValue: 4,
          confidence: 0.6,
          source: 'pattern',
          reason: 'Detected high-effort complexity keywords'
        });
      } else if (lowEffortKeywords.some(k => text.includes(k))) {
        suggestions.push({
          field: 'effort',
          suggestedValue: 0.5,
          confidence: 0.65,
          source: 'pattern',
          reason: 'Detected low-effort keywords'
        });
      } else {
        suggestions.push({
          field: 'effort',
          suggestedValue: 2,
          confidence: 0.4,
          source: 'metrics',
          reason: 'Default medium effort estimate'
        });
      }
    }

    const overallConfidence = suggestions.length > 0
      ? suggestions.reduce((sum, s) => sum + s.confidence, 0) / suggestions.length
      : 0;

    return {
      framework,
      suggestions,
      overallConfidence
    };
  }
}

export default PrioritizationService;
