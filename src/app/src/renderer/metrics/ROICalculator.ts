/**
 * ROI Calculator
 * Calculates Return on Investment for work items and strategic initiatives
 */

import type {
  WorkItem,
  StrategicInitiative,
  PortfolioROI,
  ROICategoryType,
} from '../types/workItem';

interface EffortMetrics {
  totalStoryPoints: number;
  totalHours: number;
  teamCost: number;
  opportunityCost: number;
}

interface ImpactMetrics {
  revenueImpact: number;
  costReduction: number;
  riskReduction: number;
  strategicValue: number;
  developerProductivity: number;
  total: number;
}

interface ROIMetrics {
  ratio: number;
  percentage: number;
  paybackPeriod: number;
  npv: number;
}

interface CategoryROI {
  investment: number;
  impact: number;
  roi: number;
  itemCount: number;
}

/**
 * ROI Calculator Class
 *
 * Calculates:
 * - Investment (effort) metrics
 * - Impact metrics by category
 * - ROI ratios and percentages
 * - Payback period and NPV
 */
export class ROICalculator {
  // Configuration constants
  private readonly HOUR_TO_STORY_POINT_RATIO = 4; // 1 story point = 4 hours
  private readonly AVERAGE_HOURLY_RATE = 75; // USD per hour
  private readonly OPPORTUNITY_COST_PREMIUM = 0.2; // 20%
  private readonly DISCOUNT_RATE = 0.1; // 10% annual
  private readonly ANNUALIZATION_MONTHS = 12;

  // Impact scaling factors (impact score to dollars)
  private readonly IMPACT_SCALING = {
    'revenue-impact': 10000,
    'cost-reduction': 5000,
    'risk-mitigation': 8000,
    'strategic-capability': 7000,
    'developer-productivity': 3000,
    'customer-experience': 6000,
  } as const;

  /**
   * Calculate ROI for a portfolio of work items
   */
  calculatePortfolio(workItems: WorkItem[]): PortfolioROI {
    const effort = this.calculateEffort(workItems);
    const impact = this.calculateImpact(workItems);
    const roi = this.calculateROIMetrics(effort, impact);
    const byCategory = this.calculateByCategory(workItems);

    return {
      totalInvestment: effort,
      totalImpact: impact,
      roi,
      byCategory,
      byInitiative: this.calculateByInitiative(workItems),
    };
  }

  /**
   * Calculate ROI for a specific strategic initiative
   */
  calculateInitiativeROI(
    initiative: StrategicInitiative,
    workItems: WorkItem[]
  ): PortfolioROI {
    const linkedItems = workItems.filter(
      (item) =>
        item.parentInitiative === initiative.id ||
        initiative.linkedWorkItems.includes(item.id)
    );

    return this.calculatePortfolio(linkedItems);
  }

  // ==========================================================================
  // EFFORT CALCULATIONS
  // ==========================================================================

  /**
   * Calculate total effort from work items
   */
  private calculateEffort(workItems: WorkItem[]): EffortMetrics {
    const totalStoryPoints = this.calculateTotalStoryPoints(workItems);
    const totalHours = totalStoryPoints * this.HOUR_TO_STORY_POINT_RATIO;
    const teamCost = totalHours * this.AVERAGE_HOURLY_RATE;
    const opportunityCost = teamCost * this.OPPORTUNITY_COST_PREMIUM;

    return {
      totalStoryPoints,
      totalHours,
      teamCost,
      opportunityCost,
    };
  }

  /**
   * Calculate total story points from work items
   */
  private calculateTotalStoryPoints(workItems: WorkItem[]): number {
    return workItems.reduce((sum, item) => {
      // Use explicit story points if available
      if (item.roiCategory?.estimatedEffort) {
        return sum + item.roiCategory.estimatedEffort;
      }

      // Use metrics-based estimation
      return sum + this.estimatePoints(item);
    }, 0);
  }

  /**
   * Estimate story points from work item characteristics
   */
  private estimatePoints(item: WorkItem): number {
    if (item.metrics) {
      const linesChanged = (item.metrics.linesAdded || 0) + (item.metrics.linesDeleted || 0);
      const filesChanged = item.metrics.filesChanged || 1;
      const complexity = item.metrics.complexityScore || 50;

      // Base points from code changes
      let points = 2;
      if (linesChanged > 1000 || filesChanged > 20) points = 13;
      else if (linesChanged > 500 || filesChanged > 10) points = 8;
      else if (linesChanged > 200 || filesChanged > 5) points = 5;
      else if (linesChanged > 50 || filesChanged > 2) points = 3;

      // Adjust for complexity
      const complexityMultiplier = 0.5 + (complexity / 100);
      return Math.round(points * complexityMultiplier);
    }

    // Default based on type
    switch (item.type) {
      case 'epic':
        return 21;
      case 'initiative':
        return 13;
      case 'pr':
        return 5;
      case 'commit':
        return 1;
      default:
        return 3;
    }
  }

  // ==========================================================================
  // IMPACT CALCULATIONS
  // ==========================================================================

  /**
   * Calculate total impact from work items
   */
  private calculateImpact(workItems: WorkItem[]): ImpactMetrics {
    const revenueImpact = this.calculateRevenueImpact(workItems);
    const costReduction = this.calculateCostReduction(workItems);
    const riskReduction = this.calculateRiskReduction(workItems);
    const strategicValue = this.calculateStrategicValue(workItems);
    const developerProductivity = this.calculateDeveloperProductivity(workItems);

    const total =
      revenueImpact +
      costReduction +
      riskReduction +
      strategicValue +
      developerProductivity;

    return {
      revenueImpact,
      costReduction,
      riskReduction,
      strategicValue,
      developerProductivity,
      total,
    };
  }

  /**
   * Calculate revenue impact from work items
   */
  private calculateRevenueImpact(workItems: WorkItem[]): number {
    const revenueItems = workItems.filter(
      (item) => item.roiCategory?.name === 'revenue-impact'
    );

    return revenueItems.reduce((sum, item) => {
      const impactScore = item.impactScore || item.roiCategory?.estimatedImpact || 5;
      const confidence = item.roiCategory?.confidenceLevel || 0.7;
      return sum + impactScore * this.IMPACT_SCALING['revenue-impact'] * confidence;
    }, 0);
  }

  /**
   * Calculate cost reduction from work items
   */
  private calculateCostReduction(workItems: WorkItem[]): number {
    const costItems = workItems.filter(
      (item) => item.roiCategory?.name === 'cost-reduction'
    );

    return costItems.reduce((sum, item) => {
      const impactScore = item.impactScore || item.roiCategory?.estimatedImpact || 5;
      const confidence = item.roiCategory?.confidenceLevel || 0.7;
      return sum + impactScore * this.IMPACT_SCALING['cost-reduction'] * confidence;
    }, 0);
  }

  /**
   * Calculate risk reduction from work items
   */
  private calculateRiskReduction(workItems: WorkItem[]): number {
    const riskItems = workItems.filter(
      (item) => item.roiCategory?.name === 'risk-mitigation'
    );

    return riskItems.reduce((sum, item) => {
      const impactScore = item.impactScore || item.roiCategory?.estimatedImpact || 5;
      const confidence = item.roiCategory?.confidenceLevel || 0.8; // Higher default for risk
      return sum + impactScore * this.IMPACT_SCALING['risk-mitigation'] * confidence;
    }, 0);
  }

  /**
   * Calculate strategic value from work items
   */
  private calculateStrategicValue(workItems: WorkItem[]): number {
    const strategicItems = workItems.filter(
      (item) =>
        item.strategicTags?.some(
          (tag) => tag.strategicAlignment === 'core'
        )
    );

    // Strategic value is based on alignment and importance
    return strategicItems.reduce((sum, item) => {
      const coreTags = item.strategicTags?.filter(
        (tag) => tag.strategicAlignment === 'core'
      ).length || 0;
      const impactScore = item.impactScore || 5;
      return sum + coreTags * impactScore * 1000;
    }, 0);
  }

  /**
   * Calculate developer productivity impact
   */
  private calculateDeveloperProductivity(workItems: WorkItem[]): number {
    const productivityItems = workItems.filter(
      (item) => item.roiCategory?.name === 'developer-productivity'
    );

    // Also include tech-debt items as they improve productivity
    const techDebtItems = workItems.filter(
      (item) =>
        item.strategicTags?.some((tag) => tag.category === 'tech-debt') &&
        !productivityItems.includes(item)
    );

    const allItems = [...productivityItems, ...techDebtItems];

    return allItems.reduce((sum, item) => {
      const impactScore = item.impactScore || item.roiCategory?.estimatedImpact || 5;
      const confidence = item.roiCategory?.confidenceLevel || 0.6;
      return sum + impactScore * this.IMPACT_SCALING['developer-productivity'] * confidence;
    }, 0);
  }

  // ==========================================================================
  // ROI METRICS CALCULATIONS
  // ==========================================================================

  /**
   * Calculate ROI metrics from effort and impact
   */
  private calculateROIMetrics(
    effort: EffortMetrics,
    impact: ImpactMetrics
  ): ROIMetrics {
    const totalGain = impact.total;
    const totalCost = effort.teamCost + effort.opportunityCost;

    const ratio = totalCost > 0 ? (totalGain - totalCost) / totalCost : 0;
    const percentage = ratio * 100;

    // Payback period calculation
    const monthlyBenefit = totalGain / this.ANNUALIZATION_MONTHS;
    const paybackPeriod = monthlyBenefit > 0 ? totalCost / monthlyBenefit : Infinity;

    // Net Present Value (simplified, assuming 3-year horizon)
    const years = 3;
    const annualBenefit = totalGain / this.ANNUALIZATION_MONTHS * 12;
    let npv = -totalCost;
    for (let year = 1; year <= years; year++) {
      npv += annualBenefit / Math.pow(1 + this.DISCOUNT_RATE, year);
    }

    return { ratio, percentage, paybackPeriod, npv };
  }

  // ==========================================================================
  // CATEGORY BREAKDOWN
  // ==========================================================================

  /**
   * Calculate ROI breakdown by category
   */
  private calculateByCategory(workItems: WorkItem[]): Record<ROICategoryType, CategoryROI> {
    const categories: ROICategoryType[] = [
      'revenue-impact',
      'cost-reduction',
      'risk-mitigation',
      'strategic-capability',
      'developer-productivity',
      'customer-experience',
    ];

    const result: Record<ROICategoryType, CategoryROI> = {} as Record<ROICategoryType, CategoryROI>;

    for (const category of categories) {
      const categoryItems = workItems.filter(
        (item) =>
          item.roiCategory?.name === category ||
          item.strategicTags?.some((tag) => {
            const tagToCategoryMap: Record<string, ROICategoryType> = {
              feature: 'customer-experience',
              'tech-debt': 'developer-productivity',
              performance: 'cost-reduction',
              security: 'risk-mitigation',
              compliance: 'risk-mitigation',
              infrastructure: 'strategic-capability',
            };
            return tagToCategoryMap[tag.category] === category;
          })
      );

      const investment = this.calculateEffort(categoryItems).teamCost;
      const impact = this.calculateCategoryImpact(categoryItems, category);
      const roi = investment > 0 ? (impact - investment) / investment : 0;

      result[category] = {
        investment,
        impact,
        roi,
        itemCount: categoryItems.length,
      };
    }

    return result;
  }

  /**
   * Calculate impact for a specific category
   */
  private calculateCategoryImpact(
    workItems: WorkItem[],
    category: ROICategoryType
  ): number {
    if (workItems.length === 0) return 0;

    const scalingFactor = this.IMPACT_SCALING[category] || 5000;

    return workItems.reduce((sum, item) => {
      const impactScore = item.impactScore || item.roiCategory?.estimatedImpact || 5;
      const confidence = item.roiCategory?.confidenceLevel || 0.7;
      return sum + impactScore * scalingFactor * confidence;
    }, 0);
  }

  // ==========================================================================
  // INITIATIVE BREAKDOWN
  // ==========================================================================

  /**
   * Calculate ROI breakdown by initiative
   */
  private calculateByInitiative(
    workItems: WorkItem[]
  ): Array<{
    initiativeId: string;
    name: string;
    roi: number;
    status: 'on_track' | 'at_risk' | 'off_track';
  }> {
    // Group items by initiative
    const byInitiative = new Map<string, WorkItem[]>();

    for (const item of workItems) {
      const initiativeId = item.parentInitiative;
      if (initiativeId) {
        const existing = byInitiative.get(initiativeId) || [];
        byInitiative.set(initiativeId, [...existing, item]);
      }
    }

    // Calculate ROI for each initiative
    return Array.from(byInitiative.entries()).map(([initiativeId, items]) => {
      const effort = this.calculateEffort(items);
      const impact = this.calculateImpact(items);
      const roi = effort.teamCost > 0 ? (impact.total - effort.teamCost) / effort.teamCost : 0;

      // Determine status based on ROI
      let status: 'on_track' | 'at_risk' | 'off_track';
      if (roi > 0.5) {
        status = 'on_track';
      } else if (roi > 0) {
        status = 'at_risk';
      } else {
        status = 'off_track';
      }

      return {
        initiativeId,
        name: `Initiative ${initiativeId.slice(-6)}`, // Truncated name
        roi,
        status,
      };
    });
  }

  // ==========================================================================
  // UTILITY FUNCTIONS
  // ==========================================================================

  /**
   * Classify ROI into categories
   */
  classifyROI(ratio: number): 'high-roi' | 'medium-roi' | 'low-roi' | 'strategic-investment' {
    if (ratio > 2) return 'high-roi';
    if (ratio > 0.5) return 'medium-roi';
    if (ratio > 0) return 'low-roi';
    return 'strategic-investment';
  }

  /**
   * Generate recommendation based on ROI
   */
  generateRecommendation(
    roi: number,
    targetMet: boolean
  ): 'prioritize' | 'maintain' | 'reconsider' | 'deprecated' {
    if (roi > 1 && targetMet) return 'prioritize';
    if (roi > 0 && targetMet) return 'maintain';
    if (roi < 0 && !targetMet) return 'reconsider';
    return 'deprecated';
  }
}

// ============================================================================
// Export Singleton Instance
// ============================================================================

export const roiCalculator = new ROICalculator();
