/**
 * Industry Relevance Scorer
 * Calculates industry relevance scores for work items and initiatives
 *
 * Measures alignment with:
 * - Technical innovation
 * - Market alignment
 * - Competitive parity
 * - Future proofing
 * - Ecosystem integration
 */

import type {
  WorkItem,
  IndustryRelevanceScore,
  StrategicTag,
  ROICategoryType,
} from '../../types/workItem';

// ============================================================================
// Tech Trend Types
// ============================================================================

export interface TechTrend {
  id: string;
  name: string;
  category: 'ai-ml' | 'cloud-native' | 'security' | 'performance' | 'dx' | 'npu';
  maturity: 'emerging' | 'growing' | 'mainstream' | 'mature' | 'declining';
  adoptionRate: number; // Industry adoption percentage
  growthTrajectory: 'accelerating' | 'steady' | 'plateauing' | 'declining';
  relevanceWeights: Record<string, number>; // Work item type -> relevance weight
}

// ============================================================================
// Industry Relevance Scorer Class
// ============================================================================

export class IndustryRelevanceScorer {
  private techTrends: TechTrend[] = this.loadTechTrends();

  /**
   * Calculate industry relevance for work items
   */
  calculateRelevance(workItems: WorkItem[]): IndustryRelevanceScore {
    if (workItems.length === 0) {
      return this.getEmptyScore();
    }

    return {
      overall: this.calculateOverallScore(workItems),
      dimensions: this.calculateDimensions(workItems),
      trends: this.identifyTrends(workItems),
      benchmarks: this.compareWithBenchmarks(workItems),
    };
  }

  /**
   * Calculate relevance for a specific strategic initiative
   */
  calculateInitiativeRelevance(
    initiativeWorkItems: WorkItem[]
  ): IndustryRelevanceScore {
    return this.calculateRelevance(initiativeWorkItems);
  }

  // ==========================================================================
  // Overall Score Calculation
  // ==========================================================================

  private calculateOverallScore(workItems: WorkItem[]): number {
    const dimensions = this.calculateDimensions(workItems);

    const weights = {
      technicalInnovation: 0.25,
      marketAlignment: 0.25,
      competitiveParity: 0.2,
      futureProofing: 0.2,
      ecosystemIntegration: 0.1,
    };

    const overall =
      dimensions.technicalInnovation * weights.technicalInnovation +
      dimensions.marketAlignment * weights.marketAlignment +
      dimensions.competitiveParity * weights.competitiveParity +
      dimensions.futureProofing * weights.futureProofing +
      dimensions.ecosystemIntegration * weights.ecosystemIntegration;

    return Math.round(overall);
  }

  // ==========================================================================
  // Dimension Calculations
  // ==========================================================================

  private calculateDimensions(workItems: WorkItem[]): IndustryRelevanceScore['dimensions'] {
    return {
      technicalInnovation: this.scoreTechnicalInnovation(workItems),
      marketAlignment: this.scoreMarketAlignment(workItems),
      competitiveParity: this.scoreCompetitiveParity(workItems),
      futureProofing: this.scoreFutureProofing(workItems),
      ecosystemIntegration: this.scoreEcosystemIntegration(workItems),
    };
  }

  /**
   * Score technical innovation
   * Based on use of cutting-edge technology, novel approaches, research backing
   */
  private scoreTechnicalInnovation(workItems: WorkItem[]): number {
    const innovationKeywords = [
      'npu',
      'ai',
      'ml',
      'transformer',
      'quantization',
      'edge',
      'neural',
      'inference',
      'acceleration',
      'tensor',
    ];

    const matchCount = workItems.filter((item) =>
      this.matchesKeywords(item, innovationKeywords)
    ).length;

    // Also consider strategic tags
    const techDebtItems = workItems.filter(
      (item) =>
        item.strategicTags?.some((tag) => tag.category === 'tech-debt')
    ).length;

    // Innovation score based on AI/ML focus and tech debt ratio
    const baseScore = (matchCount / workItems.length) * 100 * 2;
    const techDebtPenalty = (techDebtItems / workItems.length) * 20;

    return Math.min(100, Math.max(0, baseScore - techDebtPenalty));
  }

  /**
   * Score market alignment
   * Based on addressing market demands, user requests, pain points
   */
  private scoreMarketAlignment(workItems: WorkItem[]): number {
    const marketKeywords = [
      'performance',
      'compatibility',
      'api',
      'integration',
      'enterprise',
      'customer',
      'user',
      'experience',
      'speed',
      'reliability',
    ];

    const matchCount = workItems.filter((item) =>
      this.matchesKeywords(item, marketKeywords)
    ).length;

    // Consider priority as a signal of market importance
    const highPriorityItems = workItems.filter(
      (item) => item.priority === 'high' || item.priority === 'critical'
    ).length;

    const baseScore = (matchCount / workItems.length) * 100 * 1.5;
    const priorityBonus = (highPriorityItems / workItems.length) * 20;

    return Math.min(100, Math.max(0, baseScore + priorityBonus));
  }

  /**
   * Score competitive parity
   * Based on feature parity with competitors, unique differentiators, standards compliance
   */
  private scoreCompetitiveParity(workItems: WorkItem[]): number {
    const parityKeywords = [
      'compatibility',
      'standard',
      'ollama',
      'openai-api',
      'onnx',
      'interop',
      'protocol',
      'sdk',
      'client',
    ];

    const matchCount = workItems.filter((item) =>
      this.matchesKeywords(item, parityKeywords)
    ).length;

    // Consider labels that indicate standards compliance
    const standardsLabels = ['standard', 'compatibility', 'api', 'protocol'];
    const standardsMatch = workItems.filter((item) =>
      item.labels.some((l) =>
        standardsLabels.some((sl) => l.name.toLowerCase().includes(sl))
      )
    ).length;

    const baseScore = (matchCount / workItems.length) * 100 * 1.5;
    const standardsBonus = (standardsMatch / workItems.length) * 30;

    return Math.min(100, Math.max(0, baseScore + standardsBonus));
  }

  /**
   * Score future proofing
   * Based on using growing vs declining technologies, sustainability, maintenance burden
   */
  private scoreFutureProofing(workItems: WorkItem[]): number {
    const growingTechs = ['npu', 'rust', 'webgpu', 'quantization', 'edge-ai', 'llm'];
    const decliningTechs = ['deprecated', 'legacy', 'v1', 'obsolete', 'sunset'];

    const growingCount = workItems.filter((item) =>
      this.matchesKeywords(item, growingTechs)
    ).length;

    const decliningCount = workItems.filter((item) =>
      this.matchesKeywords(item, decliningTechs)
    ).length;

    // Base score starts at 50 (neutral)
    const baseScore = 50;

    // Adjust based on growing vs declining ratio
    const growthBonus = (growingCount / workItems.length) * 50;
    const decliningPenalty = (decliningCount / workItems.length) * 50;

    return Math.min(100, Math.max(0, baseScore + growthBonus - decliningPenalty));
  }

  /**
   * Score ecosystem integration
   * Based on API compatibility, standard formats, plugin/extension points
   */
  private scoreEcosystemIntegration(workItems: WorkItem[]): number {
    const integrationKeywords = [
      'api',
      'sdk',
      'plugin',
      'extension',
      'webhook',
      'rest',
      'integration',
      'connector',
      'adapter',
      'bridge',
    ];

    const matchCount = workItems.filter((item) =>
      this.matchesKeywords(item, integrationKeywords)
    ).length;

    // Consider strategic tags for infrastructure
    const infrastructureItems = workItems.filter((item) =>
      item.strategicTags?.some((tag) => tag.category === 'infrastructure')
    ).length;

    const baseScore = (matchCount / workItems.length) * 100 * 1.5;
    const infrastructureBonus = (infrastructureItems / workItems.length) * 30;

    return Math.min(100, Math.max(0, baseScore + infrastructureBonus));
  }

  // ==========================================================================
  // Trend Identification
  // ==========================================================================

  private identifyTrends(workItems: WorkItem[]): IndustryRelevanceScore['trends'] {
    const emerging: string[] = [];
    const declining: string[] = [];

    for (const trend of this.techTrends) {
      const relevance = workItems.some(
        (item) => this.matchesKeywords(item, [trend.name.toLowerCase()])
      );

      if (relevance) {
        if (trend.maturity === 'emerging' || trend.maturity === 'growing') {
          emerging.push(trend.name);
        } else if (trend.maturity === 'declining') {
          declining.push(trend.name);
        }
      }
    }

    // Add detected technology trends from work items
    const detectedTechs = this.detectTechnologies(workItems);
    for (const tech of detectedTechs.growing) {
      if (!emerging.includes(tech)) {
        emerging.push(tech);
      }
    }
    for (const tech of detectedTechs.declining) {
      if (!declining.includes(tech)) {
        declining.push(tech);
      }
    }

    return { emerging, declining };
  }

  private detectTechnologies(workItems: WorkItem[]): {
    growing: string[];
    declining: string[];
  } {
    const growingTechs = ['NPU', 'Edge AI', 'Quantization', 'LLM', 'Transformer'];
    const decliningTechs: string[] = [];

    const foundGrowing: string[] = [];

    for (const item of workItems) {
      for (const tech of growingTechs) {
        if (
          this.matchesKeywords(item, [tech.toLowerCase()]) &&
          !foundGrowing.includes(tech)
        ) {
          foundGrowing.push(tech);
        }
      }
    }

    return { growing: foundGrowing, declining: decliningTechs };
  }

  // ==========================================================================
  // Benchmark Comparison
  // ==========================================================================

  private compareWithBenchmarks(workItems: WorkItem[]): IndustryRelevanceScore['benchmarks'] {
    const industryAverage = 65;
    const leaderAverage = 85;
    const ourScore = this.calculateOverallScore(workItems);

    return {
      industryAverage,
      leaderAverage,
      ourScore,
    };
  }

  // ==========================================================================
  // Utility Methods
  // ==========================================================================

  private matchesKeywords(item: WorkItem, keywords: string[]): boolean {
    const searchText = `
      ${item.title.toLowerCase()}
      ${item.description?.toLowerCase() || ''}
      ${item.labels.map((l) => l.name.toLowerCase()).join(' ')}
      ${item.strategicTags?.map((t) => t.name.toLowerCase()).join(' ') || ''}
    `.toLowerCase();

    return keywords.some((keyword) => searchText.includes(keyword.toLowerCase()));
  }

  private loadTechTrends(): TechTrend[] {
    // Technology trends database
    // In production, this would be loaded from an external source
    return [
      {
        id: 'npu-acceleration',
        name: 'NPU/AI Acceleration',
        category: 'npu',
        maturity: 'growing',
        adoptionRate: 35,
        growthTrajectory: 'accelerating',
        relevanceWeights: { pr: 0.9, issue: 0.7, commit: 0.5, initiative: 0.8 },
      },
      {
        id: 'edge-ai',
        name: 'Edge AI',
        category: 'ai-ml',
        maturity: 'growing',
        adoptionRate: 45,
        growthTrajectory: 'accelerating',
        relevanceWeights: { pr: 0.8, issue: 0.8, commit: 0.6, initiative: 0.9 },
      },
      {
        id: 'llm-inference',
        name: 'LLM Inference',
        category: 'ai-ml',
        maturity: 'mainstream',
        adoptionRate: 60,
        growthTrajectory: 'steady',
        relevanceWeights: { pr: 0.9, issue: 0.7, commit: 0.5, initiative: 0.8 },
      },
      {
        id: 'quantization',
        name: 'Model Quantization',
        category: 'ai-ml',
        maturity: 'growing',
        adoptionRate: 40,
        growthTrajectory: 'accelerating',
        relevanceWeights: { pr: 0.8, issue: 0.6, commit: 0.7, initiative: 0.7 },
      },
      {
        id: 'openai-compatibility',
        name: 'OpenAI API Compatibility',
        category: 'cloud-native',
        maturity: 'mainstream',
        adoptionRate: 75,
        growthTrajectory: 'steady',
        relevanceWeights: { pr: 0.7, issue: 0.8, commit: 0.4, initiative: 0.6 },
      },
      {
        id: 'rust-rewrite',
        name: 'Rust Migration',
        category: 'dx',
        maturity: 'growing',
        adoptionRate: 30,
        growthTrajectory: 'accelerating',
        relevanceWeights: { pr: 0.9, issue: 0.7, commit: 0.8, initiative: 0.8 },
      },
    ];
  }

  private getEmptyScore(): IndustryRelevanceScore {
    return {
      overall: 0,
      dimensions: {
        technicalInnovation: 0,
        marketAlignment: 0,
        competitiveParity: 0,
        futureProofing: 0,
        ecosystemIntegration: 0,
      },
      trends: {
        emerging: [],
        declining: [],
      },
      benchmarks: {
        industryAverage: 65,
        leaderAverage: 85,
        ourScore: 0,
      },
    };
  }

  // ==========================================================================
  // Public Analysis Methods
  // ==========================================================================

  /**
   * Get trend analysis for a specific work item
   */
  analyzeWorkItemTrend(item: WorkItem): {
    alignedTrends: string[];
    score: number;
    recommendations: string[];
  } {
    const alignedTrends: string[] = [];
    const recommendations: string[] = [];

    for (const trend of this.techTrends) {
      if (this.matchesKeywords(item, [trend.name.toLowerCase()])) {
        alignedTrends.push(trend.name);
      }
    }

    // Generate recommendations based on gaps
    if (alignedTrends.length === 0) {
      recommendations.push(
        'Consider aligning work with emerging technology trends like NPU acceleration or Edge AI'
      );
    }

    const score = alignedTrends.length * 20;

    return {
      alignedTrends,
      score: Math.min(100, score),
      recommendations,
    };
  }

  /**
   * Get portfolio-level trend analysis
   */
  analyzePortfolio(workItems: WorkItem[]): {
    topAlignedTrends: { trend: string; count: number }[];
    missingTrends: string[];
    overallAlignment: number;
  } {
    const trendCounts = new Map<string, number>();

    for (const item of workItems) {
      const analysis = this.analyzeWorkItemTrend(item);
      for (const trend of analysis.alignedTrends) {
        trendCounts.set(trend, (trendCounts.get(trend) || 0) + 1);
      }
    }

    const topAlignedTrends = Array.from(trendCounts.entries())
      .map(([trend, count]) => ({ trend, count }))
      .sort((a, b) => b.count - a.count)
      .slice(0, 5);

    const alignedTrendNames = new Set(trendCounts.keys());
    const missingTrends = this.techTrends
      .filter((t) => !alignedTrendNames.has(t.name))
      .filter((t) => t.maturity === 'growing' || t.maturity === 'emerging')
      .map((t) => t.name);

    const overallAlignment =
      workItems.length > 0
        ? (trendCounts.size / this.techTrends.length) * 100
        : 0;

    return {
      topAlignedTrends,
      missingTrends,
      overallAlignment: Math.round(overallAlignment),
    };
  }
}

// ============================================================================
// Export singleton instance
// ============================================================================

export const industryRelevanceScorer = new IndustryRelevanceScorer();
