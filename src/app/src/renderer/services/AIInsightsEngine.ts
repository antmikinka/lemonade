/**
 * AI Insights Engine
 * Generates actionable insights from work items and metrics
 *
 * Provides:
 * - Trend analysis
 * - Anomaly detection
 * - Recommendations
 * - Risk identification
 * - Opportunity surfacing
 */

import type {
  WorkItem,
  AIInsight,
  DataPoint,
  DevelopmentKPIs,
  PortfolioROI,
  IndustryRelevanceScore,
  StrategicInitiative,
} from '../../types/workItem';

// ============================================================================
// Insight Generation Configuration
// ============================================================================

interface InsightConfig {
  velocityDropThreshold: number; // Percentage drop to trigger alert
  cycleTimeIncreaseThreshold: number; // Percentage increase to trigger alert
  prReviewTimeThreshold: number; // Hours
  codeChurnThreshold: number; // Percentage
  wipLimitPerDeveloper: number;
  techDebtThreshold: number; // Percentage
}

const DEFAULT_CONFIG: InsightConfig = {
  velocityDropThreshold: 0.15, // 15% drop
  cycleTimeIncreaseThreshold: 0.25, // 25% increase
  prReviewTimeThreshold: 48, // hours
  codeChurnThreshold: 0.20, // 20%
  wipLimitPerDeveloper: 3,
  techDebtThreshold: 0.30, // 30%
};

// ============================================================================
// AI Insights Engine Class
// ============================================================================

export class AIInsightsEngine {
  private config: InsightConfig;

  constructor(config?: Partial<InsightConfig>) {
    this.config = { ...DEFAULT_CONFIG, ...config };
  }

  /**
   * Generate insights from work items and metrics
   */
  generateInsights(
    workItems: WorkItem[],
    devKPIs: DevelopmentKPIs,
    roiAnalysis: PortfolioROI,
    relevanceScore: IndustryRelevanceScore,
    initiatives?: StrategicInitiative[]
  ): AIInsight[] {
    const insights: AIInsight[] = [];

    // Analyze trends
    insights.push(...this.analyzeTrends(workItems, devKPIs));

    // Detect anomalies
    insights.push(...this.detectAnomalies(workItems, devKPIs));

    // Generate recommendations
    insights.push(...this.generateRecommendations(devKPIs, roiAnalysis));

    // Identify risks
    insights.push(...this.identifyRisks(workItems, devKPIs, initiatives));

    // Surface opportunities
    insights.push(...this.identifyOpportunities(workItems, relevanceScore));

    // Sort by severity and confidence
    return insights.sort((a, b) => {
      const severityOrder = { critical: 4, high: 3, medium: 2, low: 1 };
      const severityDiff = severityOrder[b.severity] - severityOrder[a.severity];
      if (severityDiff !== 0) return severityDiff;
      return b.confidence - a.confidence;
    });
  }

  // ==========================================================================
  // Trend Analysis
  // ==========================================================================

  private analyzeTrends(
    workItems: WorkItem[],
    devKPIs: DevelopmentKPIs
  ): AIInsight[] {
    const insights: AIInsight[] = [];

    // Velocity trend analysis
    if (devKPIs.velocity.trend === 'increasing') {
      insights.push({
        id: this.generateId('trend-velocity-up'),
        type: 'trend',
        severity: 'low',
        title: 'Velocity Increasing',
        description: `Team velocity has increased to ${devKPIs.velocity.current} points this sprint, above the average of ${devKPIs.velocity.average.toFixed(1)}.`,
        dataPoints: [
          {
            label: 'Current Velocity',
            value: devKPIs.velocity.current,
            trend: 'up',
          },
          {
            label: 'Average Velocity',
            value: devKPIs.velocity.average,
          },
        ],
        confidence: 0.9,
        generatedAt: new Date().toISOString(),
      });
    } else if (devKPIs.velocity.trend === 'decreasing') {
      insights.push({
        id: this.generateId('trend-velocity-down'),
        type: 'trend',
        severity: 'medium',
        title: 'Velocity Decreasing',
        description: `Team velocity has decreased to ${devKPIs.velocity.current} points, below the average of ${devKPIs.velocity.average.toFixed(1)}. Consider investigating blockers.`,
        dataPoints: [
          {
            label: 'Current Velocity',
            value: devKPIs.velocity.current,
            trend: 'down',
          },
          {
            label: 'Average Velocity',
            value: devKPIs.velocity.average,
          },
        ],
        recommendedAction:
          'Review sprint retrospectives for identified blockers. Check team capacity and workload distribution.',
        confidence: 0.85,
        generatedAt: new Date().toISOString(),
      });
    }

    // Cycle time trend analysis
    if (devKPIs.cycleTime.trend === 'degrading') {
      insights.push({
        id: this.generateId('trend-cycle-time-degrading'),
        type: 'trend',
        severity: 'high',
        title: 'Cycle Time Degrading',
        description: `Average cycle time has increased to ${devKPIs.cycleTime.average.toFixed(1)} days. This may indicate process inefficiencies or scope creep.`,
        dataPoints: [
          {
            label: 'Current Cycle Time',
            value: devKPIs.cycleTime.average,
            trend: 'up',
          },
          {
            label: 'Median Cycle Time',
            value: devKPIs.cycleTime.median,
          },
          {
            label: '90th Percentile',
            value: devKPIs.cycleTime.percentile90,
          },
        ],
        recommendedAction:
          'Analyze work items with longest cycle times. Consider breaking down large items and reducing WIP.',
        confidence: 0.8,
        generatedAt: new Date().toISOString(),
      });
    } else if (devKPIs.cycleTime.trend === 'improving') {
      insights.push({
        id: this.generateId('trend-cycle-time-improving'),
        type: 'trend',
        severity: 'low',
        title: 'Cycle Time Improving',
        description: `Average cycle time has improved to ${devKPIs.cycleTime.average.toFixed(1)} days, showing better delivery efficiency.`,
        dataPoints: [
          {
            label: 'Current Cycle Time',
            value: devKPIs.cycleTime.average,
            trend: 'down',
          },
          {
            label: 'Median Cycle Time',
            value: devKPIs.cycleTime.median,
          },
        ],
        confidence: 0.85,
        generatedAt: new Date().toISOString(),
      });
    }

    return insights;
  }

  // ==========================================================================
  // Anomaly Detection
  // ==========================================================================

  private detectAnomalies(
    workItems: WorkItem[],
    devKPIs: DevelopmentKPIs
  ): AIInsight[] {
    const insights: AIInsight[] = [];

    // Detect PR review bottlenecks
    if (devKPIs.pullRequest.averageTimeToFirstReview > this.config.prReviewTimeThreshold) {
      const avgReviewTime = devKPIs.pullRequest.averageTimeToFirstReview;
      insights.push({
        id: this.generateId('anomaly-pr-review-bottleneck'),
        type: 'anomaly',
        severity: 'high',
        title: 'PR Review Bottleneck Detected',
        description: `Average time to first review is ${avgReviewTime.toFixed(1)} hours, indicating a potential review bottleneck.`,
        dataPoints: [
          {
            label: 'Time to First Review',
            value: avgReviewTime,
            trend: 'up',
            comparison: {
              baseline: 24,
              change: avgReviewTime - 24,
              changePercent: ((avgReviewTime - 24) / 24) * 100,
            },
          },
        ],
        recommendedAction:
          'Consider implementing review rotation, reducing PR sizes, or adding reviewers to the team.',
        confidence: 0.9,
        generatedAt: new Date().toISOString(),
      });
    }

    // Detect high code churn
    if (devKPIs.codeQuality.codeChurn > this.config.codeChurnThreshold * 100) {
      const churn = devKPIs.codeQuality.codeChurn;
      insights.push({
        id: this.generateId('anomaly-high-code-churn'),
        type: 'anomaly',
        severity: 'medium',
        title: 'High Code Churn Detected',
        description: `${churn.toFixed(1)}% of code is being rewritten within 2 weeks of completion, suggesting requirements instability or technical debt.`,
        dataPoints: [
          {
            label: 'Code Churn Rate',
            value: churn,
            trend: 'up',
            comparison: {
              baseline: 15,
              change: churn - 15,
              changePercent: ((churn - 15) / 15) * 100,
            },
          },
        ],
        recommendedAction:
          'Review requirements gathering process. Consider more thorough design reviews before implementation.',
        confidence: 0.75,
        generatedAt: new Date().toISOString(),
      });
    }

    // Detect large PR sizes
    if (devKPIs.codeQuality.averagePRSize > 400) {
      insights.push({
        id: this.generateId('anomaly-large-prs'),
        type: 'anomaly',
        severity: 'medium',
        title: 'Large Pull Requests Detected',
        description: `Average PR size is ${devKPIs.codeQuality.averagePRSize.toFixed(0)} lines, which may lead to slower reviews and more bugs.`,
        dataPoints: [
          {
            label: 'Average PR Size',
            value: devKPIs.codeQuality.averagePRSize,
            trend: 'up',
            comparison: {
              baseline: 200,
              change: devKPIs.codeQuality.averagePRSize - 200,
              changePercent: ((devKPIs.codeQuality.averagePRSize - 200) / 200) * 100,
            },
          },
        ],
        recommendedAction:
          'Encourage smaller, focused PRs. Consider splitting large features into incremental deliverables.',
        confidence: 0.8,
        generatedAt: new Date().toISOString(),
      });
    }

    return insights;
  }

  // ==========================================================================
  // Recommendation Generation
  // ==========================================================================

  private generateRecommendations(
    devKPIs: DevelopmentKPIs,
    roiAnalysis: PortfolioROI
  ): AIInsight[] {
    const insights: AIInsight[] = [];

    // ROI-based recommendations
    if (roiAnalysis.roi.ratio < 0.5) {
      insights.push({
        id: this.generateId('rec-low-roi'),
        type: 'recommendation',
        severity: 'high',
        title: 'Low ROI Alert',
        description: `Current initiatives show an ROI ratio of ${roiAnalysis.roi.ratio.toFixed(2)}, below the target of 0.5. Consider reprioritizing work.`,
        dataPoints: [
          { label: 'Current ROI', value: roiAnalysis.roi.ratio },
          { label: 'Target ROI', value: 0.5 },
          { label: 'Total Investment', value: roiAnalysis.totalInvestment.teamCost },
        ],
        recommendedAction:
          'Review strategic initiatives and focus on high-impact work. Consider sunsetting low-impact projects.',
        confidence: 0.8,
        generatedAt: new Date().toISOString(),
      });
    }

    // WIP optimization recommendation
    if (devKPIs.throughput.workInProgress > 10) {
      insights.push({
        id: this.generateId('rec-high-wip'),
        type: 'recommendation',
        severity: 'medium',
        title: 'High Work in Progress',
        description: `Team has ${devKPIs.throughput.workInProgress} items in progress. Research suggests limiting WIP improves throughput and quality.`,
        dataPoints: [
          { label: 'Current WIP', value: devKPIs.throughput.workInProgress },
          {
            label: 'Recommended WIP',
            value: Math.max(3, devKPIs.throughput.itemsCompleted / 2),
          },
        ],
        recommendedAction:
          'Limit WIP to 2-3 items per developer. Finish current work before starting new items.',
        confidence: 0.85,
        generatedAt: new Date().toISOString(),
      });
    }

    // Review depth recommendation
    if (devKPIs.codeQuality.reviewDepth < 2) {
      insights.push({
        id: this.generateId('rec-low-review-depth'),
        type: 'recommendation',
        severity: 'medium',
        title: 'Shallow Code Reviews',
        description: `Average review depth is ${devKPIs.codeQuality.reviewDepth.toFixed(1)} comments per PR, which may indicate superficial reviews.`,
        dataPoints: [
          {
            label: 'Current Review Depth',
            value: devKPIs.codeQuality.reviewDepth,
          },
          { label: 'Recommended Range', value: 5, unit: '3-8 comments' },
        ],
        recommendedAction:
          'Encourage thorough code reviews. Consider review checklists and pair programming.',
        confidence: 0.7,
        generatedAt: new Date().toISOString(),
      });
    }

    return insights;
  }

  // ==========================================================================
  // Risk Identification
  // ==========================================================================

  private identifyRisks(
    workItems: WorkItem[],
    devKPIs: DevelopmentKPIs,
    initiatives?: StrategicInitiative[]
  ): AIInsight[] {
    const insights: AIInsight[] = [];

    // Single point of failure risk
    const assigneeDistribution = this.calculateAssigneeDistribution(workItems);
    for (const [assignee, percentage] of Object.entries(assigneeDistribution)) {
      if (percentage > 40) {
        insights.push({
          id: this.generateId(`risk-sPOF-${assignee.replace(/\s+/g, '-').toLowerCase()}`),
          type: 'risk',
          severity: 'high',
          title: `Single Point of Failure: ${assignee}`,
          description: `${assignee} is assigned to ${percentage.toFixed(0)}% of active work items, creating a bus factor risk.`,
          dataPoints: [
            { label: `${assignee}'s Workload`, value: percentage },
            { label: 'Recommended Max', value: 25 },
          ],
          recommendedAction:
            'Distribute work more evenly. Implement pair programming and knowledge sharing.',
          confidence: 0.9,
          generatedAt: new Date().toISOString(),
        });
      }
    }

    // Technical debt accumulation
    const techDebtItems = workItems.filter((item) =>
      item.strategicTags?.some((tag) => tag.category === 'tech-debt')
    ).length;

    if (workItems.length > 0 && techDebtItems / workItems.length > this.config.techDebtThreshold) {
      const debtRatio = (techDebtItems / workItems.length) * 100;
      insights.push({
        id: this.generateId('risk-tech-debt'),
        type: 'risk',
        severity: 'medium',
        title: 'Technical Debt Accumulation',
        description: `${debtRatio.toFixed(0)}% of work items are tagged as technical debt.`,
        dataPoints: [
          { label: 'Tech Debt Items', value: techDebtItems },
          { label: 'Total Items', value: workItems.length },
          { label: 'Debt Ratio', value: debtRatio },
        ],
        recommendedAction:
          'Allocate 20-30% of capacity to technical debt reduction. Track debt paydown as a metric.',
        confidence: 0.8,
        generatedAt: new Date().toISOString(),
      });
    }

    // Initiative health risk
    if (initiatives) {
      const atRiskInitiatives = initiatives.filter(
        (i) => i.healthStatus === 'red' || i.healthStatus === 'yellow'
      );

      if (atRiskInitiatives.length > 0) {
        insights.push({
          id: this.generateId('risk-initiatives'),
          type: 'risk',
          severity: 'high',
          title: 'Initiatives at Risk',
          description: `${atRiskInitiatives.length} strategic initiative(s) are showing warning or critical status.`,
          dataPoints: atRiskInitiatives.map((i) => ({
            label: i.name,
            value: i.progress,
            trend: i.healthStatus === 'red' ? 'down' : 'stable',
          })),
          recommendedAction:
            'Review initiative status with stakeholders. Identify blockers and allocate additional resources if needed.',
          confidence: 0.85,
          generatedAt: new Date().toISOString(),
        });
      }
    }

    // Burnout risk from high utilization
    const capacity = devKPIs.team?.capacity;
    if (capacity && capacity.utilizationRate > 0.9) {
      insights.push({
        id: this.generateId('risk-burnout'),
        type: 'risk',
        severity: 'high',
        title: 'Team Burnout Risk',
        description: `Team utilization is at ${Math.round(capacity.utilizationRate * 100)}%, which may lead to burnout.`,
        dataPoints: [
          { label: 'Current Utilization', value: capacity.utilizationRate * 100 },
          { label: 'Sustainable Utilization', value: 80 },
        ],
        recommendedAction:
          'Reduce workload or increase capacity. Ensure team members have time for breaks and recovery.',
        confidence: 0.75,
        generatedAt: new Date().toISOString(),
      });
    }

    return insights;
  }

  // ==========================================================================
  // Opportunity Identification
  // ==========================================================================

  private identifyOpportunities(
    workItems: WorkItem[],
    relevanceScore: IndustryRelevanceScore
  ): AIInsight[] {
    const insights: AIInsight[] = [];

    // Emerging trend alignment opportunity
    if (relevanceScore.trends.emerging.length > 0) {
      insights.push({
        id: this.generateId('opp-emerging-trends'),
        type: 'opportunity',
        severity: 'low',
        title: 'Emerging Trend Alignment',
        description: `Current work aligns with ${relevanceScore.trends.emerging.length} emerging industry trends: ${relevanceScore.trends.emerging.join(', ')}.`,
        dataPoints: [
          { label: 'Emerging Trends', value: relevanceScore.trends.emerging.length },
          { label: 'Industry Relevance Score', value: relevanceScore.overall },
        ],
        recommendedAction:
          'Consider documenting and sharing these innovations. Leverage for marketing and recruiting.',
        confidence: 0.7,
        generatedAt: new Date().toISOString(),
      });
    }

    // Benchmark gap opportunity
    if (relevanceScore.benchmarks.ourScore < relevanceScore.benchmarks.leaderAverage) {
      const gap = relevanceScore.benchmarks.leaderAverage - relevanceScore.benchmarks.ourScore;
      const lowestDimension = this.getLowestDimension(relevanceScore.dimensions);
      insights.push({
        id: this.generateId('opp-benchmark-gap'),
        type: 'opportunity',
        severity: 'medium',
        title: 'Industry Leadership Opportunity',
        description: `Your industry relevance score (${relevanceScore.benchmarks.ourScore}) is ${gap} points below industry leaders (${relevanceScore.benchmarks.leaderAverage}).`,
        dataPoints: [
          { label: 'Our Score', value: relevanceScore.benchmarks.ourScore },
          { label: 'Leader Average', value: relevanceScore.benchmarks.leaderAverage },
          { label: 'Gap', value: gap },
        ],
        recommendedAction: `Focus on ${lowestDimension} to close the gap with industry leaders.`,
        confidence: 0.75,
        generatedAt: new Date().toISOString(),
      });
    }

    // High-ROI initiatives opportunity
    if (relevanceScore.overall >= 70) {
      insights.push({
        id: this.generateId('opp-high-relevance'),
        type: 'opportunity',
        severity: 'low',
        title: 'Strong Industry Alignment',
        description: `Your industry relevance score of ${relevanceScore.overall} positions you competitively in the market.`,
        dataPoints: [
          { label: 'Our Score', value: relevanceScore.overall },
          { label: 'Industry Average', value: relevanceScore.benchmarks.industryAverage },
        ],
        recommendedAction:
          'Document best practices and consider open-sourcing non-core innovations to build brand recognition.',
        confidence: 0.8,
        generatedAt: new Date().toISOString(),
      });
    }

    return insights;
  }

  // ==========================================================================
  // Utility Methods
  // ==========================================================================

  private calculateAssigneeDistribution(workItems: WorkItem[]): Record<string, number> {
    const activeItems = workItems.filter(
      (item) => item.status === 'in_progress' || item.status === 'in_review'
    );

    if (activeItems.length === 0) return {};

    const distribution: Record<string, number> = {};
    const total = activeItems.length;

    for (const item of activeItems) {
      for (const assignee of item.assignees) {
        distribution[assignee.name] = (distribution[assignee.name] || 0) + 1;
      }
    }

    for (const name of Object.keys(distribution)) {
      distribution[name] = (distribution[name] / total) * 100;
    }

    return distribution;
  }

  private getLowestDimension(
    dimensions: IndustryRelevanceScore['dimensions']
  ): string {
    const entries = Object.entries(dimensions);
    const lowest = entries.reduce((min, [key, value]) =>
      value < min[1] ? [key, value] : min
    , entries[0]);
    return lowest[0].replace(/([A-Z])/g, ' $1').trim();
  }

  private generateId(prefix: string): string {
    return `${prefix}-${Date.now()}-${Math.random().toString(36).substr(2, 5)}`;
  }
}

// ============================================================================
// Export singleton instance
// ============================================================================

export const aiInsightsEngine = new AIInsightsEngine();
