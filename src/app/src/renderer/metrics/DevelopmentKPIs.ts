/**
 * Development KPIs Calculator
 * Calculates development key performance indicators from work items
 */

import type {
  WorkItem,
  DevelopmentKPIs,
  WorkItemType,
} from '../types/workItem';

interface DateRange {
  start: Date;
  end: Date;
}

interface Sprint {
  start: Date;
  end: Date;
  items: WorkItem[];
}

/**
 * Development KPI Calculator Class
 *
 * Calculates metrics for:
 * - Velocity tracking
 * - Cycle time analysis
 * - PR performance
 * - Code quality
 * - Throughput measurement
 */
export class DevelopmentKPICalculator {
  private readonly SPRINT_DURATION_DAYS = 14;
  private readonly MILLISECONDS_PER_DAY = 1000 * 60 * 60 * 24;

  /**
   * Calculate all development KPIs from work items
   */
  calculate(workItems: WorkItem[], period?: DateRange): DevelopmentKPIs {
    const filteredItems = period ? this.filterByPeriod(workItems, period) : workItems;
    const completedItems = this.getCompletedItems(filteredItems);

    return {
      velocity: this.calculateVelocity(completedItems),
      cycleTime: this.calculateCycleTime(filteredItems, completedItems),
      pullRequest: this.calculatePRMetrics(filteredItems),
      codeQuality: this.calculateCodeQuality(filteredItems),
      throughput: this.calculateThroughput(filteredItems),
    };
  }

  /**
   * Filter work items by date range
   */
  private filterByPeriod(items: WorkItem[], period: DateRange): WorkItem[] {
    return items.filter((item) => {
      const itemDate = new Date(item.createdAt);
      return itemDate >= period.start && itemDate <= period.end;
    });
  }

  /**
   * Get completed work items
   */
  private getCompletedItems(items: WorkItem[]): WorkItem[] {
    return items.filter(
      (item) =>
        item.status === 'done' ||
        item.status === 'merged' ||
        item.status === 'closed'
    );
  }

  // ==========================================================================
  // VELOCITY CALCULATIONS
  // ==========================================================================

  /**
   * Calculate velocity metrics
   */
  private calculateVelocity(completedItems: WorkItem[]): DevelopmentKPIs['velocity'] {
    const currentSprint = this.getCurrentSprintItems(completedItems);
    const previousSprints = this.getPreviousSprints(completedItems, 4);

    const current = this.calculateSprintVelocity(currentSprint);
    const sprintVelocities = previousSprints.map((sprint) =>
      this.calculateSprintVelocity(sprint.items)
    );

    const average = this.calculateAverage([current, ...sprintVelocities]);
    const trend = this.calculateVelocityTrend(current, sprintVelocities);

    // Build history for charting
    const history = [...sprintVelocities, current].reverse();

    return { current, average, trend, history };
  }

  /**
   * Get items completed in current sprint
   */
  private getCurrentSprintItems(items: WorkItem[]): WorkItem[] {
    const now = new Date();
    const sprintStart = new Date(now.getTime() - this.SPRINT_DURATION_DAYS * this.MILLISECONDS_PER_DAY);

    return items.filter((item) => {
      const completedDate = new Date(item.resolvedAt || item.mergedAt || item.closedAt || item.updatedAt);
      return completedDate >= sprintStart;
    });
  }

  /**
   * Get items from previous sprints
   */
  private getPreviousSprints(items: WorkItem[], count: number): Sprint[] {
    const sprints: Sprint[] = [];
    const now = new Date();

    for (let i = 1; i <= count; i++) {
      const sprintEnd = new Date(now.getTime() - i * this.SPRINT_DURATION_DAYS * this.MILLISECONDS_PER_DAY);
      const sprintStart = new Date(sprintEnd.getTime() - this.SPRINT_DURATION_DAYS * this.MILLISECONDS_PER_DAY);

      const sprintItems = items.filter((item) => {
        const completedDate = new Date(item.resolvedAt || item.mergedAt || item.closedAt || item.updatedAt);
        return completedDate >= sprintStart && completedDate < sprintEnd;
      });

      sprints.push({ start: sprintStart, end: sprintEnd, items: sprintItems });
    }

    return sprints;
  }

  /**
   * Calculate velocity for a sprint (story points or item count)
   */
  private calculateSprintVelocity(items: WorkItem[]): number {
    const totalPoints = items.reduce((sum, item) => {
      return sum + (item.metrics?.estimatedPoints || this.estimatePoints(item));
    }, 0);

    return totalPoints;
  }

  /**
   * Calculate velocity trend
   */
  private calculateVelocityTrend(current: number, previous: number[]): 'increasing' | 'stable' | 'decreasing' {
    if (previous.length === 0) return 'stable';

    const averagePrevious = this.calculateAverage(previous);
    const threshold = 0.15; // 15% change threshold

    if (current > averagePrevious * (1 + threshold)) return 'increasing';
    if (current < averagePrevious * (1 - threshold)) return 'decreasing';
    return 'stable';
  }

  // ==========================================================================
  // CYCLE TIME CALCULATIONS
  // ==========================================================================

  /**
   * Calculate cycle time metrics
   */
  private calculateCycleTime(
    allItems: WorkItem[],
    completedItems: WorkItem[]
  ): DevelopmentKPIs['cycleTime'] {
    const cycleTimes = this.calculateIndividualCycleTimes(completedItems);

    const average = this.calculateAverage(cycleTimes);
    const median = this.calculateMedian(cycleTimes);
    const percentile90 = this.calculatePercentile(cycleTimes, 90);

    const byType = {
      issue: this.calculateAverageForType(cycleTimes, completedItems, 'issue'),
      pr: this.calculateAverageForType(cycleTimes, completedItems, 'pr'),
      commit: this.calculateAverageForType(cycleTimes, completedItems, 'commit'),
    };

    const trend = this.calculateCycleTimeTrend(cycleTimes, completedItems);

    return { average, median, percentile90, byType, trend };
  }

  /**
   * Calculate cycle time for each item
   */
  private calculateIndividualCycleTimes(items: WorkItem[]): number[] {
    return items
      .filter((item) => item.resolvedAt || item.mergedAt || item.closedAt)
      .map((item) => {
        const start = new Date(item.createdAt).getTime();
        const end = new Date(item.resolvedAt || item.mergedAt || item.closedAt!).getTime();
        return (end - start) / this.MILLISECONDS_PER_DAY; // Days
      });
  }

  /**
   * Calculate cycle time trend by comparing recent vs older items
   */
  private calculateCycleTimeTrend(cycleTimes: number[], items: WorkItem[]): 'improving' | 'stable' | 'degrading' {
    if (cycleTimes.length < 4) return 'stable';

    const sortedItems = [...items].sort((a, b) =>
      new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime()
    );

    const midpoint = Math.floor(sortedItems.length / 2);
    const recent = sortedItems.slice(0, midpoint);
    const older = sortedItems.slice(midpoint);

    const recentAvg = this.calculateAverage(
      this.calculateIndividualCycleTimes(recent)
    );
    const olderAvg = this.calculateAverage(
      this.calculateIndividualCycleTimes(older)
    );

    const threshold = 0.1; // 10% change threshold

    if (recentAvg < olderAvg * (1 - threshold)) return 'improving';
    if (recentAvg > olderAvg * (1 + threshold)) return 'degrading';
    return 'stable';
  }

  /**
   * Calculate average cycle time for a specific work item type
   */
  private calculateAverageForType(
    cycleTimes: number[],
    items: WorkItem[],
    type: WorkItemType
  ): number {
    const typeItems = items.filter((item) => item.type === type);
    const typeCycleTimes = this.calculateIndividualCycleTimes(typeItems);
    return this.calculateAverage(typeCycleTimes) || 0;
  }

  // ==========================================================================
  // PR METRICS CALCULATIONS
  // ==========================================================================

  /**
   * Calculate pull request specific metrics
   */
  private calculatePRMetrics(items: WorkItem[]): DevelopmentKPIs['pullRequest'] {
    const prItems = items.filter((item) => item.type === 'pr');

    if (prItems.length === 0) {
      return {
        averageTimeToFirstReview: 0,
        averageTimeToMerge: 0,
        mergeRate: 0,
        reworkRate: 0,
      };
    }

    const timeToFirstReview = this.calculateTimeToFirstReview(prItems);
    const timeToMerge = this.calculateTimeToMerge(prItems);
    const merged = prItems.filter((item) => item.mergeStatus === 'merged').length;

    return {
      averageTimeToFirstReview: this.calculateAverage(timeToFirstReview) || 0,
      averageTimeToMerge: this.calculateAverage(timeToMerge) || 0,
      mergeRate: (merged / prItems.length) * 100,
      reworkRate: this.calculateReworkRate(prItems),
    };
  }

  /**
   * Calculate time to first review for PRs
   */
  private calculateTimeToFirstReview(prItems: WorkItem[]): number[] {
    return prItems
      .filter((item) => item.metrics?.timeToFirstReview !== undefined)
      .map((item) => item.metrics!.timeToFirstReview!);
  }

  /**
   * Calculate time to merge for PRs
   */
  private calculateTimeToMerge(prItems: WorkItem[]): number[] {
    return prItems
      .filter((item) => item.mergedAt)
      .map((item) => {
        const start = new Date(item.createdAt).getTime();
        const end = new Date(item.mergedAt!).getTime();
        return (end - start) / (1000 * 60 * 60); // Hours
      });
  }

  /**
   * Calculate rework rate (PRs with changes requested)
   */
  private calculateReworkRate(prItems: WorkItem[]): number {
    const prsWithChangesRequested = prItems.filter(
      (item) => (item.metrics?.changesRequested || 0) > 0
    ).length;

    return (prsWithChangesRequested / prItems.length) * 100;
  }

  // ==========================================================================
  // CODE QUALITY CALCULATIONS
  // ==========================================================================

  /**
   * Calculate code quality metrics
   */
  private calculateCodeQuality(items: WorkItem[]): DevelopmentKPIs['codeQuality'] {
    return {
      codeChurn: this.calculateCodeChurn(items),
      averagePRSize: this.calculateAveragePRSize(items),
      reviewDepth: this.calculateReviewDepth(items),
      defectRate: this.calculateDefectRate(items),
    };
  }

  /**
   * Calculate code churn (percentage of code rewritten within 2 weeks)
   */
  private calculateCodeChurn(items: WorkItem[]): number {
    const TWO_WEEKS_MS = 14 * this.MILLISECONDS_PER_DAY;

    const churnedItems = items.filter((item) => {
      if (!item.resolvedAt && !item.closedAt) return false;

      const resolvedDate = new Date(item.resolvedAt || item.closedAt!).getTime();
      const twoWeeksLater = resolvedDate + TWO_WEEKS_MS;

      // Look for related items created within 2 weeks of resolution
      return items.some((other) => {
        if (other.id === item.id) return false;
        if (new Date(other.createdAt).getTime() > twoWeeksLater) return false;

        // Check if related (same author, similar labels, or linked)
        return this.isRelatedTo(other, item);
      });
    });

    return items.length > 0 ? (churnedItems.length / items.length) * 100 : 0;
  }

  /**
   * Check if two work items are related
   */
  private isRelatedTo(a: WorkItem, b: WorkItem): boolean {
    // Same author
    if (a.author.id === b.author.id) return true;

    // Shared labels
    const sharedLabels = a.labels.filter((la) =>
      b.labels.some((lb) => lb.name === la.name)
    );
    if (sharedLabels.length >= 2) return true;

    // Linked items
    if (a.linkedItems?.some((rel) => rel.targetId === b.id)) return true;
    if (b.linkedItems?.some((rel) => rel.targetId === a.id)) return true;

    return false;
  }

  /**
   * Calculate average PR size (lines changed)
   */
  private calculateAveragePRSize(items: WorkItem[]): number {
    const prItems = items.filter((item) => item.type === 'pr');

    if (prItems.length === 0) return 0;

    const totalLines = prItems.reduce((sum, item) => {
      return (
        sum +
        (item.metrics?.linesAdded || 0) +
        (item.metrics?.linesDeleted || 0)
      );
    }, 0);

    return totalLines / prItems.length;
  }

  /**
   * Calculate review depth (average comments per PR)
   */
  private calculateReviewDepth(items: WorkItem[]): number {
    const prItems = items.filter((item) => item.type === 'pr');

    if (prItems.length === 0) return 0;

    const totalComments = prItems.reduce(
      (sum, item) => sum + (item.metrics?.reviewComments || 0),
      0
    );

    return totalComments / prItems.length;
  }

  /**
   * Calculate defect rate (reopened issues percentage)
   */
  private calculateDefectRate(items: WorkItem[]): number {
    const issueItems = items.filter((item) => item.type === 'issue');

    // Count reopened issues (simplified - would need more data in production)
    const reopenedCount = issueItems.filter((item) => {
      // Check if issue was closed then reopened (would need timeline data)
      return item.status === 'done' && (item.metrics?.commentCount || 0) > 10;
    }).length;

    return issueItems.length > 0 ? (reopenedCount / issueItems.length) * 100 : 0;
  }

  // ==========================================================================
  // THROUGHPUT CALCULATIONS
  // ==========================================================================

  /**
   * Calculate throughput metrics
   */
  private calculateThroughput(items: WorkItem[]): DevelopmentKPIs['throughput'] {
    const completed = items.filter(
      (item) =>
        item.status === 'done' || item.status === 'merged'
    ).length;

    const started = items.filter(
      (item) =>
        item.status === 'in_progress' || item.status === 'in_review'
    ).length;

    const wip = items.filter(
      (item) =>
        item.status === 'in_progress' || item.status === 'in_review'
    ).length;

    const throughputTrend = this.calculateThroughputTrend(items);

    return {
      itemsCompleted: completed,
      itemsStarted: started,
      workInProgress: wip,
      throughputTrend,
    };
  }

  /**
   * Calculate throughput trend (last N periods)
   */
  private calculateThroughputTrend(items: WorkItem[]): number[] {
    const trend: number[] = [];
    const now = new Date();

    // Calculate throughput for last 10 periods (each period = 1 week)
    for (let i = 9; i >= 0; i--) {
      const periodStart = new Date(
        now.getTime() - (i + 1) * 7 * this.MILLISECONDS_PER_DAY
      );
      const periodEnd = new Date(
        now.getTime() - i * 7 * this.MILLISECONDS_PER_DAY
      );

      const completed = items.filter((item) => {
        const completedDate = new Date(
          item.resolvedAt || item.mergedAt || item.closedAt || item.updatedAt
        );
        return completedDate >= periodStart && completedDate < periodEnd;
      }).length;

      trend.push(completed);
    }

    return trend;
  }

  // ==========================================================================
  // UTILITY FUNCTIONS
  // ==========================================================================

  /**
   * Calculate average of an array of numbers
   */
  private calculateAverage(values: number[]): number {
    if (values.length === 0) return 0;
    return values.reduce((sum, v) => sum + v, 0) / values.length;
  }

  /**
   * Calculate median of an array of numbers
   */
  private calculateMedian(values: number[]): number {
    if (values.length === 0) return 0;

    const sorted = [...values].sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);

    if (sorted.length % 2 === 0) {
      return (sorted[mid - 1] + sorted[mid]) / 2;
    }
    return sorted[mid];
  }

  /**
   * Calculate percentile of an array of numbers
   */
  private calculatePercentile(values: number[], percentile: number): number {
    if (values.length === 0) return 0;

    const sorted = [...values].sort((a, b) => a - b);
    const index = Math.ceil((percentile / 100) * sorted.length) - 1;
    return sorted[Math.max(0, index)];
  }

  /**
   * Estimate story points from work item characteristics
   */
  private estimatePoints(item: WorkItem): number {
    if (item.metrics) {
      const linesChanged = (item.metrics.linesAdded || 0) + (item.metrics.linesDeleted || 0);
      const filesChanged = item.metrics.filesChanged || 1;

      if (linesChanged > 1000 || filesChanged > 20) return 13;
      if (linesChanged > 500 || filesChanged > 10) return 8;
      if (linesChanged > 200 || filesChanged > 5) return 5;
      if (linesChanged > 50 || filesChanged > 2) return 3;
      return 2;
    }

    // Default based on type
    return item.type === 'pr' ? 5 : 3;
  }
}

// ============================================================================
// Export Singleton Instance
// ============================================================================

export const developmentKPICalculator = new DevelopmentKPICalculator();
