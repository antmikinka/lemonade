# Team Tracking Dashboard: AI-Automated Re-Architecture

## Executive Summary

This document outlines the fundamental transformation of the Team Tracking Dashboard from a **manual issue management tool** to an **AI-powered strategic intelligence platform**. The new architecture passively syncs from GitHub/Git repositories, requiring minimal human intervention beyond claiming ownership of work items.

### Key Paradigm Shifts

| Current State | Future State |
|---------------|--------------|
| Manual issue creation | Auto-synced from GitHub Issues/PRs |
| Manual status updates | Git-state driven status transitions |
| Basic velocity tracking | Multi-dimensional strategic metrics |
| localStorage persistence | Enterprise data layer with sync |
| Reactive reporting | AI-generated insights & recommendations |
| Individual issue focus | Portfolio-level ROI & impact analysis |

---

## 1. Revised Technical Architecture

### 1.1 High-Level Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    STRATEGIC INTELLIGENCE DASHBOARD                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐      │
│  │   GitHub API     │  │   Git History    │  │  Manual Override │      │
│  │   Integration    │  │   Analyzer       │  │  Layer           │      │
│  │                  │  │                  │  │                  │      │
│  │  • Issues        │  │  • Commits       │  │  • Claim assign  │      │
│  │  • Pull Requests │  │  • Code churn    │  │  • Set priorities│      │
│  │  • Reviews       │  │  • Velocity      │  │  • ROI tags      │      │
│  │  • Comments      │  │  • Contributors  │  │  • Impact scores │      │
│  └────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘      │
│           │                     │                     │                 │
│           └─────────────────────┼─────────────────────┘                 │
│                                 ▼                                       │
│                   ┌─────────────────────────┐                           │
│                   │   DATA SYNCHRONIZATION  │                           │
│                   │   ORCHESTRATION LAYER   │                           │
│                   │                         │                           │
│                   │  • Scheduled sync jobs  │                           │
│                   │  • Incremental updates  │                           │
│                   │  • Conflict resolution  │                           │
│                   │  • Rate limit handling  │                           │
│                   └───────────┬─────────────┘                           │
│                               │                                         │
│                               ▼                                         │
│                   ┌─────────────────────────┐                           │
│                   │    UNIFIED DATA MODEL   │                           │
│                   │                         │                           │
│                   │  • WorkItem (unified)   │                           │
│                   │  • Contribution         │                           │
│                   │  • Metric               │                           │
│                   │  • StrategicTag         │                           │
│                   └───────────┬─────────────┘                           │
│                               │                                         │
│         ┌─────────────────────┼─────────────────────┐                   │
│         │                     │                     │                   │
│         ▼                     ▼                     ▼                   │
│ ┌──────────────┐     ┌──────────────┐     ┌──────────────┐             │
│ │  METRICS     │     │  STRATEGIC   │     │   AI         │             │
│ │  ENGINE      │     │  SCORING     │     │   INSIGHTS   │             │
│ │              │     │              │     │              │             │
│ │ • Dev KPIs   │     │ • ROI calc   │     │ • Trend detect│            │
│ │ • Cycle time │     │ • Impact score│    │ • Anomalies  │             │
│ │ • Quality    │     │ • Relevance  │     │ • Recommendations│         │
│ └──────────────┘     └──────────────┘     └──────────────┘             │
│                               │                                         │
│                               ▼                                         │
│                   ┌─────────────────────────┐                           │
│                   │    PRESENTATION LAYER   │                           │
│                   │                         │                           │
│                   │  • Strategic Dashboard  │                           │
│                   │  • Portfolio Views      │                           │
│                   │  • Team Performance     │                           │
│                   │  • ROI/Impact Reports   │                           │
│                   └─────────────────────────┘                           │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Layer Descriptions

#### Layer 1: Data Sources (Auto-Sync)

| Source | Data Collected | Sync Frequency | Human Input Required |
|--------|---------------|----------------|---------------------|
| GitHub API | Issues, PRs, Reviews, Comments | 5-min polling | Claim assignee only |
| Git History | Commits, code churn, file changes | On-demand refresh | None |
| Manual Override | Priority adjustments, ROI tags, Strategic alignment | Real-time | Strategic tagging only |

#### Layer 2: Data Synchronization Orchestration

```typescript
// Proposed: src/app/src/renderer/services/sync/SyncOrchestrator.ts

interface SyncConfig {
  githubToken: string;
  repository: string;
  syncInterval: number; // milliseconds
  enableGitAnalysis: boolean;
}

interface SyncJob {
  id: string;
  type: 'full' | 'incremental' | 'git-analysis';
  status: 'pending' | 'running' | 'completed' | 'failed';
  lastRun?: Date;
  nextRun: Date;
  result?: SyncResult;
}
```

#### Layer 3: Unified Data Model

The current `Issue` interface expands to a `WorkItem` that unifies:
- GitHub Issues
- Pull Requests
- Local manual items
- Strategic initiatives

---

## 2. GitHub API Integration Strategy

### 2.1 Integration Architecture

```typescript
// Proposed: src/app/src/renderer/types/github.ts

/**
 * GitHub API Response Types (normalized)
 */
interface GitHubIssue {
  id: number;
  number: number;
  title: string;
  body: string;
  state: 'open' | 'closed';
  labels: GitHubLabel[];
  assignees: GitHubUser[];
  author: GitHubUser;
  created_at: string;
  updated_at: string;
  closed_at?: string;
  milestone?: GitHubMilestone;
  pull_request?: {
    url: string;
    merged_at?: string;
    diff_url: string;
  };
}

interface GitHubPullRequest {
  id: number;
  number: number;
  title: string;
  body: string;
  state: 'open' | 'closed';
  merged: boolean;
  merged_at?: string;
  merge_commit_sha?: string;
  additions: number;
  deletions: number;
  changed_files: number;
  commits: number;
  review_comments: number;
  author: GitHubUser;
  requested_reviewers: GitHubUser[];
  reviews: GitHubReview[];
  created_at: string;
  updated_at: string;
}

interface GitHubCommit {
  sha: string;
  message: string;
  author: GitHubUser;
  committer: GitHubUser;
  timestamp: string;
  files: GitHubFile[];
  stats: {
    total: number;
    additions: number;
    deletions: number;
  };
  parents: string[];
}

/**
 * Normalized Work Item - Unifies Issues, PRs, and manual items
 */
interface WorkItem {
  // Identity
  id: string;
  source: 'github-issue' | 'github-pr' | 'github-commit' | 'manual';
  externalId?: string; // GitHub ID
  number?: number; // GitHub number for display

  // Content
  title: string;
  description: string;
  type: 'issue' | 'pr' | 'commit' | 'initiative';

  // Status (auto-derived from GitHub state)
  status: WorkItemStatus;
  mergeStatus?: 'pending' | 'merged' | 'closed';

  // Priority (human-overridable)
  priority: Priority;
  autoPriority?: Priority; // AI-suggested priority

  // People
  author: Contributor;
  assignees: Contributor[];
  claimedBy?: string; // Human claim (overrides auto-assign)
  reviewers?: Contributor[];

  // Metrics
  metrics: WorkItemMetrics;

  // Strategic tagging (human input)
  strategicTags?: StrategicTag[];
  roiCategory?: ROICategory;
  impactScore?: number; // 1-10, human-adjustable

  // Timeline
  createdAt: string;
  updatedAt: string;
  resolvedAt?: string;
  mergedAt?: string;

  // Relationships
  linkedPRs?: string[]; // PR IDs linked to this issue
  linkedIssues?: string[]; // Issues linked to this PR
  parentInitiative?: string; // Parent strategic initiative
  childItems?: string[];

  // Labels
  labels: Label[];
}

interface WorkItemMetrics {
  // Time metrics
  timeToFirstReview?: number; // hours
  timeToMerge?: number; // hours
  cycleTime?: number; // hours (open to close/merge)
  age: number; // hours

  // Code metrics (for PRs/commits)
  linesAdded?: number;
  linesDeleted?: number;
  filesChanged?: number;
  commitsCount?: number;

  // Quality metrics
  reviewComments?: number;
  approvalCount?: number;
  changesRequested?: number;
  rebaseCount?: number;

  // Contribution metrics
  contributorCount?: number;
  commentCount?: number;
}

type WorkItemStatus =
  | 'backlog'      // Open issue, no activity
  | 'in_progress'  // Has commits/active development
  | 'in_review'    // PR open, awaiting review
  | 'merged'       // PR merged
  | 'done'         // Issue closed
  | 'closed';      // Closed without merging

type Priority = 'low' | 'medium' | 'high' | 'critical';

interface StrategicTag {
  id: string;
  name: string;
  category: 'feature' | 'tech-debt' | 'performance' | 'security' | 'compliance';
  strategicAlignment: 'core' | 'adjacent' | 'experimental';
  addedBy: string;
  addedAt: string;
}

interface ROICategory {
  id: string;
  name: 'revenue-impact' | 'cost-reduction' | 'risk-mitigation' | 'strategic-capability' | 'developer-productivity';
  estimatedEffort: number; // story points
  estimatedImpact: number; // 1-10
  actualImpact?: number; // Post-completion measurement
}
```

### 2.2 GitHub Service Implementation

```typescript
// Proposed: src/app/src/renderer/services/github/GitHubService.ts

import { Octokit } from '@octokit/rest';

interface GitHubServiceConfig {
  token: string;
  owner: string;
  repo: string;
}

class GitHubService {
  private octokit: Octokit;
  private config: GitHubServiceConfig;
  private rateLimitInfo: {
    remaining: number;
    resetAt: Date;
  };

  constructor(config: GitHubServiceConfig) {
    this.config = config;
    this.octokit = new Octokit({ auth: config.token });
    this.rateLimitInfo = { remaining: 5000, resetAt: new Date() };
  }

  /**
   * Fetch all open and recently closed issues
   */
  async fetchIssues(since?: Date): Promise<NormalizedIssue[]> {
    await this.checkRateLimit();

    const { data } = await this.octokit.issues.listForRepo({
      owner: this.config.owner,
      repo: this.config.repo,
      state: 'all',
      since: since?.toISOString(),
      per_page: 100,
    });

    return data.map(this.normalizeIssue.bind(this));
  }

  /**
   * Fetch all pull requests with review data
   */
  async fetchPullRequests(since?: Date): Promise<NormalizedPullRequest[]> {
    await this.checkRateLimit();

    const { data } = await this.octokit.pulls.list({
      owner: this.config.owner,
      repo: this.config.repo,
      state: 'all',
      per_page: 100,
    });

    // Enrich with review data
    const prsWithReviews = await Promise.all(
      data.map(async (pr) => {
        const reviews = await this.fetchPRReviews(pr.number);
        return { ...pr, reviews };
      })
    );

    return prsWithReviews.map(this.normalizePullRequest.bind(this));
  }

  /**
   * Fetch commits for git history analysis
   */
  async fetchCommits(since?: Date, until?: Date): Promise<NormalizedCommit[]> {
    await this.checkRateLimit();

    const { data } = await this.octokit.repos.listCommits({
      owner: this.config.owner,
      repo: this.config.repo,
      since: since?.toISOString(),
      until: until?.toISOString(),
      per_page: 100,
    });

    return data.map(this.normalizeCommit.bind(this));
  }

  /**
   * Link issues to their PRs using closing keywords
   */
  async linkIssuesToPRs(issues: NormalizedIssue[], prs: NormalizedPullRequest[]): Promise<LinkMap> {
    const linkMap = new Map<string, string[]>();

    for (const pr of prs) {
      // Check for "Fixes #123" patterns in PR body and commits
      const linkedIssues = this.extractIssueLinks(pr.body, pr.commits);
      for (const issueNumber of linkedIssues) {
        const issueId = issues.find(i => i.number === issueNumber)?.id;
        if (issueId) {
          const current = linkMap.get(issueId) || [];
          linkMap.set(issueId, [...current, pr.id]);
        }
      }
    }

    return linkMap;
  }

  private normalizeIssue(issue: OctokitIssue): NormalizedIssue {
    // Normalization logic
  }

  private normalizePullRequest(pr: OctokitPR): NormalizedPullRequest {
    // Normalization logic
  }

  private normalizeCommit(commit: OctokitCommit): NormalizedCommit {
    // Normalization logic
  }

  private extractIssueLinks(body: string, commits: string[]): number[] {
    const issuePattern = /(?:fix|fixes|fixed|close|closes|closed|resolve|resolves|resolved)\s+#(\d+)/gi;
    const issues = new Set<number>();

    // Search in PR body
    let match;
    while ((match = issuePattern.exec(body)) !== null) {
      issues.add(parseInt(match[1], 10));
    }

    // Search in commit messages
    for (const commit of commits) {
      while ((match = issuePattern.exec(commit)) !== null) {
        issues.add(parseInt(match[1], 10));
      }
    }

    return Array.from(issues);
  }

  private async checkRateLimit(): Promise<void> {
    const { data } = await this.octokit.rateLimit.get();
    this.rateLimitInfo = {
      remaining: data.rate.remaining,
      resetAt: new Date(data.rate.reset * 1000),
    };

    if (this.rateLimitInfo.remaining < 100) {
      throw new Error(`Rate limit nearly exhausted. Resets at ${this.rateLimitInfo.resetAt}`);
    }
  }
}
```

### 2.3 Sync Orchestration

```typescript
// Proposed: src/app/src/renderer/services/sync/SyncOrchestrator.ts

interface SyncState {
  lastFullSync: Date | null;
  lastIncrementalSync: Date | null;
  lastGitAnalysis: Date | null;
  pendingChanges: number;
  syncInProgress: boolean;
  error?: string;
}

class SyncOrchestrator {
  private githubService: GitHubService;
  private state: SyncState = {
    lastFullSync: null,
    lastIncrementalSync: null,
    lastGitAnalysis: null,
    pendingChanges: 0,
    syncInProgress: false,
  };

  constructor(githubService: GitHubService) {
    this.githubService = githubService;
  }

  /**
   * Initialize sync with configuration
   */
  async initialize(config: SyncConfig): Promise<void> {
    // Perform initial full sync
    await this.performFullSync(config);

    // Set up periodic incremental sync
    setInterval(() => {
      this.performIncrementalSync(config);
    }, config.syncInterval || 300000); // Default 5 minutes
  }

  /**
   * Full sync - fetches all data
   */
  private async performFullSync(config: SyncConfig): Promise<SyncResult> {
    this.state.syncInProgress = true;

    try {
      const [issues, pullRequests, commits] = await Promise.all([
        this.githubService.fetchIssues(),
        this.githubService.fetchPullRequests(),
        this.githubService.fetchCommits(),
      ]);

      const linkMap = await this.githubService.linkIssuesToPRs(issues, pullRequests);

      const workItems = this.mergeDataSources(issues, pullRequests, commits, linkMap);

      this.state = {
        ...this.state,
        lastFullSync: new Date(),
        lastIncrementalSync: new Date(),
        pendingChanges: workItems.length,
        syncInProgress: false,
      };

      return { success: true, workItems, syncType: 'full' };
    } catch (error) {
      this.state.syncInProgress = false;
      this.state.error = (error as Error).message;
      throw error;
    }
  }

  /**
   * Incremental sync - only fetches changes since last sync
   */
  private async performIncrementalSync(config: SyncConfig): Promise<SyncResult> {
    if (this.state.syncInProgress) return { success: false, reason: 'sync_in_progress' };
    if (!this.state.lastIncrementalSync) return this.performFullSync(config);

    this.state.syncInProgress = true;

    try {
      const since = this.state.lastIncrementalSync;

      const [issues, pullRequests, commits] = await Promise.all([
        this.githubService.fetchIssues(since),
        this.githubService.fetchPullRequests(since),
        this.githubService.fetchCommits(since),
      ]);

      const workItems = this.mergeDataSources(issues, pullRequests, commits);

      this.state = {
        ...this.state,
        lastIncrementalSync: new Date(),
        pendingChanges: workItems.length,
        syncInProgress: false,
      };

      return { success: true, workItems, syncType: 'incremental' };
    } catch (error) {
      this.state.syncInProgress = false;
      this.state.error = (error as Error).message;
      throw error;
    }
  }

  /**
   * Merge data from multiple sources into unified WorkItems
   */
  private mergeDataSources(
    issues: NormalizedIssue[],
    pullRequests: NormalizedPullRequest[],
    commits: NormalizedCommit[],
    linkMap?: LinkMap
  ): WorkItem[] {
    const workItems: WorkItem[] = [];

    // Convert issues to work items
    for (const issue of issues) {
      workItems.push(this.issueToWorkItem(issue, linkMap?.get(issue.id)));
    }

    // Convert PRs to work items
    for (const pr of pullRequests) {
      workItems.push(this.pullRequestToWorkItem(pr));
    }

    // Convert significant commits to work items (for standalone commits)
    for (const commit of commits) {
      if (this.isSignificantCommit(commit)) {
        workItems.push(this.commitToWorkItem(commit));
      }
    }

    return workItems;
  }

  private isSignificantCommit(commit: NormalizedCommit): boolean {
    // Only create work items for commits with substantial changes
    return commit.stats.total > 50 || commit.files.length > 5;
  }

  private issueToWorkItem(issue: NormalizedIssue, linkedPRs?: string[]): WorkItem {
    return {
      id: `gh-issue-${issue.id}`,
      source: 'github-issue',
      externalId: String(issue.id),
      number: issue.number,
      title: issue.title,
      description: issue.body,
      type: 'issue',
      status: this.mapIssueStatus(issue),
      priority: this.extractPriority(issue.labels),
      author: this.normalizeContributor(issue.author),
      assignees: issue.assignees.map(this.normalizeContributor.bind(this)),
      metrics: this.calculateIssueMetrics(issue),
      labels: issue.labels.map(this.normalizeLabel.bind(this)),
      linkedPRs,
      createdAt: issue.created_at,
      updatedAt: issue.updated_at,
      resolvedAt: issue.closed_at,
    };
  }

  private pullRequestToWorkItem(pr: NormalizedPullRequest): WorkItem {
    return {
      id: `gh-pr-${pr.id}`,
      source: 'github-pr',
      externalId: String(pr.id),
      number: pr.number,
      title: pr.title,
      description: pr.body,
      type: 'pr',
      status: this.mapPRStatus(pr),
      mergeStatus: pr.merged ? 'merged' : pr.state === 'closed' ? 'closed' : 'pending',
      priority: this.extractPriority(pr.labels),
      author: this.normalizeContributor(pr.author),
      reviewers: pr.requested_reviewers?.map(this.normalizeContributor.bind(this)),
      metrics: this.calculatePRMetrics(pr),
      labels: pr.labels.map(this.normalizeLabel.bind(this)),
      linkedIssues: this.extractLinkedIssues(pr.body),
      createdAt: pr.created_at,
      updatedAt: pr.updated_at,
      mergedAt: pr.merged_at,
    };
  }

  private commitToWorkItem(commit: NormalizedCommit): WorkItem {
    return {
      id: `gh-commit-${commit.sha}`,
      source: 'github-commit',
      externalId: commit.sha,
      title: commit.message.split('\n')[0],
      description: commit.message,
      type: 'commit',
      status: 'done',
      priority: 'low',
      author: this.normalizeContributor(commit.author),
      metrics: {
        age: Date.now() - new Date(commit.timestamp).getTime(),
        linesAdded: commit.stats.additions,
        linesDeleted: commit.stats.deletions,
        filesChanged: commit.files.length,
      },
      labels: [],
      createdAt: commit.timestamp,
      updatedAt: commit.timestamp,
    };
  }
}
```

---

## 3. Strategic Metrics Framework

### 3.1 Development KPIs (Auto-Calculated)

```typescript
// Proposed: src/app/src/renderer/metrics/DevelopmentKPIs.ts

interface DevelopmentKPIs {
  // Velocity Metrics
  velocity: {
    current: number; // Work items completed this sprint
    average: number; // Rolling average
    trend: 'increasing' | 'stable' | 'decreasing';
  };

  // Cycle Time Metrics
  cycleTime: {
    average: number; // Days from open to close
    median: number;
    percentile90: number;
    byType: {
      issue: number;
      pr: number;
    };
    trend: 'improving' | 'stable' | 'degrading';
  };

  // PR Metrics
  pullRequestMetrics: {
    averageTimeToFirstReview: number; // Hours
    averageTimeToMerge: number; // Hours
    mergeRate: number; // Percentage merged
    reworkRate: number; // Percentage with changes requested
  };

  // Code Quality Metrics
  codeQuality: {
    codeChurn: number; // Percentage of code rewritten within 2 weeks
    averagePRSize: number; // Lines changed
    reviewDepth: number; // Comments per PR
    defectRate: number; // Issues reopened after closing
  };

  // Throughput Metrics
  throughput: {
    itemsCompleted: number;
    itemsStarted: number;
    workInProgress: number;
    throughputTrend: number[]; // Last 10 periods
  };
}

class DevelopmentKPICalculator {
  calculate(workItems: WorkItem[], period?: DateRange): DevelopmentKPIs {
    const filteredItems = this.filterByPeriod(workItems, period);
    const completedItems = filteredItems.filter(
      item => item.status === 'done' || item.status === 'merged'
    );

    return {
      velocity: this.calculateVelocity(completedItems),
      cycleTime: this.calculateCycleTime(filteredItems, completedItems),
      pullRequestMetrics: this.calculatePRMetrics(filteredItems),
      codeQuality: this.calculateCodeQuality(filteredItems),
      throughput: this.calculateThroughput(filteredItems),
    };
  }

  private calculateVelocity(completedItems: WorkItem[]): DevelopmentKPIs['velocity'] {
    const currentSprint = this.getCurrentSprintItems(completedItems);
    const previousSprints = this.getPreviousSprintItems(completedItems, 4);

    const current = currentSprint.length;
    const average = this.calculateAverage([
      ...previousSprints.map(s => s.length),
      current,
    ]);

    const trend = current > average * 1.1 ? 'increasing'
      : current < average * 0.9 ? 'decreasing'
      : 'stable';

    return { current, average, trend };
  }

  private calculateCycleTime(
    allItems: WorkItem[],
    completedItems: WorkItem[]
  ): DevelopmentKPIs['cycleTime'] {
    const cycleTimes = completedItems
      .filter(item => item.resolvedAt || item.mergedAt)
      .map(item => {
        const start = new Date(item.createdAt).getTime();
        const end = new Date(item.resolvedAt || item.mergedAt!).getTime();
        return (end - start) / (1000 * 60 * 60 * 24); // Days
      });

    return {
      average: this.calculateAverage(cycleTimes),
      median: this.calculateMedian(cycleTimes),
      percentile90: this.calculatePercentile(cycleTimes, 90),
      byType: {
        issue: this.calculateAverageForType(cycleTimes, completedItems, 'issue'),
        pr: this.calculateAverageForType(cycleTimes, completedItems, 'pr'),
      },
      trend: this.calculateCycleTimeTrend(cycleTimes),
    };
  }

  private calculatePRMetrics(items: WorkItem[]): DevelopmentKPIs['pullRequestMetrics'] {
    const prItems = items.filter(item => item.type === 'pr');

    const timeToFirstReview = prItems
      .filter(item => item.metrics?.timeToFirstReview)
      .map(item => item.metrics!.timeToFirstReview!);

    const timeToMerge = prItems
      .filter(item => item.mergedAt)
      .map(item => {
        const start = new Date(item.createdAt).getTime();
        const end = new Date(item.mergedAt!).getTime();
        return (end - start) / (1000 * 60 * 60); // Hours
      });

    const merged = prItems.filter(item => item.mergeStatus === 'merged').length;

    return {
      averageTimeToFirstReview: this.calculateAverage(timeToFirstReview) || 0,
      averageTimeToMerge: this.calculateAverage(timeToMerge) || 0,
      mergeRate: (merged / prItems.length) * 100 || 0,
      reworkRate: this.calculateReworkRate(prItems),
    };
  }

  private calculateCodeQuality(items: WorkItem[]): DevelopmentKPIs['codeQuality'] {
    // Code churn: items with significant follow-up changes within 2 weeks
    const churnedItems = items.filter(item => {
      if (!item.resolvedAt) return false;
      const resolvedDate = new Date(item.resolvedAt);
      const twoWeeksLater = new Date(resolvedDate.getTime() + 14 * 24 * 60 * 60 * 1000);

      return items.some(other => {
        if (other.id === item.id) return false;
        if (new Date(other.createdAt) > twoWeeksLater) return false;
        // Check if this item references the original (via linked issues or commit messages)
        return this.isRelatedTo(other, item);
      });
    });

    return {
      codeChurn: (churnedItems.length / items.length) * 100 || 0,
      averagePRSize: this.calculateAveragePRSize(items),
      reviewDepth: this.calculateReviewDepth(items),
      defectRate: this.calculateDefectRate(items),
    };
  }

  private calculateThroughput(items: WorkItem[]): DevelopmentKPIs['throughput'] {
    const completed = items.filter(item =>
      item.status === 'done' || item.status === 'merged'
    ).length;

    const started = items.filter(item =>
      item.status === 'in_progress' || item.status === 'in_review'
    ).length;

    const wip = items.filter(item =>
      item.status === 'in_progress' || item.status === 'in_review'
    ).length;

    return {
      itemsCompleted: completed,
      itemsStarted: started,
      workInProgress: wip,
      throughputTrend: this.calculateThroughputTrend(items),
    };
  }
}
```

### 3.2 ROI/Impact Calculation Framework

```typescript
// Proposed: src/app/src/renderer/metrics/ROICalculator.ts

interface ROIAnalysis {
  // Investment (Effort)
  effort: {
    totalStoryPoints: number;
    totalHours: number;
    teamCost: number; // Calculated from team rates
    opportunityCost: number; // What else could have been done
  };

  // Returns (Impact)
  impact: {
    revenueImpact: number; // Estimated revenue influence
    costReduction: number; // Estimated cost savings
    riskReduction: number; // Quantified risk mitigation
    strategicValue: number; // Capability enablement score
    developerProductivity: number; // Time saved for devs
  };

  // Calculated Metrics
  roi: {
    ratio: number; // (Gain - Cost) / Cost
    percentage: number;
    paybackPeriod: number; // Days to break even
    npv: number; // Net present value
  };

  // Classification
  category: 'high-roi' | 'medium-roi' | 'low-roi' | 'strategic-investment';
  recommendation: 'prioritize' | 'maintain' | 'reconsider' | 'deprecated';
}

interface StrategicInitiative {
  id: string;
  name: string;
  description: string;
  timeframe: {
    start: Date;
    end: Date;
  };
  targets: {
    expectedROI: number;
    expectedImpact: number;
    budgetPoints: number;
  };
  actuals?: {
    actualROI: number;
    actualImpact: number;
    actualPoints: number;
  };
  linkedWorkItems: string[];
  status: 'planning' | 'active' | 'completed' | 'cancelled';
}

class ROICalculator {
  /**
   * Calculate ROI for a completed initiative
   */
  calculateROI(initiative: StrategicInitiative, workItems: WorkItem[]): ROIAnalysis {
    const linkedItems = workItems.filter(item =>
      initiative.linkedWorkItems.includes(item.id) ||
      item.parentInitiative === initiative.id
    );

    const effort = this.calculateEffort(linkedItems);
    const impact = this.calculateImpact(linkedItems, initiative);

    const roi = this.calculateROIMetrics(effort, impact);
    const category = this.classifyROI(roi);
    const recommendation = this.generateRecommendation(roi, initiative);

    return { effort, impact, roi, category, recommendation };
  }

  /**
   * Calculate effort from work items
   */
  private calculateEffort(workItems: WorkItem[]): ROIAnalysis['effort'] {
    const totalStoryPoints = workItems.reduce(
      (sum, item) => sum + (item.estimatedPoints || this.estimatePoints(item)),
      0
    );

    // Assuming 1 story point ≈ 4 hours
    const totalHours = totalStoryPoints * 4;

    // Calculate team cost (configurable rates)
    const averageHourlyRate = 75; // Configurable
    const teamCost = totalHours * averageHourlyRate;

    // Opportunity cost: what else could have been done
    const opportunityCost = teamCost * 0.2; // 20% premium for opportunity

    return {
      totalStoryPoints,
      totalHours,
      teamCost,
      opportunityCost,
    };
  }

  /**
   * Calculate impact from work items and strategic context
   */
  private calculateImpact(
    workItems: WorkItem[],
    initiative: StrategicInitiative
  ): ROIAnalysis['impact'] {
    // Revenue impact: features that directly affect revenue
    const revenueItems = workItems.filter(item =>
      item.roiCategory?.name === 'revenue-impact'
    );
    const revenueImpact = revenueItems.reduce((sum, item) => {
      return sum + (item.impactScore || 5) * 10000; // Scale impact score to dollars
    }, 0);

    // Cost reduction: automation, efficiency improvements
    const costItems = workItems.filter(item =>
      item.roiCategory?.name === 'cost-reduction'
    );
    const costReduction = costItems.reduce((sum, item) => {
      return sum + (item.impactScore || 5) * 5000;
    }, 0);

    // Risk reduction: security, compliance
    const riskItems = workItems.filter(item =>
      item.roiCategory?.name === 'risk-mitigation'
    );
    const riskReduction = riskItems.reduce((sum, item) => {
      return sum + (item.impactScore || 5) * 8000;
    }, 0);

    // Strategic value: new capabilities
    const strategicItems = workItems.filter(item =>
      item.strategicTags?.some(tag => tag.strategicAlignment === 'core')
    );
    const strategicValue = strategicItems.length * initiative.targets.expectedImpact * 1000;

    // Developer productivity: internal tooling, DX improvements
    const productivityItems = workItems.filter(item =>
      item.roiCategory?.name === 'developer-productivity'
    );
    const developerProductivity = productivityItems.reduce((sum, item) => {
      return sum + (item.impactScore || 5) * 3000;
    }, 0);

    return {
      revenueImpact,
      costReduction,
      riskReduction,
      strategicValue,
      developerProductivity,
    };
  }

  /**
   * Calculate ROI metrics
   */
  private calculateROIMetrics(
    effort: ROIAnalysis['effort'],
    impact: ROIAnalysis['impact']
  ): ROIAnalysis['roi'] {
    const totalGain =
      impact.revenueImpact +
      impact.costReduction +
      impact.riskReduction +
      impact.strategicValue +
      impact.developerProductivity;

    const totalCost = effort.teamCost + effort.opportunityCost;

    const ratio = (totalGain - totalCost) / totalCost;
    const percentage = ratio * 100;

    // Payback period: how long to recover investment
    const monthlyBenefit = totalGain / 12; // Annualized
    const paybackPeriod = totalCost / (monthlyBenefit || 1);

    // NPV: Net Present Value (simplified, 10% discount rate)
    const discountRate = 0.10;
    const npv = -totalCost + totalGain / (1 + discountRate);

    return { ratio, percentage, paybackPeriod, npv };
  }

  /**
   * Classify ROI into categories
   */
  private classifyROI(roi: ROIAnalysis['roi']): ROIAnalysis['category'] {
    if (roi.ratio > 2) return 'high-roi';
    if (roi.ratio > 0.5) return 'medium-roi';
    if (roi.ratio > 0) return 'low-roi';
    return 'strategic-investment'; // Negative ROI but may be strategically necessary
  }

  /**
   * Generate actionable recommendation
   */
  private generateRecommendation(
    roi: ROIAnalysis['roi'],
    initiative: StrategicInitiative
  ): ROIAnalysis['recommendation'] {
    // Check if actuals meet targets
    const targetMet = initiative.actuals
      ? initiative.actuals.actualROI >= initiative.targets.expectedROI * 0.8
      : true;

    if (roi.ratio > 1 && targetMet) return 'prioritize';
    if (roi.ratio > 0 && targetMet) return 'maintain';
    if (roi.ratio < 0 && !targetMet) return 'reconsider';
    return 'deprecated';
  }

  /**
   * Estimate story points from work item characteristics
   */
  private estimatePoints(item: WorkItem): number {
    if (item.metrics) {
      // Use code metrics to estimate complexity
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
```

### 3.3 Industry Relevance Scoring

```typescript
// Proposed: src/app/src/renderer/metrics/IndustryRelevanceScorer.ts

interface IndustryRelevanceScore {
  overall: number; // 0-100
  dimensions: {
    technicalInnovation: number; // 0-100
    marketAlignment: number; // 0-100
    competitiveParity: number; // 0-100
    futureProofing: number; // 0-100
    ecosystemIntegration: number; // 0-100
  };
  trends: {
    emerging: string[]; // Trends this work aligns with
    declining: string[]; // Technologies being phased out
  };
  benchmarks: {
    industryAverage: number;
    leaderAverage: number;
    ourScore: number;
  };
}

interface TechTrend {
  id: string;
  name: string;
  category: 'ai-ml' | 'cloud-native' | 'security' | 'performance' | 'dx';
  maturity: 'emerging' | 'growing' | 'mainstream' | 'mature' | 'declining';
  adoptionRate: number; // Industry adoption percentage
  growthTrajectory: 'accelerating' | 'steady' | 'plateauing' | 'declining';
  relevanceWeights: Record<string, number>; // Work item type -> relevance weight
}

class IndustryRelevanceScorer {
  private techTrends: TechTrend[] = this.loadTechTrends();

  /**
   * Calculate industry relevance for a work item or initiative
   */
  calculateRelevance(workItems: WorkItem[]): IndustryRelevanceScore {
    return {
      overall: this.calculateOverallScore(workItems),
      dimensions: this.calculateDimensions(workItems),
      trends: this.identifyTrends(workItems),
      benchmarks: this.compareWithBenchmarks(workItems),
    };
  }

  private calculateOverallScore(workItems: WorkItem[]): number {
    const weights = {
      technicalInnovation: 0.25,
      marketAlignment: 0.25,
      competitiveParity: 0.20,
      futureProofing: 0.20,
      ecosystemIntegration: 0.10,
    };

    const dimensions = this.calculateDimensions(workItems);

    return Math.round(
      dimensions.technicalInnovation * weights.technicalInnovation +
      dimensions.marketAlignment * weights.marketAlignment +
      dimensions.competitiveParity * weights.competitiveParity +
      dimensions.futureProofing * weights.futureProofing +
      dimensions.ecosystemIntegration * weights.ecosystemIntegration
    );
  }

  private calculateDimensions(workItems: WorkItem[]): IndustryRelevanceScore['dimensions'] {
    // Technical Innovation: How novel/advanced is the technology?
    const technicalInnovation = this.scoreTechnicalInnovation(workItems);

    // Market Alignment: Does this address market demands?
    const marketAlignment = this.scoreMarketAlignment(workItems);

    // Competitive Parity: Does this match/exceed competitor capabilities?
    const competitiveParity = this.scoreCompetitiveParity(workItems);

    // Future Proofing: Is this using sustainable, growing technologies?
    const futureProofing = this.scoreFutureProofing(workItems);

    // Ecosystem Integration: Does this work with industry standards?
    const ecosystemIntegration = this.scoreEcosystemIntegration(workItems);

    return {
      technicalInnovation,
      marketAlignment,
      competitiveParity,
      futureProofing,
      ecosystemIntegration,
    };
  }

  private scoreTechnicalInnovation(workItems: WorkItem[]): number {
    // Score based on:
    // - Use of ML/AI features (NPU acceleration, etc.)
    // - Novel algorithms or approaches
    // - Research-backed implementations

    const innovationKeywords = ['npu', 'ai', 'ml', 'transformer', 'quantization', 'edge'];
    const matchCount = workItems.filter(item =>
      innovationKeywords.some(keyword =>
        item.title.toLowerCase().includes(keyword) ||
        item.description.toLowerCase().includes(keyword) ||
        item.labels.some(l => l.name.toLowerCase().includes(keyword))
      )
    ).length;

    return Math.min(100, Math.round((matchCount / workItems.length) * 100 * 2));
  }

  private scoreMarketAlignment(workItems: WorkItem[]): number {
    // Score based on:
    // - Feature requests from users
    // - Common pain points addressed
    // - Market demand signals

    const marketKeywords = ['performance', 'compatibility', 'api', 'integration', 'enterprise'];
    const matchCount = workItems.filter(item =>
      marketKeywords.some(keyword =>
        item.title.toLowerCase().includes(keyword) ||
        item.labels.some(l => l.name.toLowerCase().includes(keyword))
      )
    ).length;

    return Math.min(100, Math.round((matchCount / workItems.length) * 100 * 1.5));
  }

  private scoreCompetitiveParity(workItems: WorkItem[]): number {
    // Score based on:
    // - Feature parity with competitors
    // - Unique differentiators
    // - Standard compliance

    const parityLabels = ['compatibility', 'standard', 'ollama', 'openai-api', 'onnx'];
    const matchCount = workItems.filter(item =>
      item.labels.some(l =>
        parityLabels.some(pl => l.name.toLowerCase().includes(pl))
      )
    ).length;

    return Math.min(100, Math.round((matchCount / workItems.length) * 100 * 1.5));
  }

  private scoreFutureProofing(workItems: WorkItem[]): number {
    // Score based on:
    // - Using growing vs declining technologies
    // - Sustainability of approach
    // - Maintenance burden

    const growingTechs = ['npu', 'rust', 'webgpu', 'quantization'];
    const decliningTechs = ['deprecated', 'legacy', 'v1'];

    const growingCount = workItems.filter(item =>
      growingTechs.some(tech =>
        item.labels.some(l => l.name.toLowerCase().includes(tech))
      )
    ).length;

    const decliningCount = workItems.filter(item =>
      decliningTechs.some(tech =>
        item.labels.some(l => l.name.toLowerCase().includes(tech))
      )
    ).length;

    return Math.min(100, Math.round(((growingCount - decliningCount) / workItems.length) * 100 + 50));
  }

  private scoreEcosystemIntegration(workItems: WorkItem[]): number {
    // Score based on:
    // - API compatibility
    // - Standard formats support
    // - Plugin/extension points

    const integrationKeywords = ['api', 'sdk', 'plugin', 'extension', 'webhook', 'rest'];
    const matchCount = workItems.filter(item =>
      integrationKeywords.some(keyword =>
        item.title.toLowerCase().includes(keyword) ||
        item.labels.some(l => l.name.toLowerCase().includes(keyword))
      )
    ).length;

    return Math.min(100, Math.round((matchCount / workItems.length) * 100 * 1.5));
  }

  private identifyTrends(workItems: WorkItem[]): IndustryRelevanceScore['trends'] {
    const emerging: string[] = [];
    const declining: string[] = [];

    for (const trend of this.techTrends) {
      const relevance = workItems.some(item =>
        trend.relevanceWeights[item.type] > 0.5
      );

      if (relevance) {
        if (trend.maturity === 'emerging' || trend.maturity === 'growing') {
          emerging.push(trend.name);
        } else if (trend.maturity === 'declining') {
          declining.push(trend.name);
        }
      }
    }

    return { emerging, declining };
  }

  private compareWithBenchmarks(workItems: WorkItem[]): IndustryRelevanceScore['benchmarks'] {
    // Industry benchmarks (would be loaded from external data)
    const industryAverage = 65;
    const leaderAverage = 85;
    const ourScore = this.calculateOverallScore(workItems);

    return {
      industryAverage,
      leaderAverage,
      ourScore,
    };
  }

  private loadTechTrends(): TechTrend[] {
    // This would be loaded from a trends database or API
    return [
      {
        id: 'npu-acceleration',
        name: 'NPU/AI Acceleration',
        category: 'ai-ml',
        maturity: 'growing',
        adoptionRate: 35,
        growthTrajectory: 'accelerating',
        relevanceWeights: { pr: 0.9, issue: 0.7, commit: 0.5 },
      },
      {
        id: 'edge-ai',
        name: 'Edge AI',
        category: 'ai-ml',
        maturity: 'growing',
        adoptionRate: 45,
        growthTrajectory: 'accelerating',
        relevanceWeights: { pr: 0.8, issue: 0.8, commit: 0.6 },
      },
      // ... more trends
    ];
  }
}
```

---

## 4. AI-Generated Insights Engine

```typescript
// Proposed: src/app/src/renderer/insights/AIInsightsEngine.ts

interface AIInsight {
  id: string;
  type: 'trend' | 'anomaly' | 'recommendation' | 'risk' | 'opportunity';
  severity: 'low' | 'medium' | 'high' | 'critical';
  title: string;
  description: string;
  dataPoints: DataPoint[];
  recommendedAction?: string;
  confidence: number; // 0-1
  generatedAt: string;
}

interface DataPoint {
  label: string;
  value: number;
  trend?: 'up' | 'down' | 'stable';
  comparison?: {
    baseline: number;
    change: number;
    changePercent: number;
  };
}

class AIInsightsEngine {
  /**
   * Generate insights from work items and metrics
   */
  generateInsights(
    workItems: WorkItem[],
    devKPIs: DevelopmentKPIs,
    roiAnalysis: ROIAnalysis,
    relevanceScore: IndustryRelevanceScore
  ): AIInsight[] {
    const insights: AIInsight[] = [];

    // Analyze trends
    insights.push(...this.analyzeTrends(workItems, devKPIs));

    // Detect anomalies
    insights.push(...this.detectAnomalies(workItems, devKPIs));

    // Generate recommendations
    insights.push(...this.generateRecommendations(devKPIs, roiAnalysis));

    // Identify risks
    insights.push(...this.identifyRisks(workItems, devKPIs));

    // Surface opportunities
    insights.push(...this.identifyOpportunities(workItems, relevanceScore));

    // Sort by severity and confidence
    return insights.sort((a, b) => {
      const severityOrder = { critical: 4, high: 3, medium: 2, low: 1 };
      return severityOrder[b.severity] - severityOrder[a.severity];
    });
  }

  private analyzeTrends(workItems: WorkItem[], devKPIs: DevelopmentKPIs): AIInsight[] {
    const insights: AIInsight[] = [];

    // Velocity trend analysis
    if (devKPIs.velocity.trend === 'increasing') {
      insights.push({
        id: 'trend-velocity-up',
        type: 'trend',
        severity: 'low',
        title: 'Velocity Increasing',
        description: `Team velocity has increased to ${devKPIs.velocity.current} items this sprint, above the average of ${devKPIs.velocity.average}.`,
        dataPoints: [
          { label: 'Current Velocity', value: devKPIs.velocity.current, trend: 'up' },
          { label: 'Average Velocity', value: devKPIs.velocity.average },
        ],
        confidence: 0.9,
        generatedAt: new Date().toISOString(),
      });
    } else if (devKPIs.velocity.trend === 'decreasing') {
      insights.push({
        id: 'trend-velocity-down',
        type: 'trend',
        severity: 'medium',
        title: 'Velocity Decreasing',
        description: `Team velocity has decreased to ${devKPIs.velocity.current} items, below the average of ${devKPIs.velocity.average}. Consider investigating blockers.`,
        dataPoints: [
          { label: 'Current Velocity', value: devKPIs.velocity.current, trend: 'down' },
          { label: 'Average Velocity', value: devKPIs.velocity.average },
        ],
        recommendedAction: 'Review sprint retrospectives for identified blockers. Check team capacity and workload distribution.',
        confidence: 0.85,
        generatedAt: new Date().toISOString(),
      });
    }

    // Cycle time trend analysis
    if (devKPIs.cycleTime.trend === 'degrading') {
      insights.push({
        id: 'trend-cycle-time-degrading',
        type: 'trend',
        severity: 'high',
        title: 'Cycle Time Degrading',
        description: `Average cycle time has increased to ${devKPIs.cycleTime.average.toFixed(1)} days. This may indicate process inefficiencies or scope creep.`,
        dataPoints: [
          { label: 'Current Cycle Time', value: devKPIs.cycleTime.average, trend: 'up' },
          { label: 'Median Cycle Time', value: devKPIs.cycleTime.median },
          { label: '90th Percentile', value: devKPIs.cycleTime.percentile90 },
        ],
        recommendedAction: 'Analyze work items with longest cycle times. Consider breaking down large items and reducing WIP.',
        confidence: 0.8,
        generatedAt: new Date().toISOString(),
      });
    }

    return insights;
  }

  private detectAnomalies(workItems: WorkItem[], devKPIs: DevelopmentKPIs): AIInsight[] {
    const insights: AIInsight[] = [];

    // Detect PR review bottlenecks
    if (devKPIs.pullRequestMetrics.averageTimeToFirstReview > 48) {
      insights.push({
        id: 'anomaly-pr-review-bottleneck',
        type: 'anomaly',
        severity: 'high',
        title: 'PR Review Bottleneck Detected',
        description: `Average time to first review is ${devKPIs.pullRequestMetrics.averageTimeToFirstReview.toFixed(1)} hours, indicating a potential review bottleneck.`,
        dataPoints: [
          {
            label: 'Time to First Review',
            value: devKPIs.pullRequestMetrics.averageTimeToFirstReview,
            trend: 'up',
            comparison: { baseline: 24, change: devKPIs.pullRequestMetrics.averageTimeToFirstReview - 24, changePercent: ((devKPIs.pullRequestMetrics.averageTimeToFirstReview - 24) / 24) * 100 },
          },
        ],
        recommendedAction: 'Consider implementing review rotation, reducing PR sizes, or adding reviewers to the team.',
        confidence: 0.9,
        generatedAt: new Date().toISOString(),
      });
    }

    // Detect high code churn
    if (devKPIs.codeQuality.codeChurn > 20) {
      insights.push({
        id: 'anomaly-high-code-churn',
        type: 'anomaly',
        severity: 'medium',
        title: 'High Code Churn Detected',
        description: `${devKPIs.codeQuality.codeChurn.toFixed(1)}% of code is being rewritten within 2 weeks of completion, suggesting requirements instability or technical debt.`,
        dataPoints: [
          {
            label: 'Code Churn Rate',
            value: devKPIs.codeQuality.codeChurn,
            trend: 'up',
            comparison: { baseline: 15, change: devKPIs.codeQuality.codeChurn - 15, changePercent: ((devKPIs.codeQuality.codeChurn - 15) / 15) * 100 },
          },
        ],
        recommendedAction: 'Review requirements gathering process. Consider more thorough design reviews before implementation.',
        confidence: 0.75,
        generatedAt: new Date().toISOString(),
      });
    }

    return insights;
  }

  private generateRecommendations(devKPIs: DevelopmentKPIs, roiAnalysis: ROIAnalysis): AIInsight[] {
    const insights: AIInsight[] = [];

    // ROI-based recommendations
    if (roiAnalysis.roi.ratio < 0.5) {
      insights.push({
        id: 'rec-low-roi',
        type: 'recommendation',
        severity: 'high',
        title: 'Low ROI Alert',
        description: `Current initiatives show an ROI ratio of ${roiAnalysis.roi.ratio.toFixed(2)}, below the target of 0.5. Consider reprioritizing work.`,
        dataPoints: [
          { label: 'Current ROI', value: roiAnalysis.roi.ratio },
          { label: 'Target ROI', value: 0.5 },
          { label: 'Total Investment', value: roiAnalysis.effort.teamCost },
        ],
        recommendedAction: 'Review strategic initiatives and focus on high-impact work. Consider sunsetting low-impact projects.',
        confidence: 0.8,
        generatedAt: new Date().toISOString(),
      });
    }

    // WIP optimization recommendation
    if (devKPIs.throughput.workInProgress > 10) {
      insights.push({
        id: 'rec-high-wip',
        type: 'recommendation',
        severity: 'medium',
        title: 'High Work in Progress',
        description: `Team has ${devKPIs.throughput.workInProgress} items in progress. Research suggests limiting WIP improves throughput and quality.`,
        dataPoints: [
          { label: 'Current WIP', value: devKPIs.throughput.workInProgress },
          { label: 'Recommended WIP', value: Math.max(3, devKPIs.throughput.itemsCompleted / 2) },
        ],
        recommendedAction: 'Limit WIP to 2-3 items per developer. Finish current work before starting new items.',
        confidence: 0.85,
        generatedAt: new Date().toISOString(),
      });
    }

    return insights;
  }

  private identifyRisks(workItems: WorkItem[], devKPIs: DevelopmentKPIs): AIInsight[] {
    const insights: AIInsight[] = [];

    // Single point of failure risk
    const assigneeDistribution = this.calculateAssigneeDistribution(workItems);
    for (const [assignee, percentage] of Object.entries(assigneeDistribution)) {
      if (percentage > 40) {
        insights.push({
          id: `risk-sPOF-${assignee}`,
          type: 'risk',
          severity: 'high',
          title: `Single Point of Failure: ${assignee}`,
          description: `${assignee} is assigned to ${percentage.toFixed(0)}% of active work items, creating a bus factor risk.`,
          dataPoints: [
            { label: `${assignee}'s Workload`, value: percentage },
            { label: 'Recommended Max', value: 25 },
          ],
          recommendedAction: 'Distribute work more evenly. Implement pair programming and knowledge sharing.',
          confidence: 0.9,
          generatedAt: new Date().toISOString(),
        });
      }
    }

    // Technical debt accumulation
    const techDebtItems = workItems.filter(item =>
      item.strategicTags?.some(tag => tag.category === 'tech-debt')
    ).length;

    if (techDebtItems / workItems.length > 0.3) {
      insights.push({
        id: 'risk-tech-debt',
        type: 'risk',
        severity: 'medium',
        title: 'Technical Debt Accumulation',
        description: `${((techDebtItems / workItems.length) * 100).toFixed(0)}% of work items are tagged as technical debt.`,
        dataPoints: [
          { label: 'Tech Debt Items', value: techDebtItems },
          { label: 'Total Items', value: workItems.length },
          { label: 'Debt Ratio', value: (techDebtItems / workItems.length) * 100 },
        ],
        recommendedAction: 'Allocate 20-30% of capacity to technical debt reduction. Track debt paydown as a metric.',
        confidence: 0.8,
        generatedAt: new Date().toISOString(),
      });
    }

    return insights;
  }

  private identifyOpportunities(workItems: WorkItem[], relevanceScore: IndustryRelevanceScore): AIInsight[] {
    const insights: AIInsight[] = [];

    // Emerging trend alignment opportunity
    if (relevanceScore.trends.emerging.length > 0) {
      insights.push({
        id: 'opp-emerging-trends',
        type: 'opportunity',
        severity: 'low',
        title: 'Emerging Trend Alignment',
        description: `Current work aligns with ${relevanceScore.trends.emerging.length} emerging industry trends: ${relevanceScore.trends.emerging.join(', ')}.`,
        dataPoints: [
          { label: 'Emerging Trends', value: relevanceScore.trends.emerging.length },
          { label: 'Industry Relevance Score', value: relevanceScore.overall },
        ],
        recommendedAction: 'Consider documenting and sharing these innovations. Leverage for marketing and recruiting.',
        confidence: 0.7,
        generatedAt: new Date().toISOString(),
      });
    }

    // Benchmark gap opportunity
    if (relevanceScore.benchmarks.ourScore < relevanceScore.benchmarks.leaderAverage) {
      const gap = relevanceScore.benchmarks.leaderAverage - relevanceScore.benchmarks.ourScore;
      insights.push({
        id: 'opp-benchmark-gap',
        type: 'opportunity',
        severity: 'medium',
        title: 'Industry Leadership Opportunity',
        description: `Your industry relevance score (${relevanceScore.benchmarks.ourScore}) is ${gap} points below industry leaders (${relevanceScore.benchmarks.leaderAverage}).`,
        dataPoints: [
          { label: 'Our Score', value: relevanceScore.benchmarks.ourScore },
          { label: 'Leader Average', value: relevanceScore.benchmarks.leaderAverage },
          { label: 'Gap', value: gap },
        ],
        recommendedAction: `Focus on ${this.getLowestDimension(relevanceScore.dimensions)} to close the gap with industry leaders.`,
        confidence: 0.75,
        generatedAt: new Date().toISOString(),
      });
    }

    return insights;
  }

  private calculateAssigneeDistribution(workItems: WorkItem[]): Record<string, number> {
    const activeItems = workItems.filter(item =>
      item.status === 'in_progress' || item.status === 'in_review'
    );
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

  private getLowestDimension(dimensions: IndustryRelevanceScore['dimensions']): string {
    const entries = Object.entries(dimensions);
    const lowest = entries.reduce((min, [key, value]) =>
      value < min[1] ? [key, value] : min
    , entries[0]);
    return lowest[0].replace(/([A-Z])/g, ' $1').trim();
  }
}
```

---

## 5. Revised Component Architecture

### 5.1 New Component Structure

```
src/app/src/renderer/
├── components/
│   └── panels/
│       └── TeamDashboard/          # RESTRUCTURED
│           ├── StrategicDashboard.tsx    # NEW: Top-level strategic view
│           ├── PortfolioView.tsx         # NEW: Initiative-level tracking
│           ├── DevelopmentKPIs.tsx       # REDESIGNED: Dev metrics focus
│           ├── ROIAnalysis.tsx           # NEW: ROI/Impact visualization
│           ├── IndustryRelevance.tsx     # NEW: Industry scoring display
│           ├── AIInsights.tsx            # NEW: AI-generated insights panel
│           ├── WorkItemList.tsx          # REPLACES: IssueBoard (auto-synced)
│           ├── WorkItemDetail.tsx        # REPLACES: IssueDetailPanel
│           ├── GitHubSyncStatus.tsx      # NEW: Sync status indicator
│           └── StrategicInitiatives/     # NEW: Initiative management
│               ├── InitiativeCard.tsx
│               ├── InitiativeDetail.tsx
│               └── CreateInitiative.tsx
│
├── services/                     # NEW: Service layer
│   ├── github/
│   │   ├── GitHubService.ts
│   │   └── GitHubTypes.ts
│   ├── sync/
│   │   ├── SyncOrchestrator.ts
│   │   └── SyncScheduler.ts
│   └── analytics/
│       ├── MetricsCollector.ts
│       └── InsightsEngine.ts
│
├── metrics/                      # NEW: Metrics calculation
│   ├── DevelopmentKPIs.ts
│   ├── ROICalculator.ts
│   └── IndustryRelevanceScorer.ts
│
├── types/
│   ├── workItem.ts               # REPLACES: teamDashboard.ts (expanded)
│   ├── strategicInitiative.ts    # NEW
│   ├── metrics.ts                # NEW
│   └── insights.ts               # NEW
│
├── contexts/
│   └── TeamDashboardContext.tsx  # EXPANDED: Additional providers
│
└── hooks/
    ├── useWorkItems.ts           # REPLACES: useIssues
    ├── useMetrics.ts             # NEW
    ├── useSync.ts                # NEW: Sync management
    └── useInsights.ts            # NEW: AI insights
```

### 5.2 Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         USER INTERACTION LAYER                       │
│                                                                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │ Claim Owner  │  │ Set Priority │  │ Tag Strategic│              │
│  │ (Human Only) │  │ (Override)   │  │ (Human Only) │              │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘              │
│         │                 │                 │                        │
└─────────┼─────────────────┼─────────────────┼────────────────────────┘
          │                 │                 │
          ▼                 ▼                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      STATE MANAGEMENT LAYER                          │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │              TeamDashboardContext (useReducer)              │    │
│  │                                                              │    │
│  │  State: {                                                    │    │
│  │    workItems: WorkItem[]        // Auto-synced from GitHub  │    │
│  │    initiatives: StrategicInitiative[]  // Managed            │    │
│  │    metrics: ComputedMetrics       // Derived                │    │
│  │    insights: AIInsight[]          // Generated              │    │
│  │    syncState: SyncState           // Sync status            │    │
│  │    humanOverrides: Map<string, any>  // Manual adjustments  │    │
│  │  }                                                           │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
└─────────────────────────────┬───────────────────────────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          │                   │                   │
          ▼                   ▼                   ▼
┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
│   GitHub Sync    │ │  Metrics Engine  │ │  Insights Engine │
│   (Auto)         │ │  (Auto)          │ │  (Auto)          │
│                  │ │                  │ │                  │
│ • Poll GitHub    │ │ • Calculate KPIs │ │ • Detect Trends  │
│ • Parse Git      │ │ • Calculate ROI  │ │ • Find Anomalies │
│ • Normalize Data │ │ • Score Relevance│ │ • Generate Recs  │
│ • Update State   │ │ • Update State   │ │ • Update State   │
└──────────────────┘ └──────────────────┘ └──────────────────┘
```

---

## 6. Human Input vs. Auto-Collection Matrix

| Data Element | Source | Auto-Collected | Human Override |
|--------------|--------|----------------|----------------|
| Work Item Title | GitHub Issue/PR | ✓ | ✗ (source of truth) |
| Description | GitHub Issue/PR | ✓ | ✗ |
| Status | GitHub state | ✓ (mapped) | ✗ |
| Assignee | GitHub assignee | ✓ | ✓ (claim ownership) |
| Priority | GitHub labels | ✓ (parsed) | ✓ (override) |
| Story Points | N/A | ✓ (estimated) | ✓ (adjust) |
| Due Date | GitHub milestone | ✓ | ✓ |
| Labels | GitHub labels | ✓ | ✗ |
| **ROI Category** | N/A | ✗ | ✓ (must tag) |
| **Impact Score** | N/A | ✓ (calculated) | ✓ (adjust) |
| **Strategic Tags** | N/A | ✗ | ✓ (must tag) |
| **Parent Initiative** | N/A | ✗ | ✓ (must link) |
| Cycle Time | Git timestamps | ✓ | ✗ |
| Code Metrics | Git diff | ✓ | ✗ |
| Review Metrics | GitHub API | ✓ | ✗ |

---

## 7. Implementation Phases

### Phase 1: GitHub Integration Foundation (Weeks 1-3)
- [ ] Implement GitHubService with Octokit
- [ ] Create WorkItem unified data model
- [ ] Build SyncOrchestrator for polling
- [ ] Update TeamDashboardContext for new data model
- [ ] Create GitHubSyncStatus component

### Phase 2: Metrics Engine (Weeks 4-5)
- [ ] Implement DevelopmentKPICalculator
- [ ] Implement ROICalculator
- [ ] Implement IndustryRelevanceScorer
- [ ] Create useMetrics hook
- [ ] Build DevelopmentKPIs component

### Phase 3: AI Insights (Weeks 6-7)
- [ ] Implement AIInsightsEngine
- [ ] Create insight generation rules
- [ ] Build AIInsights component
- [ ] Add trend detection algorithms
- [ ] Add anomaly detection

### Phase 4: Strategic Dashboard UI (Weeks 8-10)
- [ ] Build StrategicDashboard component
- [ ] Build PortfolioView component
- [ ] Build ROIAnalysis component
- [ ] Build IndustryRelevance component
- [ ] Create StrategicInitiatives management

### Phase 5: Polish & Optimization (Weeks 11-12)
- [ ] Performance optimization for large datasets
- [ ] Caching layer for API responses
- [ ] Offline support with local cache
- [ ] User testing and refinement
- [ ] Documentation

---

## 8. Key API Endpoints (GitHub)

| Endpoint | Purpose | Rate Limit |
|----------|---------|------------|
| `GET /repos/{owner}/{repo}/issues` | Fetch issues | 1 request per call |
| `GET /repos/{owner}/{repo}/pulls` | Fetch PRs | 1 request per call |
| `GET /repos/{owner}/{repo}/pulls/{number}/reviews` | Fetch PR reviews | 1 request per PR |
| `GET /repos/{owner}/{repo}/commits` | Fetch commits | 1 request per call |
| `GET /repos/{owner}/{repo}/issues/{number}/timeline` | Fetch issue timeline | 1 request per issue |
| `GET /rate_limit` | Check rate limit | 1 request per sync |

---

## 9. Success Metrics for the New Dashboard

| Metric | Target | Measurement |
|--------|--------|-------------|
| Time to insight | < 5 seconds | Time from open to metrics displayed |
| Sync latency | < 2 minutes | Time from GitHub event to dashboard update |
| Manual input reduction | > 80% | Reduction in manual data entry actions |
| Strategic alignment visibility | 100% | % of work items linked to initiatives |
| AI insight accuracy | > 85% | User confirmation rate of insights |
| ROI calculation coverage | > 90% | % of initiatives with ROI analysis |

---

## 10. Migration Path from Current Implementation

1. **Preserve existing localStorage data** - Migrate manual issues to WorkItems with `source: 'manual'`
2. **Parallel run period** - Run old and new dashboards side-by-side during transition
3. **Feature flag rollout** - Enable new features incrementally
4. **User training** - Document new workflows (claiming vs. creating)
5. **Deprecation timeline** - 90-day deprecation of manual issue creation

---

## Appendix A: Complete Type Definitions

See accompanying file: `src/app/src/renderer/types/workItem.ts`

## Appendix B: GitHub OAuth Flow

See accompanying file: `docs/GITHUB_OAUTH_SETUP.md`

## Appendix C: Metrics Calculation Reference

See accompanying file: `docs/METRICS_REFERENCE.md`
