/**
 * GitHub Service
 * Provides integration with GitHub API for auto-syncing issues, PRs, and commits
 *
 * This service handles:
 * - GitHub API authentication
 * - Fetching issues, pull requests, and commits
 * - Rate limit handling
 * - Data normalization
 * - Event emission for sync state updates
 */

import type {
  GitHubConfig,
  GitHubAuthState,
  IFetchOptions,
  IFetchResult,
  NormalizedIssue,
  NormalizedPullRequest,
  NormalizedCommit,
  GitHubReview,
  GitHubTimelineEvent,
  LinkMap,
  IGitHubService,
  GitHubServiceEvent,
  GitHubServiceListener,
  GitHubIssue,
  GitHubPullRequest,
  GitHubCommit,
  GitHubUser,
  GitHubLabel,
  GitHubCommitAuthor,
} from '../../types/github';

import type { WorkItem, Contributor, Label } from '../../types/workItem';

// ============================================================================
// Error Classes
// ============================================================================

export class GitHubRateLimitError extends Error {
  constructor(
    message: string,
    public readonly resetAt: Date,
    public readonly remaining: number
  ) {
    super(message);
    this.name = 'GitHubRateLimitError';
  }
}

export class GitHubAuthenticationError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'GitHubAuthenticationError';
  }
}

export class GitHubNotFoundError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'GitHubNotFoundError';
  }
}

// ============================================================================
// GitHub Normalizer
// ============================================================================

import { Octokit } from '@octokit/rest';

export const GitHubNormalizer = {
  normalizeIssue(issue: GitHubIssue): NormalizedIssue {
    return {
      id: issue.id,
      nodeId: issue.node_id,
      number: issue.number,
      title: issue.title,
      body: issue.body || '',
      state: issue.state,
      locked: issue.locked,
      labels: issue.labels,
      assignees: issue.assignees || [],
      author: issue.user,
      milestone: issue.milestone,
      reactions: issue.reactions,
      created_at: issue.created_at,
      updated_at: issue.updated_at,
      closed_at: issue.closed_at,
      pull_request: issue.pull_request,
    };
  },

  normalizePullRequest(pr: GitHubPullRequest): NormalizedPullRequest {
    return {
      id: pr.id,
      nodeId: pr.node_id,
      number: pr.number,
      title: pr.title,
      body: pr.body || '',
      state: pr.state,
      merged: pr.merged,
      merged_at: pr.merged_at,
      merge_commit_sha: pr.merge_commit_sha,
      draft: pr.draft || false,
      labels: pr.labels,
      assignees: pr.assignees || [],
      author: pr.user,
      requested_reviewers: pr.requested_reviewers || [],
      reviews: (pr as GitHubPullRequest & { reviews?: GitHubReview[] }).reviews || [],
      additions: pr.additions,
      deletions: pr.deletions,
      changed_files: pr.changed_files,
      commits: pr.commits,
      review_comments: pr.review_comments,
      milestone: pr.milestone,
      created_at: pr.created_at,
      updated_at: pr.updated_at,
      closed_at: pr.closed_at,
    };
  },

  normalizeCommit(commit: GitHubCommit): NormalizedCommit {
    const defaultUser: GitHubUser = { login: 'unknown', id: 0, node_id: '', avatar_url: '', gravatar_id: null, url: '', html_url: '', followers_url: '', following_url: '', gists_url: '', starred_url: '', subscriptions_url: '', organizations_url: '', repos_url: '', events_url: '', received_events_url: '', type: 'User', site_admin: false, created_at: '', updated_at: '' };

    // Helper to convert GitHubCommitAuthor to GitHubUser
    const toGitHubUser = (author: GitHubUser | GitHubCommitAuthor | undefined): GitHubUser => {
      if (!author) return defaultUser;
      if ('login' in author && 'id' in author) return author as GitHubUser;
      // Convert GitHubCommitAuthor to GitHubUser
      const commitAuthor = author as GitHubCommitAuthor;
      return {
        ...defaultUser,
        login: commitAuthor.name || commitAuthor.email || 'unknown',
        name: commitAuthor.name,
        email: commitAuthor.email,
      };
    };

    return {
      sha: commit.sha,
      message: commit.commit.message,
      author: toGitHubUser(commit.author ?? commit.commit.author),
      committer: toGitHubUser(commit.committer ?? commit.commit.committer),
      timestamp: commit.commit.author?.date ?? commit.commit.committer?.date ?? new Date().toISOString(),
      files: commit.files?.map((f) => ({
        filename: f.filename,
        status: f.status,
        additions: f.additions,
        deletions: f.deletions,
        changes: f.changes,
        patch: f.patch,
      })) || [],
      stats: commit.stats || { total: 0, additions: 0, deletions: 0 },
      parents: commit.parents.map((p) => p.sha),
      verified: commit.commit.verification?.verified || false,
    };
  },

  normalizeContributor(user: GitHubUser): Contributor {
    return {
      id: user.id?.toString() || user.login,
      name: user.name || user.login,
      email: user.email,
      avatar: user.avatar_url,
      githubUsername: user.login,
    };
  },

  normalizeLabel(label: GitHubLabel): Label {
    return {
      id: label.id?.toString() || label.name,
      name: label.name,
      color: label.color,
      description: label.description,
      source: 'github' as const,
    };
  },

  extractMetrics(item: GitHubIssue | GitHubPullRequest): { age: number; commentCount: number; commitsCount?: number } {
    const metrics: { age: number; commentCount: number; commitsCount?: number } = {
      age: Date.now() - new Date(item.created_at).getTime(),
      commentCount: 'comments' in item ? item.comments : 0,
    };

    if ('commits' in item) {
      metrics.commitsCount = item.commits;
    }

    return metrics;
  },

  extractPriority(labels: GitHubLabel[]): 'low' | 'medium' | 'high' | 'critical' {
    const priorityLabels = labels.map((l) => l.name.toLowerCase());

    if (priorityLabels.some((l) => l.includes('critical') || l.includes('blocker'))) {
      return 'critical';
    }
    if (priorityLabels.some((l) => l.includes('high') || l.includes('urgent'))) {
      return 'high';
    }
    if (priorityLabels.some((l) => l.includes('medium') || l.includes('normal'))) {
      return 'medium';
    }
    return 'low';
  },

  normalizeToWorkItem(
    item: NormalizedIssue | NormalizedPullRequest | NormalizedCommit
  ): WorkItem {
    const now = new Date().toISOString();

    if ('body' in item && 'merged' in item) {
      // Pull Request
      const pr = item as NormalizedPullRequest;
      return {
        id: `gh-pr-${pr.id}`,
        source: 'github-pr',
        externalId: String(pr.id),
        number: pr.number,
        title: pr.title,
        description: pr.body,
        type: 'pr',
        status: pr.merged ? 'merged' : pr.state === 'closed' ? 'closed' : 'in_review',
        mergeStatus: pr.merged ? 'merged' : pr.state === 'closed' ? 'closed' : 'pending',
        priority: this.extractPriority(pr.labels),
        author: this.normalizeContributor(pr.author),
        assignees: pr.assignees.map(this.normalizeContributor.bind(this)),
        metrics: {
          age: Date.now() - new Date(pr.created_at).getTime(),
          linesAdded: pr.additions,
          linesDeleted: pr.deletions,
          filesChanged: pr.changed_files,
          commitsCount: pr.commits,
          reviewComments: pr.review_comments,
        },
        labels: pr.labels.map(this.normalizeLabel.bind(this)),
        createdAt: pr.created_at,
        updatedAt: pr.updated_at,
        mergedAt: pr.merged_at,
        linkedItems: [],
      };
    } else if ('body' in item) {
      // Issue
      const issue = item as NormalizedIssue;
      return {
        id: `gh-issue-${issue.id}`,
        source: 'github-issue',
        externalId: String(issue.id),
        number: issue.number,
        title: issue.title,
        description: issue.body,
        type: 'issue',
        status: issue.state === 'closed' ? 'done' : 'backlog',
        priority: this.extractPriority(issue.labels),
        author: this.normalizeContributor(issue.author),
        assignees: (issue.assignees || []).map(this.normalizeContributor.bind(this)),
        metrics: {
          age: Date.now() - new Date(issue.created_at).getTime(),
          commentCount: issue.comments,
        },
        labels: issue.labels.map(this.normalizeLabel.bind(this)),
        createdAt: issue.created_at,
        updatedAt: issue.updated_at,
        resolvedAt: issue.closed_at,
        linkedItems: [],
      };
    } else {
      // Commit
      const commit = item as NormalizedCommit;
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
        assignees: [],
        metrics: {
          age: Date.now() - new Date(commit.timestamp).getTime(),
          linesAdded: commit.stats.additions,
          linesDeleted: commit.stats.deletions,
          filesChanged: commit.files.length,
        },
        labels: [],
        createdAt: commit.timestamp,
        updatedAt: commit.timestamp,
        linkedItems: [],
      };
    }
  },
};

// ============================================================================
// GitHub Service Implementation
// ============================================================================

export class GitHubService implements IGitHubService {
  private octokit: Octokit | null = null;
  private config: GitHubConfig | null = null;
  private authState: GitHubAuthState = { authenticated: false };
  private listeners: Set<GitHubServiceListener> = new Set();
  private rateLimitInfo: { remaining: number; resetAt: Date; limit: number } | null = null;

  /**
   * Initialize the GitHub service with configuration
   */
  async initialize(config: GitHubConfig): Promise<void> {
    this.config = config;

    // Dynamically import Octokit to avoid bundling issues
    const { Octokit } = await import('@octokit/rest');

    this.octokit = new Octokit({
      auth: config.token,
      baseUrl: config.baseUrl || 'https://api.github.com',
    });

    // Validate authentication
    const valid = await this.validateToken();
    if (!valid) {
      throw new GitHubAuthenticationError('Invalid GitHub token');
    }

    // Fetch initial rate limit info
    await this.updateRateLimitInfo();
  }

  /**
   * Dispose of service resources
   */
  dispose(): void {
    this.octokit = null;
    this.config = null;
    this.listeners.clear();
  }

  /**
   * Get current authentication state
   */
  getAuthState(): GitHubAuthState {
    return this.authState;
  }

  /**
   * Validate the GitHub token
   */
  async validateToken(): Promise<boolean> {
    if (!this.octokit) return false;

    try {
      const { data: user } = await this.octokit.users.getAuthenticated();
      this.authState = {
        authenticated: true,
        user: user as GitHubUser,
        scopes: this.extractScopes(),
      };
      return true;
    } catch (error) {
      this.authState = { authenticated: false };
      return false;
    }
  }

  /**
   * Fetch issues from GitHub
   */
  async fetchIssues(options?: IFetchOptions): Promise<IFetchResult<NormalizedIssue>> {
    if (!this.octokit || !this.config) {
      throw new Error('GitHubService not initialized');
    }

    this.emitEvent({ type: 'sync_started', timestamp: new Date() });

    try {
      const { data: issues } = await this.octokit.issues.listForRepo({
        owner: this.config.owner,
        repo: this.config.repo,
        state: options?.state || 'all',
        since: options?.since?.toISOString(),
        until: options?.until?.toISOString(),
        per_page: options?.perPage || 100,
        page: options?.page || 1,
        labels: options?.labels?.join(','),
        assignee: options?.assignee,
        creator: options?.creator,
      });

      await this.updateRateLimitInfo();

      const normalizedIssues = issues.map((issue) =>
        GitHubNormalizer.normalizeIssue(issue as GitHubIssue)
      );

      this.emitEvent({
        type: 'sync_completed',
        timestamp: new Date(),
        itemsSynced: normalizedIssues.length,
      });

      return {
        data: normalizedIssues,
        pagination: {
          page: options?.page || 1,
          perPage: options?.perPage || 100,
          total: issues.length,
          totalPages: Math.ceil(issues.length / (options?.perPage || 100)),
          hasNextPage: issues.length === (options?.perPage || 100),
        },
        rateLimit: await this.getRateLimitInfo(),
      };
    } catch (error) {
      this.handleError(error);
      throw error;
    }
  }

  /**
   * Fetch pull requests from GitHub
   */
  async fetchPullRequests(options?: IFetchOptions): Promise<IFetchResult<NormalizedPullRequest>> {
    if (!this.octokit || !this.config) {
      throw new Error('GitHubService not initialized');
    }

    try {
      const { data: prs } = await this.octokit.pulls.list({
        owner: this.config.owner,
        repo: this.config.repo,
        state: options?.state || 'all',
        per_page: options?.perPage || 100,
        page: options?.page || 1,
      });

      // Enrich with review data
      const prsWithReviews = await Promise.all(
        prs.map(async (pr) => {
          const reviews = await this.fetchPRReviews(pr.number);
          return { ...pr, reviews };
        })
      );

      const normalizedPRs = prsWithReviews.map((pr) =>
        GitHubNormalizer.normalizePullRequest(pr as unknown as GitHubPullRequest)
      );

      return {
        data: normalizedPRs,
        pagination: {
          page: options?.page || 1,
          perPage: options?.perPage || 100,
          total: prs.length,
          totalPages: Math.ceil(prs.length / (options?.perPage || 100)),
          hasNextPage: prs.length === (options?.perPage || 100),
        },
        rateLimit: await this.getRateLimitInfo(),
      };
    } catch (error) {
      this.handleError(error);
      throw error;
    }
  }

  /**
   * Fetch commits from GitHub
   */
  async fetchCommits(options?: IFetchOptions): Promise<IFetchResult<NormalizedCommit>> {
    if (!this.octokit || !this.config) {
      throw new Error('GitHubService not initialized');
    }

    try {
      const { data: commits } = await this.octokit.repos.listCommits({
        owner: this.config.owner,
        repo: this.config.repo,
        since: options?.since?.toISOString(),
        until: options?.until?.toISOString(),
        per_page: options?.perPage || 100,
        page: options?.page || 1,
      });

      const normalizedCommits = commits.map((commit) =>
        GitHubNormalizer.normalizeCommit(commit as GitHubCommit)
      );

      return {
        data: normalizedCommits,
        pagination: {
          page: options?.page || 1,
          perPage: options?.perPage || 100,
          total: commits.length,
          totalPages: Math.ceil(commits.length / (options?.perPage || 100)),
          hasNextPage: commits.length === (options?.perPage || 100),
        },
        rateLimit: await this.getRateLimitInfo(),
      };
    } catch (error) {
      this.handleError(error);
      throw error;
    }
  }

  /**
   * Fetch PR reviews
   */
  async fetchPRReviews(prNumber: number): Promise<GitHubReview[]> {
    if (!this.octokit || !this.config) {
      throw new Error('GitHubService not initialized');
    }

    try {
      const { data: reviews } = await this.octokit.pulls.listReviews({
        owner: this.config.owner,
        repo: this.config.repo,
        pull_number: prNumber,
      });

      return reviews as unknown as GitHubReview[];
    } catch (error) {
      console.error(`Failed to fetch reviews for PR #${prNumber}:`, error);
      return [];
    }
  }

  /**
   * Fetch issue timeline events
   */
  async fetchIssueTimeline(issueNumber: number): Promise<GitHubTimelineEvent[]> {
    if (!this.octokit || !this.config) {
      throw new Error('GitHubService not initialized');
    }

    try {
      const { data: timeline } = await this.octokit.issues.listEvents({
        owner: this.config.owner,
        repo: this.config.repo,
        issue_number: issueNumber,
      });

      return timeline as unknown as GitHubTimelineEvent[];
    } catch (error) {
      console.error(`Failed to fetch timeline for issue #${issueNumber}:`, error);
      return [];
    }
  }

  /**
   * Link issues to their PRs using closing keywords
   */
  async linkIssuesToPRs(
    issues: NormalizedIssue[],
    prs: NormalizedPullRequest[]
  ): Promise<LinkMap> {
    const linkMap = new Map<string, string[]>();

    for (const pr of prs) {
      const linkedIssues = this.extractIssueLinks(pr.body, pr.number);

      for (const issueNumber of linkedIssues) {
        const issue = issues.find((i) => i.number === issueNumber);
        if (issue) {
          const current = linkMap.get(issue.id.toString()) || [];
          linkMap.set(issue.id.toString(), [...current, pr.id.toString()]);
        }
      }
    }

    return linkMap;
  }

  /**
   * Get rate limit information
   */
  async getRateLimitInfo(): Promise<{ remaining: number; resetAt: Date; limit: number }> {
    if (!this.octokit) {
      throw new Error('GitHubService not initialized');
    }

    if (this.rateLimitInfo && this.rateLimitInfo.resetAt > new Date()) {
      return this.rateLimitInfo;
    }

    await this.updateRateLimitInfo();
    return this.rateLimitInfo!;
  }

  /**
   * Add event listener
   */
  addListener(listener: GitHubServiceListener): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  /**
   * Remove event listener
   */
  removeListener(listener: GitHubServiceListener): void {
    this.listeners.delete(listener);
  }

  /**
   * Normalize GitHub data to WorkItem
   */
  normalizeToWorkItem(
    item: NormalizedIssue | NormalizedPullRequest | NormalizedCommit
  ): WorkItem {
    return GitHubNormalizer.normalizeToWorkItem(item);
  }

  // ==========================================================================
  // Private Methods
  // ==========================================================================

  private async updateRateLimitInfo(): Promise<void> {
    if (!this.octokit) return;

    try {
      const { data } = await this.octokit.rateLimit.get();
      const core = data.resources.core;

      this.rateLimitInfo = {
        remaining: core.remaining,
        resetAt: new Date(core.reset * 1000),
        limit: core.limit,
      };

      // Emit warning if running low
      if (core.remaining < 500) {
        this.emitEvent({
          type: 'rate_limit_warning',
          remaining: core.remaining,
          resetAt: new Date(core.reset * 1000),
        });
      }
    } catch (error) {
      console.error('Failed to fetch rate limit info:', error);
    }
  }

  private extractIssueLinks(body: string, prNumber: number): number[] {
    const patterns = [
      /(?:fix|fixes|fixed|fixing)\s+#(\d+)/gi,
      /(?:close|closes|closed|closing)\s+#(\d+)/gi,
      /(?:resolve|resolves|resolved|resolving)\s+#(\d+)/gi,
    ];

    const issues = new Set<number>();

    for (const pattern of patterns) {
      let match;
      while ((match = pattern.exec(body)) !== null) {
        issues.add(parseInt(match[1], 10));
      }
    }

    return Array.from(issues);
  }

  private extractScopes(): string[] {
    // OAuth scopes would be available in the response headers
    // For PAT flow, we return empty array as scopes are predefined
    return [];
  }

  private handleError(error: unknown): void {
    if (error instanceof GitHubRateLimitError) {
      this.emitEvent({
        type: 'rate_limit_exceeded',
        resetAt: error.resetAt,
      });
    }

    this.emitEvent({
      type: 'sync_error',
      error: error instanceof Error ? error : new Error('Unknown error'),
    });
  }

  private emitEvent(event: GitHubServiceEvent): void {
    for (const listener of this.listeners) {
      try {
        listener(event);
      } catch (error) {
        console.error('Error in GitHub event listener:', error);
      }
    }
  }
}

// ============================================================================
// Export singleton instance
// ============================================================================

export const gitHubService = new GitHubService();
