# GitHub Integration Implementation Guide

## Overview

This guide provides step-by-step instructions for implementing the GitHub API integration layer for the AI-automated Team Tracking Dashboard.

## Prerequisites

1. **GitHub App or Personal Access Token**
   - For production: Create a GitHub App
   - For development: Use a Personal Access Token (classic)

2. **Required Permissions**
   - `repo` scope (full control of private repositories)
   - `read:org` scope (read organization data)
   - `read:user` scope (read user data)

## Step 1: GitHub App Setup (Recommended for Production)

### 1.1 Create GitHub App

1. Go to GitHub Settings > Developer Settings > GitHub Apps
2. Click "New GitHub App"
3. Fill in:
   - **Name**: `Lemonade Team Dashboard`
   - **Homepage URL**: `https://github.com/lemonade-org/lemonade`
   - **Callback URL**: (not needed for PAT flow)
   - Uncheck "Webhook" (we're polling, not receiving webhooks)

### 1.2 Set Permissions

**Repository Permissions:**
| Permission | Access |
|------------|--------|
| Contents | Read |
| Issues | Read |
| Pull requests | Read |
| Commit statuses | Read |
| Repository Projects | Read |
| Discussions | Read |

**Organization Permissions:**
| Permission | Access |
|------------|--------|
| Members | Read |

### 1.3 Generate Client Secret

1. Click "Generate a client secret"
2. Copy and securely store the client secret
3. Note the App ID

### 1.4 Install App on Repository

1. Go to your GitHub App settings
2. Click "Install App"
3. Select your organization/account
4. Choose repositories to grant access to

## Step 2: Personal Access Token Setup (Development)

### 2.1 Create Token

1. Go to GitHub Settings > Developer Settings > Personal Access Tokens > Tokens (classic)
2. Click "Generate new token (classic)"
3. Select scopes:
   - `repo` (Full control of private repositories)
   - `read:org` (Read organization data)
   - `read:user` (Read user data)
4. Click "Generate token"
5. **Copy and securely store the token** (won't be shown again)

### 2.2 Store Token Securely

For Electron app, use secure storage:

```typescript
// src/app/src/renderer/services/github/tokenStorage.ts

import { ipcRenderer } from 'electron';

export const tokenStorage = {
  async saveToken(token: string): Promise<void> {
    // Store in system keychain via Electron main process
    await ipcRenderer.invoke('secure-storage:set', 'github-token', token);
  },

  async getToken(): Promise<string | null> {
    return await ipcRenderer.invoke('secure-storage:get', 'github-token');
  },

  async deleteToken(): Promise<void> {
    await ipcRenderer.invoke('secure-storage:delete', 'github-token');
  },
};
```

## Step 3: Implement GitHub Service

### 3.1 Install Dependencies

```bash
cd src/app
npm install @octokit/rest @octokit/types
```

### 3.2 Create GitHub Service

```typescript
// src/app/src/renderer/services/github/GitHubService.ts

import { Octokit } from '@octokit/rest';
import type {
  GitHubConfig,
  GitHubAuthState,
  IFetchOptions,
  IFetchResult,
  NormalizedIssue,
  NormalizedPullRequest,
  NormalizedCommit,
  GitHubReview,
  LinkMap,
  IGitHubService,
  GitHubServiceEvent,
  GitHubServiceListener,
} from '../../types/github';
import {
  GitHubRateLimitError,
  GitHubAuthenticationError,
  GitHubNotFoundError,
} from '../../types/github';
import { GitHubNormalizer } from './GitHubNormalizer';

export class GitHubService implements IGitHubService {
  private octokit: Octokit | null = null;
  private config: GitHubConfig | null = null;
  private authState: GitHubAuthState = { authenticated: false };
  private listeners: Set<GitHubServiceListener> = new Set();

  async initialize(config: GitHubConfig): Promise<void> {
    this.config = config;

    this.octokit = new Octokit({
      auth: config.token,
      baseUrl: config.baseUrl || 'https://api.github.com',
    });

    // Validate authentication
    const valid = await this.validateToken();
    if (!valid) {
      throw new GitHubAuthenticationError('Invalid GitHub token');
    }
  }

  dispose(): void {
    this.octokit = null;
    this.listeners.clear();
  }

  getAuthState(): GitHubAuthState {
    return this.authState;
  }

  async validateToken(): Promise<boolean> {
    if (!this.octokit) return false;

    try {
      const { data: user } = await this.octokit.users.getAuthenticated();
      this.authState = {
        authenticated: true,
        user,
        scopes: this.extractScopes(),
      };
      return true;
    } catch (error) {
      this.authState = { authenticated: false };
      return false;
    }
  }

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

      const rateLimit = await this.getRateLimitInfo();

      const normalizedIssues = issues.map((issue) =>
        GitHubNormalizer.normalizeIssue(issue)
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
        rateLimit,
      };
    } catch (error) {
      this.handleError(error);
      throw error;
    }
  }

  async fetchPullRequests(
    options?: IFetchOptions
  ): Promise<IFetchResult<NormalizedPullRequest>> {
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
        GitHubNormalizer.normalizePullRequest(pr)
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
        GitHubNormalizer.normalizeCommit(commit)
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

      return reviews;
    } catch (error) {
      console.error(`Failed to fetch reviews for PR #${prNumber}:`, error);
      return [];
    }
  }

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

  async getRateLimitInfo(): Promise<{ remaining: number; resetAt: Date; limit: number }> {
    if (!this.octokit) {
      throw new Error('GitHubService not initialized');
    }

    const { data } = await this.octokit.rateLimit.get();
    const core = data.resources.core;

    const resetAt = new Date(core.reset * 1000);

    // Emit warning if running low
    if (core.remaining < 500) {
      this.emitEvent({
        type: 'rate_limit_warning',
        remaining: core.remaining,
        resetAt,
      });
    }

    return {
      remaining: core.remaining,
      resetAt,
      limit: core.limit,
    };
  }

  addListener(listener: GitHubServiceListener): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  removeListener(listener: GitHubServiceListener): void {
    this.listeners.delete(listener);
  }

  normalizeToWorkItem(
    item: NormalizedIssue | NormalizedPullRequest | NormalizedCommit
  ) {
    return GitHubNormalizer.normalizeToWorkItem(item);
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
    // Extract from OAuth scopes if available
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
```

### 3.3 Create Normalizer

```typescript
// src/app/src/renderer/services/github/GitHubNormalizer.ts

import type {
  GitHubIssue,
  GitHubPullRequest,
  GitHubCommit,
  NormalizedIssue,
  NormalizedPullRequest,
  NormalizedCommit,
} from '../../types/github';
import type { Contributor, Label, WorkItem, WorkItemMetrics } from '../../types/workItem';

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
      reviews: pr.reviews || [],
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
    return {
      sha: commit.sha,
      message: commit.commit.message,
      author: commit.author || commit.commit.author || { login: 'unknown', id: 0 },
      committer: commit.committer || commit.commit.committer || { login: 'unknown', id: 0 },
      timestamp: commit.commit.author?.date || commit.commit.committer?.date || new Date().toISOString(),
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

  normalizeContributor(user: any): Contributor {
    return {
      id: user.id?.toString() || user.login,
      name: user.name || user.login,
      email: user.email,
      avatar: user.avatar_url,
      githubUsername: user.login,
    };
  },

  normalizeLabel(label: any): Label {
    return {
      id: label.id?.toString() || label.name,
      name: label.name,
      color: label.color,
      description: label.description,
      source: 'github',
    };
  },

  extractMetrics(item: GitHubIssue | GitHubPullRequest): WorkItemMetrics {
    const metrics: WorkItemMetrics = {
      age: Date.now() - new Date(item.created_at).getTime(),
      commentCount: 'comments' in item ? item.comments : 0,
    };

    if ('commit' in item && 'commits' in item) {
      metrics.commitsCount = item.commits;
    }

    return metrics;
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
        metrics: this.extractMetrics(pr),
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
        assignees: issue.assignees.map(this.normalizeContributor.bind(this)),
        metrics: this.extractMetrics(issue),
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

  extractPriority(labels: any[]): 'low' | 'medium' | 'high' | 'critical' {
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
};
```

## Step 4: Configure Sync Orchestrator

See `src/app/src/renderer/services/sync/SyncOrchestrator.ts` in the main architecture document.

## Step 5: Testing

### 5.1 Unit Tests

```typescript
// src/app/src/renderer/services/github/__tests__/GitHubService.test.ts

import { GitHubService } from '../GitHubService';

describe('GitHubService', () => {
  let service: GitHubService;

  beforeEach(() => {
    service = new GitHubService();
  });

  afterEach(() => {
    service.dispose();
  });

  it('should initialize with valid config', async () => {
    await service.initialize({
      token: process.env.GITHUB_TOKEN!,
      owner: 'test-owner',
      repo: 'test-repo',
    });

    expect(service.getAuthState().authenticated).toBe(true);
  });

  it('should fetch issues', async () => {
    await service.initialize({
      token: process.env.GITHUB_TOKEN!,
      owner: 'lemonade-org',
      repo: 'lemonade',
    });

    const result = await service.fetchIssues();

    expect(result.data).toBeDefined();
    expect(result.pagination.total).toBeGreaterThanOrEqual(0);
  });
});
```

## Step 6: Security Considerations

1. **Never commit tokens** - Use environment variables or secure storage
2. **Use minimum required scopes** - Request only what's needed
3. **Implement rate limit handling** - Respect GitHub's rate limits
4. **Cache responses** - Reduce API calls
5. **Use conditional requests** - Leverage ETags for caching

## Troubleshooting

| Issue | Solution |
|-------|----------|
| 401 Unauthorized | Token expired or invalid |
| 403 Rate Limit | Implement backoff, check rate limits |
| 404 Not Found | Check repo name, ensure app is installed |
| Empty results | Verify repository visibility and permissions |
