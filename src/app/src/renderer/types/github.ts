/**
 * GitHub Service Types
 * Type definitions for GitHub API integration
 */

import type {
  Contributor,
  WorkItem,
  Label,
  WorkItemMetrics,
  WorkItemRelationship,
} from '../types/workItem';

// ============================================================================
// GitHub API Configuration
// ============================================================================

export interface GitHubConfig {
  token: string;
  owner: string;
  repo: string;
  baseUrl?: string; // For GitHub Enterprise
  uploadUrl?: string; // For GitHub Enterprise
}

export interface GitHubAuthState {
  authenticated: boolean;
  user?: GitHubUser;
  scopes?: string[];
  tokenExpiresAt?: string;
}

// ============================================================================
// GitHub API Response Types (Raw)
// ============================================================================

export interface GitHubUser {
  login: string;
  id: number;
  node_id: string;
  avatar_url: string;
  gravatar_id: string | null;
  url: string;
  html_url: string;
  followers_url: string;
  following_url: string;
  gists_url: string;
  starred_url: string;
  subscriptions_url: string;
  organizations_url: string;
  repos_url: string;
  events_url: string;
  received_events_url: string;
  type: 'User' | 'Bot';
  site_admin: boolean;
  name?: string;
  email?: string;
  company?: string;
  blog?: string;
  location?: string;
  hireable?: boolean;
  bio?: string;
  twitter_username?: string;
  public_repos?: number;
  public_gists?: number;
  followers?: number;
  following?: number;
  created_at: string;
  updated_at: string;
}

export interface GitHubLabel {
  id: number;
  node_id: string;
  url: string;
  name: string;
  color: string;
  default: boolean;
  description?: string;
}

export interface GitHubMilestone {
  url: string;
  html_url: string;
  labels_url: string;
  id: number;
  node_id: string;
  number: number;
  title: string;
  description: string;
  creator: GitHubUser;
  open_issues: number;
  closed_issues: number;
  state: 'open' | 'closed';
  created_at: string;
  updated_at: string;
  due_on?: string;
  closed_at?: string;
}

export interface GitHubIssue {
  url: string;
  repository_url: string;
  labels_url: string;
  comments_url: string;
  events_url: string;
  html_url: string;
  id: number;
  node_id: string;
  number: number;
  title: string;
  user: GitHubUser;
  labels: GitHubLabel[];
  state: 'open' | 'closed';
  locked: boolean;
  assignee?: GitHubUser;
  assignees?: GitHubUser[];
  milestone?: GitHubMilestone;
  comments: number;
  created_at: string;
  updated_at: string;
  closed_at?: string;
  author_association: 'COLLABORATOR' | 'CONTRIBUTOR' | 'FIRST_TIMER' | 'FIRST_TIME_CONTRIBUTOR' | 'MANNEQUIN' | 'MEMBER' | 'NONE' | 'OWNER';
  body?: string;
  body_html?: string;
  body_text?: string;
  timeline_url?: string;
  reactions?: GitHubReactions;
  draft?: boolean;
  pull_request?: {
    url: string;
    html_url: string;
    diff_url: string;
    patch_url: string;
    merged_at?: string;
  };
}

export interface GitHubPullRequest {
  url: string;
  id: number;
  node_id: string;
  html_url: string;
  diff_url: string;
  patch_url: string;
  issue_url: string;
  number: number;
  state: 'open' | 'closed';
  locked: boolean;
  title: string;
  user: GitHubUser;
  body?: string;
  body_html?: string;
  body_text?: string;
  labels: GitHubLabel[];
  milestone?: GitHubMilestone;
  active_lock_reason?: string;
  created_at: string;
  updated_at: string;
  closed_at?: string;
  merged_at?: string;
  merge_commit_sha?: string;
  assignee?: GitHubUser;
  assignees?: GitHubUser[];
  requested_reviewers?: GitHubUser[];
  requested_teams?: GitHubTeam[];
  head: GitHubPullRequestBranch;
  base: GitHubPullRequestBranch;
  author_association: string;
  draft?: boolean;
  merged: boolean;
  mergeable?: boolean;
  rebaseable?: boolean;
  mergeable_state: 'behind' | 'blocked' | 'clean' | 'dirty' | 'draft' | 'has_hooks' | 'unknown' | 'unstable';
  merged_by?: GitHubUser;
  comments: number;
  review_comments: number;
  maintainer_can_modify: boolean;
  commits: number;
  additions: number;
  deletions: number;
  changed_files: number;
}

export interface GitHubPullRequestBranch {
  label: string;
  ref: string;
  sha: string;
  user: GitHubUser;
  repo: GitHubRepository;
}

export interface GitHubRepository {
  id: number;
  node_id: string;
  name: string;
  full_name: string;
  private: boolean;
  owner: GitHubUser;
  html_url: string;
  description: string;
  fork: boolean;
  url: string;
  forks_url: string;
  keys_url: string;
  collaborators_url: string;
  teams_url: string;
  hooks_url: string;
  created_at: string;
  updated_at: string;
  pushed_at: string;
  homepage?: string;
  size: number;
  stargazers_count: number;
  watchers_count: number;
  language?: string;
  forks_count: number;
  open_issues_count: number;
  master_branch?: string;
  default_branch: string;
  topics?: string[];
}

export interface GitHubTeam {
  id: number;
  node_id: string;
  url: string;
  html_url: string;
  name: string;
  slug: string;
  description?: string;
  privacy: 'secret' | 'closed' | 'visible';
  notification_setting: 'notifications_enabled' | 'notifications_disabled';
  permission: 'pull' | 'push' | 'admin' | 'maintain' | 'triage';
  members_url: string;
  repositories_url: string;
}

export interface GitHubCommit {
  url: string;
  sha: string;
  node_id: string;
  html_url: string;
  comments_url: string;
  commit: GitHubCommitDetail;
  author?: GitHubUser;
  committer?: GitHubUser;
  parents: GitHubCommitParent[];
  stats?: GitHubCommitStats;
  files?: GitHubCommitFile[];
}

export interface GitHubCommitAuthor {
  name: string;
  email: string;
  date?: string;
}

export interface GitHubCommitDetail {
  url: string;
  author?: GitHubCommitAuthor;
  committer?: GitHubCommitAuthor;
  message: string;
  tree: {
    sha: string;
    url: string;
  };
  comment_count: number;
  verification?: {
    verified: boolean;
    reason: string;
    signature?: string;
    payload?: string;
  };
}

export interface GitHubCommitParent {
  url: string;
  sha: string;
  html_url?: string;
}

export interface GitHubCommitStats {
  total: number;
  additions: number;
  deletions: number;
}

export interface GitHubCommitFile {
  sha: string;
  filename: string;
  status: 'added' | 'removed' | 'modified' | 'renamed' | 'copied' | 'changed' | 'unchanged';
  additions: number;
  deletions: number;
  changes: number;
  blob_url: string;
  raw_url: string;
  contents_url: string;
  patch?: string;
  previous_filename?: string;
}

export interface GitHubReview {
  id: number;
  node_id: string;
  user: GitHubUser;
  body?: string;
  body_html?: string;
  body_text?: string;
  html_url: string;
  pull_request_url: string;
  author_association: string;
  state: 'APPROVED' | 'CHANGES_REQUESTED' | 'COMMENTED' | 'DISMISSED' | 'PENDING';
  submitted_at: string;
  commit_id: string;
}

export interface GitHubReactions {
  url: string;
  total_count: number;
  '+1': number;
  '-1': number;
  laugh: number;
  hooray: number;
  confused: number;
  heart: number;
  rocket: number;
  eyes: number;
}

export interface GitHubTimelineEvent {
  id: number;
  node_id: string;
  url: string;
  actor?: GitHubUser;
  commit_id?: string;
  event: string;
  created_at: string;
  data?: unknown;
}

export interface GitHubRateLimit {
  resources: {
    core: GitHubRateLimitDetail;
    search: GitHubRateLimitDetail;
    graphql: GitHubRateLimitDetail;
    integration_manifest: GitHubRateLimitDetail;
    source_import: GitHubRateLimitDetail;
  };
}

export interface GitHubRateLimitDetail {
  limit: number;
  used: number;
  remaining: number;
  reset: number; // Unix timestamp
}

// ============================================================================
// Normalized Types (for internal use)
// ============================================================================

export interface NormalizedIssue {
  id: number;
  nodeId: string;
  number: number;
  title: string;
  body: string;
  state: 'open' | 'closed';
  locked: boolean;
  labels: GitHubLabel[];
  assignees: GitHubUser[];
  author: GitHubUser;
  milestone?: GitHubMilestone;
  reactions?: GitHubReactions;
  comments?: number;
  created_at: string;
  updated_at: string;
  closed_at?: string;
  pull_request?: {
    url: string;
    merged_at?: string;
  };
}

export interface NormalizedPullRequest {
  id: number;
  nodeId: string;
  number: number;
  title: string;
  body: string;
  state: 'open' | 'closed';
  merged: boolean;
  merged_at?: string;
  merge_commit_sha?: string;
  draft: boolean;
  labels: GitHubLabel[];
  assignees: GitHubUser[];
  author: GitHubUser;
  requested_reviewers: GitHubUser[];
  reviews: GitHubReview[];
  additions: number;
  deletions: number;
  changed_files: number;
  commits: number;
  review_comments: number;
  milestone?: GitHubMilestone;
  created_at: string;
  updated_at: string;
  closed_at?: string;
}

export interface NormalizedCommit {
  sha: string;
  message: string;
  author: GitHubUser;
  committer: GitHubUser;
  timestamp: string;
  files: Array<{
    filename: string;
    status: string;
    additions: number;
    deletions: number;
    changes: number;
    patch?: string;
  }>;
  stats: {
    total: number;
    additions: number;
    deletions: number;
  };
  parents: string[];
  verified: boolean;
}

// ============================================================================
// Service Interface
// ============================================================================

export interface IFetchOptions {
  since?: Date;
  until?: Date;
  state?: 'all' | 'open' | 'closed';
  perPage?: number;
  page?: number;
  labels?: string[];
  assignee?: string;
  creator?: string;
  mentioned?: string;
}

export interface IFetchResult<T> {
  data: T[];
  pagination: {
    page: number;
    perPage: number;
    total: number;
    totalPages: number;
    hasNextPage: boolean;
  };
  rateLimit: {
    remaining: number;
    resetAt: Date;
  };
}

export interface LinkMap {
  get(issueId: string): string[] | undefined;
  set(issueId: string, prIds: string[]): void;
  has(issueId: string): boolean;
  entries(): IterableIterator<[string, string[]]>;
}

// ============================================================================
// Service Events
// ============================================================================

export type GitHubServiceEvent =
  | { type: 'sync_started'; timestamp: Date }
  | { type: 'sync_progress'; progress: number; total: number; current: string }
  | { type: 'sync_completed'; timestamp: Date; itemsSynced: number }
  | { type: 'sync_error'; error: Error }
  | { type: 'rate_limit_warning'; remaining: number; resetAt: Date }
  | { type: 'rate_limit_exceeded'; resetAt: Date };

export type GitHubServiceListener = (event: GitHubServiceEvent) => void;

// ============================================================================
// Normalization Functions Type
// ============================================================================

export interface INormalizationService {
  normalizeIssue(issue: GitHubIssue): NormalizedIssue;
  normalizePullRequest(pr: GitHubPullRequest): NormalizedPullRequest;
  normalizeCommit(commit: GitHubCommit): NormalizedCommit;
  normalizeContributor(user: GitHubUser): Contributor;
  normalizeLabel(label: GitHubLabel): Label;
  extractMetrics(issue: GitHubIssue | GitHubPullRequest): WorkItemMetrics;
  extractRelationships(item: GitHubIssue | GitHubPullRequest): WorkItemRelationship[];
}

// ============================================================================
// GitHub Service Class Type
// ============================================================================

export interface IGitHubService {
  // Lifecycle
  initialize(config: GitHubConfig): Promise<void>;
  dispose(): void;

  // Authentication
  getAuthState(): GitHubAuthState;
  validateToken(): Promise<boolean>;

  // Data Fetching
  fetchIssues(options?: IFetchOptions): Promise<IFetchResult<NormalizedIssue>>;
  fetchPullRequests(options?: IFetchOptions): Promise<IFetchResult<NormalizedPullRequest>>;
  fetchCommits(options?: IFetchOptions): Promise<IFetchResult<NormalizedCommit>>;
  fetchPRReviews(prNumber: number): Promise<GitHubReview[]>;
  fetchIssueTimeline(issueNumber: number): Promise<GitHubTimelineEvent[]>;

  // Linking
  linkIssuesToPRs(
    issues: NormalizedIssue[],
    prs: NormalizedPullRequest[]
  ): Promise<LinkMap>;

  // Rate Limiting
  getRateLimitInfo(): Promise<{ remaining: number; resetAt: Date; limit: number }>;

  // Event Handling
  addListener(listener: GitHubServiceListener): () => void;
  removeListener(listener: GitHubServiceListener): void;

  // Utilities
  normalizeToWorkItem(item: NormalizedIssue | NormalizedPullRequest | NormalizedCommit): WorkItem;
}

// ============================================================================
// Issue Link Patterns
// ============================================================================

export const ISSUE_LINK_PATTERNS = {
  // GitHub closing keywords
  FIXES: /(?:fix|fixes|fixed|fixing)\s+#(\d+)/gi,
  CLOSES: /(?:close|closes|closed|closing)\s+#(\d+)/gi,
  RESOLVES: /(?:resolve|resolves|resolved|resolving)\s+#(\d+)/gi,

  // Alternative formats
  FIXES_WITH_SPACE: /(?:fix|fixes|fixed)\s+#[\s]*(\d+)/gi,
  ISSUE_REFERENCE: /#(\d+)/g,

  // Jira-style references (for integration)
  JIRA_STYLE: /\b([A-Z]+-\d+)\b/g,
} as const;

// ============================================================================
// GitHub API Rate Limits (for reference)
// ============================================================================

export const RATE_LIMITS = {
  // GitHub App
  GITHUB_APP: {
    core: 5000,
    search: 30,
    graphql: 5000,
  },
  // Personal Access Token
  PAT: {
    core: 5000,
    search: 10,
    graphql: 5000,
  },
  // Unauthenticated
  UNAUTHENTICATED: {
    core: 60,
    search: 10,
  },
} as const;

// ============================================================================
// Error Types
// ============================================================================

export class GitHubError extends Error {
  constructor(
    message: string,
    public readonly statusCode?: number,
    public readonly response?: unknown
  ) {
    super(message);
    this.name = 'GitHubError';
  }
}

export class GitHubRateLimitError extends GitHubError {
  constructor(
    message: string,
    public readonly resetAt: Date,
    public readonly remaining: number
  ) {
    super(message, 403);
    this.name = 'GitHubRateLimitError';
  }
}

export class GitHubAuthenticationError extends GitHubError {
  constructor(message: string) {
    super(message, 401);
    this.name = 'GitHubAuthenticationError';
  }
}

export class GitHubNotFoundError extends GitHubError {
  constructor(message: string) {
    super(message, 404);
    this.name = 'GitHubNotFoundError';
  }
}

// ============================================================================
// Sync State Types (for SyncService)
// ============================================================================

export interface SyncState {
  status: 'idle' | 'syncing' | 'completed' | 'error' | 'rate_limited';
  lastFullSync?: string;
  lastIncrementalSync?: string;
  lastGitAnalysis?: string;
  nextScheduledSync?: string;
  pendingChanges: number;
  error?: string;
  rateLimitInfo?: {
    remaining: number;
    resetAt: string;
    limit: number;
  };
}
