/**
 * Work Item TypeScript Interfaces
 * Unified data model for AI-Automated Team Tracking Dashboard
 *
 * This replaces the legacy Issue interface with a comprehensive WorkItem type
 * that unifies GitHub Issues, Pull Requests, Commits, and manual items.
 */

// ============================================================================
// Core Work Item Types
// ============================================================================

export type WorkItemSource = 'github-issue' | 'github-pr' | 'github-commit' | 'manual';

export type WorkItemType = 'issue' | 'pr' | 'commit' | 'initiative' | 'epic';

export type WorkItemStatus =
  | 'backlog'      // Open issue, no active development
  | 'in_progress'  // Active development (has commits or assigned)
  | 'in_review'    // PR open, awaiting review/merge
  | 'merged'       // PR successfully merged
  | 'done'         // Issue closed as completed
  | 'closed';      // Closed without completion (duplicate, wontfix, etc.)

export type PRMergeStatus = 'pending' | 'merged' | 'closed' | 'draft';

export type Priority = 'low' | 'medium' | 'high' | 'critical';

export type PriorityChangeReason =
  | 'manual_override'
  | 'ai_suggested'
  | 'deadline_approaching'
  | 'blocker_identified'
  | 'dependency_changed';

// ============================================================================
// Contributor Types
// ============================================================================

export interface Contributor {
  id: string;
  name: string;
  email?: string;
  avatar?: string;
  githubUsername?: string;
  role?: string;
  team?: string;
}

export interface TeamMember extends Contributor {
  githubId?: number;
  capacity?: number; // Weekly capacity in hours
  skills?: string[];
  active: boolean;
}

// ============================================================================
// Label Types
// ============================================================================

export interface Label {
  id: string;
  name: string;
  color: string;
  description?: string;
  source: 'github' | 'manual';
}

// ============================================================================
// Work Item Metrics (Auto-Calculated)
// ============================================================================

export interface WorkItemMetrics {
  // Time metrics (in hours unless specified)
  age: number;
  timeToFirstReview?: number;
  timeToMerge?: number;
  cycleTime?: number;
  waitTime?: number; // Time spent blocked or waiting

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
  commentCount?: number;

  // Contribution metrics
  contributorCount?: number;
  reactionCount?: number;

  // Complexity estimates (auto-calculated)
  estimatedPoints?: number;
  complexityScore?: number; // 1-100
}

// ============================================================================
// Strategic Tagging (Human Input)
// ============================================================================

export type StrategicCategory = 'feature' | 'tech-debt' | 'performance' | 'security' | 'compliance' | 'infrastructure';

export type StrategicAlignment = 'core' | 'adjacent' | 'experimental';

export interface StrategicTag {
  id: string;
  name: string;
  category: StrategicCategory;
  strategicAlignment: StrategicAlignment;
  description?: string;
  addedBy: string; // User ID
  addedAt: string; // ISO timestamp
}

export type ROICategoryType =
  | 'revenue-impact'
  | 'cost-reduction'
  | 'risk-mitigation'
  | 'strategic-capability'
  | 'developer-productivity'
  | 'customer-experience';

export interface ROICategory {
  id: string;
  name: ROICategoryType;
  description?: string;
  estimatedEffort: number; // Story points
  estimatedImpact: number; // 1-10 scale
  actualImpact?: number; // Post-completion measurement
  confidenceLevel?: number; // 0-1, how confident in estimates
}

// ============================================================================
// Work Item Relationships
// ============================================================================

export interface WorkItemRelationship {
  type: 'blocks' | 'blocked_by' | 'relates_to' | 'depends_on' | 'duplicate_of' | 'parent_of' | 'child_of';
  targetId: string;
  source: 'github' | 'manual';
}

export interface LinkedEntity {
  id: string;
  type: 'issue' | 'pr' | 'commit' | 'external';
  url?: string;
  title?: string;
}

// ============================================================================
// Main Work Item Interface
// ============================================================================

export interface WorkItem {
  // Identity
  id: string;
  source: WorkItemSource;
  externalId?: string; // GitHub ID
  number?: number; // GitHub number for display
  url?: string; // GitHub URL

  // Content
  title: string;
  description: string;
  type: WorkItemType;

  // Status (auto-derived from GitHub state)
  status: WorkItemStatus;
  mergeStatus?: PRMergeStatus;

  // Priority (human-overridable with tracking)
  priority: Priority;
  autoPriority?: Priority; // AI-suggested priority
  priorityHistory?: PriorityChange[];

  // People
  author: Contributor;
  assignees: Contributor[];
  claimedBy?: string; // Human claim (overrides auto-assign)
  reviewers?: Contributor[];
  mentionedUsers?: Contributor[];

  // Metrics (auto-calculated)
  metrics: WorkItemMetrics;

  // Strategic tagging (human input required for ROI tracking)
  strategicTags?: StrategicTag[];
  roiCategory?: ROICategory;
  impactScore?: number; // 1-10, human-adjustable
  autoImpactScore?: number; // AI-calculated baseline

  // Timeline
  createdAt: string;
  updatedAt: string;
  resolvedAt?: string;
  mergedAt?: string;
  closedAt?: string;
  lastActivityAt?: string;

  // Relationships
  linkedItems: WorkItemRelationship[];
  parentInitiative?: string; // Parent strategic initiative ID
  epicId?: string;

  // Labels
  labels: Label[];

  // GitHub-specific fields
  github?: GitHubMetadata;

  // Local-only fields (for manual items)
  local?: LocalMetadata;
}

export interface PriorityChange {
  from: Priority;
  to: Priority;
  reason: PriorityChangeReason;
  changedBy: string;
  changedAt: string;
  note?: string;
}

export interface GitHubMetadata {
  nodeId: string;
  htmlUrl: string;
  state: 'open' | 'closed';
  locked: boolean;
  activeLockReason?: string;
  draft?: boolean;
  milestone?: GitHubMilestone;
  repository: {
    owner: string;
    name: string;
    fullName: string;
  };
  reactions?: {
    '+1': number;
    '-1': number;
    laugh: number;
    hooray: number;
    confused: number;
    heart: number;
    rocket: number;
    eyes: number;
  };
  timeline?: TimelineEvent[];
}

export interface GitHubMilestone {
  id: number;
  number: number;
  title: string;
  description?: string;
  state: 'open' | 'closed';
  dueOn?: string;
  closedAt?: string;
  progress: number; // Percentage complete
}

export interface TimelineEvent {
  id: string;
  type: string;
  createdAt: string;
  actor?: Contributor;
  data?: unknown;
}

export interface LocalMetadata {
  createdBy: string;
  private: boolean; // Only visible to creator
  notes?: string; // Private notes
}

// ============================================================================
// Strategic Initiative Types
// ============================================================================

export interface StrategicInitiative {
  id: string;
  name: string;
  description: string;
  objective: string; // Clear objective statement
  keyResults?: string[]; // Measurable key results

  // Timeframe
  timeframe: {
    start: string; // ISO date
    end: string;   // ISO date
    milestones?: Milestone[];
  };

  // Targets
  targets: {
    expectedROI: number; // Expected ROI ratio
    expectedImpact: number; // 1-10 scale
    budgetPoints: number; // Allocated story points
    targetMetrics?: Record<string, number>;
  };

  // Actuals (populated as work completes)
  actuals?: {
    actualROI?: number;
    actualImpact?: number;
    actualPoints: number;
    completedAt?: string;
  };

  // Linked work
  linkedWorkItems: string[]; // WorkItem IDs
  childInitiatives?: string[]; // Sub-initiative IDs
  parentInitiative?: string;

  // Ownership
  sponsor: Contributor;
  owner?: Contributor;
  contributors?: Contributor[];

  // Status tracking
  status: 'planning' | 'active' | 'on_hold' | 'completed' | 'cancelled';
  healthStatus: 'green' | 'yellow' | 'red';
  progress: number; // 0-100 percentage

  // Metadata
  tags?: string[];
  createdAt: string;
  updatedAt: string;
  createdBy: string;
}

export interface Milestone {
  id: string;
  name: string;
  description?: string;
  dueDate: string;
  completed?: boolean;
  completedAt?: string;
  linkedWorkItems?: string[];
}

// ============================================================================
// Dashboard State Types
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

export interface DashboardFilters {
  // Source filters
  sources?: WorkItemSource[];
  types?: WorkItemType[];

  // Status filters
  statuses?: WorkItemStatus[];
  mergeStatuses?: PRMergeStatus[];

  // Priority filters
  priorities?: Priority[];

  // People filters
  assignees?: string[]; // Contributor IDs
  reviewers?: string[];

  // Strategic filters
  strategicCategories?: StrategicCategory[];
  strategicAlignments?: StrategicAlignment[];
  roiCategories?: ROICategoryType[];
  initiativeId?: string;

  // Time filters
  createdAfter?: string;
  createdBefore?: string;
  updatedAfter?: string;
  updatedBefore?: string;

  // Search
  searchQuery?: string;
  labels?: string[];

  // Metric filters
  minImpactScore?: number;
  maxCycleTime?: number;
  minComplexity?: number;
}

export interface DashboardState {
  // Core data
  workItems: WorkItem[];
  initiatives: StrategicInitiative[];
  teamMembers: TeamMember[];

  // Derived data
  metrics: DashboardMetrics;
  insights: AIInsight[];

  // UI state
  filters: DashboardFilters;
  viewMode: 'strategic' | 'portfolio' | 'board' | 'list';
  selectedWorkItem?: WorkItem;
  selectedInitiative?: StrategicInitiative;

  // Sync state
  syncState: SyncState;

  // Human overrides (tracked separately from auto-data)
  humanOverrides: {
    priorityOverrides: Map<string, { value: Priority; reason: string }>;
    impactOverrides: Map<string, { value: number; reason: string }>;
    claimOverrides: Map<string, { claimedBy: string; claimedAt: string }>;
  };

  // Modal/dialog states
  isCreatingInitiative: boolean;
  isConfiguringSync: boolean;
  isViewingInsights: boolean;
}

// ============================================================================
// Metrics Types
// ============================================================================

export interface DashboardMetrics {
  // Development KPIs
  development: DevelopmentKPIs;

  // ROI Analysis
  roi: PortfolioROI;

  // Industry Relevance
  relevance: IndustryRelevanceScore;

  // Team Metrics
  team: TeamMetrics;
}

export interface DevelopmentKPIs {
  // Velocity
  velocity: {
    current: number;
    average: number;
    trend: 'increasing' | 'stable' | 'decreasing';
    history: number[]; // Last N sprints
  };

  // Cycle Time
  cycleTime: {
    average: number; // Days
    median: number;
    percentile90: number;
    trend: 'improving' | 'stable' | 'degrading';
    byType: Record<WorkItemType, number>;
  };

  // PR Metrics
  pullRequest: {
    averageTimeToFirstReview: number; // Hours
    averageTimeToMerge: number; // Hours
    mergeRate: number; // Percentage
    reworkRate: number; // Percentage with changes requested
  };

  // Code Quality
  codeQuality: {
    codeChurn: number; // Percentage
    averagePRSize: number; // Lines changed
    reviewDepth: number; // Comments per PR
    defectRate: number; // Reopened issues percentage
  };

  // Throughput
  throughput: {
    itemsCompleted: number;
    itemsStarted: number;
    workInProgress: number;
    throughputTrend: number[]; // Last N periods
  };
}

export interface PortfolioROI {
  // Aggregate metrics
  totalInvestment: {
    storyPoints: number;
    hours: number;
    cost: number;
  };

  totalImpact: {
    revenueImpact: number;
    costReduction: number;
    riskReduction: number;
    strategicValue: number;
    developerProductivity: number;
    total: number;
  };

  // Calculated ROI
  roi: {
    ratio: number;
    percentage: number;
    paybackPeriod: number; // Days
    npv: number; // Net present value
  };

  // By category
  byCategory: Record<ROICategoryType, {
    investment: number;
    impact: number;
    roi: number;
    itemCount: number;
  }>;

  // By initiative
  byInitiative: Array<{
    initiativeId: string;
    name: string;
    roi: number;
    status: 'on_track' | 'at_risk' | 'off_track';
  }>;
}

export interface IndustryRelevanceScore {
  overall: number; // 0-100
  dimensions: {
    technicalInnovation: number;
    marketAlignment: number;
    competitiveParity: number;
    futureProofing: number;
    ecosystemIntegration: number;
  };
  trends: {
    emerging: string[];
    declining: string[];
  };
  benchmarks: {
    industryAverage: number;
    leaderAverage: number;
    ourScore: number;
  };
}

export interface TeamMetrics {
  // Workload distribution
  workloadDistribution: Array<{
    member: string;
    assignedItems: number;
    completedItems: number;
    averageCycleTime: number;
    utilization: number; // Percentage
  }>;

  // Collaboration metrics
  collaboration: {
    averageReviewersPerPR: number;
    crossTeamDependencies: number;
    pairProgrammingSessions?: number;
  };

  // Capacity tracking
  capacity: {
    totalCapacity: number; // Hours available
    allocatedCapacity: number; // Hours assigned
    availableCapacity: number;
    utilizationRate: number;
  };
}

// ============================================================================
// AI Insights Types
// ============================================================================

export type InsightType = 'trend' | 'anomaly' | 'recommendation' | 'risk' | 'opportunity';

export type InsightSeverity = 'low' | 'medium' | 'high' | 'critical';

export interface AIInsight {
  id: string;
  type: InsightType;
  severity: InsightSeverity;
  title: string;
  description: string;
  detailedAnalysis?: string;

  // Supporting data
  dataPoints: DataPoint[];
  relatedWorkItems?: string[]; // WorkItem IDs
  relatedInitiatives?: string[]; // Initiative IDs

  // Actionability
  recommendedAction?: string;
  confidence: number; // 0-1

  // Metadata
  generatedAt: string;
  expiresAt?: string; // When insight becomes stale
  acknowledgedBy?: string[]; // Users who acknowledged this insight
  dismissedBy?: string[]; // Users who dismissed this insight
}

export interface DataPoint {
  label: string;
  value: number;
  unit?: string;
  trend?: 'up' | 'down' | 'stable';
  comparison?: {
    baseline: number;
    change: number;
    changePercent: number;
  };
  context?: string;
}

// ============================================================================
// Action Types (for Reducer)
// ============================================================================

export type DashboardAction =
  // Work item actions
  | { type: 'SET_WORK_ITEMS'; payload: WorkItem[] }
  | { type: 'ADD_WORK_ITEM'; payload: WorkItem }
  | { type: 'UPDATE_WORK_ITEM'; payload: WorkItem }
  | { type: 'DELETE_WORK_ITEM'; payload: string }
  | { type: 'MERGE_WORK_ITEMS'; payload: { sourceId: string; targetId: string } }

  // Initiative actions
  | { type: 'SET_INITIATIVES'; payload: StrategicInitiative[] }
  | { type: 'ADD_INITIATIVE'; payload: StrategicInitiative }
  | { type: 'UPDATE_INITIATIVE'; payload: StrategicInitiative }
  | { type: 'DELETE_INITIATIVE'; payload: string }
  | { type: 'LINK_WORK_TO_INITIATIVE'; payload: { workItemId: string; initiativeId: string } }
  | { type: 'UNLINK_WORK_FROM_INITIATIVE'; payload: { workItemId: string; initiativeId: string } }

  // Team member actions
  | { type: 'SET_TEAM_MEMBERS'; payload: TeamMember[] }
  | { type: 'ADD_TEAM_MEMBER'; payload: TeamMember }
  | { type: 'UPDATE_TEAM_MEMBER'; payload: TeamMember }

  // Selection actions
  | { type: 'SELECT_WORK_ITEM'; payload?: WorkItem }
  | { type: 'SELECT_INITIATIVE'; payload?: StrategicInitiative }

  // Filter actions
  | { type: 'SET_FILTERS'; payload: Partial<DashboardFilters> }
  | { type: 'RESET_FILTERS' }
  | { type: 'SET_VIEW_MODE'; payload: DashboardState['viewMode'] }

  // Sync actions
  | { type: 'SET_SYNC_STATE'; payload: Partial<SyncState> }
  | { type: 'SYNC_STARTED' }
  | { type: 'SYNC_COMPLETED'; payload: { workItems: WorkItem[]; initiatives: StrategicInitiative[] } }
  | { type: 'SYNC_FAILED'; payload: string }

  // Human override actions
  | { type: 'CLAIM_WORK_ITEM'; payload: { workItemId: string; userId: string } }
  | { type: 'SET_PRIORITY_OVERRIDE'; payload: { workItemId: string; priority: Priority; reason: string } }
  | { type: 'SET_IMPACT_OVERRIDE'; payload: { workItemId: string; impactScore: number; reason: string } }
  | { type: 'ADD_STRATEGIC_TAG'; payload: { workItemId: string; tag: StrategicTag } }
  | { type: 'REMOVE_STRATEGIC_TAG'; payload: { workItemId: string; tagId: string } }
  | { type: 'SET_ROI_CATEGORY'; payload: { workItemId: string; roiCategory: ROICategory } }

  // Metrics actions
  | { type: 'UPDATE_METRICS'; payload: DashboardMetrics }

  // Insights actions
  | { type: 'SET_INSIGHTS'; payload: AIInsight[] }
  | { type: 'ACKNOWLEDGE_INSIGHT'; payload: { insightId: string; userId: string } }
  | { type: 'DISMISS_INSIGHT'; payload: { insightId: string; userId: string } }

  // Modal actions
  | { type: 'SET_CREATING_INITIATIVE'; payload: boolean }
  | { type: 'SET_CONFIGURING_SYNC'; payload: boolean }
  | { type: 'SET_VIEWING_INSIGHTS'; payload: boolean }

  // Bulk actions
  | { type: 'BULK_UPDATE_STATUS'; payload: { workItemIds: string[]; status: WorkItemStatus } }
  | { type: 'BULK_ASSIGN'; payload: { workItemIds: string[]; assigneeId: string } }

  // Reset
  | { type: 'RESET_DASHBOARD' };

// ============================================================================
// Context Type
// ============================================================================

export interface TeamDashboardContextType {
  state: DashboardState;
  dispatch: React.Dispatch<DashboardAction>;

  // Work item operations
  addWorkItem: (item: Omit<WorkItem, 'id' | 'createdAt' | 'updatedAt'>) => void;
  updateWorkItem: (item: WorkItem) => void;
  deleteWorkItem: (id: string) => void;
  claimWorkItem: (id: string) => void;
  linkWorkToInitiative: (workItemId: string, initiativeId: string) => void;

  // Initiative operations
  addInitiative: (initiative: Omit<StrategicInitiative, 'id' | 'createdAt' | 'updatedAt'>) => void;
  updateInitiative: (initiative: StrategicInitiative) => void;
  deleteInitiative: (id: string) => void;

  // Filter operations
  setFilters: (filters: Partial<DashboardFilters>) => void;
  resetFilters: () => void;
  setViewMode: (mode: DashboardState['viewMode']) => void;

  // Selection operations
  selectWorkItem: (item?: WorkItem) => void;
  selectInitiative: (initiative?: StrategicInitiative) => void;

  // Override operations
  setPriorityOverride: (id: string, priority: Priority, reason: string) => void;
  setImpactOverride: (id: string, impactScore: number, reason: string) => void;
  addStrategicTag: (id: string, tag: StrategicTag) => void;

  // Insight operations
  acknowledgeInsight: (id: string) => void;
  dismissInsight: (id: string) => void;

  // Modal operations
  showCreateInitiative: () => void;
  hideCreateInitiative: () => void;
  showSyncConfig: () => void;
  hideSyncConfig: () => void;
  toggleInsights: () => void;
}

// ============================================================================
// Constants
// ============================================================================

export const WORK_ITEM_STATUS_COLORS: Record<WorkItemStatus, string> = {
  backlog: '#6b7280',
  in_progress: '#3b82f6',
  in_review: '#f59e0b',
  merged: '#8b5cf6',
  done: '#10b981',
  closed: '#ef4444',
};

export const PRIORITY_COLORS: Record<Priority, string> = {
  low: '#6b7280',
  medium: '#3b82f6',
  high: '#f59e0b',
  critical: '#ef4444',
};

export const PRIORITY_LABELS: Record<Priority, string> = {
  low: 'Low',
  medium: 'Medium',
  high: 'High',
  critical: 'Critical',
};

export const STRATEGIC_CATEGORY_COLORS: Record<StrategicCategory, string> = {
  feature: '#3b82f6',
  'tech-debt': '#f59e0b',
  performance: '#8b5cf6',
  security: '#ef4444',
  compliance: '#ec4899',
  infrastructure: '#6b7280',
};

export const INITIATIVE_STATUS_COLORS: Record<StrategicInitiative['status'], string> = {
  planning: '#6b7280',
  active: '#3b82f6',
  on_hold: '#f59e0b',
  completed: '#10b981',
  cancelled: '#ef4444',
};

export const INITIATIVE_HEALTH_COLORS: Record<StrategicInitiative['healthStatus'], string> = {
  green: '#10b981',
  yellow: '#f59e0b',
  red: '#ef4444',
};

// ============================================================================
// Legacy Type Aliases (for migration)
// ============================================================================

/**
 * @deprecated Use WorkItem instead
 */
export type Issue = WorkItem;

/**
 * @deprecated Use WorkItemStatus instead
 */
export type IssueStatus = WorkItemStatus;

/**
 * @deprecated Use Priority instead
 */
export type IssuePriority = Priority;

/**
 * @deprecated Use TeamMember instead
 */
export type { TeamMember as Member };

/**
 * @deprecated Use DashboardState instead
 */
export type { DashboardState as MemberState };

/**
 * @deprecated Use DashboardAction instead
 */
export type { DashboardAction as MemberAction };
