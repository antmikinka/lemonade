/**
 * Types Module Exports
 * Centralized exports for all dashboard types
 */

// Work Item types (main types)
export type {
  WorkItem,
  WorkItemSource,
  WorkItemType,
  WorkItemStatus,
  WorkItemMetrics,
  WorkItemRelationship,
  Priority,
  PriorityChangeReason,
  PriorityChange,
  StrategicTag,
  StrategicCategory,
  StrategicAlignment,
  ROICategory,
  ROICategoryType,
  LinkedEntity,
  GitHubMetadata,
  GitHubMilestone,
  TimelineEvent,
  LocalMetadata,
} from './workItem';

// Strategic Initiative types
export type {
  StrategicInitiative,
  Milestone,
} from './workItem';

// Dashboard State types
export type {
  DashboardState,
  DashboardAction,
  DashboardFilters,
  DashboardMetrics,
  SyncState,
} from './workItem';

// Metrics types
export type {
  DevelopmentKPIs,
  PortfolioROI,
  IndustryRelevanceScore,
  TeamMetrics,
} from './workItem';

// AI Insights types
export type {
  AIInsight,
  DataPoint,
  InsightType,
  InsightSeverity,
} from './workItem';

// Team types
export type {
  Contributor,
  TeamMember,
  Label,
} from './workItem';

// Context types
export type {
  TeamDashboardContextType,
} from '../contexts/TeamDashboardContext';

// GitHub types
export type {
  GitHubIssue,
  GitHubPullRequest,
  GitHubCommit,
  GitHubReview,
  GitHubTimelineEvent,
  GitHubConfig,
  GitHubAuthState,
  IFetchOptions,
  IFetchResult,
  NormalizedIssue,
  NormalizedPullRequest,
  NormalizedCommit,
  LinkMap,
  IGitHubService,
  GitHubServiceEvent,
  GitHubServiceListener,
} from './github';
