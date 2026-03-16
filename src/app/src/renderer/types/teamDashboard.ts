/**
 * Team Dashboard TypeScript Interfaces
 * Defines the core data structures for the Team Tracking Dashboard
 */

export interface TeamMember {
  id: string;
  name: string;
  avatar?: string;
  role?: string;
}

export interface Issue {
  id: string;
  title: string;
  description: string;
  status: IssueStatus;
  priority: IssuePriority;
  assignee?: TeamMember;
  labels: string[];
  createdAt: string;
  updatedAt: string;
  resolvedAt?: string;
  dueDate?: string;
  storyPoints?: number;
}

export type IssueStatus = 'backlog' | 'in_progress' | 'review' | 'done';

export type IssuePriority = 'low' | 'medium' | 'high' | 'critical';

export interface KPIMetrics {
  quarter: string;
  issuesCompleted: number;
  avgResolutionTime: number; // days
  velocity: number; // issues per sprint
  completionRate: number; // percentage
  totalIssues: number;
  issuesInBacklog: number;
  issuesInProgress: number;
  issuesInReview: number;
}

export interface DashboardState {
  issues: Issue[];
  teamMembers: TeamMember[];
  selectedIssue?: Issue;
  filters: {
    assignee?: string;
    priority?: IssuePriority;
    labels?: string[];
    searchQuery?: string;
  };
  viewMode: 'board' | 'list';
  isCreatingIssue: boolean;
}

export type DashboardAction =
  | { type: 'SET_ISSUES'; payload: Issue[] }
  | { type: 'ADD_ISSUE'; payload: Issue }
  | { type: 'UPDATE_ISSUE'; payload: Issue }
  | { type: 'DELETE_ISSUE'; payload: string }
  | { type: 'MOVE_ISSUE'; payload: { issueId: string; status: IssueStatus } }
  | { type: 'SET_TEAM_MEMBERS'; payload: TeamMember[] }
  | { type: 'SELECT_ISSUE'; payload?: Issue }
  | { type: 'SET_FILTERS'; payload: Partial<DashboardState['filters']> }
  | { type: 'SET_VIEW_MODE'; payload: 'board' | 'list' }
  | { type: 'SET_CREATING_ISSUE'; payload: boolean }
  | { type: 'LOAD_FROM_STORAGE'; payload: Partial<DashboardState> };

export const STATUS_COLUMNS: { id: IssueStatus; label: string; color: string }[] = [
  { id: 'backlog', label: 'Backlog', color: '#6b7280' },
  { id: 'in_progress', label: 'In Progress', color: '#3b82f6' },
  { id: 'review', label: 'Review', color: '#f59e0b' },
  { id: 'done', label: 'Done', color: '#10b981' },
];

export const PRIORITY_COLORS: Record<IssuePriority, string> = {
  low: '#6b7280',
  medium: '#3b82f6',
  high: '#f59e0b',
  critical: '#ef4444',
};

export const PRIORITY_LABELS: Record<IssuePriority, string> = {
  low: 'Low',
  medium: 'Medium',
  high: 'High',
  critical: 'Critical',
};
