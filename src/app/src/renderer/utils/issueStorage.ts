/**
 * Issue Storage Utility
 * Handles localStorage persistence for Team Dashboard data
 */

import { Issue, TeamMember, DashboardState, KPIMetrics } from '../types/teamDashboard';

const STORAGE_KEYS = {
  ISSUES: 'team_dashboard_issues',
  TEAM_MEMBERS: 'team_dashboard_members',
  SETTINGS: 'team_dashboard_settings',
};

/**
 * Generate a unique ID for issues
 */
export const generateId = (): string => {
  return `issue_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
};

/**
 * Issue persistence functions
 */
export const issueStorage = {
  // Issues
  saveIssues: (issues: Issue[]): void => {
    try {
      localStorage.setItem(STORAGE_KEYS.ISSUES, JSON.stringify(issues));
    } catch (error) {
      console.error('Failed to save issues to localStorage:', error);
    }
  },

  loadIssues: (): Issue[] => {
    try {
      const stored = localStorage.getItem(STORAGE_KEYS.ISSUES);
      if (stored) {
        return JSON.parse(stored) as Issue[];
      }
    } catch (error) {
      console.error('Failed to load issues from localStorage:', error);
    }
    return [];
  },

  // Team Members
  saveTeamMembers: (members: TeamMember[]): void => {
    try {
      localStorage.setItem(STORAGE_KEYS.TEAM_MEMBERS, JSON.stringify(members));
    } catch (error) {
      console.error('Failed to save team members to localStorage:', error);
    }
  },

  loadTeamMembers: (): TeamMember[] => {
    try {
      const stored = localStorage.getItem(STORAGE_KEYS.TEAM_MEMBERS);
      if (stored) {
        return JSON.parse(stored) as TeamMember[];
      }
    } catch (error) {
      console.error('Failed to load team members from localStorage:', error);
    }
    return [];
  },

  // Settings (view mode, filters)
  saveSettings: (settings: { viewMode: 'board' | 'list' }): void => {
    try {
      localStorage.setItem(STORAGE_KEYS.SETTINGS, JSON.stringify(settings));
    } catch (error) {
      console.error('Failed to save settings to localStorage:', error);
    }
  },

  loadSettings: (): { viewMode: 'board' | 'list' } | null => {
    try {
      const stored = localStorage.getItem(STORAGE_KEYS.SETTINGS);
      if (stored) {
        return JSON.parse(stored) as { viewMode: 'board' | 'list' };
      }
    } catch (error) {
      console.error('Failed to load settings from localStorage:', error);
    }
    return null;
  },

  // Clear all storage
  clearAll: (): void => {
    try {
      localStorage.removeItem(STORAGE_KEYS.ISSUES);
      localStorage.removeItem(STORAGE_KEYS.TEAM_MEMBERS);
      localStorage.removeItem(STORAGE_KEYS.SETTINGS);
    } catch (error) {
      console.error('Failed to clear localStorage:', error);
    }
  },
};

/**
 * Calculate KPI metrics from issues
 */
export const calculateKPIMetrics = (issues: Issue[], quarter?: string): KPIMetrics => {
  const currentQuarter = quarter || getCurrentQuarter();

  // Filter issues for the current quarter
  const quarterIssues = issues.filter((issue) => {
    if (!issue.resolvedAt) return false;
    const issueDate = new Date(issue.resolvedAt);
    const issueQuarter = getQuarterFromDate(issueDate);
    return issueQuarter === currentQuarter;
  });

  const issuesCompleted = quarterIssues.length;

  // Calculate average resolution time
  const resolutionTimes = quarterIssues
    .filter((issue) => issue.resolvedAt)
    .map((issue) => {
      const created = new Date(issue.createdAt);
      const resolved = new Date(issue.resolvedAt!);
      return (resolved.getTime() - created.getTime()) / (1000 * 60 * 60 * 24); // days
    });

  const avgResolutionTime =
    resolutionTimes.length > 0
      ? resolutionTimes.reduce((sum, time) => sum + time, 0) / resolutionTimes.length
      : 0;

  // Calculate velocity (issues completed in quarter)
  const velocity = issuesCompleted;

  // Calculate completion rate
  const totalResolved = issues.filter((i) => i.status === 'done').length;
  const completionRate =
    issues.length > 0 ? (totalResolved / issues.length) * 100 : 0;

  return {
    quarter: currentQuarter,
    issuesCompleted,
    avgResolutionTime: Math.round(avgResolutionTime * 10) / 10,
    velocity,
    completionRate: Math.round(completionRate * 10) / 10,
    totalIssues: issues.length,
    issuesInBacklog: issues.filter((i) => i.status === 'backlog').length,
    issuesInProgress: issues.filter((i) => i.status === 'in_progress').length,
    issuesInReview: issues.filter((i) => i.status === 'review').length,
  };
};

/**
 * Get current quarter string (e.g., "Q1 2026")
 */
export const getCurrentQuarter = (): string => {
  const date = new Date();
  const quarter = Math.floor(date.getMonth() / 3) + 1;
  return `Q${quarter} ${date.getFullYear()}`;
};

/**
 * Get quarter string from date
 */
export const getQuarterFromDate = (date: Date): string => {
  const quarter = Math.floor(date.getMonth() / 3) + 1;
  return `Q${quarter} ${date.getFullYear()}`;
};

/**
 * Get the last 4 quarters for KPI display
 */
export const getLast4Quarters = (): string[] => {
  const quarters: string[] = [];
  const now = new Date();

  for (let i = 0; i < 4; i++) {
    const date = new Date(now.getFullYear(), now.getMonth() - i * 3, 1);
    quarters.unshift(getQuarterFromDate(date));
  }

  return quarters;
};
