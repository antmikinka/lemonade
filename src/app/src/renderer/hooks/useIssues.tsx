/**
 * useIssues Hook
 * Provides convenient issue management operations
 */

import { useCallback } from 'react';
import { useTeamDashboard } from '../contexts/TeamDashboardContext';
import { Issue, IssueStatus, IssuePriority, TeamMember } from '../types/teamDashboard';

/**
 * Custom hook for issue management operations
 */
export const useIssues = () => {
  const { state, dispatch, addIssue, updateIssue, deleteIssue, moveIssue } = useTeamDashboard();

  /**
   * Create a new issue with full type safety
   */
  const createIssue = useCallback(
    (
      title: string,
      description: string,
      priority: IssuePriority = 'medium',
      status: IssueStatus = 'backlog',
      assignee?: TeamMember,
      labels?: string[],
      storyPoints?: number,
      dueDate?: string,
    ): Issue => {
      const now = new Date().toISOString();
      const newIssue: Issue = {
        id: `issue_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
        title,
        description,
        priority,
        status,
        assignee,
        labels: labels || [],
        storyPoints,
        dueDate,
        createdAt: now,
        updatedAt: now,
      };

      addIssue({
        title,
        description,
        priority,
        status,
        assignee,
        labels: labels || [],
        storyPoints,
        dueDate,
      });

      return newIssue;
    },
    [addIssue],
  );

  /**
   * Update issue status and move to appropriate column
   */
  const transitionIssue = useCallback(
    (issueId: string, newStatus: IssueStatus): void => {
      moveIssue(issueId, newStatus);
    },
    [moveIssue],
  );

  /**
   * Assign a team member to an issue
   */
  const assignIssue = useCallback(
    (issueId: string, member: TeamMember | undefined): void => {
      const issue = state.issues.find((i) => i.id === issueId);
      if (issue) {
        updateIssue({ ...issue, assignee: member, updatedAt: new Date().toISOString() });
      }
    },
    [state.issues, updateIssue],
  );

  /**
   * Add a label to an issue
   */
  const addLabelToIssue = useCallback(
    (issueId: string, label: string): void => {
      const issue = state.issues.find((i) => i.id === issueId);
      if (issue && !issue.labels.includes(label)) {
        updateIssue({
          ...issue,
          labels: [...issue.labels, label],
          updatedAt: new Date().toISOString(),
        });
      }
    },
    [state.issues, updateIssue],
  );

  /**
   * Remove a label from an issue
   */
  const removeLabelFromIssue = useCallback(
    (issueId: string, label: string): void => {
      const issue = state.issues.find((i) => i.id === issueId);
      if (issue) {
        updateIssue({
          ...issue,
          labels: issue.labels.filter((l) => l !== label),
          updatedAt: new Date().toISOString(),
        });
      }
    },
    [state.issues, updateIssue],
  );

  /**
   * Mark an issue as resolved
   */
  const resolveIssue = useCallback(
    (issueId: string): void => {
      const issue = state.issues.find((i) => i.id === issueId);
      if (issue) {
        updateIssue({
          ...issue,
          status: 'done',
          resolvedAt: new Date().toISOString(),
          updatedAt: new Date().toISOString(),
        });
      }
    },
    [state.issues, updateIssue],
  );

  /**
   * Reopen a resolved issue
   */
  const reopenIssue = useCallback(
    (issueId: string): void => {
      const issue = state.issues.find((i) => i.id === issueId);
      if (issue) {
        updateIssue({
          ...issue,
          status: 'in_progress',
          resolvedAt: undefined,
          updatedAt: new Date().toISOString(),
        });
      }
    },
    [state.issues, updateIssue],
  );

  /**
   * Get issues by status
   */
  const getIssuesByStatus = useCallback(
    (status: IssueStatus): Issue[] => {
      return state.issues.filter((issue) => issue.status === status);
    },
    [state.issues],
  );

  /**
   * Get issues by assignee
   */
  const getIssuesByAssignee = useCallback(
    (assigneeId: string): Issue[] => {
      return state.issues.filter((issue) => issue.assignee?.id === assigneeId);
    },
    [state.issues],
  );

  /**
   * Search issues by query
   */
  const searchIssues = useCallback(
    (query: string): Issue[] => {
      const lowerQuery = query.toLowerCase();
      return state.issues.filter(
        (issue) =>
          issue.title.toLowerCase().includes(lowerQuery) ||
          issue.description.toLowerCase().includes(lowerQuery) ||
          issue.labels.some((label) => label.toLowerCase().includes(lowerQuery)),
      );
    },
    [state.issues],
  );

  return {
    // State
    issues: state.issues,
    teamMembers: state.teamMembers,
    selectedIssue: state.selectedIssue,
    filters: state.filters,

    // CRUD operations
    createIssue,
    updateIssue,
    deleteIssue,
    transitionIssue,

    // Assignment operations
    assignIssue,
    addLabelToIssue,
    removeLabelFromIssue,

    // Status operations
    resolveIssue,
    reopenIssue,

    // Query operations
    getIssuesByStatus,
    getIssuesByAssignee,
    searchIssues,
  };
};

export default useIssues;
