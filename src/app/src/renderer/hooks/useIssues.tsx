/**
 * useIssues Hook
 * Provides convenient work item management operations
 */

import { useCallback } from 'react';
import { useTeamDashboard } from '../contexts/TeamDashboardContext';
import type { WorkItem, WorkItemStatus, Priority, TeamMember, Label } from '../types/workItem';

/**
 * Custom hook for work item management operations
 */
export const useIssues = () => {
  const { state, dispatch, addWorkItem, updateWorkItem, deleteWorkItem } = useTeamDashboard();

  /**
   * Create a new work item with full type safety
   */
  const createWorkItem = useCallback(
    (
      title: string,
      description: string,
      priority: Priority = 'medium',
      status: WorkItemStatus = 'backlog',
      assignees?: TeamMember[],
      labels?: Label[],
    ): WorkItem => {
      const now = new Date().toISOString();
      const author: { id: string; name: string } = { id: 'current-user', name: 'Current User' };
      const newWorkItem: WorkItem = {
        id: `manual_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
        source: 'manual',
        title,
        description,
        type: 'issue',
        status,
        priority,
        author,
        assignees: assignees || [],
        labels: labels || [],
        linkedItems: [],
        createdAt: now,
        updatedAt: now,
        metrics: {
          age: 0,
          estimatedPoints: 0,
        },
      };

      addWorkItem({
        source: 'manual',
        title,
        description,
        type: 'issue',
        status,
        priority,
        author,
        assignees: assignees || [],
        labels: labels || [],
        linkedItems: [],
        metrics: {
          age: 0,
          estimatedPoints: 0,
        },
      });

      return newWorkItem;
    },
    [addWorkItem],
  );

  /**
   * Update work item status and move to appropriate column
   */
  const transitionWorkItem = useCallback(
    (workItemId: string, newStatus: WorkItemStatus): void => {
      const workItem = state.workItems.find((i) => i.id === workItemId);
      if (workItem) {
        updateWorkItem({ ...workItem, status: newStatus, updatedAt: new Date().toISOString() });
      }
    },
    [state.workItems, updateWorkItem],
  );

  /**
   * Assign team members to a work item
   */
  const assignWorkItem = useCallback(
    (workItemId: string, members: TeamMember[] | undefined): void => {
      const workItem = state.workItems.find((i) => i.id === workItemId);
      if (workItem) {
        updateWorkItem({ ...workItem, assignees: members || [], updatedAt: new Date().toISOString() });
      }
    },
    [state.workItems, updateWorkItem],
  );

  /**
   * Add a label to a work item
   */
  const addLabelToWorkItem = useCallback(
    (workItemId: string, label: Label): void => {
      const workItem = state.workItems.find((i) => i.id === workItemId);
      if (workItem && !workItem.labels.some((l) => l.id === label.id)) {
        updateWorkItem({
          ...workItem,
          labels: [...workItem.labels, label],
          updatedAt: new Date().toISOString(),
        });
      }
    },
    [state.workItems, updateWorkItem],
  );

  /**
   * Remove a label from a work item
   */
  const removeLabelFromWorkItem = useCallback(
    (workItemId: string, labelId: string): void => {
      const workItem = state.workItems.find((i) => i.id === workItemId);
      if (workItem) {
        updateWorkItem({
          ...workItem,
          labels: workItem.labels.filter((l) => l.id !== labelId),
          updatedAt: new Date().toISOString(),
        });
      }
    },
    [state.workItems, updateWorkItem],
  );

  /**
   * Mark a work item as resolved
   */
  const resolveWorkItem = useCallback(
    (workItemId: string): void => {
      const workItem = state.workItems.find((i) => i.id === workItemId);
      if (workItem) {
        updateWorkItem({
          ...workItem,
          status: 'done',
          resolvedAt: new Date().toISOString(),
          updatedAt: new Date().toISOString(),
        });
      }
    },
    [state.workItems, updateWorkItem],
  );

  /**
   * Reopen a resolved work item
   */
  const reopenWorkItem = useCallback(
    (workItemId: string): void => {
      const workItem = state.workItems.find((i) => i.id === workItemId);
      if (workItem) {
        updateWorkItem({
          ...workItem,
          status: 'in_progress',
          resolvedAt: undefined,
          updatedAt: new Date().toISOString(),
        });
      }
    },
    [state.workItems, updateWorkItem],
  );

  /**
   * Get work items by status
   */
  const getWorkItemsByStatus = useCallback(
    (status: WorkItemStatus): WorkItem[] => {
      return state.workItems.filter((item: WorkItem) => item.status === status);
    },
    [state.workItems],
  );

  /**
   * Get work items by assignee
   */
  const getWorkItemsByAssignee = useCallback(
    (assigneeId: string): WorkItem[] => {
      return state.workItems.filter((item: WorkItem) => item.assignees?.some((a) => a.id === assigneeId));
    },
    [state.workItems],
  );

  /**
   * Search work items by query
   */
  const searchWorkItems = useCallback(
    (query: string): WorkItem[] => {
      const lowerQuery = query.toLowerCase();
      return state.workItems.filter(
        (item: WorkItem) =>
          item.title.toLowerCase().includes(lowerQuery) ||
          item.description.toLowerCase().includes(lowerQuery) ||
          item.labels.some((label: Label) => label.name.toLowerCase().includes(lowerQuery)),
      );
    },
    [state.workItems],
  );

  return {
    // State
    workItems: state.workItems,
    teamMembers: state.teamMembers,
    selectedWorkItem: state.selectedWorkItem,
    filters: state.filters,

    // CRUD operations
    createWorkItem,
    updateWorkItem,
    deleteWorkItem,
    transitionWorkItem,

    // Assignment operations
    assignWorkItem,
    addLabelToWorkItem,
    removeLabelFromWorkItem,

    // Status operations
    resolveWorkItem,
    reopenWorkItem,

    // Query operations
    getWorkItemsByStatus,
    getWorkItemsByAssignee,
    searchWorkItems,
  };
};

export default useIssues;
