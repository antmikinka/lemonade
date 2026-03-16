/**
 * IssueBoard Component
 * Kanban board displaying issues in columns by status
 *
 * Uses react-window for virtualized scrolling of issue lists
 */

import React, { useCallback, useEffect, useRef, useMemo } from 'react';
import { FixedSizeList } from 'react-window';
import { useTeamDashboard } from '../../../contexts/TeamDashboardContext';
import { Issue, IssueStatus, STATUS_COLUMNS } from '../../../types/teamDashboard';
import IssueCard from './IssueCard';

interface IssueBoardProps {
  issues: Issue[];
}

const ISSUE_CARD_HEIGHT = 120; // Height of each issue card in pixels
const COLUMN_HEADER_HEIGHT = 48; // Height of column header

interface IssueBoardProps {
  issues: Issue[];
}

const IssueBoard: React.FC<IssueBoardProps> = ({ issues }) => {
  const { moveIssue, selectIssue, state } = useTeamDashboard();
  const draggedIssueRef = useRef<string | null>(null);
  const boardRef = useRef<HTMLDivElement>(null);

  // Filter issues based on current filters
  const filteredIssues = useMemoIssues(issues, state.filters);

  // Group issues by status
  const issuesByStatus = useMemo(() => {
    const grouped: Record<IssueStatus, Issue[]> = {
      backlog: [],
      in_progress: [],
      review: [],
      done: [],
    };

    filteredIssues.forEach((issue) => {
      grouped[issue.status].push(issue);
    });

    return grouped;
  }, [filteredIssues]);

  // Handle drag start
  const handleDragStart = useCallback((e: React.DragEvent, issueId: string) => {
    draggedIssueRef.current = issueId;
    e.dataTransfer.effectAllowed = 'move';
    e.dataTransfer.setData('text/plain', issueId);
  }, []);

  // Handle drag over
  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = 'move';
  }, []);

  // Handle drop
  const handleDrop = useCallback(
    (e: React.DragEvent, status: IssueStatus) => {
      e.preventDefault();
      const issueId = draggedIssueRef.current;
      if (issueId) {
        moveIssue(issueId, status);
        draggedIssueRef.current = null;
      }
    },
    [moveIssue],
  );

  // Handle drag end cleanup
  const handleDragEnd = useCallback(() => {
    draggedIssueRef.current = null;
  }, []);

  // Calculate board height based on viewport
  const getBoardHeight = () => {
    if (boardRef.current) {
      return boardRef.current.clientHeight - COLUMN_HEADER_HEIGHT;
    }
    return 600; // Default height
  };

  return (
    <div ref={boardRef} className="issue-board" role="region" aria-label="Issue Kanban Board">
      {STATUS_COLUMNS.map((column) => {
        const columnIssues = issuesByStatus[column.id];
        return (
          <div
            key={column.id}
            className="issue-board-column"
            onDragOver={handleDragOver}
            onDrop={(e) => handleDrop(e, column.id)}
          >
            <div className="issue-board-column-header">
              <div className="issue-board-column-title">
                <span
                  className="issue-board-column-color"
                  style={{ backgroundColor: column.color }}
                />
                <h3>{column.label}</h3>
                <span className="issue-board-column-count">{columnIssues.length}</span>
              </div>
            </div>

            <div
              className="issue-board-column-content"
              role="list"
              aria-label={`${column.label} issues`}
            >
              {columnIssues.length === 0 ? (
                <div className="issue-board-empty-state">
                  <span>No issues</span>
                </div>
              ) : (
                <FixedSizeList
                  height={Math.min(getBoardHeight(), columnIssues.length * ISSUE_CARD_HEIGHT + 8)}
                  itemCount={columnIssues.length}
                  itemSize={ISSUE_CARD_HEIGHT}
                  itemData={columnIssues}
                  overscanCount={5}
                  width="100%"
                >
                  {({ index, style, data }: { index: number; style: React.CSSProperties; data: Issue[] }) => {
                    const issue = data[index];
                    return (
                      <div
                        style={style}
                        draggable
                        onDragStart={(e) => handleDragStart(e, issue.id)}
                        onDragEnd={handleDragEnd}
                      >
                        <IssueCard issue={issue} />
                      </div>
                    );
                  }}
                </FixedSizeList>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
};

/**
 * Memoized filter function for issues
 */
const useMemoIssues = (issues: Issue[], filters: {
  assignee?: string;
  priority?: 'low' | 'medium' | 'high' | 'critical';
  labels?: string[];
  searchQuery?: string;
}): Issue[] => {
  return React.useMemo(() => {
    return issues.filter((issue) => {
      // Filter by assignee
      if (filters.assignee && issue.assignee?.id !== filters.assignee) {
        return false;
      }

      // Filter by priority
      if (filters.priority && issue.priority !== filters.priority) {
        return false;
      }

      // Filter by labels
      if (filters.labels && filters.labels.length > 0) {
        const hasMatchingLabel = filters.labels.some((label) =>
          issue.labels.includes(label),
        );
        if (!hasMatchingLabel) {
          return false;
        }
      }

      // Filter by search query
      if (filters.searchQuery) {
        const query = filters.searchQuery.toLowerCase();
        const matchesTitle = issue.title.toLowerCase().includes(query);
        const matchesDescription = issue.description.toLowerCase().includes(query);
        const matchesLabels = issue.labels.some((label) =>
          label.toLowerCase().includes(query),
        );
        if (!matchesTitle && !matchesDescription && !matchesLabels) {
          return false;
        }
      }

      return true;
    });
  }, [issues, filters]);
};

export default IssueBoard;
