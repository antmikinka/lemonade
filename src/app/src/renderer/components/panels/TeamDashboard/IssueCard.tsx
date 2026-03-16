/**
 * IssueCard Component
 * Displays an individual issue card in the Kanban board
 */

import React from 'react';
import { Issue, PRIORITY_COLORS, PRIORITY_LABELS } from '../../../types/teamDashboard';
import { useTeamDashboard } from '../../../contexts/TeamDashboardContext';

interface IssueCardProps {
  issue: Issue;
  onClick?: (issue: Issue) => void;
}

const IssueCard: React.FC<IssueCardProps> = ({ issue, onClick }) => {
  const { selectIssue, state } = useTeamDashboard();

  const handleClick = () => {
    selectIssue(issue);
    if (onClick) {
      onClick(issue);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      handleClick();
    }
  };

  const priorityColor = PRIORITY_COLORS[issue.priority];
  const isSelected = state.selectedIssue?.id === issue.id;

  return (
    <div
      className={`issue-card ${isSelected ? 'selected' : ''}`}
      onClick={handleClick}
      onKeyDown={handleKeyDown}
      role="button"
      tabIndex={0}
      aria-pressed={isSelected}
      aria-label={`Issue: ${issue.title}, Priority: ${PRIORITY_LABELS[issue.priority]}`}
    >
      <div className="issue-card-header">
        <div
          className="issue-card-priority"
          style={{ backgroundColor: priorityColor }}
          aria-label={`Priority: ${PRIORITY_LABELS[issue.priority]}`}
        />
        <span className="issue-card-id">{issue.id.split('_').pop()}</span>
      </div>

      <h4 className="issue-card-title">{issue.title}</h4>

      {issue.description && (
        <p className="issue-card-description">
          {issue.description.length > 80
            ? `${issue.description.substring(0, 80)}...`
            : issue.description}
        </p>
      )}

      {issue.labels && issue.labels.length > 0 && (
        <div className="issue-card-labels">
          {issue.labels.slice(0, 3).map((label, index) => (
            <span key={index} className="issue-card-label">
              {label}
            </span>
          ))}
          {issue.labels.length > 3 && (
            <span className="issue-card-label-more">+{issue.labels.length - 3}</span>
          )}
        </div>
      )}

      <div className="issue-card-footer">
        {issue.assignee && (
          <div className="issue-card-assignee" title={issue.assignee.name}>
            {issue.assignee.avatar ? (
              <img
                src={issue.assignee.avatar}
                alt={issue.assignee.name}
                className="issue-card-avatar"
              />
            ) : (
              <div className="issue-card-avatar-placeholder">
                {issue.assignee.name.charAt(0).toUpperCase()}
              </div>
            )}
            <span className="issue-card-assignee-name">{issue.assignee.name}</span>
          </div>
        )}

        {issue.storyPoints && (
          <span className="issue-card-points" aria-label={`${issue.storyPoints} story points`}>
            {issue.storyPoints} pts
          </span>
        )}

        {issue.dueDate && (
          <span
            className={`issue-card-due-date ${isOverdue(issue.dueDate) ? 'overdue' : ''}`}
            title={`Due: ${formatDate(issue.dueDate)}`}
          >
            {formatRelativeDate(issue.dueDate)}
          </span>
        )}
      </div>
    </div>
  );
};

/**
 * Check if a date is overdue
 */
const isOverdue = (dueDate: string): boolean => {
  return new Date(dueDate) < new Date();
};

/**
 * Format date for display
 */
const formatDate = (dateString: string): string => {
  const date = new Date(dateString);
  return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
};

/**
 * Format date as relative (e.g., "in 2 days", "yesterday")
 */
const formatRelativeDate = (dateString: string): string => {
  const due = new Date(dateString);
  const now = new Date();
  const diffTime = due.getTime() - now.getTime();
  const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24));

  if (diffDays < 0) {
    return `${Math.abs(diffDays)}d overdue`;
  } else if (diffDays === 0) {
    return 'Today';
  } else if (diffDays === 1) {
    return 'Tomorrow';
  } else {
    return `in ${diffDays}d`;
  }
};

export default IssueCard;
