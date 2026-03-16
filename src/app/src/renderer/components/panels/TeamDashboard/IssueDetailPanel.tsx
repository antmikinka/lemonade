/**
 * IssueDetailPanel Component
 * Displays detailed view of a selected issue with edit capabilities
 */

import React, { useState, useEffect, useRef } from 'react';
import { useTeamDashboard } from '../../../contexts/TeamDashboardContext';
import { Issue, IssueStatus, IssuePriority, TeamMember, PRIORITY_LABELS } from '../../../types/teamDashboard';
import { PRIORITY_COLORS } from '../../../types/teamDashboard';
import { CloseIcon } from '../../../components/Icons';

interface IssueDetailPanelProps {
  issue: Issue;
  onClose: () => void;
}

const IssueDetailPanel: React.FC<IssueDetailPanelProps> = ({ issue, onClose }) => {
  const { updateIssue, deleteIssue, state, selectIssue } = useTeamDashboard();
  const [isEditing, setIsEditing] = useState(false);
  const [editedIssue, setEditedIssue] = useState<Issue>(issue);
  const [newLabel, setNewLabel] = useState('');
  const panelRef = useRef<HTMLDivElement>(null);

  // Sync edited issue with prop changes
  useEffect(() => {
    setEditedIssue(issue);
  }, [issue]);

  // Handle escape key to close
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        if (isEditing) {
          setIsEditing(false);
          setEditedIssue(issue);
        } else {
          onClose();
        }
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [isEditing, issue, onClose]);

  // Handle clicks outside panel
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (panelRef.current && !panelRef.current.contains(e.target as Node)) {
        onClose();
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [onClose]);

  const handleSave = () => {
    updateIssue(editedIssue);
    setIsEditing(false);
  };

  const handleCancel = () => {
    setEditedIssue(issue);
    setIsEditing(false);
  };

  const handleDelete = () => {
    if (window.confirm(`Are you sure you want to delete issue "${issue.title}"?`)) {
      deleteIssue(issue.id);
      onClose();
    }
  };

  const handleStatusChange = (newStatus: IssueStatus) => {
    setEditedIssue({ ...editedIssue, status: newStatus, updatedAt: new Date().toISOString() });
  };

  const handlePriorityChange = (newPriority: IssuePriority) => {
    setEditedIssue({ ...editedIssue, priority: newPriority, updatedAt: new Date().toISOString() });
  };

  const handleAssigneeChange = (memberId: string) => {
    const member = state.teamMembers.find((m) => m.id === memberId);
    setEditedIssue({
      ...editedIssue,
      assignee: member || undefined,
      updatedAt: new Date().toISOString(),
    });
  };

  const handleAddLabel = () => {
    if (newLabel.trim() && !editedIssue.labels.includes(newLabel.trim())) {
      setEditedIssue({
        ...editedIssue,
        labels: [...editedIssue.labels, newLabel.trim()],
        updatedAt: new Date().toISOString(),
      });
      setNewLabel('');
    }
  };

  const handleRemoveLabel = (labelToRemove: string) => {
    setEditedIssue({
      ...editedIssue,
      labels: editedIssue.labels.filter((l) => l !== labelToRemove),
      updatedAt: new Date().toISOString(),
    });
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleAddLabel();
    }
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  return (
    <div className="issue-detail-overlay" role="dialog" aria-modal="true" aria-labelledby="issue-detail-title">
      <div className="issue-detail-panel" ref={panelRef}>
        <div className="issue-detail-header">
          <h3 id="issue-detail-title">Issue Details</h3>
          <button
            className="issue-detail-close"
            onClick={onClose}
            aria-label="Close issue details"
          >
            <CloseIcon size={18} />
          </button>
        </div>

        <div className="issue-detail-content">
          {isEditing ? (
            <div className="issue-detail-edit">
              <div className="issue-detail-field">
                <label htmlFor="edit-title">Title</label>
                <input
                  id="edit-title"
                  type="text"
                  value={editedIssue.title}
                  onChange={(e) =>
                    setEditedIssue({ ...editedIssue, title: e.target.value })
                  }
                  className="issue-detail-input"
                />
              </div>

              <div className="issue-detail-field">
                <label htmlFor="edit-description">Description</label>
                <textarea
                  id="edit-description"
                  value={editedIssue.description}
                  onChange={(e) =>
                    setEditedIssue({ ...editedIssue, description: e.target.value })
                  }
                  className="issue-detail-textarea"
                  rows={4}
                />
              </div>

              <div className="issue-detail-field">
                <label htmlFor="edit-status">Status</label>
                <select
                  id="edit-status"
                  value={editedIssue.status}
                  onChange={(e) => handleStatusChange(e.target.value as IssueStatus)}
                  className="issue-detail-select"
                >
                  <option value="backlog">Backlog</option>
                  <option value="in_progress">In Progress</option>
                  <option value="review">Review</option>
                  <option value="done">Done</option>
                </select>
              </div>

              <div className="issue-detail-field">
                <label htmlFor="edit-priority">Priority</label>
                <select
                  id="edit-priority"
                  value={editedIssue.priority}
                  onChange={(e) => handlePriorityChange(e.target.value as IssuePriority)}
                  className="issue-detail-select"
                >
                  <option value="low">Low</option>
                  <option value="medium">Medium</option>
                  <option value="high">High</option>
                  <option value="critical">Critical</option>
                </select>
              </div>

              <div className="issue-detail-field">
                <label htmlFor="edit-assignee">Assignee</label>
                <select
                  id="edit-assignee"
                  value={editedIssue.assignee?.id || ''}
                  onChange={(e) => handleAssigneeChange(e.target.value)}
                  className="issue-detail-select"
                >
                  <option value="">Unassigned</option>
                  {state.teamMembers.map((member) => (
                    <option key={member.id} value={member.id}>
                      {member.name}
                    </option>
                  ))}
                </select>
              </div>

              <div className="issue-detail-field">
                <label htmlFor="edit-labels">Labels</label>
                <div className="issue-detail-labels-edit">
                  <div className="issue-detail-labels-list">
                    {editedIssue.labels.map((label) => (
                      <span key={label} className="issue-detail-label">
                        {label}
                        <button
                          className="issue-detail-label-remove"
                          onClick={() => handleRemoveLabel(label)}
                          aria-label={`Remove label ${label}`}
                        >
                          ×
                        </button>
                      </span>
                    ))}
                  </div>
                  <div className="issue-detail-label-add">
                    <input
                      id="edit-labels"
                      type="text"
                      value={newLabel}
                      onChange={(e) => setNewLabel(e.target.value)}
                      onKeyDown={handleKeyDown}
                      placeholder="Add label..."
                      className="issue-detail-input issue-detail-input-small"
                    />
                    <button
                      className="issue-detail-button issue-detail-button-small"
                      onClick={handleAddLabel}
                    >
                      Add
                    </button>
                  </div>
                </div>
              </div>

              <div className="issue-detail-field">
                <label htmlFor="edit-story-points">Story Points</label>
                <input
                  id="edit-story-points"
                  type="number"
                  min="0"
                  max="13"
                  value={editedIssue.storyPoints || ''}
                  onChange={(e) =>
                    setEditedIssue({
                      ...editedIssue,
                      storyPoints: e.target.value ? parseInt(e.target.value, 10) : undefined,
                    })
                  }
                  className="issue-detail-input"
                />
              </div>

              <div className="issue-detail-field">
                <label htmlFor="edit-due-date">Due Date</label>
                <input
                  id="edit-due-date"
                  type="date"
                  value={editedIssue.dueDate ? editedIssue.dueDate.split('T')[0] : ''}
                  onChange={(e) =>
                    setEditedIssue({
                      ...editedIssue,
                      dueDate: e.target.value ? new Date(e.target.value).toISOString() : undefined,
                    })
                  }
                  className="issue-detail-input"
                />
              </div>
            </div>
          ) : (
            <div className="issue-detail-view">
              <div className="issue-detail-title-section">
                <div
                  className="issue-detail-priority-badge"
                  style={{ backgroundColor: PRIORITY_COLORS[issue.priority] }}
                >
                  {PRIORITY_LABELS[issue.priority]}
                </div>
                <h4>{issue.title}</h4>
              </div>

              <div className="issue-detail-meta">
                <span className="issue-detail-id">#{issue.id.split('_').pop()}</span>
                <span className="issue-detail-status">{formatStatus(issue.status)}</span>
              </div>

              {issue.description && (
                <div className="issue-detail-section">
                  <h5>Description</h5>
                  <p className="issue-detail-description">{issue.description}</p>
                </div>
              )}

              {issue.labels && issue.labels.length > 0 && (
                <div className="issue-detail-section">
                  <h5>Labels</h5>
                  <div className="issue-detail-labels">
                    {issue.labels.map((label) => (
                      <span key={label} className="issue-detail-label">
                        {label}
                      </span>
                    ))}
                  </div>
                </div>
              )}

              <div className="issue-detail-section">
                <h5>Details</h5>
                <div className="issue-detail-grid">
                  <div className="issue-detail-grid-item">
                    <span className="issue-detail-grid-label">Assignee:</span>
                    <span className="issue-detail-grid-value">
                      {issue.assignee?.name || 'Unassigned'}
                    </span>
                  </div>
                  {issue.storyPoints && (
                    <div className="issue-detail-grid-item">
                      <span className="issue-detail-grid-label">Story Points:</span>
                      <span className="issue-detail-grid-value">{issue.storyPoints}</span>
                    </div>
                  )}
                  {issue.dueDate && (
                    <div className="issue-detail-grid-item">
                      <span className="issue-detail-grid-label">Due Date:</span>
                      <span className="issue-detail-grid-value">{formatDate(issue.dueDate)}</span>
                    </div>
                  )}
                  <div className="issue-detail-grid-item">
                    <span className="issue-detail-grid-label">Created:</span>
                    <span className="issue-detail-grid-value">{formatDate(issue.createdAt)}</span>
                  </div>
                  <div className="issue-detail-grid-item">
                    <span className="issue-detail-grid-label">Updated:</span>
                    <span className="issue-detail-grid-value">{formatDate(issue.updatedAt)}</span>
                  </div>
                  {issue.resolvedAt && (
                    <div className="issue-detail-grid-item">
                      <span className="issue-detail-grid-label">Resolved:</span>
                      <span className="issue-detail-grid-value">{formatDate(issue.resolvedAt)}</span>
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}
        </div>

        <div className="issue-detail-footer">
          {isEditing ? (
            <div className="issue-detail-actions">
              <button className="issue-detail-button issue-detail-button-secondary" onClick={handleCancel}>
                Cancel
              </button>
              <button className="issue-detail-button issue-detail-button-primary" onClick={handleSave}>
                Save Changes
              </button>
            </div>
          ) : (
            <div className="issue-detail-actions">
              <button
                className="issue-detail-button issue-detail-button-danger"
                onClick={handleDelete}
              >
                Delete Issue
              </button>
              <button
                className="issue-detail-button issue-detail-button-primary"
                onClick={() => setIsEditing(true)}
              >
                Edit Issue
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

/**
 * Format status for display
 */
const formatStatus = (status: IssueStatus): string => {
  const statusLabels: Record<IssueStatus, string> = {
    backlog: 'Backlog',
    in_progress: 'In Progress',
    review: 'Review',
    done: 'Done',
  };
  return statusLabels[status];
};

export default IssueDetailPanel;
