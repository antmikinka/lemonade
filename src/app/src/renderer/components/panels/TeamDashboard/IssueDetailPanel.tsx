/**
 * IssueDetailPanel Component
 * Displays detailed view of a selected issue with edit capabilities
 */

import React, { useState, useEffect, useRef } from 'react';
import { useTeamDashboard } from '../../../contexts/TeamDashboardContext';
import type { WorkItem, WorkItemStatus, Priority, Contributor, Label } from '../../../types/workItem';
import { PRIORITY_COLORS, PRIORITY_LABELS } from '../../../types/workItem';
import { CloseIcon } from '../../../components/Icons';

interface IssueDetailPanelProps {
  issue: WorkItem;
  onClose: () => void;
}

const IssueDetailPanel: React.FC<IssueDetailPanelProps> = ({ issue, onClose }) => {
  const { updateWorkItem, deleteWorkItem, state, selectWorkItem } = useTeamDashboard();
  const [isEditing, setIsEditing] = useState(false);
  const [editedIssue, setEditedIssue] = useState<WorkItem>(issue);
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
    updateWorkItem(editedIssue);
    setIsEditing(false);
  };

  const handleCancel = () => {
    setEditedIssue(issue);
    setIsEditing(false);
  };

  const handleDelete = () => {
    if (window.confirm(`Are you sure you want to delete issue "${issue.title}"?`)) {
      deleteWorkItem(issue.id);
      onClose();
    }
  };

  const handleStatusChange = (newStatus: WorkItemStatus) => {
    setEditedIssue({ ...editedIssue, status: newStatus, updatedAt: new Date().toISOString() });
  };

  const handlePriorityChange = (newPriority: Priority) => {
    setEditedIssue({ ...editedIssue, priority: newPriority, updatedAt: new Date().toISOString() });
  };

  const handleAssigneeChange = (memberId: string) => {
    const member = state.teamMembers.find((m) => m.id === memberId);
    setEditedIssue({
      ...editedIssue,
      assignees: member ? [member] : [],
      updatedAt: new Date().toISOString(),
    });
  };

  const handleAddLabel = () => {
    if (newLabel.trim()) {
      const existingLabel = editedIssue.labels.find(l => l.name === newLabel.trim());
      if (!existingLabel) {
        const newLabelObj: Label = {
          id: `label_${Date.now()}`,
          name: newLabel.trim(),
          color: '#' + Math.floor(Math.random() * 16777215).toString(16).padStart(6, '0'),
          source: 'manual',
        };
        setEditedIssue({
          ...editedIssue,
          labels: [...editedIssue.labels, newLabelObj],
          updatedAt: new Date().toISOString(),
        });
        setNewLabel('');
      }
    }
  };

  const handleRemoveLabel = (labelToRemove: Label) => {
    setEditedIssue({
      ...editedIssue,
      labels: editedIssue.labels.filter((l) => l.id !== labelToRemove.id),
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
                  onChange={(e) => handleStatusChange(e.target.value as WorkItemStatus)}
                  className="issue-detail-select"
                >
                  <option value="backlog">Backlog</option>
                  <option value="in_progress">In Progress</option>
                  <option value="in_review">In Review</option>
                  <option value="merged">Merged</option>
                  <option value="done">Done</option>
                  <option value="closed">Closed</option>
                </select>
              </div>

              <div className="issue-detail-field">
                <label htmlFor="edit-priority">Priority</label>
                <select
                  id="edit-priority"
                  value={editedIssue.priority}
                  onChange={(e) => handlePriorityChange(e.target.value as Priority)}
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
                  value={editedIssue.assignees?.[0]?.id || ''}
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
                      <span key={label.id} className="issue-detail-label">
                        {label.name}
                        <button
                          className="issue-detail-label-remove"
                          onClick={() => handleRemoveLabel(label)}
                          aria-label={`Remove label ${label.name}`}
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
                  value={editedIssue.metrics?.estimatedPoints || ''}
                  onChange={(e) =>
                    setEditedIssue({
                      ...editedIssue,
                      metrics: {
                        ...editedIssue.metrics,
                        estimatedPoints: e.target.value ? parseInt(e.target.value, 10) : undefined,
                      },
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
                      <span key={label.id} className="issue-detail-label" style={{ backgroundColor: label.color + '20', color: label.color }}>
                        {label.name}
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
                      {issue.assignees?.[0]?.name || 'Unassigned'}
                    </span>
                  </div>
                  {issue.metrics?.estimatedPoints && (
                    <div className="issue-detail-grid-item">
                      <span className="issue-detail-grid-label">Story Points:</span>
                      <span className="issue-detail-grid-value">{issue.metrics.estimatedPoints}</span>
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
const formatStatus = (status: WorkItemStatus): string => {
  const statusLabels: Record<WorkItemStatus, string> = {
    backlog: 'Backlog',
    in_progress: 'In Progress',
    in_review: 'In Review',
    merged: 'Merged',
    done: 'Done',
    closed: 'Closed',
  };
  return statusLabels[status];
};

export default IssueDetailPanel;
