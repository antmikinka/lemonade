/**
 * CreateIssueModal Component
 * Modal dialog for creating new issues
 */

import React, { useState, useRef, useEffect } from 'react';
import { useTeamDashboard } from '../../../contexts/TeamDashboardContext';
import type { WorkItem, WorkItemStatus, Priority, Label, Contributor } from '../../../types/workItem';
import { PRIORITY_LABELS } from '../../../types/workItem';
import { CloseIcon } from '../../../components/Icons';

/**
 * Focus trap hook for modal dialogs
 * Traps focus within the modal and restores focus on close
 */
function useFocusTrap(isActive: boolean, onClose: () => void, containerRef: React.RefObject<HTMLElement | null>) {
  const previousFocusRef = useRef<HTMLElement | null>(null);

  useEffect(() => {
    if (!isActive) {
      return;
    }

    // Store previously focused element
    previousFocusRef.current = document.activeElement as HTMLElement;

    // Focus first focusable element in modal
    const container = containerRef.current;
    if (container) {
      const focusableElements = container.querySelectorAll<HTMLElement>(
        'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
      );
      if (focusableElements.length > 0) {
        focusableElements[0].focus();
      }
    }

    // Handle tab key for focus trapping
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key !== 'Tab') return;

      const containerElem = containerRef.current;
      if (!containerElem) return;

      const focusableElements = containerElem.querySelectorAll<HTMLElement>(
        'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
      );

      if (focusableElements.length === 0) return;

      const firstElement = focusableElements[0];
      const lastElement = focusableElements[focusableElements.length - 1];

      if (e.shiftKey) {
        // Shift + Tab: go to last element if on first
        if (document.activeElement === firstElement) {
          e.preventDefault();
          lastElement.focus();
        }
      } else {
        // Tab: go to first element if on last
        if (document.activeElement === lastElement) {
          e.preventDefault();
          firstElement.focus();
        }
      }
    };

    document.addEventListener('keydown', handleKeyDown);

    return () => {
      document.removeEventListener('keydown', handleKeyDown);

      // Restore focus to previously focused element
      previousFocusRef.current?.focus();
    };
  }, [isActive, onClose, containerRef]);
}

interface CreateIssueModalProps {
  isOpen: boolean;
  onClose: () => void;
}

const CreateIssueModal: React.FC<CreateIssueModalProps> = ({ isOpen, onClose }) => {
  const { addWorkItem, state } = useTeamDashboard();
  const modalRef = useRef<HTMLDivElement>(null);

  // Apply focus trap when modal is open
  useFocusTrap(isOpen, onClose, modalRef);

  // Handle clicks outside modal
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (modalRef.current && !modalRef.current.contains(e.target as Node)) {
        onClose();
      }
    };

    if (isOpen) {
      document.addEventListener('mousedown', handleClickOutside);
    }
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [isOpen, onClose]);

  const [formData, setFormData] = useState({
    title: '',
    description: '',
    status: 'backlog' as WorkItemStatus,
    priority: 'medium' as Priority,
    assigneeId: '',
    labels: [] as Label[],
    storyPoints: undefined as number | undefined,
    dueDate: undefined as string | undefined,
  });

  const [newLabel, setNewLabel] = useState('');
  const [errors, setErrors] = useState<{ title?: string }>({});

  // Reset form when modal opens/closes
  useEffect(() => {
    if (isOpen) {
      setFormData({
        title: '',
        description: '',
        status: 'backlog',
        priority: 'medium',
        assigneeId: '',
        labels: [],
        storyPoints: undefined,
        dueDate: undefined,
      });
      setErrors({});
      setNewLabel('');
    }
  }, [isOpen]);

  const validateForm = (): boolean => {
    const newErrors: { title?: string } = {};

    if (!formData.title.trim()) {
      newErrors.title = 'Title is required';
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();

    if (!validateForm()) {
      return;
    }

    const assignee: Contributor | undefined = formData.assigneeId
      ? state.teamMembers.find((m) => m.id === formData.assigneeId)
      : undefined;

    const now = new Date().toISOString();
    addWorkItem({
      title: formData.title,
      description: formData.description,
      status: formData.status,
      priority: formData.priority,
      type: 'issue',
      source: 'manual',
      author: {
        id: 'current-user',
        name: 'Current User',
      },
      assignees: assignee ? [assignee] : [],
      labels: formData.labels,
      metrics: {
        age: 0,
        estimatedPoints: formData.storyPoints,
      } as any,
      linkedItems: [],
    });

    onClose();
  };

  const handleAddLabel = () => {
    if (newLabel.trim()) {
      const existingLabel = formData.labels.find(l => l.name === newLabel.trim());
      if (!existingLabel) {
        const newLabelObj: Label = {
          id: `label_${Date.now()}`,
          name: newLabel.trim(),
          color: '#' + Math.floor(Math.random() * 16777215).toString(16).padStart(6, '0'),
          source: 'manual',
        };
        setFormData({ ...formData, labels: [...formData.labels, newLabelObj] });
        setNewLabel('');
      }
    }
  };

  const handleRemoveLabel = (labelToRemove: Label) => {
    setFormData({
      ...formData,
      labels: formData.labels.filter((l) => l.id !== labelToRemove.id),
    });
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleAddLabel();
    }
  };

  if (!isOpen) {
    return null;
  }

  return (
    <div
      className="create-issue-overlay"
      role="dialog"
      aria-modal="true"
      aria-labelledby="create-issue-title"
    >
      <div className="create-issue-modal" ref={modalRef}>
        <div className="create-issue-header">
          <h3 id="create-issue-title">Create New Issue</h3>
          <button
            className="create-issue-close"
            onClick={onClose}
            aria-label="Close modal"
          >
            <CloseIcon size={18} />
          </button>
        </div>

        <form onSubmit={handleSubmit} className="create-issue-form">
          <div className="create-issue-body">
            <div className="create-issue-field">
              <label htmlFor="new-issue-title">Title *</label>
              <input
                id="new-issue-title"
                type="text"
                value={formData.title}
                onChange={(e) => setFormData({ ...formData, title: e.target.value })}
                className={`create-issue-input ${errors.title ? 'error' : ''}`}
                placeholder="Enter issue title"
                autoFocus
                aria-required="true"
                aria-invalid={!!errors.title}
                aria-describedby={errors.title ? 'title-error' : undefined}
              />
              {errors.title && (
                <span id="title-error" className="create-issue-error">
                  {errors.title}
                </span>
              )}
            </div>

            <div className="create-issue-field">
              <label htmlFor="new-issue-description">Description</label>
              <textarea
                id="new-issue-description"
                value={formData.description}
                onChange={(e) => setFormData({ ...formData, description: e.target.value })}
                className="create-issue-textarea"
                rows={4}
                placeholder="Describe the issue..."
              />
            </div>

            <div className="create-issue-row">
              <div className="create-issue-field">
                <label htmlFor="new-issue-status">Status</label>
                <select
                  id="new-issue-status"
                  value={formData.status}
                  onChange={(e) =>
                    setFormData({ ...formData, status: e.target.value as WorkItemStatus })
                  }
                  className="create-issue-select"
                >
                  <option value="backlog">Backlog</option>
                  <option value="in_progress">In Progress</option>
                  <option value="in_review">In Review</option>
                  <option value="merged">Merged</option>
                  <option value="done">Done</option>
                  <option value="closed">Closed</option>
                </select>
              </div>

              <div className="create-issue-field">
                <label htmlFor="new-issue-priority">Priority</label>
                <select
                  id="new-issue-priority"
                  value={formData.priority}
                  onChange={(e) =>
                    setFormData({ ...formData, priority: e.target.value as Priority })
                  }
                  className="create-issue-select"
                >
                  <option value="low">Low</option>
                  <option value="medium">Medium</option>
                  <option value="high">High</option>
                  <option value="critical">Critical</option>
                </select>
              </div>
            </div>

            <div className="create-issue-row">
              <div className="create-issue-field">
                <label htmlFor="new-issue-assignee">Assignee</label>
                <select
                  id="new-issue-assignee"
                  value={formData.assigneeId}
                  onChange={(e) => setFormData({ ...formData, assigneeId: e.target.value })}
                  className="create-issue-select"
                >
                  <option value="">Unassigned</option>
                  {state.teamMembers.map((member) => (
                    <option key={member.id} value={member.id}>
                      {member.name}
                    </option>
                  ))}
                </select>
              </div>

              <div className="create-issue-field">
                <label htmlFor="new-issue-points">Story Points</label>
                <input
                  id="new-issue-points"
                  type="number"
                  min="0"
                  max="13"
                  value={formData.storyPoints || ''}
                  onChange={(e) =>
                    setFormData({
                      ...formData,
                      storyPoints: e.target.value ? parseInt(e.target.value, 10) : undefined,
                    })
                  }
                  className="create-issue-input"
                  placeholder="Optional"
                />
              </div>
            </div>

            <div className="create-issue-field">
              <label htmlFor="new-issue-due-date">Due Date</label>
              <input
                id="new-issue-due-date"
                type="date"
                value={formData.dueDate ? formData.dueDate.split('T')[0] : ''}
                onChange={(e) =>
                  setFormData({
                    ...formData,
                    dueDate: e.target.value ? new Date(e.target.value).toISOString() : undefined,
                  })
                }
                className="create-issue-input"
              />
            </div>

            <div className="create-issue-field">
              <label htmlFor="new-issue-labels">Labels</label>
              <div className="create-issue-labels">
                <div className="create-issue-labels-list">
                  {formData.labels.map((label) => (
                    <span key={label.id} className="create-issue-label" style={{ backgroundColor: label.color + '20', color: label.color }}>
                      {label.name}
                      <button
                        type="button"
                        className="create-issue-label-remove"
                        onClick={() => handleRemoveLabel(label)}
                        aria-label={`Remove label ${label.name}`}
                      >
                        ×
                      </button>
                    </span>
                  ))}
                </div>
                <div className="create-issue-label-add">
                  <input
                    id="new-issue-labels"
                    type="text"
                    value={newLabel}
                    onChange={(e) => setNewLabel(e.target.value)}
                    onKeyDown={handleKeyDown}
                    placeholder="Add label..."
                    className="create-issue-input create-issue-input-small"
                  />
                  <button
                    type="button"
                    className="create-issue-button create-issue-button-small"
                    onClick={handleAddLabel}
                  >
                    Add
                  </button>
                </div>
              </div>
            </div>
          </div>

          <div className="create-issue-footer">
            <button
              type="button"
              className="create-issue-button create-issue-button-secondary"
              onClick={onClose}
            >
              Cancel
            </button>
            <button type="submit" className="create-issue-button create-issue-button-primary">
              Create Issue
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

export default CreateIssueModal;
