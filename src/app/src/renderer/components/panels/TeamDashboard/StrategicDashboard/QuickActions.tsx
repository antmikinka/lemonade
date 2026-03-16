/**
 * Quick Actions Component
 * Provides quick actions for human input on strategic items
 */

import React, { useState, useEffect, useRef } from 'react';
import { useTeamDashboard } from '../../../contexts/TeamDashboardContext';
import type { WorkItem, StrategicInitiative, StrategicTag, ROICategory } from '../../../types/workItem';
import type { StrategicCategory } from '../../../types/workItem';

/**
 * Focus trap hook for modal dialogs
 * Traps focus within the modal and restores focus on close
 */
function useFocusTrap(isActive: boolean, onClose: () => void) {
  const containerRef = useRef<HTMLDivElement>(null);
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

      const focusableElements = container?.querySelectorAll<HTMLElement>(
        'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
      );

      if (!focusableElements || focusableElements.length === 0) return;

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

    // Handle escape key
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        onClose();
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    document.addEventListener('keydown', handleEscape);

    return () => {
      document.removeEventListener('keydown', handleKeyDown);
      document.removeEventListener('keydown', handleEscape);

      // Restore focus to previously focused element
      previousFocusRef.current?.focus();
    };
  }, [isActive, onClose]);

  return containerRef;
}

interface QuickActionsProps {
  workItems: WorkItem[];
  initiatives: StrategicInitiative[];
}

export const QuickActions: React.FC<QuickActionsProps> = ({
  workItems,
  initiatives,
}) => {
  const {
    claimWorkItem,
    addStrategicTag,
    setROICategory,
  } = useTeamDashboard();

  const [selectedAction, setSelectedAction] = useState<string | null>(null);
  const panelRef = useFocusTrap(!!selectedAction, () => setSelectedAction(null));

  // Find items needing human input
  const unclaimedItems = workItems.filter(
    (item) =>
      !item.claimedBy &&
      (item.status === 'in_progress' || item.status === 'in_review')
  );

  const untaggedHighImpact = workItems.filter(
    (item) =>
      (item.priority === 'high' || item.priority === 'critical') &&
      (!item.strategicTags || item.strategicTags.length === 0) &&
      !item.roiCategory
  );

  const actions = [
    {
      id: 'claim-work',
      title: 'Claim Work Items',
      description: `Take ownership of ${unclaimedItems.length} in-progress items`,
      count: unclaimedItems.length,
      color: 'blue',
      icon: (
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <path d="M20 21v-2a4 4 0 00-4-4H8a4 4 0 00-4 4v2" />
          <circle cx="12" cy="7" r="4" />
        </svg>
      ),
    },
    {
      id: 'tag-strategic',
      title: 'Tag Strategic Items',
      description: `Add strategic tags to ${untaggedHighImpact.length} high-priority items`,
      count: untaggedHighImpact.length,
      color: 'purple',
      icon: (
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <path d="M20.59 13.41l-7.17 7.17a2 2 0 01-2.83 0L2 12V2h10l8.59 8.59a2 2 0 010 2.82z" />
          <circle cx="7" cy="7" r="2" />
        </svg>
      ),
    },
    {
      id: 'assign-roi',
      title: 'Assign ROI Categories',
      description: 'Categorize items by return on investment',
      count: workItems.filter((i) => !i.roiCategory).length,
      color: 'green',
      icon: (
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <line x1="12" y1="1" x2="12" y2="23" />
          <path d="M17 5H9.5a3.5 3.5 0 000 7h5a3.5 3.5 0 010 7H6" />
        </svg>
      ),
    },
    {
      id: 'review-initiatives',
      title: 'Review Initiatives',
      description: `Check progress on ${initiatives.filter((i) => i.status === 'active').length} active initiatives`,
      count: initiatives.filter((i) => i.status === 'active').length,
      color: 'orange',
      icon: (
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <rect x="3" y="4" width="18" height="18" rx="2" ry="2" />
          <line x1="16" y1="2" x2="16" y2="6" />
          <line x1="8" y1="2" x2="8" y2="6" />
          <line x1="3" y1="10" x2="21" y2="10" />
        </svg>
      ),
    },
  ];

  return (
    <div className="quick-actions" role="region" aria-label="Quick Actions">
      <h3 className="quick-actions-title">Quick Actions</h3>
      <p className="quick-actions-subtitle">
        Human input needed for strategic tracking
      </p>

      <div className="quick-actions-grid">
        {actions.map((action) => (
          <button
            key={action.id}
            className={`quick-action-card quick-action-${action.color} ${action.count > 0 ? 'has-count' : ''}`}
            onClick={() => setSelectedAction(action.id)}
            disabled={action.count === 0}
            aria-label={`${action.title}: ${action.count} items need attention`}
          >
            <div className="quick-action-icon">{action.icon}</div>

            <div className="quick-action-content">
              <h4 className="quick-action-title">{action.title}</h4>
              <p className="quick-action-description">{action.description}</p>
            </div>

            {action.count > 0 && (
              <span className="quick-action-badge">{action.count}</span>
            )}

            <div className="quick-action-arrow">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M9 18l6-6-6-6" />
              </svg>
            </div>
          </button>
        ))}
      </div>

      {/* Action Detail Panel (shown when action is selected) */}
      {selectedAction && (
        <div
          ref={panelRef}
          className="quick-action-panel"
          role="dialog"
          aria-modal="true"
        >
          <div className="quick-action-panel-header">
            <h3>
              {actions.find((a) => a.id === selectedAction)?.title}
            </h3>
            <button
              className="quick-action-panel-close"
              onClick={() => setSelectedAction(null)}
              aria-label="Close panel"
            >
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M18 6L6 18M6 6l12 12" />
              </svg>
            </button>
          </div>

          <div className="quick-action-panel-content">
            {selectedAction === 'claim-work' && (
              <ClaimWorkPanel
                items={unclaimedItems}
                onClose={() => setSelectedAction(null)}
                onClaim={claimWorkItem}
              />
            )}
            {selectedAction === 'tag-strategic' && (
              <TagStrategicPanel
                items={untaggedHighImpact}
                onClose={() => setSelectedAction(null)}
                onTag={addStrategicTag}
              />
            )}
            {selectedAction === 'assign-roi' && (
              <AssignROIPanel
                items={workItems.filter((i) => !i.roiCategory)}
                onClose={() => setSelectedAction(null)}
                onAssignROI={(id, roiCategory) => setROICategory(id, roiCategory)}
              />
            )}
            {selectedAction === 'review-initiatives' && (
              <ReviewInitiativesPanel initiatives={initiatives} onClose={() => setSelectedAction(null)} />
            )}
          </div>
        </div>
      )}
    </div>
  );
};

// ============================================================================
// Sub-components for Action Panels
// ============================================================================

interface ClaimWorkPanelProps {
  items: WorkItem[];
  onClose: () => void;
  onClaim: (id: string) => void;
}

const ClaimWorkPanel: React.FC<ClaimWorkPanelProps> = ({ items, onClose, onClaim }) => {
  if (items.length === 0) {
    return <p className="empty-message">No items available to claim</p>;
  }

  const handleClaim = (id: string) => {
    onClaim(id);
  };

  return (
    <div className="claim-work-list">
      {items.map((item) => (
        <div key={item.id} className="claim-work-item">
          <div className="claim-work-info">
            <span className="claim-work-title">{item.title}</span>
            <span className="claim-work-type">{item.type}</span>
          </div>
          <button
            className="claim-work-btn"
            onClick={() => handleClaim(item.id)}
          >
            Claim
          </button>
        </div>
      ))}
    </div>
  );
};

interface TagStrategicPanelProps {
  items: WorkItem[];
  onClose: () => void;
  onTag: (id: string, tag: StrategicTag) => void;
}

const TagStrategicPanel: React.FC<TagStrategicPanelProps> = ({ items, onClose, onTag }) => {
  const strategicCategories: Array<{ id: string; name: string; color: string }> = [
    { id: 'feature', name: 'Feature', color: '#3b82f6' },
    { id: 'tech-debt', name: 'Tech Debt', color: '#f59e0b' },
    { id: 'performance', name: 'Performance', color: '#8b5cf6' },
    { id: 'security', name: 'Security', color: '#ef4444' },
    { id: 'compliance', name: 'Compliance', color: '#ec4899' },
    { id: 'infrastructure', name: 'Infrastructure', color: '#6b7280' },
  ];

  const handleTagSelect = (itemId: string, categoryId: string) => {
    const category = strategicCategories.find((cat) => cat.id === categoryId);
    if (category) {
      const tag: StrategicTag = {
        id: `tag_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
        name: category.name,
        category: category.id as StrategicCategory,
        strategicAlignment: 'core',
        addedBy: 'current-user',
        addedAt: new Date().toISOString(),
      };
      onTag(itemId, tag);
    }
  };

  return (
    <div className="tag-strategic-panel">
      <p className="tag-instruction">Select a strategic category for each item:</p>
      <div className="strategic-categories">
        {strategicCategories.map((cat) => (
          <button
            key={cat.id}
            className="category-tag"
            style={{ borderColor: cat.color }}
          >
            <span
              className="category-dot"
              style={{ backgroundColor: cat.color }}
            />
            {cat.name}
          </button>
        ))}
      </div>
      <div className="tag-items-list">
        {items.slice(0, 5).map((item) => (
          <div key={item.id} className="tag-item">
            <span className="tag-item-title">{item.title}</span>
            <select
              className="tag-item-select"
              onChange={(e) => handleTagSelect(item.id, e.target.value)}
              defaultValue=""
            >
              <option value="">Select category...</option>
              {strategicCategories.map((cat) => (
                <option key={cat.id} value={cat.id}>
                  {cat.name}
                </option>
              ))}
            </select>
          </div>
        ))}
      </div>
    </div>
  );
};

interface AssignROIPanelProps {
  items: WorkItem[];
  onClose: () => void;
  onAssignROI: (id: string, roiCategory: ROICategory) => void;
}

const AssignROIPanel: React.FC<AssignROIPanelProps> = ({ items, onClose, onAssignROI }) => {
  const roiCategories = [
    { id: 'revenue-impact', name: 'Revenue Impact' },
    { id: 'cost-reduction', name: 'Cost Reduction' },
    { id: 'risk-mitigation', name: 'Risk Mitigation' },
    { id: 'strategic-capability', name: 'Strategic Capability' },
    { id: 'developer-productivity', name: 'Developer Productivity' },
    { id: 'customer-experience', name: 'Customer Experience' },
  ];

  const handleROISelect = (itemId: string, categoryId: string) => {
    const category = roiCategories.find((cat) => cat.id === categoryId);
    if (category) {
      const roiCategory: ROICategory = {
        id: `roi_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
        name: category.id as ROICategory['name'],
        estimatedEffort: 0,
        estimatedImpact: 5,
      };
      onAssignROI(itemId, roiCategory);
    }
  };

  return (
    <div className="assign-roi-panel">
      <p className="roi-instruction">Categorize items by their primary ROI type:</p>
      <div className="roi-items-list">
        {items.slice(0, 5).map((item) => (
          <div key={item.id} className="roi-item">
            <span className="roi-item-title">{item.title}</span>
            <select
              className="roi-item-select"
              onChange={(e) => handleROISelect(item.id, e.target.value)}
              defaultValue=""
            >
              <option value="">Select ROI category...</option>
              {roiCategories.map((cat) => (
                <option key={cat.id} value={cat.id}>
                  {cat.name}
                </option>
              ))}
            </select>
          </div>
        ))}
      </div>
    </div>
  );
};

interface ReviewInitiativesPanelProps {
  initiatives: StrategicInitiative[];
  onClose: () => void;
}

const ReviewInitiativesPanel: React.FC<ReviewInitiativesPanelProps> = ({
  initiatives,
  onClose,
}) => {
  return (
    <div className="review-initiatives-list">
      {initiatives
        .filter((i) => i.status === 'active')
        .map((initiative) => (
          <div key={initiative.id} className="review-initiative-item">
            <div className="review-initiative-header">
              <span className="review-initiative-name">{initiative.name}</span>
              <span
                className={`review-initiative-health health-${initiative.healthStatus}`}
              >
                {initiative.healthStatus}
              </span>
            </div>
            <div className="review-initiative-progress">
              <div className="progress-bar">
                <div
                  className="progress-fill"
                  style={{ width: `${initiative.progress}%` }}
                />
              </div>
              <span className="progress-label">{initiative.progress}% complete</span>
            </div>
            <button className="review-initiative-btn">Review Status</button>
          </div>
        ))}
    </div>
  );
};
