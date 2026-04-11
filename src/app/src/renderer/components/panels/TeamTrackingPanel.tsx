/**
 * TeamTrackingPanel Component
 * Main panel for the AI-Automated Team Tracking Dashboard
 * Provides Kanban board, KPI metrics, and work item management with GitHub sync
 */

import React, { useState, useMemo } from 'react';
import { useTeamDashboard } from '../../contexts/TeamDashboardContext';
import IssueBoard from './TeamDashboard/IssueBoard';
import KPIDashboard from './TeamDashboard/KPIDashboard';
import IssueDetailPanel from './TeamDashboard/IssueDetailPanel';
import CreateIssueModal from './TeamDashboard/CreateIssueModal';
import StrategicDashboard from './TeamDashboard/StrategicDashboard';
import type { WorkItem, Priority, WorkItemStatus } from '../../types/workItem';
import {
  PlusIcon,
  BoardIcon,
  ListIcon,
  SearchIcon,
  FilterIcon,
  RefreshCwIcon,
  TargetIcon,
  BarChartIcon,
} from '../../components/Icons';

interface TeamTrackingPanelProps {
  isVisible: boolean;
}

const TeamTrackingPanel: React.FC<TeamTrackingPanelProps> = ({ isVisible }) => {
  const {
    state,
    setViewMode,
    setFilters,
    showCreateInitiative,
    hideCreateInitiative,
    selectWorkItem,
    triggerSync,
    showSyncConfig,
  } = useTeamDashboard();

  const [showFilters, setShowFilters] = useState(false);
  const [searchInput, setSearchInput] = useState('');
  const [activeView, setActiveView] = useState<'board' | 'strategic'>('board');

  // Debounced search
  const debouncedSearch = useMemo(() => {
    const timer = setTimeout(() => {
      setFilters({ searchQuery: searchInput || undefined });
    }, 300);
    return () => clearTimeout(timer);
  }, [searchInput, setFilters]);

  const handleSearchChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setSearchInput(e.target.value);
  };

  const handleClearFilters = () => {
    setSearchInput('');
    setFilters({});
    setShowFilters(false);
  };

  const handlePriorityFilter = (priority: Priority) => {
    const currentPriorities = state.filters.priorities || [];
    const newPriorities = currentPriorities.includes(priority)
      ? currentPriorities.filter((p) => p !== priority)
      : [...currentPriorities, priority];

    setFilters({
      priorities: newPriorities.length > 0 ? newPriorities : undefined,
    });
  };

  const activeFiltersCount = useMemo(() => {
    let count = 0;
    if (state.filters.priorities) count++;
    if (state.filters.statuses) count++;
    if (state.filters.labels && state.filters.labels.length > 0) count++;
    if (state.filters.searchQuery) count++;
    return count;
  }, [state.filters]);

  const handleSync = async () => {
    await triggerSync();
  };

  if (!isVisible) {
    return null;
  }

  return (
    <div className="team-tracking-panel">
      {/* Header Section */}
      <div className="team-tracking-header">
        <div className="team-tracking-title-section">
          <h2 className="team-tracking-title">
            {activeView === 'strategic' ? (
              <>
                <TargetIcon size={24} />
                Strategic Intelligence
              </>
            ) : (
              <>
                <BoardIcon size={24} />
                Team Dashboard
              </>
            )}
          </h2>
          <span className="team-tracking-issue-count">
            {state.workItems.length}{' '}
            {state.workItems.length === 1 ? 'item' : 'items'}
          </span>
        </div>

        <div className="team-tracking-actions">
          {/* Sync Button */}
          <button
            className="team-tracking-sync-btn"
            onClick={handleSync}
            disabled={state.syncState.status === 'syncing'}
            title="Sync with GitHub"
            aria-label="Sync with GitHub"
          >
            <RefreshCwIcon
              size={18}
              className={state.syncState.status === 'syncing' ? 'spinning' : ''}
            />
            <span>Sync</span>
          </button>

          {/* View Toggle */}
          <div className="team-tracking-view-toggle">
            <button
              className={`team-tracking-view-btn ${activeView === 'strategic' ? 'active' : ''}`}
              onClick={() => setActiveView('strategic')}
              title="Strategic view"
              aria-label="Strategic view"
              aria-pressed={activeView === 'strategic'}
            >
              <BarChartIcon size={18} />
              <span>Strategic</span>
            </button>
            <button
              className={`team-tracking-view-btn ${activeView === 'board' ? 'active' : ''}`}
              onClick={() => setActiveView('board')}
              title="Board view"
              aria-label="Board view"
              aria-pressed={activeView === 'board'}
            >
              <BoardIcon size={18} />
              <span>Board</span>
            </button>
          </div>

          {/* Create Button */}
          <button
            className="team-tracking-create-btn"
            onClick={() => {
              selectWorkItem(undefined);
              showCreateInitiative();
            }}
            aria-label="Create new work item"
          >
            <PlusIcon size={18} />
            <span>New Item</span>
          </button>
        </div>
      </div>

      {/* Toolbar */}
      {activeView === 'board' && (
        <div className="team-tracking-toolbar">
          <div className="team-tracking-search">
            <span className="team-tracking-search-icon">
              <SearchIcon size={16} />
            </span>
            <input
              type="text"
              placeholder="Search work items..."
              value={searchInput}
              onChange={handleSearchChange}
              className="team-tracking-search-input"
              aria-label="Search work items"
            />
            {searchInput && (
              <button
                className="team-tracking-search-clear"
                onClick={() => setSearchInput('')}
                aria-label="Clear search"
              >
                ×
              </button>
            )}
          </div>

          <div className="team-tracking-filters">
            <button
              className={`team-tracking-filter-btn ${showFilters || activeFiltersCount > 0 ? 'active' : ''}`}
              onClick={() => setShowFilters(!showFilters)}
              aria-expanded={showFilters}
              aria-label="Toggle filters"
            >
              <FilterIcon size={16} />
              <span>Filters</span>
              {activeFiltersCount > 0 && (
                <span className="team-tracking-filter-count">{activeFiltersCount}</span>
              )}
            </button>

            {showFilters && (
              <div className="team-tracking-filter-panel">
                <div className="team-tracking-filter-section">
                  <h4>Priority</h4>
                  <div className="team-tracking-priority-filters">
                    {(['critical', 'high', 'medium', 'low'] as Priority[]).map(
                      (priority) => (
                        <button
                          key={priority}
                          className={`team-tracking-priority-btn ${
                            state.filters.priorities?.includes(priority) ? 'active' : ''
                          }`}
                          onClick={() => handlePriorityFilter(priority)}
                          style={{
                            borderColor:
                              state.filters.priorities?.includes(priority)
                                ? getPriorityColor(priority)
                                : '#333',
                            backgroundColor: state.filters.priorities?.includes(priority)
                              ? getPriorityColor(priority) + '20'
                              : 'transparent',
                          }}
                        >
                          <span
                            className="team-tracking-priority-dot"
                            style={{ backgroundColor: getPriorityColor(priority) }}
                          />
                          {priority.charAt(0).toUpperCase() + priority.slice(1)}
                        </button>
                      )
                    )}
                  </div>
                </div>

                {activeFiltersCount > 0 && (
                  <button
                    className="team-tracking-clear-filters"
                    onClick={handleClearFilters}
                  >
                    Clear all filters
                  </button>
                )}
              </div>
            )}
          </div>
        </div>
      )}

      {/* Content Section */}
      <div className="team-tracking-content">
        {activeView === 'strategic' ? (
          <StrategicDashboard isVisible={true} />
        ) : (
          <>
            <div className="team-tracking-kpi-section">
              <KPIDashboard />
            </div>

            <div className="team-tracking-board-section">
              {state.viewMode === 'board' ? (
                <IssueBoard issues={state.workItems} />
              ) : (
                <div className="team-tracking-list-view">
                  <div className="team-tracking-list-header">
                    <span>Title</span>
                    <span>Status</span>
                    <span>Priority</span>
                    <span>Assignee</span>
                    <span>Labels</span>
                  </div>
                  {state.workItems.length === 0 ? (
                    <div className="team-tracking-empty-state">
                      <p>No work items yet</p>
                      <p className="team-tracking-empty-hint">
                        Sync with GitHub or create your first item manually
                      </p>
                      <button
                        className="team-tracking-connect-btn"
                        onClick={showSyncConfig}
                      >
                        <RefreshCwIcon size={18} />
                        <span>Configure GitHub Sync</span>
                      </button>
                    </div>
                  ) : (
                    state.workItems.map((item) => (
                      <div
                        key={item.id}
                        className="team-tracking-list-item"
                        onClick={() => selectWorkItem(item)}
                        role="button"
                        tabIndex={0}
                        onKeyDown={(e) => {
                          if (e.key === 'Enter' || e.key === ' ') {
                            selectWorkItem(item);
                          }
                        }}
                      >
                        <span className="team-tracking-list-title">{item.title}</span>
                        <span
                          className={`team-tracking-list-status status-${item.status}`}
                        >
                          {formatStatus(item.status)}
                        </span>
                        <span
                          className={`team-tracking-list-priority priority-${item.priority}`}
                        >
                          {formatPriority(item.priority)}
                        </span>
                        <span className="team-tracking-list-assignee">
                          {item.assignees?.[0]?.name ||
                            item.claimedBy ||
                            'Unassigned'}
                        </span>
                        <span className="team-tracking-list-labels">
                          {item.labels.slice(0, 2).map((l) => l.name).join(', ')}
                          {item.labels.length > 2 && ` +${item.labels.length - 2}`}
                        </span>
                      </div>
                    ))
                  )}
                </div>
              )}
            </div>
          </>
        )}
      </div>

      {/* Detail Panel */}
      {state.selectedWorkItem && (
        <IssueDetailPanel
          issue={state.selectedWorkItem as any}
          onClose={() => selectWorkItem(undefined)}
        />
      )}

      {/* Create Modal */}
      <CreateIssueModal isOpen={state.isCreatingInitiative} onClose={hideCreateInitiative} />
    </div>
  );
};

/**
 * Get color for priority
 */
const getPriorityColor = (priority: Priority): string => {
  const colors: Record<Priority, string> = {
    low: '#6b7280',
    medium: '#3b82f6',
    high: '#f59e0b',
    critical: '#ef4444',
  };
  return colors[priority];
};

/**
 * Format status for display
 */
const formatStatus = (status: WorkItemStatus): string => {
  const labels: Record<WorkItemStatus, string> = {
    backlog: 'Backlog',
    in_progress: 'In Progress',
    in_review: 'In Review',
    merged: 'Merged',
    done: 'Done',
    closed: 'Closed',
  };
  return labels[status] || status;
};

/**
 * Format priority for display
 */
const formatPriority = (priority: Priority): string => {
  const labels: Record<Priority, string> = {
    low: 'Low',
    medium: 'Medium',
    high: 'High',
    critical: 'Critical',
  };
  return labels[priority] || priority;
};

export default TeamTrackingPanel;
