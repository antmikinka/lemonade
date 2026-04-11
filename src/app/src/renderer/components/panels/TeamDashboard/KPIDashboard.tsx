/**
 * KPIDashboard Component
 * Displays key performance metrics for the team
 */

import React, { useMemo, useState } from 'react';
import { useTeamDashboard } from '../../../contexts/TeamDashboardContext';
import { getLast4Quarters } from '../../../utils/issueStorage';
import type { WorkItemStatus } from '../../../types/workItem';

const KPIDashboard: React.FC = () => {
  const { state } = useTeamDashboard();
  const quarters = useMemo(() => getLast4Quarters(), []);
  const [selectedQuarter, setSelectedQuarter] = useState(quarters[quarters.length - 1]);

  // Use metrics from state - calculated in TeamDashboardContext
  const metrics = state.metrics.development;

  // Calculate status counts from workItems
  const statusCounts = useMemo(() => {
    const total = state.workItems.length;
    const backlog = state.workItems.filter((i) => i.status === 'backlog').length;
    const inProgress = state.workItems.filter((i) => i.status === 'in_progress').length;
    const inReview = state.workItems.filter((i) => i.status === 'in_review').length;
    const done = state.workItems.filter((i) => i.status === 'done' || i.status === 'merged').length;
    return { total, backlog, inProgress, inReview, done };
  }, [state.workItems]);

  // Get percentage for status bar
  const getPercentage = (value: number, total: number): number => {
    if (total === 0) return 0;
    return Math.round((value / total) * 100);
  };

  return (
    <div className="kpi-dashboard" role="region" aria-label="Team Performance Metrics">
      <div className="kpi-dashboard-header">
        <h2>Team Performance</h2>
        <div className="kpi-quarter-selector">
          <label htmlFor="quarter-select" className="kpi-quarter-label">
            Quarter:
          </label>
          <select
            id="quarter-select"
            className="kpi-quarter-select"
            value={selectedQuarter}
            onChange={(e) => setSelectedQuarter(e.target.value)}
            aria-label="Select quarter for metrics"
          >
            {quarters.map((quarter) => (
              <option key={quarter} value={quarter}>
                {quarter}
              </option>
            ))}
          </select>
        </div>
      </div>

      <div className="kpi-metrics-grid">
        <div className="kpi-metric-card">
          <div className="kpi-metric-icon issues-completed">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          </div>
          <div className="kpi-metric-content">
            <span className="kpi-metric-value">{metrics.throughput.itemsCompleted}</span>
            <span className="kpi-metric-label">Issues Completed</span>
          </div>
        </div>

        <div className="kpi-metric-card">
          <div className="kpi-metric-icon velocity">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
            </svg>
          </div>
          <div className="kpi-metric-content">
            <span className="kpi-metric-value">{metrics.velocity.current}</span>
            <span className="kpi-metric-label">Velocity</span>
          </div>
        </div>

        <div className="kpi-metric-card">
          <div className="kpi-metric-icon resolution-time">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <circle cx="12" cy="12" r="10" />
              <path d="M12 6v6l4 2" />
            </svg>
          </div>
          <div className="kpi-metric-content">
            <span className="kpi-metric-value">{Math.round(metrics.cycleTime.average)}</span>
            <span className="kpi-metric-label">Avg Cycle Time (days)</span>
          </div>
        </div>

        <div className="kpi-metric-card">
          <div className="kpi-metric-icon completion-rate">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M11 3.055A9.001 9.001 0 1020.945 13H11V3.055z" />
              <path d="M20.488 9H15V3.512A9.025 9.025 0 0120.488 9z" />
            </svg>
          </div>
          <div className="kpi-metric-content">
            <span className="kpi-metric-value">{metrics.codeQuality.defectRate}%</span>
            <span className="kpi-metric-label">Defect Rate</span>
          </div>
        </div>
      </div>

      <div className="kpi-status-summary">
        <h3>Current Status</h3>
        <div className="kpi-status-bar">
          <div
            className="kpi-status-segment backlog"
            style={{ width: `${getPercentage(statusCounts.backlog, statusCounts.total)}%` }}
            title={`${statusCounts.backlog} in backlog`}
          />
          <div
            className="kpi-status-segment in-progress"
            style={{ width: `${getPercentage(statusCounts.inProgress, statusCounts.total)}%` }}
            title={`${statusCounts.inProgress} in progress`}
          />
          <div
            className="kpi-status-segment review"
            style={{ width: `${getPercentage(statusCounts.inReview, statusCounts.total)}%` }}
            title={`${statusCounts.inReview} in review`}
          />
          <div
            className="kpi-status-segment done"
            style={{ width: `${getPercentage(statusCounts.done, statusCounts.total)}%` }}
            title={`${statusCounts.done} completed`}
          />
        </div>
        <div className="kpi-status-legend">
          <div className="kpi-legend-item">
            <span className="kpi-legend-color backlog" />
            <span>Backlog ({statusCounts.backlog})</span>
          </div>
          <div className="kpi-legend-item">
            <span className="kpi-legend-color in-progress" />
            <span>In Progress ({statusCounts.inProgress})</span>
          </div>
          <div className="kpi-legend-item">
            <span className="kpi-legend-color review" />
            <span>Review ({statusCounts.inReview})</span>
          </div>
          <div className="kpi-legend-item">
            <span className="kpi-legend-color done" />
            <span>Done ({statusCounts.done})</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default KPIDashboard;
