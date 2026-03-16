/**
 * Strategic Dashboard Component
 * Top-level view showing strategic intelligence overview
 *
 * This is the main entry point for the AI-automated Team Tracking Dashboard.
 * It displays:
 * - Key strategic metrics at a glance
 * - Portfolio health overview
 * - AI-generated insights
 * - Quick actions for human input
 */

import React, { useMemo, useCallback } from 'react';
import { useTeamDashboard } from '../../contexts/TeamDashboardContext';
import type {
  DashboardMetrics,
  AIInsight,
  StrategicInitiative,
  WorkItem,
} from '../../types/workItem';

// Import sub-components (stubs - to be implemented)
import { PortfolioSummary } from './StrategicDashboard/PortfolioSummary';
import { KPICards } from './StrategicDashboard/KPICards';
import { InsightsPanel } from './StrategicDashboard/InsightsPanel';
import { InitiativeHealth } from './StrategicDashboard/InitiativeHealth';
import { QuickActions } from './StrategicDashboard/QuickActions';
import { SyncStatus } from './StrategicDashboard/SyncStatus';

import {
  TrendingUpIcon,
  TargetIcon,
  AlertCircleIcon,
  LightbulbIcon,
  RefreshCwIcon,
  SettingsIcon,
} from '../../components/Icons';

interface StrategicDashboardProps {
  isVisible: boolean;
}

const StrategicDashboard: React.FC<StrategicDashboardProps> = ({ isVisible }) => {
  const { state, dispatch, acknowledgeInsight, dismissInsight } = useTeamDashboard();

  // Memoized metrics calculations
  const metrics = state.metrics;
  const insights = state.insights;
  const initiatives = state.initiatives;
  const syncState = state.syncState;

  // Count insights by severity
  const insightCounts = useMemo(() => {
    return {
      critical: insights.filter((i) => i.severity === 'critical').length,
      high: insights.filter((i) => i.severity === 'high').length,
      medium: insights.filter((i) => i.severity === 'medium').length,
      low: insights.filter((i) => i.severity === 'low').length,
    };
  }, [insights]);

  // Get top initiatives by ROI
  const topInitiatives = useMemo(() => {
    return [...initiatives]
      .filter((i) => i.status === 'active')
      .sort((a, b) => {
        const aProgress = a.actuals?.actualROI || 0;
        const bProgress = b.actuals?.actualROI || 0;
        return b.progress - a.progress;
      })
      .slice(0, 5);
  }, [initiatives]);

  // Handle sync refresh
  const handleRefresh = useCallback(() => {
    dispatch({ type: 'SYNC_STARTED' });
    // Sync orchestrator will handle the actual sync
  }, [dispatch]);

  // Handle insight acknowledgment
  const handleAcknowledgeInsight = useCallback(
    (insightId: string) => {
      acknowledgeInsight(insightId);
    },
    [acknowledgeInsight]
  );

  // Handle insight dismissal
  const handleDismissInsight = useCallback(
    (insightId: string) => {
      dismissInsight(insightId);
    },
    [dismissInsight]
  );

  if (!isVisible) {
    return null;
  }

  return (
    <div className="strategic-dashboard" role="region" aria-label="Strategic Intelligence Dashboard">
      {/* Header Section */}
      <header className="strategic-dashboard-header">
        <div className="strategic-dashboard-title-section">
          <h2 className="strategic-dashboard-title">
            <TargetIcon size={24} />
            Strategic Intelligence Dashboard
          </h2>
          <p className="strategic-dashboard-subtitle">
            AI-powered insights for data-driven decision making
          </p>
        </div>

        <div className="strategic-dashboard-actions">
          <SyncStatus syncState={syncState} onRefresh={handleRefresh} />

          <button
            className="strategic-dashboard-action-btn"
            onClick={handleRefresh}
            disabled={syncState.status === 'syncing'}
            title="Refresh data from GitHub"
            aria-label="Refresh data from GitHub"
          >
            <RefreshCwIcon size={18} className={syncState.status === 'syncing' ? 'spinning' : ''} />
            <span>Sync</span>
          </button>

          <button
            className="strategic-dashboard-action-btn"
            title="Configure GitHub sync settings"
            aria-label="Configure GitHub sync settings"
          >
            <SettingsIcon size={18} />
            <span>Settings</span>
          </button>
        </div>
      </header>

      {/* Main Content Grid */}
      <div className="strategic-dashboard-grid">
        {/* Top Row: Portfolio Summary & KPI Cards */}
        <section className="strategic-dashboard-row top-row">
          <PortfolioSummary metrics={metrics} />
          <KPICards metrics={metrics} />
        </section>

        {/* Middle Row: Initiative Health & Insights */}
        <section className="strategic-dashboard-row middle-row">
          <div className="strategic-dashboard-column">
            <InitiativeHealth
              initiatives={initiatives}
              topInitiatives={topInitiatives}
              metrics={metrics}
            />
          </div>

          <div className="strategic-dashboard-column">
            <InsightsPanel
              insights={insights}
              insightCounts={insightCounts}
              onAcknowledge={handleAcknowledgeInsight}
              onDismiss={handleDismissInsight}
            />
          </div>
        </section>

        {/* Bottom Row: Quick Actions for Human Input */}
        <section className="strategic-dashboard-row bottom-row">
          <QuickActions workItems={state.workItems} initiatives={initiatives} />
        </section>
      </div>

      {/* Empty State */}
      {state.workItems.length === 0 && syncState.status !== 'syncing' && (
        <div className="strategic-dashboard-empty-state">
          <div className="strategic-dashboard-empty-icon">
            <TrendingUpIcon size={48} />
          </div>
          <h3>No Data Available</h3>
          <p>
            Connect your GitHub repository to start tracking strategic metrics.
          </p>
          <button className="strategic-dashboard-connect-btn">
            <SettingsIcon size={18} />
            <span>Configure GitHub Integration</span>
          </button>
        </div>
      )}
    </div>
  );
};

export default StrategicDashboard;

// ============================================================================
// CSS Classes Reference (for styles.css)
// ============================================================================

/*
.strategic-dashboard {
  display: flex;
  flex-direction: column;
  height: 100%;
  padding: 16px;
  background: var(--bg-primary);
  overflow: auto;
}

.strategic-dashboard-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding-bottom: 16px;
  border-bottom: 1px solid var(--border-color);
  margin-bottom: 16px;
}

.strategic-dashboard-title {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 20px;
  font-weight: 600;
  color: var(--text-primary);
}

.strategic-dashboard-subtitle {
  font-size: 13px;
  color: var(--text-secondary);
  margin-top: 4px;
}

.strategic-dashboard-actions {
  display: flex;
  gap: 8px;
  align-items: center;
}

.strategic-dashboard-action-btn {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 8px 12px;
  background: var(--bg-secondary);
  border: 1px solid var(--border-color);
  border-radius: 6px;
  color: var(--text-primary);
  font-size: 13px;
  cursor: pointer;
  transition: all 0.2s;
}

.strategic-dashboard-action-btn:hover {
  background: var(--bg-tertiary);
}

.strategic-dashboard-action-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.strategic-dashboard-action-btn .spinning {
  animation: spin 1s linear infinite;
}

@keyframes spin {
  from { transform: rotate(0deg); }
  to { transform: rotate(360deg); }
}

.strategic-dashboard-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 16px;
  flex: 1;
  min-height: 0;
}

.strategic-dashboard-row {
  display: contents;
}

.strategic-dashboard-column {
  display: flex;
  flex-direction: column;
  gap: 16px;
  min-height: 0;
}

.top-row {
  grid-column: 1 / -1;
  display: grid;
  grid-template-columns: 1fr 2fr;
  gap: 16px;
}

.middle-row {
  grid-column: 1 / -1;
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 16px;
}

.bottom-row {
  grid-column: 1 / -1;
}

.strategic-dashboard-empty-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 100%;
  text-align: center;
  color: var(--text-secondary);
}

.strategic-dashboard-empty-icon {
  margin-bottom: 16px;
  opacity: 0.5;
}

.strategic-dashboard-empty-state h3 {
  font-size: 18px;
  margin-bottom: 8px;
  color: var(--text-primary);
}

.strategic-dashboard-empty-state p {
  margin-bottom: 16px;
  max-width: 400px;
}

.strategic-dashboard-connect-btn {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 10px 16px;
  background: var(--accent-primary);
  border: none;
  border-radius: 6px;
  color: white;
  font-size: 14px;
  cursor: pointer;
  transition: all 0.2s;
}

.strategic-dashboard-connect-btn:hover {
  background: var(--accent-primary-hover);
}
*/
