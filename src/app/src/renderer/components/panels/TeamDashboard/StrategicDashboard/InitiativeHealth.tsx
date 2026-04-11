/**
 * Initiative Health Component
 * Displays strategic initiative progress and health status
 */

import React, { useMemo } from 'react';
import type { StrategicInitiative, DashboardMetrics } from '../../../../types/workItem';

interface InitiativeHealthProps {
  initiatives: StrategicInitiative[];
  topInitiatives: StrategicInitiative[];
  metrics: DashboardMetrics;
}

export const InitiativeHealth: React.FC<InitiativeHealthProps> = ({
  initiatives,
  topInitiatives,
  metrics,
}) => {
  const healthStats = useMemo(() => {
    const total = initiatives.length;
    const green = initiatives.filter((i) => i.healthStatus === 'green').length;
    const yellow = initiatives.filter((i) => i.healthStatus === 'yellow').length;
    const red = initiatives.filter((i) => i.healthStatus === 'red').length;

    return { total, green, yellow, red };
  }, [initiatives]);

  const roiStats = useMemo(() => {
    return metrics.roi.byInitiative || [];
  }, [metrics]);

  if (initiatives.length === 0) {
    return (
      <div className="initiative-health empty" role="region" aria-label="Initiative Health">
        <h3 className="initiative-health-title">Initiative Health</h3>
        <div className="initiative-health-empty">
          <p>No strategic initiatives defined</p>
          <button className="initiative-health-create-btn">
            Create Initiative
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="initiative-health" role="region" aria-label="Initiative Health">
      <div className="initiative-health-header">
        <h3 className="initiative-health-title">Initiative Health</h3>
        <span className="initiative-health-count">{initiatives.length}</span>
      </div>

      {/* Health Overview */}
      <div className="initiative-health-overview">
        <div className="health-stat">
          <div className="health-stat-value green">{healthStats.green}</div>
          <div className="health-stat-label">On Track</div>
        </div>
        <div className="health-stat">
          <div className="health-stat-value yellow">{healthStats.yellow}</div>
          <div className="health-stat-label">At Risk</div>
        </div>
        <div className="health-stat">
          <div className="health-stat-value red">{healthStats.red}</div>
          <div className="health-stat-label">Off Track</div>
        </div>
      </div>

      {/* Top Initiatives by ROI */}
      {topInitiatives.length > 0 && (
        <div className="top-initiatives">
          <h4 className="top-initiatives-title">Top Performing Initiatives</h4>
          <div className="top-initiatives-list">
            {topInitiatives.map((initiative) => (
              <div
                key={initiative.id}
                className="top-initiative-item"
                role="article"
                aria-label={`${initiative.name}: ${initiative.progress}% complete`}
              >
                <div className="top-initiative-info">
                  <span className="top-initiative-name">{initiative.name}</span>
                  <span
                    className={`top-initiative-status status-${initiative.healthStatus}`}
                  >
                    {initiative.healthStatus === 'green'
                      ? 'On Track'
                      : initiative.healthStatus === 'yellow'
                      ? 'At Risk'
                      : 'Off Track'}
                  </span>
                </div>

                <div className="top-initiative-progress">
                  <div className="progress-bar">
                    <div
                      className="progress-fill"
                      style={{
                        width: `${initiative.progress}%`,
                        backgroundColor: getProgressColor(initiative.progress),
                      }}
                      role="progressbar"
                      aria-valuenow={initiative.progress}
                      aria-valuemin={0}
                      aria-valuemax={100}
                    />
                  </div>
                  <span className="progress-percent">{initiative.progress}%</span>
                </div>

                <div className="top-initiative-metrics">
                  <span className="metric-label">ROI:</span>
                  <span className="metric-value">
                    {((initiative.actuals?.actualROI || 0) * 100).toFixed(0)}%
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* ROI by Initiative Chart */}
      {roiStats.length > 0 && (
        <div className="initiative-roi-chart">
          <h4 className="chart-title">ROI by Initiative</h4>
          <div className="roi-bar-chart">
            {roiStats.slice(0, 5).map((item, index: number) => (
              <div key={index} className="roi-bar-item">
                <div className="roi-bar-label">{item.name}</div>
                <div className="roi-bar-container">
                  <div
                    className={`roi-bar ${item.status === 'on_track' ? 'positive' : item.status === 'at_risk' ? 'warning' : 'negative'}`}
                    style={{ width: `${Math.min(100, Math.max(0, item.roi * 50))}%` }}
                  />
                </div>
                <div className="roi-bar-value">{(item.roi * 100).toFixed(0)}%</div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

// ============================================================================
// Helper Functions
// ============================================================================

function getProgressColor(progress: number): string {
  if (progress >= 80) return '#10b981';
  if (progress >= 50) return '#3b82f6';
  if (progress >= 25) return '#f59e0b';
  return '#ef4444';
}
