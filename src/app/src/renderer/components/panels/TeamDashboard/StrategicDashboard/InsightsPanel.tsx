/**
 * Insights Panel Component
 * Displays AI-generated insights with acknowledgment and dismissal actions
 */

import React, { useMemo } from 'react';
import type { AIInsight, InsightSeverity } from '../../../types/workItem';

interface InsightsPanelProps {
  insights: AIInsight[];
  insightCounts: Record<InsightSeverity, number>;
  onAcknowledge: (insightId: string) => void;
  onDismiss: (insightId: string) => void;
}

export const InsightsPanel: React.FC<InsightsPanelProps> = ({
  insights,
  insightCounts,
  onAcknowledge,
  onDismiss,
}) => {
  const filteredInsights = useMemo(() => {
    // Filter out acknowledged and dismissed insights
    return insights.filter(
      (insight) =>
        !insight.acknowledgedBy?.includes('current-user') &&
        !insight.dismissedBy?.includes('current-user')
    );
  }, [insights]);

  const severityColors: Record<InsightSeverity, string> = {
    critical: '#ef4444',
    high: '#f59e0b',
    medium: '#3b82f6',
    low: '#10b981',
  };

  const severityIcons: Record<InsightSeverity, string> = {
    critical: '!',
    high: '!',
    medium: 'i',
    low: 'i',
  };

  if (filteredInsights.length === 0) {
    return (
      <div className="insights-panel empty" role="region" aria-label="AI Insights">
        <h3 className="insights-panel-title">AI Insights</h3>
        <div className="insights-panel-empty">
          <div className="insights-panel-empty-icon">
            <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
              <path d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3M3.343 3.343l.707.707M12 21a9 9 0 110-18 9 9 0 010 18z" />
            </svg>
          </div>
          <p className="insights-panel-empty-text">No new insights</p>
          <p className="insights-panel-empty-hint">
            Check back after more work items are synced
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="insights-panel" role="region" aria-label="AI Insights">
      <div className="insights-panel-header">
        <h3 className="insights-panel-title">AI Insights</h3>
        <span className="insights-panel-count">{filteredInsights.length}</span>
      </div>

      <div className="insights-panel-list">
        {filteredInsights.map((insight) => (
          <article
            key={insight.id}
            className={`insight-card insight-${insight.severity}`}
            role="article"
            aria-label={`${insight.severity} priority insight: ${insight.title}`}
          >
            <div className="insight-card-header">
              <div
                className="insight-card-severity"
                style={{ backgroundColor: severityColors[insight.severity] }}
              >
                {severityIcons[insight.severity]}
              </div>
              <div className="insight-card-meta">
                <span className="insight-card-type">{insight.type}</span>
                <span className="insight-card-confidence">
                  {(insight.confidence * 100).toFixed(0)}% confidence
                </span>
              </div>
            </div>

            <h4 className="insight-card-title">{insight.title}</h4>

            <p className="insight-card-description">{insight.description}</p>

            {insight.dataPoints && insight.dataPoints.length > 0 && (
              <div className="insight-card-data">
                {insight.dataPoints.map((point, index) => (
                  <div key={index} className="insight-data-point">
                    <span className="insight-data-label">{point.label}</span>
                    <span className="insight-data-value">
                      {typeof point.value === 'number'
                        ? point.value.toLocaleString(undefined, { maximumFractionDigits: 1 })
                        : point.value}
                      {point.unit ? ` ${point.unit}` : ''}
                    </span>
                  </div>
                ))}
              </div>
            )}

            {insight.recommendedAction && (
              <div className="insight-card-action">
                <span className="insight-action-label">Recommended action:</span>
                <p className="insight-action-text">{insight.recommendedAction}</p>
              </div>
            )}

            <div className="insight-card-actions">
              <button
                className="insight-action-btn acknowledge"
                onClick={() => onAcknowledge(insight.id)}
                aria-label="Acknowledge insight"
              >
                Acknowledge
              </button>
              <button
                className="insight-action-btn dismiss"
                onClick={() => onDismiss(insight.id)}
                aria-label="Dismiss insight"
              >
                Dismiss
              </button>
            </div>
          </article>
        ))}
      </div>
    </div>
  );
};
