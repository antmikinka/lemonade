/**
 * KPI Cards Component
 * Displays key performance indicator cards with visualizations
 */

import React, { useMemo } from 'react';
import type { DashboardMetrics, DevelopmentKPIs } from '../../../../types/workItem';

interface KPICardsProps {
  metrics: DashboardMetrics;
}

export const KPICards: React.FC<KPICardsProps> = ({ metrics }) => {
  const kpis = metrics.development;

  const cards = useMemo(() => [
    {
      id: 'cycle-time',
      title: 'Cycle Time',
      value: kpis.cycleTime.average.toFixed(1),
      unit: 'days',
      trend: kpis.cycleTime.trend,
      trendIcon: getTrendIcon(kpis.cycleTime.trend),
      subtitle: `Median: ${kpis.cycleTime.median.toFixed(1)}d`,
      color: 'blue',
    },
    {
      id: 'velocity',
      title: 'Velocity',
      value: kpis.velocity.current.toString(),
      unit: 'points',
      trend: kpis.velocity.trend,
      trendIcon: getTrendIcon(kpis.velocity.trend),
      subtitle: `Avg: ${kpis.velocity.average.toFixed(1)}`,
      color: 'green',
    },
    {
      id: 'pr-review',
      title: 'Time to Review',
      value: (kpis.pullRequest.averageTimeToFirstReview / 24).toFixed(1),
      unit: 'days',
      trend: kpis.pullRequest.averageTimeToFirstReview > 48 ? 'degrading' : 'stable',
      trendIcon: getTrendIcon(kpis.pullRequest.averageTimeToFirstReview > 48 ? 'degrading' : 'stable'),
      subtitle: `To merge: ${(kpis.pullRequest.averageTimeToMerge / 24).toFixed(1)}d`,
      color: 'purple',
    },
    {
      id: 'throughput',
      title: 'Throughput',
      value: kpis.throughput.itemsCompleted.toString(),
      unit: 'items',
      trend: getThroughputTrend(kpis.throughput),
      trendIcon: getTrendIcon(getThroughputTrend(kpis.throughput)),
      subtitle: `WIP: ${kpis.throughput.workInProgress}`,
      color: 'orange',
    },
    {
      id: 'code-quality',
      title: 'Code Quality',
      value: (100 - kpis.codeQuality.codeChurn).toFixed(0),
      unit: '%',
      trend: kpis.codeQuality.codeChurn > 20 ? 'degrading' : 'stable',
      trendIcon: getTrendIcon(kpis.codeQuality.codeChurn > 20 ? 'degrading' : 'stable'),
      subtitle: `Churn: ${kpis.codeQuality.codeChurn.toFixed(1)}%`,
      color: 'red',
    },
    {
      id: 'merge-rate',
      title: 'Merge Rate',
      value: kpis.pullRequest.mergeRate.toFixed(0),
      unit: '%',
      trend: kpis.pullRequest.mergeRate >= 80 ? 'improving' : kpis.pullRequest.mergeRate >= 60 ? 'stable' : 'degrading',
      trendIcon: getTrendIcon(kpis.pullRequest.mergeRate >= 80 ? 'improving' : kpis.pullRequest.mergeRate >= 60 ? 'stable' : 'degrading'),
      subtitle: `Rework: ${kpis.pullRequest.reworkRate.toFixed(1)}%`,
      color: 'indigo',
    },
  ], [kpis]);

  return (
    <div className="kpi-cards" role="region" aria-label="Key Performance Indicators">
      <h3 className="kpi-cards-title">Key Metrics</h3>

      <div className="kpi-cards-grid">
        {cards.map((card) => (
          <div
            key={card.id}
            className={`kpi-card kpi-card-${card.color}`}
            role="article"
            aria-label={`${card.title}: ${card.value} ${card.unit}`}
          >
            <div className="kpi-card-header">
              <span className="kpi-card-title">{card.title}</span>
              <span className={`kpi-card-trend ${card.trend}`}>
                {card.trendIcon}
              </span>
            </div>

            <div className="kpi-card-body">
              <div className="kpi-card-value">
                {card.value}
                <span className="kpi-card-unit">{card.unit}</span>
              </div>
              <div className="kpi-card-subtitle">{card.subtitle}</div>
            </div>

            {/* Mini sparkline chart */}
            <div className="kpi-card-sparkline">
              <Sparkline data={getSparklineData(card.id, kpis)} trend={card.trend} />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

// ============================================================================
// Helper Functions
// ============================================================================

function getTrendIcon(trend: string): React.JSX.Element {
  const size = 16;
  switch (trend) {
    case 'increasing':
    case 'improving':
      return (
        <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <path d="M18 15L12 9L6 15" />
        </svg>
      );
    case 'decreasing':
    case 'degrading':
      return (
        <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <path d="M6 9L12 15L18 9" />
        </svg>
      );
    default:
      return (
        <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <path d="M5 12H19" />
        </svg>
      );
  }
}

function getThroughputTrend(throughput: DevelopmentKPIs['throughput']): 'improving' | 'stable' | 'degrading' {
  if (throughput.throughputTrend.length < 2) return 'stable';

  const recent = throughput.throughputTrend.slice(-3);
  const previous = throughput.throughputTrend.slice(-6, -3);

  const recentAvg = recent.reduce((a: number, b: number) => a + b, 0) / recent.length;
  const previousAvg = previous.reduce((a: number, b: number) => a + b, 0) / previous.length;

  if (recentAvg > previousAvg * 1.1) return 'improving';
  if (recentAvg < previousAvg * 0.9) return 'degrading';
  return 'stable';
}

function getSparklineData(cardId: string, kpis: DevelopmentKPIs): number[] {
  switch (cardId) {
    case 'velocity':
      return kpis.velocity.history;
    case 'throughput':
      return kpis.throughput.throughputTrend;
    case 'cycle-time':
      return [kpis.cycleTime.average, kpis.cycleTime.median, kpis.cycleTime.percentile90];
    default:
      return [50, 60, 55, 70, 65, 80];
  }
}

// ============================================================================
// Sparkline Component
// ============================================================================

interface SparklineProps {
  data: number[];
  trend: string;
}

const Sparkline: React.FC<SparklineProps> = ({ data, trend }) => {
  if (!data || data.length === 0) return null;

  const width = 100;
  const height = 30;
  const padding = 2;

  const max = Math.max(...data);
  const min = Math.min(...data);
  const range = max - min || 1;

  const points = data.map((value, index) => {
    const x = padding + (index / (data.length - 1)) * (width - 2 * padding);
    const y = height - padding - ((value - min) / range) * (height - 2 * padding);
    return `${x},${y}`;
  }).join(' ');

  const strokeColor = trend === 'improving' || trend === 'increasing'
    ? '#10b981'
    : trend === 'degrading' || trend === 'decreasing'
    ? '#ef4444'
    : '#6b7280';

  return (
    <svg width={width} height={height} className="kpi-card-sparkline-svg">
      <polyline
        points={points}
        fill="none"
        stroke={strokeColor}
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
};
