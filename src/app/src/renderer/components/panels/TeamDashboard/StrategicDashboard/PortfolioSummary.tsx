/**
 * Portfolio Summary Component
 * Displays high-level portfolio metrics at a glance
 */

import React, { useMemo } from 'react';
import type { DashboardMetrics, WorkItem } from '../../../../types/workItem';

interface PortfolioSummaryProps {
  metrics: DashboardMetrics;
}

export const PortfolioSummary: React.FC<PortfolioSummaryProps> = ({ metrics }) => {
  const summaryStats = useMemo(() => {
    const roi = metrics.roi;
    const dev = metrics.development;

    return {
      totalInvestment: roi.totalInvestment.cost,
      totalImpact: roi.totalImpact.total,
      roiRatio: roi.roi.ratio,
      roiPercentage: roi.roi.percentage,
      itemsCompleted: dev.throughput.itemsCompleted,
      velocity: dev.velocity.current,
      cycleTime: dev.cycleTime.average,
    };
  }, [metrics]);

  return (
    <div className="portfolio-summary" role="region" aria-label="Portfolio Summary">
      <h3 className="portfolio-summary-title">Portfolio Summary</h3>

      <div className="portfolio-summary-grid">
        <div className="portfolio-summary-card investment">
          <div className="portfolio-summary-icon">$</div>
          <div className="portfolio-summary-content">
            <div className="portfolio-summary-label">Total Investment</div>
            <div className="portfolio-summary-value">
              ${summaryStats.totalInvestment.toLocaleString(undefined, { maximumFractionDigits: 0 })}
            </div>
          </div>
        </div>

        <div className="portfolio-summary-card impact">
          <div className="portfolio-summary-icon">+</div>
          <div className="portfolio-summary-content">
            <div className="portfolio-summary-label">Total Impact</div>
            <div className="portfolio-summary-value">
              ${summaryStats.totalImpact.toLocaleString(undefined, { maximumFractionDigits: 0 })}
            </div>
          </div>
        </div>

        <div className="portfolio-summary-card roi">
          <div className="portfolio-summary-icon">%</div>
          <div className="portfolio-summary-content">
            <div className="portfolio-summary-label">ROI</div>
            <div className={`portfolio-summary-value ${summaryStats.roiPercentage >= 0 ? 'positive' : 'negative'}`}>
              {summaryStats.roiPercentage >= 0 ? '+' : ''}{summaryStats.roiPercentage.toFixed(1)}%
            </div>
          </div>
        </div>

        <div className="portfolio-summary-card velocity">
          <div className="portfolio-summary-icon">&gt;</div>
          <div className="portfolio-summary-content">
            <div className="portfolio-summary-label">Velocity</div>
            <div className="portfolio-summary-value">{summaryStats.velocity}</div>
            <div className="portfolio-summary-sub">pts/sprint</div>
          </div>
        </div>
      </div>
    </div>
  );
};
