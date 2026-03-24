/**
 * Results Card component for displaying prioritization results.
 * Shows single item results with score and category information.
 *
 * @module components/results/ResultsCard
 */

import React from 'react';
import { FrameworkResult, FrameworkType } from '../../services/prioritization/types';
import { Card, Badge } from '../common';

/**
 * Props for the ResultsCard component.
 */
export interface ResultsCardProps {
  /** The framework result to display */
  result: FrameworkResult;
  /** Optional item title */
  itemTitle?: string;
  /** Optional item description */
  itemDescription?: string;
  /** Whether to show detailed breakdown */
  showDetails?: boolean;
  /** Custom class name */
  className?: string;
  /** Action buttons to display in header */
  actions?: React.ReactNode;
}

/**
 * Get a human-readable framework name.
 */
function getFrameworkName(framework: FrameworkType): string {
  const names: Record<FrameworkType, string> = {
    RICE: 'RICE Score',
    MoSCoW: 'MoSCoW Category',
    ValueEffort: 'Value vs Effort',
    ICE: 'ICE Score',
    Eisenhower: 'Eisenhower Matrix',
    P0P4: 'Priority Level',
    WSJF: 'WSJF Score',
    Kano: 'Kano Category',
  };
  return names[framework] || framework;
}

/**
 * Get the appropriate badge variant based on score or category.
 */
function getBadgeVariant(
  framework: FrameworkType,
  score?: number,
  details?: Record<string, unknown>
): 'success' | 'warning' | 'error' | 'primary' | 'default' {
  if (framework === 'RICE' || framework === 'ICE' || framework === 'WSJF') {
    if (!score) return 'default';
    if (framework === 'RICE' && score >= 50) return 'success';
    if (framework === 'RICE' && score >= 20) return 'warning';
    if (framework === 'ICE' && score >= 500) return 'success';
    if (framework === 'ICE' && score >= 200) return 'warning';
    if (framework === 'WSJF' && score >= 10) return 'success';
    if (framework === 'WSJF' && score >= 5) return 'warning';
    return 'default';
  }

  if (framework === 'MoSCoW') {
    const category = details?.category as string;
    if (category === 'Must have') return 'error';
    if (category === 'Should have') return 'warning';
    if (category === 'Could have') return 'primary';
    return 'default';
  }

  if (framework === 'Eisenhower') {
    const quadrant = details?.quadrant as string;
    if (quadrant === 'Do First') return 'error';
    if (quadrant === 'Schedule') return 'warning';
    if (quadrant === 'Delegate') return 'primary';
    return 'default';
  }

  if (framework === 'P0P4') {
    const priority = details?.priority as string;
    if (priority === 'P0') return 'error';
    if (priority === 'P1') return 'warning';
    if (priority === 'P2') return 'primary';
    return 'default';
  }

  if (framework === 'ValueEffort') {
    const quadrant = details?.quadrant as string;
    if (quadrant === 'QuickWin') return 'success';
    if (quadrant === 'MajorProject') return 'warning';
    if (quadrant === 'FillIn') return 'primary';
    return 'default';
  }

  if (framework === 'Kano') {
    const category = details?.category as string;
    if (category === 'OneDimensional' || category === 'Attractive') return 'success';
    if (category === 'MustBe') return 'warning';
    return 'default';
  }

  return 'default';
}

/**
 * Results Card component for displaying prioritization results.
 *
 * This component displays the results of a prioritization calculation,
 * including the score, category/quadrant, and optional detailed breakdown.
 *
 * @param props - ResultsCard component props
 * @returns Rendered results card
 *
 * @example
 * ```tsx
 * <ResultsCard
 *   result={{
 *     framework: 'RICE',
 *     score: 45.5,
 *     details: { reach: 100, impact: 2, confidence: 0.8, effort: 3 },
 *   }}
 *   itemTitle="Add Dark Mode"
 *   showDetails
 * />
 * ```
 */
export function ResultsCard({
  result,
  itemTitle,
  itemDescription,
  showDetails = true,
  className = '',
  actions,
}: ResultsCardProps): React.JSX.Element {
  const badgeVariant = getBadgeVariant(
    result.framework,
    result.score,
    result.details
  );

  const getCategoryDisplay = (): string => {
    const details = result.details;

    switch (result.framework) {
      case 'MoSCoW':
        return (details.category as string) || '';
      case 'Eisenhower':
        return (details.quadrant as string) || '';
      case 'ValueEffort':
        return (details.quadrant as string) || '';
      case 'P0P4':
        return (details.priority as string) || '';
      case 'Kano':
        return (details.category as string) || '';
      default:
        return '';
    }
  };

  const categoryDisplay = getCategoryDisplay();

  return (
    <Card
      variant="elevated"
      className={`results-card ${className}`}
      headerAction={actions}
    >
      <div className="results-header">
        <div className="results-framework">
          <span className="results-framework-label">{getFrameworkName(result.framework)}</span>
        </div>
        {categoryDisplay && (
          <Badge variant={badgeVariant} size="lg">
            {categoryDisplay}
          </Badge>
        )}
      </div>

      {(itemTitle || result.score !== undefined) && (
        <div className="results-main">
          {itemTitle && <h4 className="results-title">{itemTitle}</h4>}
          {itemDescription && (
            <p className="results-description">{itemDescription}</p>
          )}
          {result.score !== undefined && (
            <div className="results-score">
              <span className="results-score-value">
                {result.score.toFixed(2)}
              </span>
              <span className="results-score-label">Score</span>
            </div>
          )}
        </div>
      )}

      {showDetails && Object.keys(result.details).length > 0 && (
        <div className="results-details">
          <h5 className="results-details-title">Breakdown</h5>
          <dl className="results-details-list">
            {Object.entries(result.details).map(([key, value]) => {
              if (typeof value === 'number') {
                return (
                  <div key={key} className="results-detail-item">
                    <dt className="results-detail-label">
                      {formatLabel(key)}
                    </dt>
                    <dd className="results-detail-value">
                      {typeof value === 'number' && (value % 1 !== 0)
                        ? value.toFixed(2)
                        : value}
                    </dd>
                  </div>
                );
              }
              return null;
            })}
          </dl>
        </div>
      )}
    </Card>
  );
}

/**
 * Format a camelCase key to a readable label.
 */
function formatLabel(key: string): string {
  return key
    .replace(/([A-Z])/g, ' $1')
    .replace(/^./, (str) => str.toUpperCase());
}

export default ResultsCard;
