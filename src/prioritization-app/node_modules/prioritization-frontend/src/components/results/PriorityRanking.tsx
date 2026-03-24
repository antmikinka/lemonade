/**
 * Priority Ranking component for displaying sorted prioritization results.
 * Shows items ranked by their priority score or category.
 *
 * @module components/results/PriorityRanking
 */

import React from 'react';
import { FrameworkType } from '../../services/prioritization/types';
import { Card, Badge } from '../common';

/**
 * Ranked item interface.
 */
export interface RankedItem {
  /** Unique identifier */
  id: string;
  /** Item title */
  title: string;
  /** Item description */
  description?: string;
  /** Rank position (1 = highest priority) */
  rank: number;
  /** Score value */
  score?: number;
  /** Category or quadrant */
  category?: string;
  /** Framework that produced this result */
  framework?: FrameworkType;
  /** Color for the rank indicator */
  color?: string;
  /** Change in rank (for trend display) */
  rankChange?: number;
}

/**
 * Props for the PriorityRanking component.
 */
export interface PriorityRankingProps {
  /** Ranking title */
  title?: string;
  /** Items to display in ranking */
  items: RankedItem[];
  /** Framework type for the ranking */
  framework?: FrameworkType;
  /** Currently selected item ID */
  selectedItemId?: string;
  /** Item selection handler */
  onItemSelected?: (itemId: string) => void;
  /** Number of items to show (for pagination) */
  limit?: number;
  /** Show rank change indicators */
  showRankChange?: boolean;
  /** Show score values */
  showScores?: boolean;
  /** Show category badges */
  showCategories?: boolean;
  /** Custom class name */
  className?: string;
  /** Empty state message */
  emptyMessage?: string;
  /** Highlight top N items */
  highlightTop?: number;
}

/**
 * Priority Ranking component for displaying sorted results.
 *
 * This component displays a ranked list of prioritization items,
 * showing their position, score, and optional trend indicators.
 *
 * @param props - PriorityRanking component props
 * @returns Rendered priority ranking
 *
 * @example
 * ```tsx
 * <PriorityRanking
 *   items={[
 *     { id: '1', title: 'Feature A', rank: 1, score: 95.5, category: 'Must have' },
 *     { id: '2', title: 'Feature B', rank: 2, score: 82.3, category: 'Should have' },
 *   ]}
 *   onItemSelected={handleSelect}
 *   highlightTop={3}
 * />
 * ```
 */
export function PriorityRanking({
  title = 'Priority Ranking',
  items,
  framework,
  selectedItemId,
  onItemSelected,
  limit,
  showRankChange = false,
  showScores = true,
  showCategories = true,
  className = '',
  emptyMessage = 'No items to rank',
  highlightTop = 3,
}: PriorityRankingProps): React.JSX.Element {
  // Apply limit if specified
  const displayItems = limit ? items.slice(0, limit) : items;

  return (
    <Card
      title={title}
      variant="default"
      className={`priority-ranking ${className}`}
    >
      {items.length === 0 ? (
        <div className="ranking-empty">
          <p>{emptyMessage}</p>
        </div>
      ) : (
        <div className="ranking-list">
          {displayItems.map((item, index) => (
            <div
              key={item.id}
              className={`ranking-item ${selectedItemId === item.id ? 'ranking-item-selected' : ''} ${
                index < highlightTop ? 'ranking-item-highlighted' : ''
              }`}
              onClick={() => onItemSelected?.(item.id)}
              role={onItemSelected ? 'button' : undefined}
              tabIndex={onItemSelected ? 0 : undefined}
            >
              {/* Rank Number */}
              <div className="ranking-rank">
                <span
                  className={`ranking-rank-number ${index < highlightTop ? 'ranking-rank-highlight' : ''}`}
                  style={{ color: item.color }}
                >
                  {item.rank}
                </span>
                {showRankChange && item.rankChange && (
                  <span
                    className={`ranking-rank-change ${item.rankChange > 0 ? 'ranking-rank-up' : 'ranking-rank-down'}`}
                  >
                    {item.rankChange > 0 ? (
                      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <path d="M18 15l-6-6-6 6" />
                      </svg>
                    ) : (
                      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <path d="M6 9l6 6 6-6" />
                      </svg>
                    )}
                    {Math.abs(item.rankChange)}
                  </span>
                )}
              </div>

              {/* Item Info */}
              <div className="ranking-info">
                <h4 className="ranking-title">{item.title}</h4>
                {item.description && (
                  <p className="ranking-description">{item.description}</p>
                )}
                <div className="ranking-meta">
                  {framework && (
                    <span className="ranking-framework">
                      {getFrameworkName(framework)}
                    </span>
                  )}
                  {showCategories && item.category && (
                    <Badge variant={getCategoryBadgeVariant(item.category)} size="sm">
                      {item.category}
                    </Badge>
                  )}
                </div>
              </div>

              {/* Score */}
              {showScores && (
                <div className="ranking-score">
                  <span className="ranking-score-value">
                    {item.score !== undefined ? item.score.toFixed(1) : '-'}
                  </span>
                  <span className="ranking-score-label">pts</span>
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      {limit && items.length > limit && (
        <div className="ranking-footer">
          <p>
            Showing {displayItems.length} of {items.length} items
          </p>
        </div>
      )}
    </Card>
  );
}

/**
 * Get framework display name.
 */
function getFrameworkName(framework: FrameworkType): string {
  const names: Record<FrameworkType, string> = {
    RICE: 'RICE',
    MoSCoW: 'MoSCoW',
    ValueEffort: 'Value vs Effort',
    ICE: 'ICE',
    Eisenhower: 'Eisenhower',
    P0P4: 'P0-P4',
    WSJF: 'WSJF',
    Kano: 'Kano',
  };
  return names[framework] || framework;
}

/**
 * Get badge variant based on category.
 */
function getCategoryBadgeVariant(category: string): 'success' | 'warning' | 'error' | 'primary' | 'default' {
  const highPriorityCategories = [
    'Must have',
    'Do First',
    'Quick Win',
    'P0',
    'P1',
    'OneDimensional',
    'Attractive',
  ];
  const mediumPriorityCategories = [
    'Should have',
    'Schedule',
    'MajorProject',
    'P2',
    'MustBe',
  ];

  if (highPriorityCategories.some((c) => category.includes(c))) return 'success';
  if (mediumPriorityCategories.some((c) => category.includes(c))) return 'warning';
  return 'primary';
}

export default PriorityRanking;
