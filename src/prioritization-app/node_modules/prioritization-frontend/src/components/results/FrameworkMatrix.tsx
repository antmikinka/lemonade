/**
 * Framework Matrix component for 2x2 matrix visualizations.
 * Displays Value-Effort and Eisenhower matrix quadrants.
 *
 * @module components/results/FrameworkMatrix
 */

import React from 'react';
import { Card } from '../common';

/**
 * Matrix item interface for plotting on the matrix.
 */
export interface MatrixItem {
  /** Unique identifier */
  id: string;
  /** Item title */
  title: string;
  /** X-axis value (e.g., Effort) */
  x: number;
  /** Y-axis value (e.g., Value) */
  y: number;
  /** Optional color for the item */
  color?: string;
  /** Optional category or quadrant */
  category?: string;
}

/**
 * Matrix type for different 2x2 frameworks.
 */
export type MatrixType = 'value-effort' | 'eisenhower';

/**
 * Props for the FrameworkMatrix component.
 */
export interface FrameworkMatrixProps {
  /** Type of matrix to display */
  type?: MatrixType;
  /** Items to plot on the matrix */
  items: MatrixItem[];
  /** X-axis label */
  xLabel?: string;
  /** Y-axis label */
  yLabel?: string;
  /** Quadrant labels (bottom-left, bottom-right, top-left, top-right) */
  quadrantLabels?: [string, string, string, string];
  /** Show grid lines */
  showGrid?: boolean;
  /** Show axis labels */
  showLabels?: boolean;
  /** Click handler for matrix items */
  onItemClick?: (item: MatrixItem) => void;
  /** Custom class name */
  className?: string;
  /** Matrix title */
  title?: string;
}

/**
 * Default quadrant labels for Value-Effort matrix.
 */
const VALUE_EFFORT_QUADRANTS: [string, string, string, string] = [
  'Fill In',    // Low Value, Low Effort (bottom-left)
  'Avoid',      // Low Value, High Effort (bottom-right)
  'Quick Win',  // High Value, Low Effort (top-left)
  'Major Project', // High Value, High Effort (top-right)
];

/**
 * Default quadrant labels for Eisenhower matrix.
 */
const EISENHOWER_QUADRANTS: [string, string, string, string] = [
  'Eliminate',  // Not Urgent, Not Important (bottom-left)
  'Delegate',   // Urgent, Not Important (bottom-right)
  'Schedule',   // Not Urgent, Important (top-left)
  'Do First',   // Urgent, Important (top-right)
];

/**
 * Framework Matrix component for 2x2 visualizations.
 *
 * This component renders an interactive 2x2 matrix for visualizing
 * prioritization frameworks like Value-Effort or Eisenhower Matrix.
 *
 * @param props - FrameworkMatrix component props
 * @returns Rendered matrix component
 *
 * @example
 * ```tsx
 * <FrameworkMatrix
 *   type="value-effort"
 *   items={[
 *     { id: '1', title: 'Feature A', x: 3, y: 8 },
 *     { id: '2', title: 'Feature B', x: 7, y: 5 },
 *   ]}
 *   onItemClick={handleItemClick}
 * />
 * ```
 */
export function FrameworkMatrix({
  type = 'value-effort',
  items,
  xLabel,
  yLabel,
  quadrantLabels,
  showGrid = true,
  showLabels = true,
  onItemClick,
  className = '',
  title,
}: FrameworkMatrixProps): React.JSX.Element {
  // Determine default labels based on matrix type
  const defaultXLabel = type === 'value-effort' ? 'Effort' : 'Urgent';
  const defaultYLabel = type === 'value-effort' ? 'Value' : 'Important';
  const defaultQuadrants = type === 'value-effort'
    ? VALUE_EFFORT_QUADRANTS
    : EISENHOWER_QUADRANTS;

  const effectiveXLabel = xLabel || defaultXLabel;
  const effectiveYLabel = yLabel || defaultYLabel;
  const effectiveQuadrants = quadrantLabels || defaultQuadrants;

  return (
    <Card
      title={title || `${type === 'value-effort' ? 'Value vs Effort' : 'Eisenhower'} Matrix`}
      variant="default"
      className={`framework-matrix ${className}`}
    >
      <div className="matrix-container">
        {/* Matrix Grid */}
        <div className="matrix-grid">
          {/* Quadrant backgrounds */}
          <div className="matrix-quadrants">
            <div className="matrix-quadrant matrix-quadrant-top-left">
              <span className="matrix-quadrant-label">{effectiveQuadrants[2]}</span>
            </div>
            <div className="matrix-quadrant matrix-quadrant-top-right">
              <span className="matrix-quadrant-label">{effectiveQuadrants[3]}</span>
            </div>
            <div className="matrix-quadrant matrix-quadrant-bottom-left">
              <span className="matrix-quadrant-label">{effectiveQuadrants[0]}</span>
            </div>
            <div className="matrix-quadrant matrix-quadrant-bottom-right">
              <span className="matrix-quadrant-label">{effectiveQuadrants[1]}</span>
            </div>
          </div>

          {/* Grid lines */}
          {showGrid && (
            <>
              <div className="matrix-grid-line matrix-grid-line-vertical" />
              <div className="matrix-grid-line matrix-grid-line-horizontal" />
            </>
          )}

          {/* Axis labels */}
          {showLabels && (
            <>
              <div className="matrix-axis-label matrix-axis-label-top">
                {effectiveYLabel}: High
              </div>
              <div className="matrix-axis-label matrix-axis-label-bottom">
                {effectiveYLabel}: Low
              </div>
              <div className="matrix-axis-label matrix-axis-label-left">
                {effectiveXLabel}: Low
              </div>
              <div className="matrix-axis-label matrix-axis-label-right">
                {effectiveXLabel}: High
              </div>
            </>
          )}

          {/* Items plotted on matrix */}
          <div className="matrix-items">
            {items.map((item) => {
              // Normalize coordinates to percentage (0-100)
              // Assuming x and y are on a 1-10 scale
              const normalizedX = Math.min(100, Math.max(0, ((item.x - 1) / 9) * 100));
              const normalizedY = Math.min(100, Math.max(0, ((item.y - 1) / 9) * 100));

              return (
                <div
                  key={item.id}
                  className={`matrix-item ${onItemClick ? 'matrix-item-clickable' : ''}`}
                  style={{
                    left: `${normalizedX}%`,
                    bottom: `${normalizedY}%`,
                    backgroundColor: item.color || 'var(--color-accent)',
                  }}
                  onClick={() => onItemClick?.(item)}
                  role={onItemClick ? 'button' : undefined}
                  tabIndex={onItemClick ? 0 : undefined}
                  aria-label={`${item.title}: ${effectiveXLabel} ${item.x}, ${effectiveYLabel} ${item.y}`}
                >
                  <span className="matrix-item-dot" />
                  <span className="matrix-item-label">{item.title}</span>
                </div>
              );
            })}
          </div>
        </div>

        {/* Legend */}
        {items.length > 0 && (
          <div className="matrix-legend">
            <span className="matrix-legend-title">Items: {items.length}</span>
          </div>
        )}
      </div>
    </Card>
  );
}

export default FrameworkMatrix;
