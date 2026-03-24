/**
 * Comparison Table component for comparing multiple prioritization items.
 * Displays items side-by-side with their scores and details.
 *
 * @module components/results/ComparisonTable
 */

import React from 'react';
import { FrameworkResult, FrameworkType } from '../../services/prioritization/types';
import { Card, Badge } from '../common';

/**
 * Table column definition.
 */
export interface TableColumn {
  /** Column key */
  key: string;
  /** Column header label */
  header: string;
  /** Column width (CSS value) */
  width?: string;
  /** Whether the column is sortable */
  sortable?: boolean;
  /** Custom cell renderer */
  render?: (value: unknown, item: TableItem) => React.ReactNode;
}

/**
 * Table item interface.
 */
export interface TableItem {
  /** Unique identifier */
  id: string;
  /** Item title */
  title: string;
  /** Item description */
  description?: string;
  /** Framework results for this item */
  results?: Partial<Record<FrameworkType, FrameworkResult>>;
  /** Additional data for rendering */
  [key: string]: unknown;
}

/**
 * Sort configuration.
 */
export interface SortConfig {
  /** Column key to sort by */
  key: string;
  /** Sort direction */
  direction: 'asc' | 'desc';
}

/**
 * Props for the ComparisonTable component.
 */
export interface ComparisonTableProps {
  /** Table title */
  title?: string;
  /** Items to display in the table */
  items: TableItem[];
  /** Column definitions */
  columns?: TableColumn[];
  /** Currently selected item ID */
  selectedItemId?: string;
  /** Item selection handler */
  onItemSelected?: (itemId: string) => void;
  /** Sort configuration */
  sortConfig?: SortConfig;
  /** Sort change handler */
  onSortChange?: (config: SortConfig) => void;
  /** Custom class name */
  className?: string;
  /** Show row numbers */
  showRowNumbers?: boolean;
  /** Show item descriptions */
  showDescriptions?: boolean;
  /** Empty state message */
  emptyMessage?: string;
}

/**
 * Comparison Table component for comparing multiple items.
 *
 * This component displays a sortable table comparing multiple prioritization
 * items with their scores, categories, and other details.
 *
 * @param props - ComparisonTable component props
 * @returns Rendered comparison table
 *
 * @example
 * ```tsx
 * <ComparisonTable
 *   items={[
 *     {
 *       id: '1',
 *       title: 'Feature A',
 *       description: 'Description A',
 *       results: { RICE: { score: 45.5, ... } },
 *     },
 *   ]}
 *   onItemSelected={handleSelect}
 *   selectedItemId="1"
 * />
 * ```
 */
export function ComparisonTable({
  title = 'Comparison',
  items,
  columns,
  selectedItemId,
  onItemSelected,
  sortConfig,
  onSortChange,
  className = '',
  showRowNumbers = true,
  showDescriptions = true,
  emptyMessage = 'No items to compare',
}: ComparisonTableProps): React.JSX.Element {
  // Default columns if not provided
  const defaultColumns: TableColumn[] = [
    {
      key: 'title',
      header: 'Item',
      width: '250px',
      render: (_, item) => (
        <div className="table-item-cell">
          <span className="table-item-title">{item.title}</span>
          {showDescriptions && item.description && (
            <span className="table-item-description">{item.description}</span>
          )}
        </div>
      ),
    },
    {
      key: 'score',
      header: 'Score',
      width: '100px',
      sortable: true,
      render: (_, item) => {
        const score = getScoreFromResults(item.results);
        return score !== null ? (
          <span className="table-score">{score.toFixed(2)}</span>
        ) : (
          <span className="table-no-data">-</span>
        );
      },
    },
    {
      key: 'category',
      header: 'Category',
      width: '150px',
      render: (_, item) => {
        const category = getCategoryFromResults(item.results);
        return category ? (
          <Badge variant={getCategoryBadgeVariant(category)} size="sm">
            {category}
          </Badge>
        ) : (
          <span className="table-no-data">-</span>
        );
      },
    },
  ];

  const effectiveColumns = columns || defaultColumns;

  // Handle sort
  const handleSort = (column: TableColumn) => {
    if (!column.sortable || !onSortChange) return;

    const newDirection: 'asc' | 'desc' =
      sortConfig?.key === column.key && sortConfig.direction === 'desc'
        ? 'asc'
        : 'desc';

    onSortChange({ key: column.key, direction: newDirection });
  };

  return (
    <Card title={title} variant="default" className={`comparison-table ${className}`}>
      <div className="table-container">
        {items.length === 0 ? (
          <div className="table-empty">
            <p>{emptyMessage}</p>
          </div>
        ) : (
          <table className="table">
            <thead className="table-head">
              <tr className="table-row">
                {showRowNumbers && (
                  <th className="table-header table-header-number">#</th>
                )}
                {effectiveColumns.map((column) => (
                  <th
                    key={column.key}
                    className={`table-header ${column.sortable ? 'table-header-sortable' : ''}`}
                    style={{ width: column.width }}
                    onClick={() => handleSort(column)}
                    role={column.sortable ? 'button' : undefined}
                    tabIndex={column.sortable ? 0 : undefined}
                    aria-sort={
                      sortConfig?.key === column.key
                        ? sortConfig.direction === 'asc'
                          ? 'ascending'
                          : 'descending'
                        : 'none'
                    }
                  >
                    {column.header}
                    {column.sortable && (
                      <span className="table-sort-icon">
                        {sortConfig?.key === column.key ? (
                          sortConfig.direction === 'asc' ? (
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                              <path d="M12 19V5M5 12l7-7 7 7" />
                            </svg>
                          ) : (
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                              <path d="M12 5v14M5 12l7 7 7-7" />
                            </svg>
                          )
                        ) : (
                          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                            <path d="M7 15l5 5 5-5M7 9l5-5 5 5" />
                          </svg>
                        )}
                      </span>
                    )}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="table-body">
              {items.map((item, index) => (
                <tr
                  key={item.id}
                  className={`table-row ${selectedItemId === item.id ? 'table-row-selected' : ''}`}
                  onClick={() => onItemSelected?.(item.id)}
                  role={onItemSelected ? 'button' : undefined}
                  tabIndex={onItemSelected ? 0 : undefined}
                >
                  {showRowNumbers && (
                    <td className="table-cell table-cell-number">{index + 1}</td>
                  )}
                  {effectiveColumns.map((column) => (
                    <td
                      key={column.key}
                      className="table-cell"
                      style={{ width: column.width }}
                    >
                      {column.render
                        ? column.render(item[column.key], item)
                        : (item[column.key] as React.ReactNode)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </Card>
  );
}

/**
 * Extract score from results object.
 */
function getScoreFromResults(
  results?: Partial<Record<FrameworkType, FrameworkResult>>
): number | null {
  if (!results) return null;

  // Try to get score from any available result
  const frameworks: FrameworkType[] = [
    'RICE',
    'WSJF',
    'ICE',
    'P0P4',
    'MoSCoW',
    'Eisenhower',
    'ValueEffort',
    'Kano',
  ];

  for (const framework of frameworks) {
    const result = results[framework];
    if (result?.score !== undefined) {
      return result.score;
    }
  }

  return null;
}

/**
 * Extract category from results object.
 */
function getCategoryFromResults(
  results?: Partial<Record<FrameworkType, FrameworkResult>>
): string | null {
  if (!results) return null;

  const frameworkCategories: Record<FrameworkType, string | null> = {
    RICE: null,
    ICE: null,
    WSJF: null,
    MoSCoW: (results.MoSCoW?.details?.category as string) || null,
    Eisenhower: (results.Eisenhower?.details?.quadrant as string) || null,
    ValueEffort: (results.ValueEffort?.details?.quadrant as string) || null,
    P0P4: (results.P0P4?.details?.priority as string) || null,
    Kano: (results.Kano?.details?.category as string) || null,
  };

  for (const key of Object.keys(frameworkCategories)) {
    const category = frameworkCategories[key as FrameworkType];
    if (category) return category;
  }

  return null;
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

export default ComparisonTable;
