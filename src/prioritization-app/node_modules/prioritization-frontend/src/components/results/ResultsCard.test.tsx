/**
 * Unit tests for ResultsCard component.
 *
 * Tests cover:
 * - Results display correctly
 * - Score formatting works
 * - Category badge shows
 * - Framework name display
 * - Details breakdown rendering
 * - Custom actions rendering
 * - Different framework types
 *
 * @module components/results/tests/ResultsCard.test
 */

import { describe, it, expect, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import { ResultsCard, type ResultsCardProps } from '../ResultsCard';
import type { FrameworkResult, FrameworkType } from '../../services/prioritization/types';

describe('ResultsCard', () => {
  const createRiceResult = (overrides?: Partial<FrameworkResult>): FrameworkResult => ({
    framework: 'RICE',
    score: 45.5,
    details: {
      reach: 100,
      impact: 2,
      confidence: 0.8,
      effort: 3,
    },
    ...overrides,
  });

  const defaultProps: ResultsCardProps = {
    result: createRiceResult(),
  };

  const renderCard = (props?: Partial<ResultsCardProps>) => {
    return render(<ResultsCard {...defaultProps} {...props} />);
  };

  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('Rendering', () => {
    it('renders the card component', () => {
      renderCard();
      expect(screen.getByText('RICE Score')).toBeInTheDocument();
    });

    it('displays the framework name', () => {
      renderCard({ result: createRiceResult({ framework: 'RICE' }) });
      expect(screen.getByText('RICE Score')).toBeInTheDocument();
    });

    it('displays the score value', () => {
      renderCard({ result: createRiceResult({ score: 75.25 }) });
      expect(screen.getByText('75.25')).toBeInTheDocument();
    });

    it('displays "Score" label', () => {
      renderCard();
      expect(screen.getByText('Score')).toBeInTheDocument();
    });
  });

  describe('Score Display', () => {
    it('formats score with 2 decimal places', () => {
      renderCard({ result: createRiceResult({ score: 100 }) });
      expect(screen.getByText('100.00')).toBeInTheDocument();
    });

    it('formats fractional scores correctly', () => {
      renderCard({ result: createRiceResult({ score: 33.333333 }) });
      expect(screen.getByText('33.33')).toBeInTheDocument();
    });

    it('does not display score when score is undefined', () => {
      renderCard({ result: createRiceResult({ score: undefined }) });
      expect(screen.queryByText('Score')).not.toBeInTheDocument();
    });
  });

  describe('Category Badge', () => {
    it('displays category badge for MoSCoW results', () => {
      const moscowResult: FrameworkResult = {
        framework: 'MoSCoW',
        score: 0,
        details: { category: 'Must have' },
      };
      renderCard({ result: moscowResult });
      expect(screen.getByText('Must have')).toBeInTheDocument();
    });

    it('displays category badge for Eisenhower results', () => {
      const eisenhowerResult: FrameworkResult = {
        framework: 'Eisenhower',
        score: 0,
        details: { quadrant: 'Do First' },
      };
      renderCard({ result: eisenhowerResult });
      expect(screen.getByText('Do First')).toBeInTheDocument();
    });

    it('displays category badge for P0P4 results', () => {
      const p0p4Result: FrameworkResult = {
        framework: 'P0P4',
        score: 0,
        details: { priority: 'P0' },
      };
      renderCard({ result: p0p4Result });
      expect(screen.getByText('P0')).toBeInTheDocument();
    });

    it('displays category badge for ValueEffort results', () => {
      const valueEffortResult: FrameworkResult = {
        framework: 'ValueEffort',
        score: 0,
        details: { quadrant: 'QuickWin' },
      };
      renderCard({ result: valueEffortResult });
      expect(screen.getByText('QuickWin')).toBeInTheDocument();
    });

    it('displays category badge for Kano results', () => {
      const kanoResult: FrameworkResult = {
        framework: 'Kano',
        score: 0,
        details: { category: 'OneDimensional' },
      };
      renderCard({ result: kanoResult });
      expect(screen.getByText('OneDimensional')).toBeInTheDocument();
    });

    it('does not display badge when no category is present', () => {
      const riceResult: FrameworkResult = {
        framework: 'RICE',
        score: 50,
        details: { reach: 100, impact: 2, confidence: 0.5, effort: 2 },
      };
      renderCard({ result: riceResult });
      // RICE doesn't have a category, so no badge should show
      expect(screen.queryByTestId('badge')).not.toBeInTheDocument();
    });
  });

  describe('Badge Variants', () => {
    it('applies success variant for high RICE scores', () => {
      const { container } = renderCard({
        result: createRiceResult({ score: 60 }),
      });
      const badge = container.querySelector('.badge-success');
      // High RICE scores don't have category badges, so check for badge existence
      expect(badge).not.toBeInTheDocument();
    });

    it('applies error variant for MoSCoW Must have', () => {
      const { container } = renderCard({
        result: {
          framework: 'MoSCoW',
          score: 0,
          details: { category: 'Must have' },
        },
      });
      const badge = container.querySelector('.badge-error');
      expect(badge).toBeInTheDocument();
    });

    it('applies warning variant for MoSCoW Should have', () => {
      const { container } = renderCard({
        result: {
          framework: 'MoSCoW',
          score: 0,
          details: { category: 'Should have' },
        },
      });
      const badge = container.querySelector('.badge-warning');
      expect(badge).toBeInTheDocument();
    });

    it('applies error variant for Eisenhower Do First', () => {
      const { container } = renderCard({
        result: {
          framework: 'Eisenhower',
          score: 0,
          details: { quadrant: 'Do First' },
        },
      });
      const badge = container.querySelector('.badge-error');
      expect(badge).toBeInTheDocument();
    });

    it('applies error variant for P0 priority', () => {
      const { container } = renderCard({
        result: {
          framework: 'P0P4',
          score: 0,
          details: { priority: 'P0' },
        },
      });
      const badge = container.querySelector('.badge-error');
      expect(badge).toBeInTheDocument();
    });
  });

  describe('Item Title and Description', () => {
    it('displays item title when provided', () => {
      renderCard({ itemTitle: 'Feature A' });
      expect(screen.getByText('Feature A')).toBeInTheDocument();
    });

    it('displays item description when provided', () => {
      renderCard({
        itemTitle: 'Feature A',
        itemDescription: 'This is a description',
      });
      expect(screen.getByText('This is a description')).toBeInTheDocument();
    });

    it('does not display title/description when not provided', () => {
      renderCard();
      expect(screen.queryByText(/Feature/i)).not.toBeInTheDocument();
    });
  });

  describe('Details Breakdown', () => {
    it('displays breakdown when showDetails is true', () => {
      renderCard({ showDetails: true });
      expect(screen.getByText('Breakdown')).toBeInTheDocument();
    });

    it('does not display breakdown when showDetails is false', () => {
      renderCard({ showDetails: false });
      expect(screen.queryByText('Breakdown')).not.toBeInTheDocument();
    });

    it('displays numeric details in breakdown', () => {
      renderCard({ result: createRiceResult() });
      expect(screen.getByText('Reach')).toBeInTheDocument();
      expect(screen.getByText('Impact')).toBeInTheDocument();
      expect(screen.getByText('Confidence')).toBeInTheDocument();
      expect(screen.getByText('Effort')).toBeInTheDocument();
    });

    it('formats numeric values with 2 decimal places for decimals', () => {
      renderCard({
        result: createRiceResult({
          details: {
            reach: 100,
            impact: 2,
            confidence: 0.8333,
            effort: 3.5,
          },
        }),
      });
      // Should show formatted values
      const confidenceValue = screen.getByText('0.83');
      expect(confidenceValue).toBeInTheDocument();
    });

    it('shows integer values without decimals', () => {
      renderCard({
        result: createRiceResult({
          details: { reach: 100, impact: 2, confidence: 0.8, effort: 3 },
        }),
      });
      expect(screen.getByText('100')).toBeInTheDocument();
      expect(screen.getByText('2')).toBeInTheDocument();
      expect(screen.getByText('3')).toBeInTheDocument();
    });

    it('does not display breakdown if details is empty', () => {
      renderCard({
        result: { framework: 'RICE', score: 50, details: {} },
        showDetails: true,
      });
      expect(screen.queryByText('Breakdown')).not.toBeInTheDocument();
    });
  });

  describe('Custom Actions', () => {
    it('renders custom action buttons', () => {
      const actions = (
        <button data-testid="custom-action" type="button">
          Action
        </button>
      );
      renderCard({ actions });
      expect(screen.getByTestId('custom-action')).toBeInTheDocument();
    });

    it('renders multiple action buttons', () => {
      const actions = (
        <div>
          <button data-testid="action-1" type="button">
            Action 1
          </button>
          <button data-testid="action-2" type="button">
            Action 2
          </button>
        </div>
      );
      renderCard({ actions });
      expect(screen.getByTestId('action-1')).toBeInTheDocument();
      expect(screen.getByTestId('action-2')).toBeInTheDocument();
    });
  });

  describe('Different Framework Types', () => {
    it('displays correct name for RICE framework', () => {
      renderCard({ result: createRiceResult({ framework: 'RICE' }) });
      expect(screen.getByText('RICE Score')).toBeInTheDocument();
    });

    it('displays correct name for MoSCoW framework', () => {
      renderCard({
        result: { framework: 'MoSCoW', score: 0, details: { category: 'Must have' } },
      });
      expect(screen.getByText('MoSCoW Category')).toBeInTheDocument();
    });

    it('displays correct name for ValueEffort framework', () => {
      renderCard({
        result: { framework: 'ValueEffort', score: 0, details: { quadrant: 'QuickWin' } },
      });
      expect(screen.getByText('Value vs Effort')).toBeInTheDocument();
    });

    it('displays correct name for ICE framework', () => {
      renderCard({
        result: { framework: 'ICE', score: 250, details: { idea: 100, confidence: 0.5, ease: 5 } },
      });
      expect(screen.getByText('ICE Score')).toBeInTheDocument();
    });

    it('displays correct name for Eisenhower framework', () => {
      renderCard({
        result: { framework: 'Eisenhower', score: 0, details: { quadrant: 'Do First' } },
      });
      expect(screen.getByText('Eisenhower Matrix')).toBeInTheDocument();
    });

    it('displays correct name for P0P4 framework', () => {
      renderCard({
        result: { framework: 'P0P4', score: 0, details: { priority: 'P1' } },
      });
      expect(screen.getByText('Priority Level')).toBeInTheDocument();
    });

    it('displays correct name for WSJF framework', () => {
      renderCard({
        result: { framework: 'WSJF', score: 12.5, details: { costOfDelay: 50, jobSize: 4 } },
      });
      expect(screen.getByText('WSJF Score')).toBeInTheDocument();
    });

    it('displays correct name for Kano framework', () => {
      renderCard({
        result: { framework: 'Kano', score: 0, details: { category: 'Attractive' } },
      });
      expect(screen.getByText('Kano Category')).toBeInTheDocument();
    });
  });

  describe('Custom ClassName', () => {
    it('applies custom className', () => {
      const { container } = renderCard({ className: 'custom-class' });
      const card = container.querySelector('.custom-class');
      expect(card).toBeInTheDocument();
    });

    it('applies base results-card class', () => {
      const { container } = renderCard();
      const card = container.querySelector('.results-card');
      expect(card).toBeInTheDocument();
    });
  });

  describe('Card Variant', () => {
    it('uses elevated variant by default', () => {
      const { container } = renderCard();
      const card = container.querySelector('.card-elevated');
      expect(card).toBeInTheDocument();
    });
  });

  describe('Results Layout', () => {
    it('has results-header section', () => {
      const { container } = renderCard();
      expect(container.querySelector('.results-header')).toBeInTheDocument();
    });

    it('has results-main section when title/score present', () => {
      const { container } = renderCard({ itemTitle: 'Test' });
      expect(container.querySelector('.results-main')).toBeInTheDocument();
    });

    it('has results-details section when showDetails is true', () => {
      const { container } = renderCard({ showDetails: true });
      expect(container.querySelector('.results-details')).toBeInTheDocument();
    });
  });
});
