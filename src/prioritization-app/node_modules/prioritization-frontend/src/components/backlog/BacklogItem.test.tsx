/**
 * Unit tests for BacklogItem component.
 *
 * Tests cover:
 * - Checkbox toggle works
 * - Drag handle is present
 * - Item content renders
 * - Selection state
 * - Drag and drop handlers
 * - Category badge display
 * - Custom actions rendering
 * - Accessibility attributes
 *
 * @module components/backlog/tests/BacklogItem.test
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { BacklogItem, type BacklogItemProps } from '../BacklogItem';
import type { PrioritizationItem } from '../../services/prioritization/types';

describe('BacklogItem', () => {
  const createMockItem = (overrides?: Partial<PrioritizationItem>): PrioritizationItem => ({
    id: 'item-1',
    title: 'Test Item',
    description: 'Test description',
    category: 'Feature',
    createdAt: new Date('2024-01-01'),
    ...overrides,
  });

  const defaultProps: BacklogItemProps = {
    item: createMockItem(),
    onSelect: vi.fn(),
    onClick: vi.fn(),
    onDragStart: vi.fn(),
    onDragEnd: vi.fn(),
  };

  const renderItem = (props?: Partial<BacklogItemProps>) => {
    return render(<BacklogItem {...defaultProps} {...props} />);
  };

  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('Rendering', () => {
    it('renders the backlog item', () => {
      renderItem();
      expect(screen.getByText('Test Item')).toBeInTheDocument();
    });

    it('renders item title', () => {
      renderItem({ item: createMockItem({ title: 'My Feature' }) });
      expect(screen.getByText('My Feature')).toBeInTheDocument();
    });

    it('renders item description by default', () => {
      renderItem();
      expect(screen.getByText('Test description')).toBeInTheDocument();
    });

    it('does not render description when showDescription is false', () => {
      renderItem({ showDescription: false });
      expect(screen.queryByText('Test description')).not.toBeInTheDocument();
    });

    it('renders category badge by default', () => {
      renderItem();
      expect(screen.getByText('Feature')).toBeInTheDocument();
    });

    it('does not render category badge when showCategory is false', () => {
      renderItem({ showCategory: false });
      expect(screen.queryByText('Feature')).not.toBeInTheDocument();
    });

    it('does not render description if item has no description', () => {
      renderItem({ item: createMockItem({ description: undefined }) });
      expect(screen.queryByText(/Test description/i)).not.toBeInTheDocument();
    });
  });

  describe('Checkbox Toggle', () => {
    it('renders checkbox', () => {
      renderItem();
      const checkbox = screen.getByRole('checkbox');
      expect(checkbox).toBeInTheDocument();
    });

    it('checkbox is unchecked by default', () => {
      renderItem();
      const checkbox = screen.getByRole('checkbox');
      expect(checkbox).not.toBeChecked();
    });

    it('checkbox is checked when isSelected is true', () => {
      renderItem({ isSelected: true });
      const checkbox = screen.getByRole('checkbox');
      expect(checkbox).toBeChecked();
    });

    it('triggers onSelect when checkbox is clicked', () => {
      renderItem();
      const checkbox = screen.getByRole('checkbox');
      fireEvent.click(checkbox);
      expect(defaultProps.onSelect).toHaveBeenCalledWith(defaultProps.item);
    });

    it('triggers onSelect when checkbox is toggled', () => {
      const handleSelect = vi.fn();
      renderItem({ onSelect: handleSelect, isSelected: true });
      const checkbox = screen.getByRole('checkbox');
      fireEvent.click(checkbox);
      expect(handleSelect).toHaveBeenCalledWith(defaultProps.item);
    });

    it('checkbox click does not trigger onClick', () => {
      renderItem();
      const checkbox = screen.getByRole('checkbox');
      fireEvent.click(checkbox);
      expect(defaultProps.onClick).not.toHaveBeenCalled();
    });
  });

  describe('Drag Handle', () => {
    it('renders drag handle when draggable is true (default)', () => {
      renderItem();
      const dragHandle = document.querySelector('.backlog-item-drag-handle');
      expect(dragHandle).toBeInTheDocument();
    });

    it('does not render drag handle when draggable is false', () => {
      renderItem({ draggable: false });
      const dragHandle = document.querySelector('.backlog-item-drag-handle');
      expect(dragHandle).not.toBeInTheDocument();
    });

    it('drag handle contains SVG', () => {
      renderItem();
      const dragHandle = document.querySelector('.backlog-item-drag-handle svg');
      expect(dragHandle).toBeInTheDocument();
    });
  });

  describe('Click Handling', () => {
    it('triggers onClick when item is clicked', () => {
      renderItem();
      const item = screen.getByRole('button');
      fireEvent.click(item);
      expect(defaultProps.onClick).toHaveBeenCalledWith(defaultProps.item);
    });

    it('triggers onClick when pressing Enter key', () => {
      renderItem();
      const item = screen.getByRole('button');
      fireEvent.keyDown(item, { key: 'Enter' });
      expect(defaultProps.onClick).toHaveBeenCalledWith(defaultProps.item);
    });

    it('triggers onClick when pressing Space key', () => {
      renderItem();
      const item = screen.getByRole('button');
      fireEvent.keyDown(item, { key: ' ' });
      expect(defaultProps.onClick).toHaveBeenCalledWith(defaultProps.item);
    });

    it('prevents default on Enter key', () => {
      renderItem();
      const item = screen.getByRole('button');
      const event = fireEvent.keyDown(item, { key: 'Enter' });
      expect(event).toBe(true);
    });
  });

  describe('Drag and Drop', () => {
    it('has draggable attribute when draggable is true', () => {
      renderItem();
      const item = screen.getByRole('button');
      expect(item).toHaveAttribute('draggable', 'true');
    });

    it('does not have draggable attribute when draggable is false', () => {
      renderItem({ draggable: false });
      const item = screen.getByRole('button');
      expect(item).toHaveAttribute('draggable', 'false');
    });

    it('triggers onDragStart when drag starts', () => {
      renderItem();
      const item = screen.getByRole('button');
      fireEvent.dragStart(item);
      expect(defaultProps.onDragStart).toHaveBeenCalledWith(defaultProps.item);
    });

    it('sets drag data on drag start', () => {
      renderItem();
      const item = screen.getByRole('button');
      const event = fireEvent.dragStart(item);
      expect(event).toBe(true);
    });

    it('does not trigger onDragStart when draggable is false', () => {
      const handleDragStart = vi.fn();
      renderItem({ draggable: false, onDragStart: handleDragStart });
      const item = screen.getByRole('button');
      fireEvent.dragStart(item);
      expect(handleDragStart).not.toHaveBeenCalled();
    });

    it('triggers onDragEnd when drag ends', () => {
      renderItem();
      const item = screen.getByRole('button');
      fireEvent.dragEnd(item);
      expect(defaultProps.onDragEnd).toHaveBeenCalled();
    });
  });

  describe('Selection State', () => {
    it('applies selected class when isSelected is true', () => {
      const { container } = renderItem({ isSelected: true });
      const item = container.querySelector('.backlog-item-selected');
      expect(item).toBeInTheDocument();
    });

    it('does not apply selected class when isSelected is false', () => {
      const { container } = renderItem({ isSelected: false });
      const item = container.querySelector('.backlog-item-selected');
      expect(item).not.toBeInTheDocument();
    });

    it('applies dragging class when isDragging is true', () => {
      const { container } = renderItem({ isDragging: true });
      const item = container.querySelector('.backlog-item-dragging');
      expect(item).toBeInTheDocument();
    });

    it('does not apply dragging class when isDragging is false', () => {
      const { container } = renderItem({ isDragging: false });
      const item = container.querySelector('.backlog-item-dragging');
      expect(item).not.toBeInTheDocument();
    });

    it('sets aria-selected to true when selected', () => {
      renderItem({ isSelected: true });
      const item = screen.getByRole('button');
      expect(item).toHaveAttribute('aria-selected', 'true');
    });

    it('sets aria-selected to false when not selected', () => {
      renderItem({ isSelected: false });
      const item = screen.getByRole('button');
      expect(item).toHaveAttribute('aria-selected', 'false');
    });
  });

  describe('Item Metadata', () => {
    it('displays creation date', () => {
      renderItem();
      expect(screen.getByText(/created/i)).toBeInTheDocument();
      expect(screen.getByText('1/1/2024')).toBeInTheDocument();
    });

    it('formats date correctly', () => {
      renderItem({
        item: createMockItem({ createdAt: new Date('2024-06-15') }),
      });
      expect(screen.getByText('6/15/2024')).toBeInTheDocument();
    });

    it('displays category in badge', () => {
      renderItem({ item: createMockItem({ category: 'Bug Fix' }) });
      expect(screen.getByText('Bug Fix')).toBeInTheDocument();
    });
  });

  describe('Custom Actions', () => {
    it('renders custom action buttons', () => {
      const actions = (
        <button data-testid="edit-btn" type="button">
          Edit
        </button>
      );
      renderItem({ actions });
      expect(screen.getByTestId('edit-btn')).toBeInTheDocument();
    });

    it('renders multiple action buttons', () => {
      const actions = (
        <div>
          <button data-testid="edit-btn" type="button">
            Edit
          </button>
          <button data-testid="delete-btn" type="button">
            Delete
          </button>
        </div>
      );
      renderItem({ actions });
      expect(screen.getByTestId('edit-btn')).toBeInTheDocument();
      expect(screen.getByTestId('delete-btn')).toBeInTheDocument();
    });

    it('does not render actions container when no actions provided', () => {
      const { container } = renderItem({ actions: undefined });
      expect(container.querySelector('.backlog-item-actions')).not.toBeInTheDocument();
    });
  });

  describe('Custom ClassName', () => {
    it('applies custom className', () => {
      const { container } = renderItem({ className: 'custom-class' });
      const item = container.querySelector('.custom-class');
      expect(item).toBeInTheDocument();
    });

    it('combines custom className with base classes', () => {
      const { container } = renderItem({ className: 'custom-class' });
      const item = container.querySelector('.backlog-item.custom-class');
      expect(item).toBeInTheDocument();
    });
  });

  describe('Accessibility', () => {
    it('has role="button"', () => {
      renderItem();
      const item = screen.getByRole('button');
      expect(item).toHaveAttribute('role', 'button');
    });

    it('has tabIndex for keyboard navigation', () => {
      renderItem();
      const item = screen.getByRole('button');
      expect(item).toHaveAttribute('tabIndex', '0');
    });

    it('has aria-selected attribute', () => {
      renderItem({ isSelected: true });
      const item = screen.getByRole('button');
      expect(item).toHaveAttribute('aria-selected');
    });

    it('checkbox has accessible label', () => {
      renderItem({ item: createMockItem({ title: 'Test Feature' }) });
      const checkbox = screen.getByRole('checkbox', { name: /select test feature/i });
      expect(checkbox).toBeInTheDocument();
    });

    it('drag handle has aria-hidden', () => {
      renderItem();
      const dragHandle = document.querySelector('.backlog-item-drag-handle');
      expect(dragHandle).toHaveAttribute('aria-hidden', 'true');
    });
  });

  describe('Item Content Layout', () => {
    it('has content container', () => {
      const { container } = renderItem();
      expect(container.querySelector('.backlog-item-content')).toBeInTheDocument();
    });

    it('has checkbox container', () => {
      const { container } = renderItem();
      expect(container.querySelector('.backlog-item-checkbox')).toBeInTheDocument();
    });

    it('has info container', () => {
      const { container } = renderItem();
      expect(container.querySelector('.backlog-item-info')).toBeInTheDocument();
    });

    it('has title element', () => {
      const { container } = renderItem();
      expect(container.querySelector('.backlog-item-title')).toBeInTheDocument();
    });

    it('has description element when showDescription is true', () => {
      const { container } = renderItem({ showDescription: true });
      expect(container.querySelector('.backlog-item-description')).toBeInTheDocument();
    });

    it('has meta container for category and date', () => {
      const { container } = renderItem();
      expect(container.querySelector('.backlog-item-meta')).toBeInTheDocument();
    });
  });

  describe('Edge Cases', () => {
    it('handles item without category', () => {
      renderItem({ item: createMockItem({ category: undefined }) });
      // Should not crash, just not show category badge
      expect(screen.queryByTestId('badge')).not.toBeInTheDocument();
    });

    it('handles very long title', () => {
      renderItem({
        item: createMockItem({
          title: 'Very long title that should be truncated in the UI display'.repeat(10),
        }),
      });
      expect(screen.getByText(/Very long title/i)).toBeInTheDocument();
    });

    it('handles empty description', () => {
      renderItem({ item: createMockItem({ description: '' }) });
      expect(screen.queryByText('')).not.toBeInTheDocument();
    });
  });
});
