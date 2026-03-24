/**
 * Unit tests for Button component.
 *
 * Tests cover:
 * - Rendering with children
 * - Click event handling
 * - Disabled state
 * - Variant classes application
 * - Loading state
 * - Icon rendering
 * - Accessibility attributes
 *
 * @module components/common/tests/Button.test
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { Button, type ButtonProps } from '../Button';

describe('Button', () => {
  const defaultProps: ButtonProps = {
    children: 'Click Me',
  };

  const renderButton = (props?: Partial<ButtonProps>) => {
    return render(<Button {...defaultProps} {...props} />);
  };

  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('Rendering', () => {
    it('renders with children', () => {
      renderButton({ children: 'Test Button' });
      expect(screen.getByRole('button', { name: 'Test Button' })).toBeInTheDocument();
    });

    it('renders with default variant (primary)', () => {
      const { container } = renderButton();
      const button = container.querySelector('.btn.btn-primary');
      expect(button).toBeInTheDocument();
    });

    it('renders with default size (md)', () => {
      const { container } = renderButton();
      const button = container.querySelector('.btn.btn-md');
      expect(button).toBeInTheDocument();
    });
  });

  describe('Variants', () => {
    it('applies primary variant classes', () => {
      const { container } = renderButton({ variant: 'primary' });
      const button = container.querySelector('.btn.btn-primary');
      expect(button).toBeInTheDocument();
    });

    it('applies secondary variant classes', () => {
      const { container } = renderButton({ variant: 'secondary' });
      const button = container.querySelector('.btn.btn-secondary');
      expect(button).toBeInTheDocument();
    });

    it('applies danger variant classes', () => {
      const { container } = renderButton({ variant: 'danger' });
      const button = container.querySelector('.btn.btn-danger');
      expect(button).toBeInTheDocument();
    });

    it('applies ghost variant classes', () => {
      const { container } = renderButton({ variant: 'ghost' });
      const button = container.querySelector('.btn.btn-ghost');
      expect(button).toBeInTheDocument();
    });
  });

  describe('Sizes', () => {
    it('applies small size classes', () => {
      const { container } = renderButton({ size: 'sm' });
      const button = container.querySelector('.btn.btn-sm');
      expect(button).toBeInTheDocument();
    });

    it('applies medium size classes', () => {
      const { container } = renderButton({ size: 'md' });
      const button = container.querySelector('.btn.btn-md');
      expect(button).toBeInTheDocument();
    });

    it('applies large size classes', () => {
      const { container } = renderButton({ size: 'lg' });
      const button = container.querySelector('.btn.btn-lg');
      expect(button).toBeInTheDocument();
    });
  });

  describe('Click Events', () => {
    it('handles click events', () => {
      const handleClick = vi.fn();
      renderButton({ onClick: handleClick });

      const button = screen.getByRole('button', { name: 'Click Me' });
      fireEvent.click(button);

      expect(handleClick).toHaveBeenCalledTimes(1);
    });

    it('passes event to click handler', () => {
      const handleClick = vi.fn();
      renderButton({ onClick: handleClick });

      const button = screen.getByRole('button', { name: 'Click Me' });
      fireEvent.click(button);

      expect(handleClick).toHaveBeenCalledWith(expect.objectContaining({
        type: 'click',
        target: expect.any(HTMLButtonElement),
      }));
    });
  });

  describe('Disabled State', () => {
    it('is disabled when disabled prop is true', () => {
      renderButton({ disabled: true });
      const button = screen.getByRole('button', { name: 'Click Me' });
      expect(button).toBeDisabled();
    });

    it('does not trigger click when disabled', () => {
      const handleClick = vi.fn();
      renderButton({ disabled: true, onClick: handleClick });

      const button = screen.getByRole('button', { name: 'Click Me' });
      fireEvent.click(button);

      expect(handleClick).not.toHaveBeenCalled();
    });

    it('applies disabled attribute', () => {
      renderButton({ disabled: true });
      const button = screen.getByRole('button', { name: 'Click Me' });
      expect(button).toHaveAttribute('disabled');
    });
  });

  describe('Loading State', () => {
    it('shows loading spinner when isLoading is true', () => {
      renderButton({ isLoading: true });
      const spinner = document.querySelector('.btn-spinner');
      expect(spinner).toBeInTheDocument();
    });

    it('is disabled when loading', () => {
      renderButton({ isLoading: true });
      const button = screen.getByRole('button', { name: 'Click Me' });
      expect(button).toBeDisabled();
    });

    it('has aria-busy attribute when loading', () => {
      renderButton({ isLoading: true });
      const button = screen.getByRole('button', { name: 'Click Me' });
      expect(button).toHaveAttribute('aria-busy', 'true');
    });

    it('does not show spinner when not loading', () => {
      renderButton({ isLoading: false });
      const spinner = document.querySelector('.btn-spinner');
      expect(spinner).not.toBeInTheDocument();
    });

    it('does not trigger click when loading', () => {
      const handleClick = vi.fn();
      renderButton({ isLoading: true, onClick: handleClick });

      const button = screen.getByRole('button', { name: 'Click Me' });
      fireEvent.click(button);

      expect(handleClick).not.toHaveBeenCalled();
    });
  });

  describe('Icons', () => {
    it('renders left icon when provided', () => {
      const testIcon = <span data-testid="left-icon">Left Icon</span>;
      renderButton({ leftIcon: testIcon });

      const icon = screen.getByTestId('left-icon');
      expect(icon).toBeInTheDocument();
    });

    it('renders right icon when provided', () => {
      const testIcon = <span data-testid="right-icon">Right Icon</span>;
      renderButton({ rightIcon: testIcon });

      const icon = screen.getByTestId('right-icon');
      expect(icon).toBeInTheDocument();
    });

    it('renders both left and right icons', () => {
      const leftIcon = <span data-testid="left-icon">Left</span>;
      const rightIcon = <span data-testid="right-icon">Right</span>;
      renderButton({ leftIcon, rightIcon });

      expect(screen.getByTestId('left-icon')).toBeInTheDocument();
      expect(screen.getByTestId('right-icon')).toBeInTheDocument();
    });
  });

  describe('Full Width', () => {
    it('applies full width class when fullWidth is true', () => {
      const { container } = renderButton({ fullWidth: true });
      const button = container.querySelector('.btn-full-width');
      expect(button).toBeInTheDocument();
    });

    it('does not apply full width class by default', () => {
      const { container } = renderButton();
      const button = container.querySelector('.btn-full-width');
      expect(button).not.toBeInTheDocument();
    });
  });

  describe('Custom ClassName', () => {
    it('applies custom className', () => {
      const { container } = renderButton({ className: 'custom-class' });
      const button = container.querySelector('.custom-class');
      expect(button).toBeInTheDocument();
    });

    it('combines custom className with base classes', () => {
      const { container } = renderButton({ className: 'custom-class' });
      const button = container.querySelector('.btn.custom-class');
      expect(button).toBeInTheDocument();
    });
  });

  describe('Accessibility', () => {
    it('has role="button"', () => {
      renderButton();
      const button = screen.getByRole('button');
      expect(button).toHaveAttribute('role', 'button');
    });

    it('has correct aria-busy when loading', () => {
      renderButton({ isLoading: true });
      const button = screen.getByRole('button');
      expect(button).toHaveAttribute('aria-busy', 'true');
    });

    it('has correct aria-busy when not loading', () => {
      renderButton({ isLoading: false });
      const button = screen.getByRole('button');
      expect(button).toHaveAttribute('aria-busy', 'false');
    });
  });

  describe('Extra Props', () => {
    it('passes through extra HTML button attributes', () => {
      renderButton({ type: 'submit', name: 'test-button' });
      const button = screen.getByRole('button', { name: 'Click Me' });
      expect(button).toHaveAttribute('type', 'submit');
      expect(button).toHaveAttribute('name', 'test-button');
    });

    it('passes through data attributes', () => {
      renderButton({ 'data-testid': 'test-button', 'data-custom': 'value' });
      const button = screen.getByTestId('test-button');
      expect(button).toHaveAttribute('data-custom', 'value');
    });
  });
});
