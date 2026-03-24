/**
 * Unit tests for Input component.
 *
 * Tests cover:
 * - Rendering with label
 * - Error state display
 * - Value changes trigger onChange
 * - Different input types
 * - Helper text display
 * - Required field indication
 * - Addon rendering
 * - Accessibility attributes
 *
 * @module components/common/tests/Input.test
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { Input, type InputProps } from '../Input';

describe('Input', () => {
  const defaultProps: InputProps = {
    label: 'Test Label',
    placeholder: 'Enter text...',
  };

  const renderInput = (props?: Partial<InputProps>) => {
    return render(<Input {...defaultProps} {...props} />);
  };

  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('Rendering', () => {
    it('renders with label', () => {
      renderInput({ label: 'Test Label' });
      expect(screen.getByText('Test Label')).toBeInTheDocument();
    });

    it('renders with placeholder', () => {
      renderInput({ placeholder: 'Enter text...' });
      expect(screen.getByPlaceholderText('Enter text...')).toBeInTheDocument();
    });

    it('renders with default type (text)', () => {
      renderInput();
      const input = screen.getByPlaceholderText('Enter text...');
      expect(input).toHaveAttribute('type', 'text');
    });

    it('renders with default size (md)', () => {
      const { container } = renderInput();
      const input = container.querySelector('.input-md');
      expect(input).toBeInTheDocument();
    });
  });

  describe('Input Types', () => {
    it('renders text input', () => {
      renderInput({ type: 'text' });
      const input = screen.getByPlaceholderText('Enter text...');
      expect(input).toHaveAttribute('type', 'text');
    });

    it('renders email input', () => {
      renderInput({ type: 'email' });
      const input = screen.getByPlaceholderText('Enter text...');
      expect(input).toHaveAttribute('type', 'email');
    });

    it('renders number input', () => {
      renderInput({ type: 'number' });
      const input = screen.getByPlaceholderText('Enter text...');
      expect(input).toHaveAttribute('type', 'number');
    });

    it('renders password input', () => {
      renderInput({ type: 'password' });
      const input = screen.getByPlaceholderText('Enter text...');
      expect(input).toHaveAttribute('type', 'password');
    });

    it('renders search input', () => {
      renderInput({ type: 'search' });
      const input = screen.getByPlaceholderText('Enter text...');
      expect(input).toHaveAttribute('type', 'search');
    });

    it('renders tel input', () => {
      renderInput({ type: 'tel' });
      const input = screen.getByPlaceholderText('Enter text...');
      expect(input).toHaveAttribute('type', 'tel');
    });

    it('renders url input', () => {
      renderInput({ type: 'url' });
      const input = screen.getByPlaceholderText('Enter text...');
      expect(input).toHaveAttribute('type', 'url');
    });

    it('renders textarea when type is textarea', () => {
      renderInput({ type: 'textarea', rows: 5 });
      const textarea = screen.getByPlaceholderText('Enter text...');
      expect(textarea.tagName).toBe('TEXTAREA');
      expect(textarea).toHaveAttribute('rows', '5');
    });
  });

  describe('Value Changes', () => {
    it('triggers onChange when value changes', () => {
      const handleChange = vi.fn();
      renderInput({ onChange: handleChange });

      const input = screen.getByPlaceholderText('Enter text...');
      fireEvent.change(input, { target: { value: 'New value' } });

      expect(handleChange).toHaveBeenCalledTimes(1);
      expect(handleChange).toHaveBeenCalledWith(
        expect.objectContaining({
          target: expect.objectContaining({
            value: 'New value',
          }),
        })
      );
    });

    it('receives correct event target value', () => {
      const handleChange = vi.fn();
      renderInput({ onChange: handleChange });

      const input = screen.getByPlaceholderText('Enter text...');
      fireEvent.change(input, { target: { value: 'Test value' } });

      expect(handleChange).toHaveBeenCalledWith(
        expect.objectContaining({
          target: expect.objectContaining({
            value: 'Test value',
          }),
        })
      );
    });

    it('handles controlled component value', () => {
      renderInput({ value: 'Controlled value' });
      const input = screen.getByPlaceholderText('Enter text...');
      expect(input).toHaveValue('Controlled value');
    });
  });

  describe('Error State', () => {
    it('displays error message when error prop is provided', () => {
      renderInput({ error: 'This is an error message' });
      expect(screen.getByText('This is an error message')).toBeInTheDocument();
    });

    it('applies error class when error is present', () => {
      const { container } = renderInput({ error: 'Error' });
      const input = container.querySelector('.input-error');
      expect(input).toBeInTheDocument();
    });

    it('sets aria-invalid to true when error is present', () => {
      renderInput({ error: 'Error' });
      const input = screen.getByPlaceholderText('Enter text...');
      expect(input).toHaveAttribute('aria-invalid', 'true');
    });

    it('links error message with aria-describedby', () => {
      renderInput({ error: 'Error message' });
      const input = screen.getByPlaceholderText('Enter text...');
      const errorId = input.getAttribute('aria-describedby');
      expect(errorId).toBeDefined();
      const errorElement = document.getElementById(errorId!);
      expect(errorElement).toHaveTextContent('Error message');
    });

    it('hides helper text when error is present', () => {
      renderInput({ error: 'Error', helperText: 'Helper text' });
      expect(screen.getByText('Error')).toBeInTheDocument();
      expect(screen.queryByText('Helper text')).not.toBeInTheDocument();
    });

    it('applies error role for accessibility', () => {
      renderInput({ error: 'Error message' });
      const errorMessage = screen.getByRole('alert');
      expect(errorMessage).toHaveTextContent('Error message');
    });
  });

  describe('Helper Text', () => {
    it('displays helper text when provided', () => {
      renderInput({ helperText: 'This is helper text' });
      expect(screen.getByText('This is helper text')).toBeInTheDocument();
    });

    it('does not display helper text when not provided', () => {
      renderInput();
      expect(screen.queryByText(/helper/i)).not.toBeInTheDocument();
    });

    it('links helper text with aria-describedby', () => {
      renderInput({ helperText: 'Helper text' });
      const input = screen.getByPlaceholderText('Enter text...');
      const helperId = input.getAttribute('aria-describedby');
      expect(helperId).toBeDefined();
      const helperElement = document.getElementById(helperId!);
      expect(helperElement).toHaveTextContent('Helper text');
    });
  });

  describe('Required Field', () => {
    it('shows required asterisk when required is true', () => {
      renderInput({ required: true });
      const asterisk = document.querySelector('.input-required');
      expect(asterisk).toBeInTheDocument();
      expect(asterisk).toHaveTextContent('*');
    });

    it('does not show required asterisk when not required', () => {
      renderInput({ required: false });
      const asterisk = document.querySelector('.input-required');
      expect(asterisk).not.toBeInTheDocument();
    });

    it('sets aria-required to true when required', () => {
      renderInput({ required: true });
      const input = screen.getByPlaceholderText('Enter text...');
      expect(input).toHaveAttribute('aria-required', 'true');
    });

    it('sets aria-required to false when not required', () => {
      renderInput({ required: false });
      const input = screen.getByPlaceholderText('Enter text...');
      expect(input).toHaveAttribute('aria-required', 'false');
    });
  });

  describe('Sizes', () => {
    it('applies small size classes', () => {
      const { container } = renderInput({ size: 'sm' });
      const input = container.querySelector('.input-sm');
      expect(input).toBeInTheDocument();
    });

    it('applies medium size classes', () => {
      const { container } = renderInput({ size: 'md' });
      const input = container.querySelector('.input-md');
      expect(input).toBeInTheDocument();
    });

    it('applies large size classes', () => {
      const { container } = renderInput({ size: 'lg' });
      const input = container.querySelector('.input-lg');
      expect(input).toBeInTheDocument();
    });
  });

  describe('Addons', () => {
    it('renders left addon when provided', () => {
      const leftAddon = <span data-testid="left-addon">Left</span>;
      renderInput({ leftAddon });
      expect(screen.getByTestId('left-addon')).toBeInTheDocument();
    });

    it('renders right addon when provided', () => {
      const rightAddon = <span data-testid="right-addon">Right</span>;
      renderInput({ rightAddon });
      expect(screen.getByTestId('right-addon')).toBeInTheDocument();
    });

    it('renders both left and right addons', () => {
      const leftAddon = <span data-testid="left-addon">Left</span>;
      const rightAddon = <span data-testid="right-addon">Right</span>;
      renderInput({ leftAddon, rightAddon });

      expect(screen.getByTestId('left-addon')).toBeInTheDocument();
      expect(screen.getByTestId('right-addon')).toBeInTheDocument();
    });

    it('applies has-addon class when addon is present', () => {
      const { container } = renderInput({ leftAddon: <span>Left</span> });
      const input = container.querySelector('.input-has-addon');
      expect(input).toBeInTheDocument();
    });
  });

  describe('Full Width', () => {
    it('applies full width class when fullWidth is true', () => {
      const { container } = renderInput({ fullWidth: true });
      const input = container.querySelector('.input-full-width');
      expect(input).toBeInTheDocument();
    });

    it('applies full width by default', () => {
      const { container } = renderInput();
      const input = container.querySelector('.input-full-width');
      expect(input).toBeInTheDocument();
    });

    it('does not apply full width when fullWidth is false', () => {
      const { container } = renderInput({ fullWidth: false });
      const input = container.querySelector('.input-full-width');
      expect(input).not.toBeInTheDocument();
    });
  });

  describe('Custom ClassName', () => {
    it('applies custom className', () => {
      const { container } = renderInput({ className: 'custom-class' });
      const input = container.querySelector('.custom-class');
      expect(input).toBeInTheDocument();
    });

    it('combines custom className with base classes', () => {
      const { container } = renderInput({ className: 'custom-class' });
      const input = container.querySelector('.input-field.custom-class');
      expect(input).toBeInTheDocument();
    });
  });

  describe('Accessibility', () => {
    it('has label associated with input via htmlFor', () => {
      const { container } = renderInput({ label: 'Test Label' });
      const label = container.querySelector('label');
      const input = screen.getByPlaceholderText('Enter text...');
      expect(label).toHaveAttribute('for', input?.id);
    });

    it('generates unique id for input', () => {
      const { container } = renderInput();
      const input = screen.getByPlaceholderText('Enter text...');
      expect(input?.id).toBeDefined();
      expect(input?.id).toMatch(/^[0-9a-f-]+$/i);
    });

    it('uses provided id when specified', () => {
      renderInput({ id: 'custom-input-id' });
      const input = screen.getByPlaceholderText('Enter text...');
      expect(input?.id).toBe('custom-input-id');
    });
  });

  describe('Extra Props', () => {
    it('passes through extra HTML input attributes', () => {
      renderInput({
        disabled: true,
        readOnly: true,
        minLength: 5,
        maxLength: 100,
      });

      const input = screen.getByPlaceholderText('Enter text...');
      expect(input).toBeDisabled();
      expect(input).toHaveAttribute('readOnly');
      expect(input).toHaveAttribute('minLength', '5');
      expect(input).toHaveAttribute('maxLength', '100');
    });

    it('passes through data attributes', () => {
      renderInput({ 'data-testid': 'test-input', 'data-custom': 'value' });
      const input = screen.getByTestId('test-input');
      expect(input).toHaveAttribute('data-custom', 'value');
    });
  });

  describe('Textarea Specific', () => {
    it('renders textarea with correct rows', () => {
      renderInput({ type: 'textarea', rows: 6 });
      const textarea = screen.getByPlaceholderText('Enter text...');
      expect(textarea).toHaveAttribute('rows', '6');
    });

    it('textarea triggers onChange', () => {
      const handleChange = vi.fn();
      renderInput({ type: 'textarea', onChange: handleChange });

      const textarea = screen.getByPlaceholderText('Enter text...');
      fireEvent.change(textarea, { target: { value: 'New textarea value' } });

      expect(handleChange).toHaveBeenCalledTimes(1);
    });
  });
});
