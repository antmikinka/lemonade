/**
 * Unit tests for Select component.
 *
 * Tests cover:
 * - Options rendering correctly
 * - Selection triggers onChange
 * - Disabled state
 * - Label and helper text display
 * - Error state display
 * - Grouped options
 * - Placeholder and empty options
 * - Accessibility attributes
 *
 * @module components/common/tests/Select.test
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import { Select, type SelectProps, type SelectOption } from '../Select';

describe('Select', () => {
  const defaultOptions: SelectOption[] = [
    { value: 'option1', label: 'Option 1' },
    { value: 'option2', label: 'Option 2' },
    { value: 'option3', label: 'Option 3' },
  ];

  const defaultProps: SelectProps = {
    label: 'Test Select',
    options: defaultOptions,
  };

  const renderSelect = (props?: Partial<SelectProps>) => {
    return render(<Select {...defaultProps} {...props} />);
  };

  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('Rendering', () => {
    it('renders with label', () => {
      renderSelect({ label: 'Test Label' });
      expect(screen.getByText('Test Label')).toBeInTheDocument();
    });

    it('renders select element', () => {
      renderSelect();
      const select = screen.getByRole('combobox');
      expect(select).toBeInTheDocument();
    });

    it('renders all options', () => {
      renderSelect({ options: defaultOptions });
      const select = screen.getByRole('combobox');

      expect(screen.getByText('Option 1')).toBeInTheDocument();
      expect(screen.getByText('Option 2')).toBeInTheDocument();
      expect(screen.getByText('Option 3')).toBeInTheDocument();
    });

    it('renders empty option by default', () => {
      renderSelect();
      const select = screen.getByRole('combobox');
      expect(screen.getByText('Select...')).toBeInTheDocument();
    });
  });

  describe('Options', () => {
    it('renders options with correct values', () => {
      const options: SelectOption[] = [
        { value: '1', label: 'One' },
        { value: '2', label: 'Two' },
        { value: '3', label: 'Three' },
      ];
      renderSelect({ options });

      const select = screen.getByRole('combobox') as HTMLSelectElement;
      expect(select.options[1]).toHaveValue('1');
      expect(select.options[2]).toHaveValue('2');
      expect(select.options[3]).toHaveValue('3');
    });

    it('renders options with correct labels', () => {
      renderSelect();
      expect(screen.getByText('Option 1')).toBeInTheDocument();
      expect(screen.getByText('Option 2')).toBeInTheDocument();
      expect(screen.getByText('Option 3')).toBeInTheDocument();
    });

    it('renders disabled options', () => {
      const options: SelectOption[] = [
        { value: '1', label: 'Enabled' },
        { value: '2', label: 'Disabled', disabled: true },
        { value: '3', label: 'Also Enabled' },
      ];
      renderSelect({ options });

      const select = screen.getByRole('combobox') as HTMLSelectElement;
      expect(select.options[1]).not.toBeDisabled();
      expect(select.options[2]).toBeDisabled();
      expect(select.options[3]).not.toBeDisabled();
    });
  });

  describe('Selection and onChange', () => {
    it('triggers onChange when selection changes', () => {
      const handleChange = vi.fn();
      renderSelect({ onChange: handleChange });

      const select = screen.getByRole('combobox');
      select.focus();
      screen.selectOptions(select, 'option2');

      expect(handleChange).toHaveBeenCalledTimes(1);
      expect(handleChange).toHaveBeenCalledWith(
        expect.objectContaining({
          target: expect.objectContaining({
            value: 'option2',
          }),
        })
      );
    });

    it('receives correct selected value', () => {
      const handleChange = vi.fn();
      renderSelect({ onChange: handleChange });

      const select = screen.getByRole('combobox');
      screen.selectOptions(select, 'option3');

      expect(handleChange).toHaveBeenCalledWith(
        expect.objectContaining({
          target: expect.objectContaining({
            value: 'option3',
          }),
        })
      );
    });

    it('has correct value when defaultValue is set', () => {
      renderSelect({ defaultValue: 'option2' });
      const select = screen.getByRole('combobox') as HTMLSelectElement;
      expect(select.value).toBe('option2');
    });

    it('has correct value when value is set (controlled)', () => {
      renderSelect({ value: 'option1' });
      const select = screen.getByRole('combobox') as HTMLSelectElement;
      expect(select.value).toBe('option1');
    });
  });

  describe('Disabled State', () => {
    it('applies disabled attribute when disabled is true', () => {
      renderSelect({ disabled: true });
      const select = screen.getByRole('combobox');
      expect(select).toBeDisabled();
    });

    it('does not trigger onChange when disabled', () => {
      const handleChange = vi.fn();
      renderSelect({ disabled: true, onChange: handleChange });

      const select = screen.getByRole('combobox');
      screen.selectOptions(select, 'option2');

      expect(handleChange).not.toHaveBeenCalled();
    });

    it('applies disabled class when disabled', () => {
      const { container } = renderSelect({ disabled: true });
      const select = container.querySelector('.select-field:disabled');
      expect(select).toBeInTheDocument();
    });
  });

  describe('Error State', () => {
    it('displays error message when error prop is provided', () => {
      renderSelect({ error: 'This is an error' });
      expect(screen.getByText('This is an error')).toBeInTheDocument();
    });

    it('applies error class when error is present', () => {
      const { container } = renderSelect({ error: 'Error' });
      const select = container.querySelector('.select-error');
      expect(select).toBeInTheDocument();
    });

    it('sets aria-invalid to true when error is present', () => {
      renderSelect({ error: 'Error' });
      const select = screen.getByRole('combobox');
      expect(select).toHaveAttribute('aria-invalid', 'true');
    });

    it('links error message with aria-describedby', () => {
      renderSelect({ error: 'Error message' });
      const select = screen.getByRole('combobox');
      const errorId = select.getAttribute('aria-describedby');
      expect(errorId).toBeDefined();
      const errorElement = document.getElementById(errorId!);
      expect(errorElement).toHaveTextContent('Error message');
    });

    it('hides helper text when error is present', () => {
      renderSelect({ error: 'Error', helperText: 'Helper text' });
      expect(screen.getByText('Error')).toBeInTheDocument();
      expect(screen.queryByText('Helper text')).not.toBeInTheDocument();
    });

    it('applies error role for accessibility', () => {
      renderSelect({ error: 'Error message' });
      const errorMessage = screen.getByRole('alert');
      expect(errorMessage).toHaveTextContent('Error message');
    });
  });

  describe('Helper Text', () => {
    it('displays helper text when provided', () => {
      renderSelect({ helperText: 'This is helper text' });
      expect(screen.getByText('This is helper text')).toBeInTheDocument();
    });

    it('does not display helper text when not provided', () => {
      renderSelect();
      expect(screen.queryByText(/helper/i)).not.toBeInTheDocument();
    });

    it('links helper text with aria-describedby', () => {
      renderSelect({ helperText: 'Helper text' });
      const select = screen.getByRole('combobox');
      const helperId = select.getAttribute('aria-describedby');
      expect(helperId).toBeDefined();
      const helperElement = document.getElementById(helperId!);
      expect(helperElement).toHaveTextContent('Helper text');
    });
  });

  describe('Required Field', () => {
    it('shows required asterisk when required is true', () => {
      renderSelect({ required: true });
      const asterisk = document.querySelector('.select-required');
      expect(asterisk).toBeInTheDocument();
      expect(asterisk).toHaveTextContent('*');
    });

    it('does not show required asterisk when not required', () => {
      renderSelect({ required: false });
      const asterisk = document.querySelector('.select-required');
      expect(asterisk).not.toBeInTheDocument();
    });

    it('sets aria-required to true when required', () => {
      renderSelect({ required: true });
      const select = screen.getByRole('combobox');
      expect(select).toHaveAttribute('aria-required', 'true');
    });
  });

  describe('Sizes', () => {
    it('applies small size classes', () => {
      const { container } = renderSelect({ size: 'sm' });
      const select = container.querySelector('.select-sm');
      expect(select).toBeInTheDocument();
    });

    it('applies medium size classes', () => {
      const { container } = renderSelect({ size: 'md' });
      const select = container.querySelector('.select-md');
      expect(select).toBeInTheDocument();
    });

    it('applies large size classes', () => {
      const { container } = renderSelect({ size: 'lg' });
      const select = container.querySelector('.select-lg');
      expect(select).toBeInTheDocument();
    });
  });

  describe('Placeholder', () => {
    it('renders placeholder as disabled option', () => {
      renderSelect({ placeholder: 'Choose an option' });
      const placeholder = screen.getByText('Choose an option');
      expect(placeholder).toBeInTheDocument();
      expect(placeholder.tagName).toBe('OPTION');
    });

    it('placeholder is disabled', () => {
      renderSelect({ placeholder: 'Choose an option' });
      const select = screen.getByRole('combobox') as HTMLSelectElement;
      const placeholderOption = select.options[0];
      expect(placeholderOption).toBeDisabled();
    });
  });

  describe('Empty Option', () => {
    it('renders empty option when allowEmpty is true (default)', () => {
      renderSelect();
      expect(screen.getByText('Select...')).toBeInTheDocument();
    });

    it('does not render empty option when allowEmpty is false', () => {
      renderSelect({ allowEmpty: false });
      expect(screen.queryByText('Select...')).not.toBeInTheDocument();
    });

    it('uses custom emptyLabel', () => {
      renderSelect({ emptyLabel: 'None' });
      expect(screen.getByText('None')).toBeInTheDocument();
    });
  });

  describe('Grouped Options', () => {
    it('renders grouped options with optgroup', () => {
      const groupedOptions: SelectOption[] = [
        { value: '1', label: 'One', group: 'Numbers' },
        { value: '2', label: 'Two', group: 'Numbers' },
        { value: 'a', label: 'A', group: 'Letters' },
        { value: 'b', label: 'B', group: 'Letters' },
      ];
      renderSelect({ options: groupedOptions, groupBy: 'group' });

      expect(screen.getByLabelText('Numbers')).toBeInTheDocument();
      expect(screen.getByLabelText('Letters')).toBeInTheDocument();
    });

    it('groups options under correct optgroup labels', () => {
      const groupedOptions: SelectOption[] = [
        { value: '1', label: 'Apple', group: 'Fruits' },
        { value: '2', label: 'Carrot', group: 'Vegetables' },
      ];
      renderSelect({ options: groupedOptions, groupBy: 'group' });

      const fruitsGroup = screen.getByLabelText('Fruits');
      const vegetablesGroup = screen.getByLabelText('Vegetables');
      expect(fruitsGroup).toBeInTheDocument();
      expect(vegetablesGroup).toBeInTheDocument();
    });

    it('puts ungrouped options in "Other" group', () => {
      const groupedOptions: SelectOption[] = [
        { value: '1', label: 'Grouped', group: 'Group' },
        { value: '2', label: 'Ungrouped' },
      ];
      renderSelect({ options: groupedOptions, groupBy: 'group' });

      expect(screen.getByLabelText('Group')).toBeInTheDocument();
      expect(screen.getByLabelText('Other')).toBeInTheDocument();
    });
  });

  describe('Full Width', () => {
    it('applies full width class when fullWidth is true', () => {
      const { container } = renderSelect({ fullWidth: true });
      const select = container.querySelector('.select-full-width');
      expect(select).toBeInTheDocument();
    });

    it('applies full width by default', () => {
      const { container } = renderSelect();
      const select = container.querySelector('.select-full-width');
      expect(select).toBeInTheDocument();
    });

    it('does not apply full width when fullWidth is false', () => {
      const { container } = renderSelect({ fullWidth: false });
      const select = container.querySelector('.select-full-width');
      expect(select).not.toBeInTheDocument();
    });
  });

  describe('Custom ClassName', () => {
    it('applies custom className', () => {
      const { container } = renderSelect({ className: 'custom-class' });
      const select = container.querySelector('.custom-class');
      expect(select).toBeInTheDocument();
    });

    it('combines custom className with base classes', () => {
      const { container } = renderSelect({ className: 'custom-class' });
      const select = container.querySelector('.select-field.custom-class');
      expect(select).toBeInTheDocument();
    });
  });

  describe('Accessibility', () => {
    it('has label associated with select via htmlFor', () => {
      const { container } = renderSelect({ label: 'Test Label' });
      const label = container.querySelector('label');
      const select = screen.getByRole('combobox');
      expect(label).toHaveAttribute('for', select?.id);
    });

    it('generates unique id for select', () => {
      const { container } = renderSelect();
      const select = screen.getByRole('combobox');
      expect(select?.id).toBeDefined();
      expect(select?.id).toMatch(/^[0-9a-f-]+$/i);
    });

    it('uses provided id when specified', () => {
      renderSelect({ id: 'custom-select-id' });
      const select = screen.getByRole('combobox');
      expect(select?.id).toBe('custom-select-id');
    });

    it('has correct role combobox', () => {
      renderSelect();
      const select = screen.getByRole('combobox');
      expect(select).toHaveAttribute('role', 'combobox');
    });
  });

  describe('Extra Props', () => {
    it('passes through extra HTML select attributes', () => {
      renderSelect({
        multiple: true,
        name: 'test-select',
        required: true,
      });

      const select = screen.getByRole('combobox');
      expect(select).toHaveAttribute('multiple');
      expect(select).toHaveAttribute('name', 'test-select');
    });

    it('passes through data attributes', () => {
      renderSelect({ 'data-testid': 'test-select', 'data-custom': 'value' });
      const select = screen.getByTestId('test-select');
      expect(select).toHaveAttribute('data-custom', 'value');
    });
  });
});
