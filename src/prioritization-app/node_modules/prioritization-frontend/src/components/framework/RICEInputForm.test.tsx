/**
 * Unit tests for RICEInputForm component.
 *
 * Tests cover:
 * - All inputs render correctly
 * - Form submission works
 * - Validation errors display
 * - onChange handler triggers
 * - Submit button state
 * - Loading state
 * - Custom class names
 *
 * @module components/framework/tests/RICEInputForm.test
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { RICEInputForm, type RICEInputFormProps } from '../RICEInputForm';

describe('RICEInputForm', () => {
  const defaultProps: RICEInputFormProps = {
    onSubmit: vi.fn(),
    onChange: vi.fn(),
  };

  const renderForm = (props?: Partial<RICEInputFormProps>) => {
    return render(<RICEInputForm {...defaultProps} {...props} />);
  };

  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('Rendering', () => {
    it('renders the form with title', () => {
      renderForm();
      expect(screen.getByText('RICE Scoring')).toBeInTheDocument();
    });

    it('renders Reach input field', () => {
      renderForm();
      const reachInput = screen.getByLabelText(/reach/i);
      expect(reachInput).toBeInTheDocument();
      expect(reachInput).toHaveAttribute('type', 'number');
    });

    it('renders Impact select dropdown', () => {
      renderForm();
      const impactSelect = screen.getByLabelText(/impact/i);
      expect(impactSelect).toBeInTheDocument();
    });

    it('renders Confidence slider', () => {
      renderForm();
      const confidenceSlider = screen.getByLabelText(/confidence/i);
      expect(confidenceSlider).toBeInTheDocument();
    });

    it('renders Effort input field', () => {
      renderForm();
      const effortInput = screen.getByLabelText(/effort/i);
      expect(effortInput).toBeInTheDocument();
      expect(effortInput).toHaveAttribute('type', 'number');
    });

    it('renders submit button by default', () => {
      renderForm();
      expect(
        screen.getByRole('button', { name: /calculate rice score/i })
      ).toBeInTheDocument();
    });

    it('does not render submit button when showSubmitButton is false', () => {
      renderForm({ showSubmitButton: false });
      expect(
        screen.queryByRole('button', { name: /calculate rice score/i })
      ).not.toBeInTheDocument();
    });
  });

  describe('Impact Options', () => {
    it('renders all impact level options', () => {
      renderForm();
      const impactSelect = screen.getByLabelText(/impact/i) as HTMLSelectElement;

      expect(impactSelect.options).toHaveLength(6); // Empty + 5 options
      expect(screen.getByText(/3 - massive impact/i)).toBeInTheDocument();
      expect(screen.getByText(/2 - high impact/i)).toBeInTheDocument();
      expect(screen.getByText(/1 - medium impact/i)).toBeInTheDocument();
      expect(screen.getByText(/0\.5 - low impact/i)).toBeInTheDocument();
      expect(screen.getByText(/0\.25 - minimal impact/i)).toBeInTheDocument();
    });

    it('has placeholder option as first option', () => {
      renderForm();
      const impactSelect = screen.getByLabelText(/impact/i) as HTMLSelectElement;
      expect(impactSelect.options[0]).toHaveTextContent(/select impact level/i);
    });
  });

  describe('Value Changes', () => {
    it('triggers onChange when Reach value changes', () => {
      const handleChange = vi.fn();
      renderForm({ onChange: handleChange });

      const reachInput = screen.getByLabelText(/reach/i);
      fireEvent.change(reachInput, { target: { value: '500' } });

      expect(handleChange).toHaveBeenCalledWith(
        expect.objectContaining({
          reach: 500,
        })
      );
    });

    it('triggers onChange when Impact value changes', () => {
      const handleChange = vi.fn();
      renderForm({ onChange: handleChange });

      const impactSelect = screen.getByLabelText(/impact/i);
      fireEvent.change(impactSelect, { target: { value: '2' } });

      expect(handleChange).toHaveBeenCalledWith(
        expect.objectContaining({
          impact: 2,
        })
      );
    });

    it('triggers onChange when Confidence value changes', () => {
      const handleChange = vi.fn();
      renderForm({ onChange: handleChange });

      const confidenceSlider = screen.getByLabelText(/confidence/i);
      fireEvent.change(confidenceSlider, { target: { value: '75' } });

      expect(handleChange).toHaveBeenCalledWith(
        expect.objectContaining({
          confidence: 75,
        })
      );
    });

    it('triggers onChange when Effort value changes', () => {
      const handleChange = vi.fn();
      renderForm({ onChange: handleChange });

      const effortInput = screen.getByLabelText(/effort/i);
      fireEvent.change(effortInput, { target: { value: '5' } });

      expect(handleChange).toHaveBeenCalledWith(
        expect.objectContaining({
          effort: 5,
        })
      );
    });

    it('accumulates values in onChange', async () => {
      const handleChange = vi.fn();
      renderForm({ onChange: handleChange });

      const reachInput = screen.getByLabelText(/reach/i);
      fireEvent.change(reachInput, { target: { value: '100' } });

      await waitFor(() => {
        expect(handleChange).toHaveBeenCalledWith(
          expect.objectContaining({
            reach: 100,
          })
        );
      });

      const effortInput = screen.getByLabelText(/effort/i);
      fireEvent.change(effortInput, { target: { value: '3' } });

      await waitFor(() => {
        expect(handleChange).toHaveBeenCalledWith(
          expect.objectContaining({
            reach: 100,
            effort: 3,
          })
        );
      });
    });
  });

  describe('Form Submission', () => {
    it('triggers onSubmit when form is submitted with valid data', () => {
      const handleSubmit = vi.fn();
      renderForm({
        onSubmit: handleSubmit,
        value: { reach: 100, impact: 2, confidence: 80, effort: 3 },
      });

      const form = screen.getByRole('form') as HTMLFormElement;
      fireEvent.submit(form);

      expect(handleSubmit).toHaveBeenCalledWith({
        reach: 100,
        impact: 2,
        confidence: 80,
        effort: 3,
      });
    });

    it('does not submit when Reach is empty', () => {
      const handleSubmit = vi.fn();
      renderForm({
        onSubmit: handleSubmit,
        value: { impact: 2, confidence: 80, effort: 3 },
      });

      const form = screen.getByRole('form') as HTMLFormElement;
      fireEvent.submit(form);

      expect(handleSubmit).not.toHaveBeenCalled();
    });

    it('does not submit when Impact is empty', () => {
      const handleSubmit = vi.fn();
      renderForm({
        onSubmit: handleSubmit,
        value: { reach: 100, confidence: 80, effort: 3 },
      });

      const form = screen.getByRole('form') as HTMLFormElement;
      fireEvent.submit(form);

      expect(handleSubmit).not.toHaveBeenCalled();
    });

    it('does not submit when Confidence is empty', () => {
      const handleSubmit = vi.fn();
      renderForm({
        onSubmit: handleSubmit,
        value: { reach: 100, impact: 2, effort: 3 },
      });

      const form = screen.getByRole('form') as HTMLFormElement;
      fireEvent.submit(form);

      expect(handleSubmit).not.toHaveBeenCalled();
    });

    it('does not submit when Effort is empty', () => {
      const handleSubmit = vi.fn();
      renderForm({
        onSubmit: handleSubmit,
        value: { reach: 100, impact: 2, confidence: 80 },
      });

      const form = screen.getByRole('form') as HTMLFormElement;
      fireEvent.submit(form);

      expect(handleSubmit).not.toHaveBeenCalled();
    });
  });

  describe('Submit Button State', () => {
    it('disables submit button when no values are provided', () => {
      renderForm();
      const submitButton = screen.getByRole('button', {
        name: /calculate rice score/i,
      });
      expect(submitButton).toBeDisabled();
    });

    it('enables submit button when all values are provided', () => {
      renderForm({
        value: { reach: 100, impact: 2, confidence: 80, effort: 3 },
      });
      const submitButton = screen.getByRole('button', {
        name: /calculate rice score/i,
      });
      expect(submitButton).not.toBeDisabled();
    });

    it('disables submit button when only some values are provided', () => {
      renderForm({
        value: { reach: 100, impact: 2 },
      });
      const submitButton = screen.getByRole('button', {
        name: /calculate rice score/i,
      });
      expect(submitButton).toBeDisabled();
    });

    it('shows loading state when isLoading is true', () => {
      renderForm({
        isLoading: true,
        value: { reach: 100, impact: 2, confidence: 80, effort: 3 },
      });
      const submitButton = screen.getByRole('button', {
        name: /calculate rice score/i,
      });
      expect(submitButton).toBeDisabled();
      const spinner = document.querySelector('.btn-spinner');
      expect(spinner).toBeInTheDocument();
    });

    it('prevents submission when loading', () => {
      const handleSubmit = vi.fn();
      renderForm({
        isLoading: true,
        onSubmit: handleSubmit,
        value: { reach: 100, impact: 2, confidence: 80, effort: 3 },
      });

      const form = screen.getByRole('form') as HTMLFormElement;
      fireEvent.submit(form);

      expect(handleSubmit).not.toHaveBeenCalled();
    });
  });

  describe('Helper Text', () => {
    it('displays Reach helper text', () => {
      renderForm();
      expect(
        screen.getByText(/number of users or events affected/i)
      ).toBeInTheDocument();
    });

    it('displays Impact helper text', () => {
      renderForm();
      expect(screen.getByText(/impact on each affected user/i)).toBeInTheDocument();
    });

    it('displays Confidence helper text', () => {
      renderForm();
      expect(
        screen.getByText(/how confident are you in your estimates/i)
      ).toBeInTheDocument();
    });

    it('displays Effort helper text', () => {
      renderForm();
      expect(
        screen.getByText(/estimated effort in person-months/i)
      ).toBeInTheDocument();
    });
  });

  describe('Placeholder Text', () => {
    it('displays Reach placeholder', () => {
      renderForm();
      const reachInput = screen.getByLabelText(/reach/i);
      expect(reachInput).toHaveAttribute(
        'placeholder',
        expect.stringContaining('users per month')
      );
    });

    it('displays Effort placeholder', () => {
      renderForm();
      const effortInput = screen.getByLabelText(/effort/i);
      expect(effortInput).toHaveAttribute(
        'placeholder',
        expect.stringContaining('person-months')
      );
    });
  });

  describe('Slider Marks', () => {
    it('displays Confidence slider marks', () => {
      renderForm();
      expect(screen.getByText('Low')).toBeInTheDocument();
      expect(screen.getByText('Medium')).toBeInTheDocument();
      expect(screen.getByText('High')).toBeInTheDocument();
    });
  });

  describe('Custom ClassName', () => {
    it('applies custom className', () => {
      const { container } = renderForm({ className: 'custom-class' });
      const formElement = container.querySelector('.custom-class');
      expect(formElement).toBeInTheDocument();
    });

    it('applies base form class', () => {
      const { container } = renderForm();
      const formElement = container.querySelector('.rice-input-form');
      expect(formElement).toBeInTheDocument();
    });
  });

  describe('Controlled Component', () => {
    it('uses value prop for controlled state', () => {
      renderForm({ value: { reach: 200, impact: 3, confidence: 90, effort: 5 } });

      const reachInput = screen.getByLabelText(/reach/i) as HTMLInputElement;
      const impactSelect = screen.getByLabelText(/impact/i) as HTMLSelectElement;
      const confidenceSlider = screen.getByLabelText(/confidence/i) as HTMLInputElement;
      const effortInput = screen.getByLabelText(/effort/i) as HTMLInputElement;

      expect(reachInput.value).toBe('200');
      expect(impactSelect.value).toBe('3');
      expect(confidenceSlider.value).toBe('90');
      expect(effortInput.value).toBe('5');
    });
  });

  describe('Form Layout', () => {
    it('renders form with grid layout class', () => {
      const { container } = renderForm();
      expect(container.querySelector('.framework-form')).toBeInTheDocument();
      expect(container.querySelector('.form-grid')).toBeInTheDocument();
    });

    it('renders two-column grid for Reach and Impact', () => {
      const { container } = renderForm();
      expect(container.querySelector('.form-grid-2')).toBeInTheDocument();
    });
  });
});
