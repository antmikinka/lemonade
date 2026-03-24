/**
 * Header component module for Prioritization Frameworks application.
 * Provides the main application header with framework selector and theme toggle.
 *
 * @module components/common/Header
 */

import React from 'react';
import { FrameworkType } from '../../services/prioritization/types';
import { Select } from './Select';
import { Button } from './Button';

/**
 * Props for the Header component.
 */
export interface HeaderProps {
  /** Current selected framework */
  selectedFramework?: FrameworkType;
  /** Framework change handler */
  onFrameworkChange?: (framework: FrameworkType) => void;
  /** Theme toggle handler */
  onThemeToggle?: () => void;
  /** Whether dark theme is active */
  isDarkTheme?: boolean;
  /** Custom logo or icon */
  logo?: React.ReactNode;
  /** Additional header actions */
  actions?: React.ReactNode;
}

/**
 * Framework selector options for the dropdown.
 */
const frameworkOptions: { value: FrameworkType; label: string }[] = [
  { value: 'RICE', label: 'RICE Scoring' },
  { value: 'MoSCoW', label: 'MoSCoW Method' },
  { value: 'Kano', label: 'Kano Model' },
  { value: 'ValueEffort', label: 'Value vs Effort' },
  { value: 'ICE', label: 'ICE Scoring' },
  { value: 'Eisenhower', label: 'Eisenhower Matrix' },
  { value: 'P0P4', label: 'P0-P4 Priority' },
  { value: 'WSJF', label: 'WSJF (Weighted Shortest Job First)' },
];

/**
 * Main application Header component with framework selector and actions.
 *
 * This component provides the primary navigation header with:
 * - Application title and branding
 * - Framework selector dropdown
 * - Theme toggle button
 * - Additional action buttons
 *
 * @param props - Header component props
 * @returns Rendered header element
 *
 * @example
 * ```tsx
 * <Header
 *   selectedFramework="RICE"
 *   onFrameworkChange={handleFrameworkChange}
 *   onThemeToggle={handleThemeToggle}
 *   isDarkTheme={true}
 * />
 * ```
 */
export function Header({
  selectedFramework,
  onFrameworkChange,
  onThemeToggle,
  isDarkTheme = true,
  logo,
  actions,
}: HeaderProps): React.JSX.Element {
  return (
    <header className="app-header-main">
      <div className="header-content">
        <div className="header-brand">
          {logo || (
            <div className="header-logo">
              <svg
                viewBox="0 0 40 40"
                fill="none"
                xmlns="http://www.w3.org/2000/svg"
                aria-hidden="true"
              >
                <rect width="40" height="40" rx="8" fill="var(--color-accent)" />
                <path
                  d="M12 20L18 26L28 14"
                  stroke="var(--color-bg-primary)"
                  strokeWidth="3"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              </svg>
            </div>
          )}
          <div className="header-title-group">
            <h1 className="header-title">Prioritization Frameworks</h1>
            <p className="header-subtitle">AI-Powered Decision Making Tools</p>
          </div>
        </div>

        <div className="header-actions">
          <div className="header-framework-selector">
            <Select
              label=""
              options={frameworkOptions}
              value={selectedFramework || ''}
              onChange={(e) =>
                onFrameworkChange?.(e.target.value as FrameworkType)
              }
              placeholder="Select Framework"
              allowEmpty={false}
              fullWidth={false}
              size="md"
            />
          </div>

          {onThemeToggle && (
            <Button
              variant="secondary"
              size="md"
              onClick={onThemeToggle}
              aria-label={isDarkTheme ? 'Switch to light theme' : 'Switch to dark theme'}
              className="header-theme-toggle"
            >
              {isDarkTheme ? (
                <svg
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  aria-hidden="true"
                >
                  <circle cx="12" cy="12" r="5" />
                  <path d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42" />
                </svg>
              ) : (
                <svg
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  aria-hidden="true"
                >
                  <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" />
                </svg>
              )}
            </Button>
          )}

          {actions && <div className="header-extra-actions">{actions}</div>}
        </div>
      </div>
    </header>
  );
}

export default Header;
