/**
 * Test setup file for Prioritization Frameworks Frontend.
 *
 * This file is executed before each test file and provides:
 * - Jest DOM matchers for better assertions
 * - Automatic cleanup of React Testing Library
 * - Global test utilities
 *
 * @module tests/setup
 */

import '@testing-library/jest-dom';
import { cleanup } from '@testing-library/react';
import { afterEach, vi } from 'vitest';

// Cleanup after each test to prevent state leakage
afterEach(() => {
  cleanup();
});

// Mock window.matchMedia for components that use media queries
Object.defineProperty(window, 'matchMedia', {
  writable: true,
  value: vi.fn().mockImplementation((query: string) => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: vi.fn(),
    removeListener: vi.fn(),
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    dispatchEvent: vi.fn(),
  })),
});

// Mock window.scrollTo for components that use scroll
window.scrollTo = vi.fn();
