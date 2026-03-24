/**
 * Test setup file for Prioritization Frameworks Backend.
 *
 * This file is executed before each test file and provides:
 * - Console mocking for clean test output
 * - Common test utilities
 * - Global test mocks
 *
 * @module tests/setup
 */

import { expect, afterEach, vi } from 'vitest';

// Mock console for clean test output
vi.spyOn(console, 'log').mockImplementation(() => {});
vi.spyOn(console, 'error').mockImplementation(() => {});
vi.spyOn(console, 'warn').mockImplementation(() => {});

// Reset mocks after each test
afterEach(() => {
  vi.clearAllMocks();
});

// Extend expect with custom matchers if needed
expect.extend({
  toBeValidDate(received: unknown) {
    const date = new Date(received as string | number | Date);
    const pass = !isNaN(date.getTime());
    return {
      pass,
      message: () =>
        `expected ${received} ${pass ? 'not ' : ''}to be a valid date`,
    };
  },
});

// Declare the custom matcher for TypeScript
declare module 'vitest' {
  interface Assertion<T = unknown> {
    toBeValidDate(): T;
  }
  interface AsymmetricMatchersContaining {
    toBeValidDate(): void;
  }
}
