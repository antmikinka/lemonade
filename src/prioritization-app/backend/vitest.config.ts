/**
 * Vitest configuration for Prioritization Frameworks Backend.
 *
 * This configuration provides:
 * - Node.js environment for backend tests
 * - Global test functions (describe, it, expect, etc.)
 * - Test setup file for common mocks
 * - Coverage reporting with thresholds
 *
 * @module vitest.config
 */

import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    globals: true,
    environment: 'node',
    setupFiles: ['./tests/setup.ts'],
    include: ['tests/**/*.test.ts'],
    exclude: ['node_modules', 'dist'],
    coverage: {
      provider: 'v8',
      reporter: ['text', 'json', 'html'],
      threshold: {
        statements: 80,
        branches: 75,
        functions: 80,
        lines: 80,
      },
      include: ['src/**/*.ts'],
      exclude: ['src/**/*.test.ts', 'tests/**/*'],
    },
  },
});
