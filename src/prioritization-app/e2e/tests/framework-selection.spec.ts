/**
 * E2E Tests for Framework Selection functionality.
 *
 * Tests cover:
 * - Framework selector displays all 8 frameworks
 * - Switching frameworks changes input form
 * - Framework-specific fields are visible
 * - Form validation works correctly
 *
 * @module e2e/tests/framework-selection.spec
 */

import { test, expect } from '@playwright/test';

test.describe('Framework Selection', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
  });

  test('displays all 8 frameworks in selector', async ({ page }) => {
    // Find the framework selector
    const frameworkSelect = page.getByLabel(/framework/i).first();
    await expect(frameworkSelect).toBeVisible();

    // Click to open dropdown
    await frameworkSelect.click();

    // Check all 8 frameworks are present
    const expectedFrameworks = [
      'RICE',
      'MoSCoW',
      'Value vs Effort',
      'ICE',
      'Eisenhower',
      'P0P4',
      'WSJF',
      'Kano',
    ];

    for (const framework of expectedFrameworks) {
      const option = page.getByRole('option', { name: new RegExp(framework, 'i') });
      await expect(option).toBeVisible();
    }
  });

  test('switches input form when framework changes from RICE to MoSCoW', async ({
    page,
  }) => {
    // Select RICE first
    await page.selectOption('[aria-label*="framework" i]', 'RICE');

    // Verify RICE-specific fields are visible
    await expect(page.getByLabel(/reach/i)).toBeVisible();
    await expect(page.getByLabel(/impact/i)).toBeVisible();
    await expect(page.getByLabel(/confidence/i)).toBeVisible();
    await expect(page.getByLabel(/effort/i)).toBeVisible();

    // Switch to MoSCoW
    await page.selectOption('[aria-label*="framework" i]', 'MoSCoW');

    // Verify MoSCoW-specific fields are visible
    await expect(page.getByLabel(/business value/i)).toBeVisible();
    await expect(page.getByText(/must have/i)).toBeVisible();

    // Verify RICE fields are hidden
    await expect(page.getByLabel(/reach/i)).not.toBeVisible();
  });

  test('switches input form when framework changes to ICE', async ({ page }) => {
    // Select ICE framework
    await page.selectOption('[aria-label*="framework" i]', 'ICE');

    // Verify ICE-specific fields are visible
    await expect(page.getByLabel(/impact/i)).toBeVisible();
    await expect(page.getByLabel(/confidence/i)).toBeVisible();
    await expect(page.getByLabel(/ease/i)).toBeVisible();
  });

  test('switches input form when framework changes to Eisenhower', async ({
    page,
  }) => {
    // Select Eisenhower framework
    await page.selectOption('[aria-label*="framework" i]', 'Eisenhower');

    // Verify Eisenhower-specific fields are visible
    await expect(page.getByLabel(/urgent/i)).toBeVisible();
    await expect(page.getByLabel(/important/i)).toBeVisible();
  });

  test('switches input form when framework changes to WSJF', async ({ page }) => {
    // Select WSJF framework
    await page.selectOption('[aria-label*="framework" i]', 'WSJF');

    // Verify WSJF-specific fields are visible
    await expect(page.getByLabel(/user.*business.*value/i)).toBeVisible();
    await expect(page.getByLabel(/time.*critical/i)).toBeVisible();
    await expect(
      page.getByLabel(/risk.*reduction/i)
    ).toBeVisible();
    await expect(page.getByLabel(/job.*size/i)).toBeVisible();
  });

  test('switches input form when framework changes to Kano', async ({ page }) => {
    // Select Kano framework
    await page.selectOption('[aria-label*="framework" i]', 'Kano');

    // Verify Kano-specific fields are visible
    await expect(page.getByLabel(/functional/i)).toBeVisible();
    await expect(page.getByLabel(/dysfunctional/i)).toBeVisible();
  });

  test('displays correct framework description/header when changed', async ({
    page,
  }) => {
    // Start with RICE
    await page.selectOption('[aria-label*="framework" i]', 'RICE');
    await expect(page.getByText(/RICE/i)).toBeVisible();

    // Change to P0P4
    await page.selectOption('[aria-label*="framework" i]', 'P0P4');
    await expect(page.getByText(/P0P4/i)).toBeVisible();
    await expect(page.getByText(/Priority/i)).toBeVisible();
  });

  test('maintains selection state when switching frameworks', async ({
    page,
  }) => {
    // Select a framework
    await page.selectOption('[aria-label*="framework" i]', 'RICE');

    // Verify the select element has the correct value
    const select = page.getByLabel(/framework/i).first();
    await expect(select).toHaveValue('RICE');

    // Switch to another framework
    await page.selectOption('[aria-label*="framework" i]', 'ICE');

    // Verify the value changed
    await expect(select).toHaveValue('ICE');
  });

  test('displays helper text for each framework', async ({ page }) => {
    const frameworks = [
      { name: 'RICE', helperText: /reach/i },
      { name: 'MoSCoW', helperText: /must have/i },
      { name: 'ICE', helperText: /impact/i },
      { name: 'Eisenhower', helperText: /urgent/i },
    ];

    for (const framework of frameworks) {
      await page.selectOption('[aria-label*="framework" i]', framework.name);
      await expect(page.getByText(framework.helperText)).toBeVisible();
    }
  });
});
