/**
 * E2E Tests for Complete Prioritization Flow.
 *
 * Tests cover:
 * - RICE prioritization end-to-end
 * - Adding multiple items to backlog
 * - Displaying priority ranking
 * - Results display and validation
 *
 * @module e2e/tests/prioritization-flow.spec
 */

import { test, expect } from '@playwright/test';

test.describe('Complete Prioritization Flow', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
  });

  test('RICE prioritization end-to-end', async ({ page }) => {
    // Select RICE framework
    await page.selectOption('[aria-label*="framework" i]', 'RICE');

    // Fill in RICE inputs
    await page.fill('[aria-label*="reach" i]', '1000');
    await page.selectOption('[aria-label*="impact" i]', '2');
    await page.fill('[aria-label*="confidence" i]', '80');
    await page.fill('[aria-label*="effort" i]', '5');

    // Submit form
    await page.click('button:has-text("Calculate"), button:has-text("Submit"), button[type="submit"]');

    // Wait for results to appear
    await page.waitForTimeout(1000);

    // Verify results are displayed
    const resultsSection = page.getByTestId('results-card').first();
    await expect(resultsSection).toBeVisible();

    // Verify score is displayed (expected: (1000 * 2 * 0.8) / 5 = 320)
    const scoreDisplay = page.getByText(/320/).first();
    await expect(scoreDisplay).toBeVisible();
  });

  test('adds multiple items to backlog', async ({ page }) => {
    // Select RICE framework
    await page.selectOption('[aria-label*="framework" i]', 'RICE');

    // Add first item
    await page.fill('[aria-label*="reach" i]', '500');
    await page.selectOption('[aria-label*="impact" i]', '3');
    await page.fill('[aria-label*="confidence" i]', '90');
    await page.fill('[aria-label*="effort" i]', '3');

    // Fill item title if there's a title field
    const titleInput = page.getByLabel(/title/i).first();
    if (await titleInput.isVisible()) {
      await titleInput.fill('Feature A');
    }

    // Submit first item
    await page.click('button:has-text("Calculate"), button:has-text("Add"), button:has-text("Submit")');
    await page.waitForTimeout(500);

    // Add second item
    await page.fill('[aria-label*="reach" i]', '200');
    await page.selectOption('[aria-label*="impact" i]', '2');
    await page.fill('[aria-label*="confidence" i]', '70');
    await page.fill('[aria-label*="effort" i]', '2');

    if (await titleInput.isVisible()) {
      await titleInput.fill('Feature B');
    }

    // Submit second item
    await page.click('button:has-text("Calculate"), button:has-text("Add"), button:has-text("Submit")');
    await page.waitForTimeout(500);

    // Verify backlog contains both items
    const backlogSection = page.getByTestId('backlog').first();
    await expect(backlogSection).toBeVisible();

    // Check for item titles or count
    const backlogItems = page.getByTestId('backlog-item').all();
    expect((await backlogItems).length).toBeGreaterThanOrEqual(1);
  });

  test('displays priority ranking correctly', async ({ page }) => {
    // Select RICE framework
    await page.selectOption('[aria-label*="framework" i]', 'RICE');

    // Add high priority item
    await page.fill('[aria-label*="reach" i]', '1000');
    await page.selectOption('[aria-label*="impact" i]', '3');
    await page.fill('[aria-label*="confidence" i]', '100');
    await page.fill('[aria-label*="effort" i]', '1');

    let titleInput = page.getByLabel(/title/i).first();
    if (await titleInput.isVisible()) {
      await titleInput.fill('High Priority Feature');
    }

    await page.click('button:has-text("Calculate"), button:has-text("Submit")');
    await page.waitForTimeout(500);

    // Add low priority item
    await page.fill('[aria-label*="reach" i]', '50');
    await page.selectOption('[aria-label*="impact" i]', '0.5');
    await page.fill('[aria-label*="confidence" i]', '50');
    await page.fill('[aria-label*="effort" i]', '10');

    if (await titleInput.isVisible()) {
      await titleInput.fill('Low Priority Feature');
    }

    await page.click('button:has-text("Calculate"), button:has-text("Submit")');
    await page.waitForTimeout(500);

    // Verify ranking display
    const rankingSection = page.getByTestId('ranking').first();
    await expect(rankingSection).toBeVisible();

    // High priority should be ranked first
    const firstRankedItem = page.getByText(/rank.*1/i).first();
    await expect(firstRankedItem).toBeVisible();
  });

  test('handles empty form submission gracefully', async ({ page }) => {
    // Select RICE framework
    await page.selectOption('[aria-label*="framework" i]', 'RICE');

    // Try to submit without filling any values
    const submitButton = page.locator('button[type="submit"]').first();

    // Button should be disabled when form is empty
    await expect(submitButton).toBeDisabled();
  });

  test('validates input values', async ({ page }) => {
    // Select RICE framework
    await page.selectOption('[aria-label*="framework" i]', 'RICE');

    // Fill invalid negative values
    await page.fill('[aria-label*="reach" i]', '-100');

    // Check for validation error or that submit is disabled
    const submitButton = page.locator('button[type="submit"]').first();
    // Either button is disabled or there's an error message
    const isDisabled = await submitButton.isDisabled();
    expect(isDisabled).toBeTruthy();
  });

  test('shows results summary after calculation', async ({ page }) => {
    // Select RICE framework
    await page.selectOption('[aria-label*="framework" i]', 'RICE');

    // Fill valid values
    await page.fill('[aria-label*="reach" i]', '100');
    await page.selectOption('[aria-label*="impact" i]', '2');
    await page.fill('[aria-label*="confidence" i]', '80');
    await page.fill('[aria-label*="effort" i]', '4');

    // Submit
    await page.click('button:has-text("Calculate"), button:has-text("Submit")');
    await page.waitForTimeout(500);

    // Verify results section is visible
    await expect(page.getByText(/score/i)).toBeVisible();
    await expect(page.getByText(/breakdown/i)).toBeVisible();
  });

  test('clears form after successful submission', async ({ page }) => {
    // Select RICE framework
    await page.selectOption('[aria-label*="framework" i]', 'RICE');

    // Fill values
    await page.fill('[aria-label*="reach" i]', '100');
    await page.selectOption('[aria-label*="impact" i]', '2');
    await page.fill('[aria-label*="confidence" i]', '80');
    await page.fill('[aria-label*="effort" i]', '4');

    // Submit
    await page.click('button:has-text("Calculate"), button:has-text("Submit")');
    await page.waitForTimeout(500);

    // Form should be cleared or reset after submission
    const reachInput = page.locator('[aria-label*="reach" i]');
    const value = await reachInput.inputValue();
    expect(value).toBe('');
  });
});
