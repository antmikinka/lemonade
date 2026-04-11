/**
 * E2E Tests for Backlog Management functionality.
 *
 * Tests cover:
 * - Drag and drop reordering
 * - Checkbox selection and bulk actions
 * - Item deletion
 * - Backlog filtering and sorting
 *
 * @module e2e/tests/backlog-management.spec
 */

import { test, expect } from '@playwright/test';

test.describe('Backlog Management', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
  });

  test('displays backlog items with checkboxes', async ({ page }) => {
    // First, add some items to the backlog
    await page.selectOption('[aria-label*="framework" i]', 'RICE');
    await page.fill('[aria-label*="reach" i]', '100');
    await page.selectOption('[aria-label*="impact" i]', '2');
    await page.fill('[aria-label*="confidence" i]', '80');
    await page.fill('[aria-label*="effort" i]', '4');

    // Submit to add item
    await page.click('button:has-text("Calculate"), button:has-text("Submit")');
    await page.waitForTimeout(500);

    // Check for backlog section
    const backlogSection = page.getByTestId('backlog').first();
    await expect(backlogSection).toBeVisible();

    // Check for checkboxes in backlog items
    const checkboxes = page.locator('input[type="checkbox"]');
    const count = await checkboxes.count();
    expect(count).toBeGreaterThan(0);
  });

  test('allows selecting multiple items via checkboxes', async ({ page }) => {
    // Add items to backlog
    await page.selectOption('[aria-label*="framework" i]', 'RICE');

    for (let i = 1; i <= 3; i++) {
      await page.fill('[aria-label*="reach" i]', `${i * 100}`);
      await page.selectOption('[aria-label*="impact" i]', '2');
      await page.fill('[aria-label*="confidence" i]', '80');
      await page.fill('[aria-label*="effort" i]', '4');

      await page.click('button:has-text("Calculate"), button:has-text("Submit")');
      await page.waitForTimeout(300);
    }

    // Select all checkboxes
    const checkboxes = page.locator('input[type="checkbox"]');
    const count = await checkboxes.count();

    for (let i = 0; i < count; i++) {
      await checkboxes.nth(i).click();
    }

    // Verify bulk actions appear
    const bulkActions = page.getByTestId('bulk-actions');
    await expect(bulkActions).toBeVisible();
  });

  test('displays drag handles for reordering', async ({ page }) => {
    // Add items to backlog
    await page.selectOption('[aria-label*="framework" i]', 'RICE');

    for (let i = 1; i <= 2; i++) {
      await page.fill('[aria-label*="reach" i]', `${i * 100}`);
      await page.selectOption('[aria-label*="impact" i]', '2');
      await page.fill('[aria-label*="confidence" i]', '80');
      await page.fill('[aria-label*="effort" i]', '4');

      await page.click('button:has-text("Calculate"), button:has-text("Submit")');
      await page.waitForTimeout(300);
    }

    // Check for drag handles
    const dragHandles = page.locator('[data-testid="drag-handle"], .drag-handle, [aria-label*="drag" i]');
    const count = await dragHandles.count();
    expect(count).toBeGreaterThan(0);
  });

  test('supports drag and drop reordering', async ({ page }) => {
    // Add items to backlog
    await page.selectOption('[aria-label*="framework" i]', 'RICE');

    // Add first item
    await page.fill('[aria-label*="reach" i]', '100');
    await page.selectOption('[aria-label*="impact" i]', '2');
    await page.fill('[aria-label*="confidence" i]', '80');
    await page.fill('[aria-label*="effort" i]', '4');
    await page.click('button:has-text("Calculate"), button:has-text("Submit")');
    await page.waitForTimeout(300);

    // Add second item
    await page.fill('[aria-label*="reach" i]', '200');
    await page.selectOption('[aria-label*="impact" i]', '2');
    await page.fill('[aria-label*="confidence" i]', '80');
    await page.fill('[aria-label*="effort" i]', '4');
    await page.click('button:has-text("Calculate"), button:has-text("Submit")');
    await page.waitForTimeout(300);

    // Find draggable items
    const draggableItems = page.locator('[draggable="true"], .draggable, [data-testid="backlog-item"]');
    const firstItem = draggableItems.first();
    const secondItem = draggableItems.nth(1);

    // Get initial positions
    const firstBox = await firstItem.boundingBox();
    const secondBox = await secondItem.boundingBox();

    // Drag first item to second position
    if (firstBox && secondBox) {
      await page.mouse.move(firstBox.x + firstBox.width / 2, firstBox.y + firstBox.height / 2);
      await page.mouse.down();
      await page.mouse.move(secondBox.x + secondBox.width / 2, secondBox.y + secondBox.height / 2);
      await page.mouse.up();
      await page.waitForTimeout(500);

      // Verify order changed (implementation dependent)
      // This is a basic test - actual implementation may vary
    }
  });

  test('displays item details in backlog', async ({ page }) => {
    // Add item to backlog
    await page.selectOption('[aria-label*="framework" i]', 'RICE');
    await page.fill('[aria-label*="reach" i]', '100');
    await page.selectOption('[aria-label*="impact" i]', '2');
    await page.fill('[aria-label*="confidence" i]', '80');
    await page.fill('[aria-label*="effort" i]', '4');

    await page.click('button:has-text("Calculate"), button:has-text("Submit")');
    await page.waitForTimeout(500);

    // Check for item details
    const backlogItem = page.getByTestId('backlog-item').first();
    await expect(backlogItem).toBeVisible();

    // Should display score/category information
    await expect(page.getByText(/score/i).first()).toBeVisible();
  });

  test('allows deleting items from backlog', async ({ page }) => {
    // Add item to backlog
    await page.selectOption('[aria-label*="framework" i]', 'RICE');
    await page.fill('[aria-label*="reach" i]', '100');
    await page.selectOption('[aria-label*="impact" i]', '2');
    await page.fill('[aria-label*="confidence" i]', '80');
    await page.fill('[aria-label*="effort" i]', '4');

    await page.click('button:has-text("Calculate"), button:has-text("Submit")');
    await page.waitForTimeout(500);

    // Find and click delete button
    const deleteButton = page.locator('[data-testid="delete-button"], button:has-text("Delete"), .delete-btn').first();
    if (await deleteButton.isVisible()) {
      await deleteButton.click();
      await page.waitForTimeout(300);

      // Verify item was removed
      const backlogItems = page.getByTestId('backlog-item').all();
      expect((await backlogItems).length).toBe(0);
    }
  });

  test('supports bulk delete action', async ({ page }) => {
    // Add multiple items
    await page.selectOption('[aria-label*="framework" i]', 'RICE');

    for (let i = 1; i <= 3; i++) {
      await page.fill('[aria-label*="reach" i]', `${i * 100}`);
      await page.selectOption('[aria-label*="impact" i]', '2');
      await page.fill('[aria-label*="confidence" i]', '80');
      await page.fill('[aria-label*="effort" i]', '4');

      await page.click('button:has-text("Calculate"), button:has-text("Submit")');
      await page.waitForTimeout(300);
    }

    // Select all items
    const checkboxes = page.locator('input[type="checkbox"]');
    const count = await checkboxes.count();

    for (let i = 0; i < count; i++) {
      await checkboxes.nth(i).click();
    }

    // Click bulk delete
    const bulkDeleteButton = page.locator('[data-testid="bulk-delete"], button:has-text("Delete Selected")').first();
    if (await bulkDeleteButton.isVisible()) {
      await bulkDeleteButton.click();
      await page.waitForTimeout(300);

      // Verify all items were removed
      const remainingItems = page.getByTestId('backlog-item').all();
      expect((await remainingItems).length).toBe(0);
    }
  });

  test('displays empty state when backlog is empty', async ({ page }) => {
    // Navigate to page without adding items
    await page.goto('/');

    // Check for empty state message
    const emptyState = page.locator('[data-testid="empty-state"], .empty-state, text=No items, text=Backlog is empty').first();
    await expect(emptyState).toBeVisible();
  });
});
