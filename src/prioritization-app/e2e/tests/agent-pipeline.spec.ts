/**
 * E2E Tests for Agent Pipeline functionality.
 *
 * Tests cover:
 * - Agent panel toggle
 * - Running the agent pipeline
 * - Progress display during execution
 * - Pipeline completion handling
 * - Available agents display
 *
 * @module e2e/tests/agent-pipeline.spec
 */

import { test, expect } from '@playwright/test';

test.describe('Agent Pipeline', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
  });

  test('displays agent panel toggle button', async ({ page }) => {
    // Find agent panel toggle
    const agentToggle = page.locator(
      '[data-testid="agent-panel-toggle"], button:has-text("Agent"), [aria-label*="agent" i]'
    ).first();
    await expect(agentToggle).toBeVisible();
  });

  test('opens agent panel when toggle is clicked', async ({ page }) => {
    // Click agent panel toggle
    const agentToggle = page.locator(
      '[data-testid="agent-panel-toggle"], button:has-text("Agent"), [aria-label*="agent" i]'
    ).first();
    await agentToggle.click();
    await page.waitForTimeout(300);

    // Verify agent panel is visible
    const agentPanel = page.locator(
      '[data-testid="agent-panel"], .agent-panel, [aria-label*="agent panel" i]'
    ).first();
    await expect(agentPanel).toBeVisible();
  });

  test('displays available agents list', async ({ page }) => {
    // Open agent panel
    const agentToggle = page.locator(
      '[data-testid="agent-panel-toggle"], button:has-text("Agent"), [aria-label*="agent" i]'
    ).first();
    await agentToggle.click();
    await page.waitForTimeout(300);

    // Check for expected agents
    const expectedAgents = ['PlanningAgent', 'DeveloperAgent', 'ReviewerAgent'];

    for (const agent of expectedAgents) {
      const agentElement = page.locator(`[data-testid="agent-${agent.toLowerCase()}"], text=${agent}`);
      await expect(agentElement).toBeVisible();
    }
  });

  test('displays run pipeline button', async ({ page }) => {
    // Open agent panel
    const agentToggle = page.locator(
      '[data-testid="agent-panel-toggle"], button:has-text("Agent"), [aria-label*="agent" i]'
    ).first();
    await agentToggle.click();
    await page.waitForTimeout(300);

    // Check for run button
    const runButton = page.locator(
      '[data-testid="run-pipeline-button"], button:has-text("Run"), button:has-text("Start Pipeline")'
    ).first();
    await expect(runButton).toBeVisible();
  });

  test('shows progress when pipeline is running', async ({ page }) => {
    // Open agent panel
    const agentToggle = page.locator(
      '[data-testid="agent-panel-toggle"], button:has-text("Agent"), [aria-label*="agent" i]'
    ).first();
    await agentToggle.click();
    await page.waitForTimeout(300);

    // Start pipeline
    const runButton = page.locator(
      '[data-testid="run-pipeline-button"], button:has-text("Run"), button:has-text("Start Pipeline")'
    ).first();
    await runButton.click();

    // Verify progress display appears
    const progressElement = page.locator(
      '[data-testid="agent-progress"], .agent-progress, [aria-label*="progress" i]'
    ).first();
    await expect(progressElement).toBeVisible();
  });

  test('completes pipeline within timeout', async ({ page }) => {
    // Open agent panel
    const agentToggle = page.locator(
      '[data-testid="agent-panel-toggle"], button:has-text("Agent"), [aria-label*="agent" i]'
    ).first();
    await agentToggle.click();
    await page.waitForTimeout(300);

    // Start pipeline
    const runButton = page.locator(
      '[data-testid="run-pipeline-button"], button:has-text("Run"), button:has-text("Start Pipeline")'
    ).first();
    await runButton.click();

    // Wait for completion (with timeout)
    const completeElement = page.locator(
      '[data-testid="pipeline-complete"], .pipeline-complete, text=Complete, text=Finished'
    ).first();
    await expect(completeElement).toBeVisible({ timeout: 30000 });
  });

  test('displays agent status during execution', async ({ page }) => {
    // Open agent panel
    const agentToggle = page.locator(
      '[data-testid="agent-panel-toggle"], button:has-text("Agent"), [aria-label*="agent" i]'
    ).first();
    await agentToggle.click();
    await page.waitForTimeout(300);

    // Start pipeline
    const runButton = page.locator(
      '[data-testid="run-pipeline-button"], button:has-text("Run"), button:has-text("Start Pipeline")'
    ).first();
    await runButton.click();
    await page.waitForTimeout(500);

    // Check for status indicators
    const statusElements = page.locator(
      '[data-testid="agent-status"], .agent-status, [aria-label*="status" i]'
    ).first();
    await expect(statusElements).toBeVisible();
  });

  test('allows cancelling running pipeline', async ({ page }) => {
    // Open agent panel
    const agentToggle = page.locator(
      '[data-testid="agent-panel-toggle"], button:has-text("Agent"), [aria-label*="agent" i]'
    ).first();
    await agentToggle.click();
    await page.waitForTimeout(300);

    // Start pipeline
    const runButton = page.locator(
      '[data-testid="run-pipeline-button"], button:has-text("Run"), button:has-text("Start Pipeline")'
    ).first();
    await runButton.click();
    await page.waitForTimeout(500);

    // Check for cancel button
    const cancelButton = page.locator(
      '[data-testid="cancel-pipeline-button"], button:has-text("Cancel"), button:has-text("Stop")'
    ).first();
    if (await cancelButton.isVisible()) {
      await cancelButton.click();
      await page.waitForTimeout(300);

      // Verify cancelled state
      const cancelledElement = page.locator(
        '[data-testid="pipeline-cancelled"], text=Cancelled, text=Stopped'
      ).first();
      await expect(cancelledElement).toBeVisible();
    }
  });

  test('displays pipeline results after completion', async ({ page }) => {
    // Open agent panel
    const agentToggle = page.locator(
      '[data-testid="agent-panel-toggle"], button:has-text("Agent"), [aria-label*="agent" i]'
    ).first();
    await agentToggle.click();
    await page.waitForTimeout(300);

    // Start pipeline
    const runButton = page.locator(
      '[data-testid="run-pipeline-button"], button:has-text("Run"), button:has-text("Start Pipeline")'
    ).first();
    await runButton.click();

    // Wait for completion
    await page.waitForTimeout(15000);

    // Check for results section
    const resultsSection = page.locator(
      '[data-testid="pipeline-results"], .pipeline-results, [aria-label*="results" i]'
    ).first();
    await expect(resultsSection).toBeVisible();
  });

  test('shows agent execution order', async ({ page }) => {
    // Open agent panel
    const agentToggle = page.locator(
      '[data-testid="agent-panel-toggle"], button:has-text("Agent"), [aria-label*="agent" i]'
    ).first();
    await agentToggle.click();
    await page.waitForTimeout(300);

    // Start pipeline
    const runButton = page.locator(
      '[data-testid="run-pipeline-button"], button:has-text("Run"), button:has-text("Start Pipeline")'
    ).first();
    await runButton.click();
    await page.waitForTimeout(1000);

    // Check for execution order/sequence display
    const executionOrder = page.locator(
      '[data-testid="execution-order"], .execution-order, [aria-label*="execution" i]'
    ).first();
    await expect(executionOrder).toBeVisible();
  });

  test('displays agent configuration options', async ({ page }) => {
    // Open agent panel
    const agentToggle = page.locator(
      '[data-testid="agent-panel-toggle"], button:has-text("Agent"), [aria-label*="agent" i]'
    ).first();
    await agentToggle.click();
    await page.waitForTimeout(300);

    // Check for configuration options
    const configElement = page.locator(
      '[data-testid="agent-config"], .agent-config, [aria-label*="config" i], [aria-label*="settings" i]'
    ).first();
    await expect(configElement).toBeVisible();
  });
});
