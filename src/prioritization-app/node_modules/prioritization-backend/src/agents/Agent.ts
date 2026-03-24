/**
 * Agent Base Class for Prioritization Framework Agent Pipeline.
 *
 * This abstract class defines the interface and common functionality
 * for all agents in the pipeline.
 *
 * @module agents/Agent
 */

import { MCPClient } from '../mcp/MCPClient';
import { MCPToolName, AgentOutput } from '../agents/types';

/**
 * Abstract base class for all pipeline agents.
 *
 * Each agent implements specific reasoning capabilities using
 * Clear Thought MCP tools.
 */
export abstract class Agent {
  /** Agent name identifier */
  protected readonly name: string;

  /** MCP client for tool invocation */
  protected readonly mcp: MCPClient;

  /**
   * Create a new agent.
   * @param name - Agent name
   * @param mcp - MCP client instance
   */
  constructor(name: string, mcp: MCPClient) {
    this.name = name;
    this.mcp = mcp;
  }

  /**
   * Execute the agent's reasoning process.
   * @param input - Agent-specific input data
   * @returns Agent output with result, reasoning, and confidence
   */
  abstract execute(input: unknown): Promise<AgentOutput>;

  /**
   * Get the list of MCP tools this agent uses.
   * @returns Array of MCP tool names
   */
  abstract getTools(): MCPToolName[];

  /**
   * Get the agent's name.
   * @returns Agent name
   */
  getName(): string {
    return this.name;
  }

  /**
   * Get the agent's description.
   * @returns Human-readable agent description
   */
  getDescription(): string {
    const descriptions: Record<string, string> = {
      PlanningAgent: 'Analyzes problems and creates structured implementation plans',
      DeveloperAgent: 'Designs technical solutions and implementation strategies',
      ReviewerAgent: 'Reviews work products and identifies improvement opportunities'
    };
    return descriptions[this.name] || 'Generic agent for pipeline processing';
  }

  /**
   * Calculate confidence score based on reasoning quality.
   * @param baseConfidence - Base confidence from agent execution
   * @param reasoningSteps - Number of reasoning steps taken
   * @returns Adjusted confidence score
   */
  protected calculateConfidence(baseConfidence: number, reasoningSteps: number): number {
    // Adjust confidence based on reasoning depth
    const depthBonus = Math.min(reasoningSteps * 0.02, 0.1);
    return Math.min(Math.max(baseConfidence + depthBonus, 0), 1);
  }

  /**
   * Format reasoning output for display.
   * @param steps - Reasoning steps
   * @returns Formatted reasoning string
   */
  protected formatReasoning(steps: string[]): string {
    return steps.map((step, index) => `Step ${index + 1}: ${step}`).join('\n');
  }
}

export default Agent;
