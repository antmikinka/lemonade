/**
 * Planning Agent for Prioritization Framework Agent Pipeline.
 *
 * This agent analyzes problems and creates structured implementation plans
 * using sequential thinking, first principles analysis, and decision frameworks.
 *
 * @module agents/PlanningAgent
 */

import { Agent } from './Agent';
import { MCPClient } from '../mcp/MCPClient';
import {
  MCPToolName,
  AgentOutput,
  PlanningInput,
  SequentialThinkingResponse,
  MentalModelResponse,
  DecisionFrameworkResponse,
  DecisionOption
} from '../agents/types';

/**
 * Planning result from the Planning Agent.
 */
export interface PlanningResult {
  /** Structured plan with steps */
  plan: string[];
  /** Key insights from analysis */
  insights: string[];
  /** Recommended approach */
  recommendedApproach: string;
  /** Identified risks */
  risks: string[];
  /** Success criteria */
  successCriteria: string[];
}

/**
 * Planning Agent that creates structured implementation plans.
 *
 * Uses sequential thinking, first principles, and decision frameworks
 * to analyze problems and generate actionable plans.
 */
export class PlanningAgent extends Agent {
  /**
   * Create a new Planning Agent.
   * @param mcp - MCP client instance
   */
  constructor(mcp: MCPClient) {
    super('PlanningAgent', mcp);
  }

  /**
   * Get the MCP tools used by this agent.
   * @returns Array of MCP tool names
   */
  getTools(): MCPToolName[] {
    return ['sequentialthinking', 'mentalmodel', 'decisionframework'];
  }

  /**
   * Execute the planning analysis.
   *
   * @param input - Planning input with problem statement and goals
   * @returns Agent output with planning result
   *
   * @example
   * ```typescript
   * const output = await planningAgent.execute({
   *   problem: 'Need to implement user authentication',
   *   goals: ['Secure', 'Scalable', 'User-friendly']
   * });
   * ```
   */
  async execute(input: unknown): Promise<AgentOutput> {
    const planningInput = input as PlanningInput;
    const reasoningSteps: string[] = [];
    const timestamp = new Date();

    try {
      // Step 1: Sequential thinking to break down the problem
      reasoningSteps.push('Starting sequential thinking analysis...');
      const thoughts = await this.runSequentialThinking(planningInput.problem);
      reasoningSteps.push(`Completed ${thoughts.thoughtNumber} thinking iterations`);

      // Step 2: First principles analysis
      reasoningSteps.push('Applying first principles mental model...');
      const firstPrinciples = await this.runFirstPrinciples(
        planningInput.problem,
        planningInput.context
      );
      reasoningSteps.push(`Derived fundamental insights: ${firstPrinciples.insights?.length || 0} key points`);

      // Step 3: Generate options based on analysis
      reasoningSteps.push('Generating solution options...');
      const options = this.generateOptions(planningInput, firstPrinciples);

      // Step 4: Decision framework for recommendations
      reasoningSteps.push('Running decision framework analysis...');
      const decision = await this.runDecisionFramework(
        planningInput.problem,
        options
      );
      reasoningSteps.push(`Selected approach: ${decision.recommendation}`);

      // Step 5: Build structured plan
      reasoningSteps.push('Building structured implementation plan...');
      const plan = this.buildPlan(decision, planningInput, firstPrinciples);

      // Calculate confidence
      const baseConfidence = 0.80;
      const confidence = this.calculateConfidence(baseConfidence, reasoningSteps.length);

      const result: PlanningResult = {
        plan,
        insights: firstPrinciples.insights || [],
        recommendedApproach: decision.recommendation,
        risks: this.identifyRisks(planningInput),
        successCriteria: this.defineSuccessCriteria(planningInput.goals)
      };

      return {
        result,
        reasoning: this.formatReasoning(reasoningSteps),
        confidence,
        agentName: this.name,
        timestamp
      };
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      return {
        result: { plan: [], insights: [], recommendedApproach: '', risks: [], successCriteria: [] },
        reasoning: `Error during planning: ${errorMessage}`,
        confidence: 0.3,
        agentName: this.name,
        timestamp
      };
    }
  }

  /**
   * Run sequential thinking analysis on the problem.
   * @param problem - Problem statement
   * @returns Sequential thinking response
   */
  private async runSequentialThinking(problem: string): Promise<SequentialThinkingResponse> {
    const totalThoughts = 5;
    let thoughtNumber = 1;
    let currentThought = `Analyzing problem: ${problem}. Breaking down into fundamental components.`;
    let nextThoughtNeeded = true;

    while (nextThoughtNeeded && thoughtNumber <= totalThoughts) {
      const response = await this.mcp.sequentialThinking(
        currentThought,
        thoughtNumber,
        totalThoughts
      );

      currentThought = response.thought;
      thoughtNumber = response.thoughtNumber || thoughtNumber + 1;
      nextThoughtNeeded = response.nextThoughtNeeded;
    }

    return {
      nextThoughtNeeded: false,
      thought: currentThought,
      thoughtNumber
    };
  }

  /**
   * Run first principles mental model analysis.
   * @param problem - Problem to analyze
   * @param _context - Additional context
   * @returns Mental model response
   */
  private async runFirstPrinciples(
    problem: string,
    _context?: string
  ): Promise<MentalModelResponse> {
    return this.mcp.mentalModel('first_principles', problem);
  }

  /**
   * Generate solution options based on analysis.
   * @param input - Planning input
   * @param _firstPrinciples - First principles analysis result
   * @returns Array of decision options
   */
  private generateOptions(
    input: PlanningInput,
    _firstPrinciples: MentalModelResponse
  ): DecisionOption[] {
    const options: DecisionOption[] = [
      {
        name: 'Minimal Viable Solution',
        description: 'Simple implementation focusing on core functionality with minimal complexity'
      },
      {
        name: 'Scalable Architecture',
        description: 'Invest in scalable design patterns and infrastructure for long-term growth'
      },
      {
        name: 'Hybrid Approach',
        description: 'Balance between quick delivery and architectural soundness with iterative improvement'
      }
    ];

    // Add constraint-based options
    if (input.constraints && input.constraints.length > 0) {
      options.push({
        name: 'Constraint-Optimized Solution',
        description: `Tailored approach considering constraints: ${input.constraints.join(', ')}`
      });
    }

    return options;
  }

  /**
   * Run decision framework analysis on options.
   * @param problem - Problem statement
   * @param options - Available options
   * @returns Decision framework response
   */
  private async runDecisionFramework(
    problem: string,
    options: DecisionOption[]
  ): Promise<DecisionFrameworkResponse> {
    return this.mcp.decisionFramework(
      `Best approach for: ${problem}`,
      options,
      'weighted-criteria'
    );
  }

  /**
   * Build structured implementation plan from decision.
   * @param decision - Decision framework response
   * @param _input - Planning input
   * @param _firstPrinciples - First principles analysis
   * @returns Array of plan steps
   */
  private buildPlan(
    decision: DecisionFrameworkResponse,
    _input: PlanningInput,
    _firstPrinciples: MentalModelResponse
  ): string[] {
    const plan: string[] = [
      `Phase 1: Foundation - ${decision.recommendation}`,
      '  - Set up project structure and tooling',
      '  - Define core interfaces and contracts',
      '  - Establish testing infrastructure',
      '',
      'Phase 2: Core Implementation',
      '  - Implement fundamental components from first principles',
      '  - Build integration points incrementally',
      '  - Validate assumptions with early testing',
      '',
      'Phase 3: Enhancement',
      '  - Add advanced features based on feedback',
      '  - Optimize performance bottlenecks',
      '  - Refine user experience',
      '',
      'Phase 4: Hardening',
      '  - Comprehensive testing and QA',
      '  - Documentation and knowledge transfer',
      '  - Deployment preparation'
    ];

    return plan;
  }

  /**
   * Identify potential risks for the plan.
   * @param input - Planning input
   * @returns Array of identified risks
   */
  private identifyRisks(input: PlanningInput): string[] {
    const risks: string[] = [
      'Technical complexity may exceed initial estimates',
      'Resource constraints could impact timeline',
      'Dependencies on external systems may introduce delays'
    ];

    if (input.constraints && input.constraints.length > 0) {
      risks.push(`Constraint-related risks: ${input.constraints.join(', ')}`);
    }

    return risks;
  }

  /**
   * Define success criteria based on goals.
   * @param goals - List of goals
   * @returns Array of success criteria
   */
  private defineSuccessCriteria(goals: string[]): string[] {
    return goals.map((goal, index) =>
      `Criterion ${index + 1}: ${goal} - measurable outcome defined`
    );
  }
}

export default PlanningAgent;
