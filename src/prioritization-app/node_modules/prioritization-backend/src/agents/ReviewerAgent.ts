/**
 * Reviewer Agent for Prioritization Framework Agent Pipeline.
 *
 * This agent reviews work products and identifies improvement opportunities
 * using metacognitive monitoring and structured argumentation.
 *
 * @module agents/ReviewerAgent
 */

import { Agent } from './Agent';
import { MCPClient } from '../mcp/MCPClient';
import {
  MCPToolName,
  AgentOutput,
  ReviewInput,
  MetacognitiveMonitoringResponse,
  StructuredArgumentationResponse,
  Argument
} from '../agents/types';

/**
 * Review result from the Reviewer Agent.
 */
export interface ReviewResult {
  /** Identified issues */
  issues: string[];
  /** Recommendations for improvement */
  recommendations: string[];
  /** Quality score (0-100) */
  qualityScore: number;
  /** Strengths identified */
  strengths: string[];
  /** Critical fixes needed */
  criticalFixes: string[];
}

/**
 * Reviewer Agent that performs quality reviews.
 *
 * Uses metacognitive monitoring and structured argumentation
 * to provide comprehensive work product reviews.
 */
export class ReviewerAgent extends Agent {
  /**
   * Create a new Reviewer Agent.
   * @param mcp - MCP client instance
   */
  constructor(mcp: MCPClient) {
    super('ReviewerAgent', mcp);
  }

  /**
   * Get the MCP tools used by this agent.
   * @returns Array of MCP tool names
   */
  getTools(): MCPToolName[] {
    return ['metacognitivemonitoring', 'structuredargumentation'];
  }

  /**
   * Execute the quality review analysis.
   *
   * @param input - Review input with work product and criteria
   * @returns Agent output with review result
   *
   * @example
   * ```typescript
   * const output = await reviewerAgent.execute({
   *   workProduct: 'Implemented authentication module',
   *   criteria: ['Security', 'Maintainability', 'Performance'],
   *   previousOutputs: [planningOutput, developmentOutput]
   * });
   * ```
   */
  async execute(input: unknown): Promise<AgentOutput> {
    const reviewInput = input as ReviewInput;
    const reasoningSteps: string[] = [];
    const timestamp = new Date();

    try {
      // Step 1: Metacognitive monitoring of the work
      reasoningSteps.push('Performing metacognitive analysis...');
      const monitoringResult = await this.runMetacognitiveMonitoring(
        reviewInput.workProduct,
        reviewInput.previousOutputs
      );
      reasoningSteps.push(`Identified ${monitoringResult.uncertaintyAreas.length} uncertainty areas`);

      // Step 2: Build arguments for structured analysis
      reasoningSteps.push('Building structured argumentation...');
      const proArguments = this.buildProArguments(reviewInput);
      const conArguments = this.buildConArguments(reviewInput, monitoringResult);

      // Step 3: Structured argumentation analysis
      reasoningSteps.push('Running structured argumentation analysis...');
      const argumentationResult = await this.runStructuredArgumentation(
        reviewInput.workProduct,
        proArguments,
        conArguments
      );
      reasoningSteps.push(`Found ${argumentationResult.weakPoints.length} weak points`);

      // Step 4: Calculate quality metrics
      reasoningSteps.push('Calculating quality metrics...');
      const qualityScore = this.calculateQualityScore(
        monitoringResult,
        argumentationResult,
        reviewInput.criteria
      );

      // Step 5: Generate review result
      reasoningSteps.push('Generating review result...');
      const result = this.generateReviewResult(
        monitoringResult,
        argumentationResult,
        reviewInput,
        qualityScore
      );

      // Calculate confidence
      const baseConfidence = 0.88;
      const confidence = this.calculateConfidence(baseConfidence, reasoningSteps.length);

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
        result: {
          issues: [],
          recommendations: [],
          qualityScore: 0,
          strengths: [],
          criticalFixes: []
        },
        reasoning: `Error during review: ${errorMessage}`,
        confidence: 0.3,
        agentName: this.name,
        timestamp
      };
    }
  }

  /**
   * Run metacognitive monitoring on the work product.
   * @param workProduct - Work product to review
   * @param previousOutputs - Previous agent outputs for context
   * @returns Metacognitive monitoring response
   */
  private async runMetacognitiveMonitoring(
    workProduct: string,
    previousOutputs: AgentOutput[]
  ): Promise<MetacognitiveMonitoringResponse> {
    // Analyze reasoning from previous outputs
    const stepsTaken = previousOutputs.map((output) => output.reasoning.split('\n')[0] || 'Unknown step');
    const avgConfidence =
      previousOutputs.length > 0
        ? previousOutputs.reduce((sum, o) => sum + o.confidence, 0) / previousOutputs.length
        : 0.5;

    const currentReasoning = `Work product: ${workProduct}. Previous agents completed ${previousOutputs.length} analysis steps.`;

    return this.mcp.metacognitiveMonitoring({
      currentReasoning,
      stepsTaken,
      confidence: avgConfidence,
      goal: 'Ensure high-quality implementation meeting all criteria'
    });
  }

  /**
   * Build pro arguments for the work product.
   * @param input - Review input
   * @returns Pro arguments array
   */
  private buildProArguments(input: ReviewInput): Argument[] {
    const proArguments: Argument[] = [];

    // Analyze work product for positive aspects
    const workProductLower = input.workProduct.toLowerCase();

    // Check for completeness indicators
    if (workProductLower.includes('implement') || workProductLower.includes('complete')) {
      proArguments.push({
        claim: 'Implementation appears complete',
        evidence: ['Work product uses implementation language', 'Describes concrete functionality']
      });
    }

    // Check for quality indicators
    if (workProductLower.includes('test') || workProductLower.includes('validate')) {
      proArguments.push({
        claim: 'Quality assurance considered',
        evidence: ['Testing mentioned in work product', 'Validation approach included']
      });
    }

    // Check for documentation
    if (workProductLower.includes('document') || workProductLower.includes('comment')) {
      proArguments.push({
        claim: 'Documentation provided',
        evidence: ['Work product includes documentation', 'Comments explain functionality']
      });
    }

    // Add criterion-based arguments
    input.criteria.forEach((criterion) => {
      proArguments.push({
        claim: `Addresses ${criterion} criterion`,
        evidence: [`Work product considers ${criterion.toLowerCase()}`]
      });
    });

    return proArguments;
  }

  /**
   * Build con arguments based on monitoring results.
   * @param input - Review input
   * @param monitoring - Metacognitive monitoring result
   * @returns Con arguments array
   */
  private buildConArguments(
    input: ReviewInput,
    monitoring: MetacognitiveMonitoringResponse
  ): Argument[] {
    const conArguments: Argument[] = [];

    // Add uncertainty areas as con arguments
    monitoring.uncertaintyAreas.forEach((area) => {
      conArguments.push({
        claim: area,
        evidence: ['Identified through metacognitive analysis'],
        counterArguments: ['May be addressed in implementation details']
      });
    });

    // Check for common issues
    const workProductLower = input.workProduct.toLowerCase();

    if (!workProductLower.includes('error') && !workProductLower.includes('handle')) {
      conArguments.push({
        claim: 'Error handling may be incomplete',
        evidence: ['No explicit error handling mentioned'],
        counterArguments: ['May be implemented but not documented']
      });
    }

    if (!workProductLower.includes('edge') && !workProductLower.includes('boundary')) {
      conArguments.push({
        claim: 'Edge cases may not be considered',
        evidence: ['No mention of edge case handling'],
        counterArguments: ['Standard edge cases may be handled by framework']
      });
    }

    // Check against criteria
    input.criteria.forEach((criterion) => {
      if (!workProductLower.includes(criterion.toLowerCase())) {
        conArguments.push({
          claim: `${criterion} aspect may need attention`,
          evidence: [`Criterion "${criterion}" not explicitly addressed`],
          counterArguments: ['May be implicitly satisfied']
        });
      }
    });

    return conArguments;
  }

  /**
   * Run structured argumentation analysis.
   * @param topic - Topic for argumentation
   * @param proArguments - Pro arguments
   * @param conArguments - Con arguments
   * @returns Structured argumentation response
   */
  private async runStructuredArgumentation(
    topic: string,
    proArguments: Argument[],
    conArguments: Argument[]
  ): Promise<StructuredArgumentationResponse> {
    return this.mcp.structuredArgumentation({
      topic: `Quality assessment: ${topic}`,
      proArguments,
      conArguments
    });
  }

  /**
   * Calculate overall quality score.
   * @param monitoring - Metacognitive monitoring result
   * @param argumentation - Structured argumentation result
   * @param criteria - Review criteria
   * @returns Quality score (0-100)
   */
  private calculateQualityScore(
    monitoring: MetacognitiveMonitoringResponse,
    argumentation: StructuredArgumentationResponse,
    criteria: string[]
  ): number {
    let score = 70; // Base score

    // Adjust based on uncertainty areas
    score -= monitoring.uncertaintyAreas.length * 5;

    // Adjust based on weak points
    score -= argumentation.weakPoints.length * 3;

    // Adjust based on pro/con ratio
    const proCount = 5; // Default pro arguments
    const conCount = monitoring.uncertaintyAreas.length + argumentation.weakPoints.length;
    if (conCount > proCount) {
      score -= (conCount - proCount) * 2;
    } else {
      score += (proCount - conCount) * 2;
    }

    // Criterion bonus
    score += criteria.length * 2;

    // Clamp to 0-100
    return Math.max(0, Math.min(100, Math.round(score)));
  }

  /**
   * Generate comprehensive review result.
   * @param monitoring - Metacognitive monitoring result
   * @param argumentation - Structured argumentation result
   * @param input - Review input
   * @param qualityScore - Calculated quality score
   * @returns Review result
   */
  private generateReviewResult(
    monitoring: MetacognitiveMonitoringResponse,
    argumentation: StructuredArgumentationResponse,
    input: ReviewInput,
    qualityScore: number
  ): ReviewResult {
    // Identify issues
    const issues: string[] = [
      ...monitoring.uncertaintyAreas,
      ...argumentation.weakPoints.slice(0, 5)
    ];

    // Identify strengths
    const strengths: string[] = [
      'Work product addresses core requirements',
      ...input.criteria.map((c) => `Considers ${c} aspect`)
    ];

    // Generate recommendations
    const recommendations: string[] = [
      ...monitoring.recommendedNextSteps,
      ...argumentation.suggestedNextTypes
    ];

    // Identify critical fixes
    const criticalFixes: string[] = [];
    if (qualityScore < 50) {
      criticalFixes.push('Address fundamental design issues before proceeding');
    }
    if (monitoring.uncertaintyAreas.length > 3) {
      criticalFixes.push('Clarify uncertain areas with stakeholders');
    }
    if (argumentation.weakPoints.length > 3) {
      criticalFixes.push('Strengthen weak arguments with additional evidence');
    }

    return {
      issues,
      recommendations,
      qualityScore,
      strengths,
      criticalFixes
    };
  }
}

export default ReviewerAgent;
