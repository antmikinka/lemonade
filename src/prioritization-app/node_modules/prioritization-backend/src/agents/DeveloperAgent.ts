/**
 * Developer Agent for Prioritization Framework Agent Pipeline.
 *
 * This agent designs technical solutions and implementation strategies
 * using design patterns, programming paradigms, and visual reasoning.
 *
 * @module agents/DeveloperAgent
 */

import { Agent } from './Agent';
import { MCPClient } from '../mcp/MCPClient';
import {
  MCPToolName,
  AgentOutput,
  DevelopmentInput,
  DesignPatternResponse,
  ProgrammingParadigmResponse,
  VisualReasoningResponse
} from '../agents/types';

/**
 * Development result from the Developer Agent.
 */
export interface DevelopmentResult {
  /** Implementation steps */
  implementation: string[];
  /** Architecture description */
  architecture: string;
  /** Recommended patterns */
  patterns: string[];
  /** Code structure suggestion */
  codeStructure: string;
  /** Technical recommendations */
  recommendations: string[];
}

/**
 * Developer Agent that designs technical solutions.
 *
 * Uses design patterns, programming paradigms, and visual reasoning
 * to create comprehensive implementation strategies.
 */
export class DeveloperAgent extends Agent {
  /**
   * Create a new Developer Agent.
   * @param mcp - MCP client instance
   */
  constructor(mcp: MCPClient) {
    super('DeveloperAgent', mcp);
  }

  /**
   * Get the MCP tools used by this agent.
   * @returns Array of MCP tool names
   */
  getTools(): MCPToolName[] {
    return ['designpattern', 'programmingparadigm', 'visualreasoning'];
  }

  /**
   * Execute the technical design analysis.
   *
   * @param input - Development input with feature requirements
   * @returns Agent output with development result
   *
   * @example
   * ```typescript
   * const output = await developerAgent.execute({
   *   feature: 'Real-time notification system',
   *   requirements: ['WebSocket support', 'Scalable', 'Persistent queue']
   * });
   * ```
   */
  async execute(input: unknown): Promise<AgentOutput> {
    const developmentInput = input as DevelopmentInput;
    const reasoningSteps: string[] = [];
    const timestamp = new Date();

    try {
      // Step 1: Design pattern selection
      reasoningSteps.push('Analyzing design pattern requirements...');
      const patternAnalysis = await this.runDesignPatternAnalysis(
        developmentInput.feature,
        developmentInput.requirements
      );
      reasoningSteps.push(`Selected pattern: ${patternAnalysis.implementation[0].split(' ')[0]}`);

      // Step 2: Programming paradigm analysis
      reasoningSteps.push('Determining optimal programming paradigm...');
      const paradigmAnalysis = await this.runProgrammingParadigmAnalysis(
        developmentInput.feature,
        developmentInput.requirements
      );
      reasoningSteps.push(`Recommended paradigm: ${paradigmAnalysis.principles[0].split(' - ')[0]}`);

      // Step 3: Visual reasoning for architecture
      reasoningSteps.push('Designing system architecture...');
      const architectureAnalysis = await this.runVisualReasoning(
        developmentInput.feature,
        developmentInput.integrations || []
      );
      reasoningSteps.push(`Architecture designed with ${architectureAnalysis.componentFlow.length} components`);

      // Step 4: Build implementation strategy
      reasoningSteps.push('Building implementation strategy...');
      const implementation = this.buildImplementationStrategy(
        patternAnalysis,
        paradigmAnalysis,
        architectureAnalysis,
        developmentInput
      );

      // Calculate confidence
      const baseConfidence = 0.85;
      const confidence = this.calculateConfidence(baseConfidence, reasoningSteps.length);

      const result: DevelopmentResult = {
        implementation,
        architecture: architectureAnalysis.diagramDescription,
        patterns: patternAnalysis.benefits.map((_, i) =>
          patternAnalysis.implementation[i]?.split(' - ')[0] || `Pattern ${i + 1}`
        ),
        codeStructure: patternAnalysis.structure || this.getDefaultCodeStructure(),
        recommendations: this.generateRecommendations(
          patternAnalysis,
          paradigmAnalysis,
          developmentInput
        )
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
        result: {
          implementation: [],
          architecture: '',
          patterns: [],
          codeStructure: '',
          recommendations: []
        },
        reasoning: `Error during development analysis: ${errorMessage}`,
        confidence: 0.3,
        agentName: this.name,
        timestamp
      };
    }
  }

  /**
   * Run design pattern analysis.
   * @param feature - Feature to implement
   * @param requirements - Feature requirements
   * @returns Design pattern response
   */
  private async runDesignPatternAnalysis(
    feature: string,
    requirements: string[]
  ): Promise<DesignPatternResponse> {
    // Determine best pattern based on requirements
    const context = `Implementing ${feature} with requirements: ${requirements.join(', ')}`;

    // Select pattern based on keywords
    const featureLower = feature.toLowerCase();
    let patternName: 'modular_architecture' | 'observer_pattern' | 'api_integration' | 'state_management' =
      'modular_architecture';

    if (featureLower.includes('event') || featureLower.includes('notification') || featureLower.includes('real-time')) {
      patternName = 'observer_pattern';
    } else if (featureLower.includes('api') || featureLower.includes('integration') || featureLower.includes('external')) {
      patternName = 'api_integration';
    } else if (featureLower.includes('state') || featureLower.includes('cache') || featureLower.includes('store')) {
      patternName = 'state_management';
    }

    return this.mcp.designPattern(patternName, context);
  }

  /**
   * Run programming paradigm analysis.
   * @param feature - Feature to implement
   * @param requirements - Feature requirements
   * @returns Programming paradigm response
   */
  private async runProgrammingParadigmAnalysis(
    feature: string,
    requirements: string[]
  ): Promise<ProgrammingParadigmResponse> {
    const context = `Implementing ${feature} with requirements: ${requirements.join(', ')}`;

    // Select paradigm based on requirements
    const requirementsLower = requirements.join(' ').toLowerCase();
    let paradigm: 'functional' | 'object_oriented' | 'event_driven' | 'async_await' = 'object_oriented';

    if (requirementsLower.includes('async') || requirementsLower.includes('concurrent')) {
      paradigm = 'async_await';
    } else if (requirementsLower.includes('event') || requirementsLower.includes('real-time')) {
      paradigm = 'event_driven';
    } else if (requirementsLower.includes('pure') || requirementsLower.includes('immutable')) {
      paradigm = 'functional';
    }

    return this.mcp.programmingParadigm(paradigm, context);
  }

  /**
   * Run visual reasoning for architecture design.
   * @param feature - Feature to implement
   * @param integrations - Integration points
   * @returns Visual reasoning response
   */
  private async runVisualReasoning(
    feature: string,
    integrations: string[]
  ): Promise<VisualReasoningResponse> {
    // Build component list based on feature and integrations
    const components = [
      { name: 'API Gateway', connections: ['Service Layer', 'Authentication'] },
      { name: 'Service Layer', connections: ['Data Access', 'External Services'] },
      { name: 'Data Access', connections: [] },
      { name: 'Event Bus', connections: integrations.length > 0 ? integrations : ['Service Layer'] }
    ];

    return this.mcp.visualReasoning({
      system: feature,
      components,
      diagramType: 'architecture'
    });
  }

  /**
   * Build implementation strategy from analyses.
   * @param pattern - Design pattern response
   * @param paradigm - Programming paradigm response
   * @param architecture - Visual reasoning response
   * @param input - Original development input
   * @returns Implementation steps array
   */
  private buildImplementationStrategy(
    pattern: DesignPatternResponse,
    paradigm: ProgrammingParadigmResponse,
    architecture: VisualReasoningResponse,
    input: DevelopmentInput
  ): string[] {
    const implementation: string[] = [
      `Step 1: Setup - Initialize project structure following ${pattern.implementation[0]?.split(' - ')[0] || 'modular'} pattern`,
      '',
      'Step 2: Core Components',
      ...pattern.implementation.map((imp, i) => `  ${i + 1}. ${imp}`),
      '',
      'Step 3: Apply Programming Paradigm',
      ...paradigm.approach.slice(0, 3).map((app, i) => `  ${i + 1}. ${app}`),
      '',
      'Step 4: Architecture Implementation',
      ...architecture.componentFlow.map((flow, i) => `  ${i + 1}. Implement ${flow}`),
      '',
      'Step 5: Integration Points',
      ...(input.integrations || []).map((integration, i) => `  ${i + 1}. Connect to ${integration}`),
      '',
      'Step 6: Testing & Validation',
      '  1. Write unit tests for core components',
      '  2. Implement integration tests',
      '  3. Performance benchmarking',
      ''
    ];

    return implementation;
  }

  /**
   * Generate technical recommendations.
   * @param pattern - Design pattern response
   * @param paradigm - Programming paradigm response
   * @param input - Development input
   * @returns Recommendations array
   */
  private generateRecommendations(
    pattern: DesignPatternResponse,
    paradigm: ProgrammingParadigmResponse,
    input: DevelopmentInput
  ): string[] {
    const recommendations: string[] = [];

    // Pattern-based recommendations
    if (pattern.tradeOffs && pattern.tradeOffs.length > 0) {
      recommendations.push(`Consider trade-offs: ${pattern.tradeOffs[0]}`);
    }

    // Paradigm-based recommendations
    if (paradigm.pitfalls && paradigm.pitfalls.length > 0) {
      recommendations.push(`Avoid pitfall: ${paradigm.pitfalls[0]}`);
    }

    // Constraint-based recommendations
    if (input.technicalConstraints && input.technicalConstraints.length > 0) {
      input.technicalConstraints.forEach((constraint) => {
        recommendations.push(`Technical constraint: ${constraint}`);
      });
    }

    // General recommendations
    recommendations.push(
      'Follow clean code principles',
      'Implement comprehensive error handling',
      'Document public APIs thoroughly',
      'Set up CI/CD pipeline early'
    );

    return recommendations;
  }

  /**
   * Get default code structure template.
   * @returns Default code structure string
   */
  private getDefaultCodeStructure(): string {
    return `
// Default Project Structure
src/
├── components/    # Reusable UI components
├── services/      # Business logic services
├── handlers/      # Request handlers
├── middleware/    # Custom middleware
├── utils/         # Utility functions
├── types/         # TypeScript type definitions
└── tests/         # Test files
`;
  }
}

export default DeveloperAgent;
