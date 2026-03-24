/**
 * Agent Pipeline Types for Prioritization Frameworks Backend.
 *
 * This module defines types for Clear Thought MCP tools integration,
 * agent pipeline orchestration, and session management.
 *
 * @module agents/types
 */

// ============================================================================
// Clear Thought MCP Tool Types
// ============================================================================

/**
 * Union type of all available Clear Thought MCP tools.
 */
export type MCPToolName =
  | 'sequentialthinking'
  | 'mentalmodel'
  | 'designpattern'
  | 'programmingparadigm'
  | 'debuggingapproach'
  | 'collaborativereasoning'
  | 'decisionframework'
  | 'metacognitivemonitoring'
  | 'scientificmethod'
  | 'structuredargumentation'
  | 'visualreasoning';

/**
 * Mental model types for first principles and strategic thinking.
 */
export type MentalModelType =
  | 'first_principles'
  | 'opportunity_cost'
  | 'pareto_principle'
  | 'occams_razor'
  | 'second_order_thinking'
  | 'inversion';

/**
 * Design pattern types for architecture analysis.
 */
export type DesignPatternType =
  | 'modular_architecture'
  | 'api_integration'
  | 'state_management'
  | 'observer_pattern'
  | 'factory_pattern'
  | 'strategy_pattern';

/**
 * Programming paradigm types.
 */
export type ProgrammingParadigmType =
  | 'functional'
  | 'object_oriented'
  | 'procedural'
  | 'event_driven'
  | 'reactive'
  | 'async_await';

/**
 * Decision framework analysis types.
 */
export type DecisionFrameworkType =
  | 'pros-cons'
  | 'weighted-criteria'
  | 'decision-tree'
  | 'cost-benefit';

/**
 * Parameters for invoking an MCP tool.
 */
export interface MCPToolCall {
  /** Tool name to invoke */
  tool: MCPToolName;
  /** Tool parameters */
  parameters: Record<string, unknown>;
}

/**
 * Response from an MCP tool invocation.
 */
export interface MCPToolResponse {
  /** Generated thought or reasoning step */
  thought?: string;
  /** Conclusion reached */
  conclusion?: string;
  /** Reasoning behind the conclusion */
  reasoning?: string;
  /** Additional tool-specific data */
  [key: string]: unknown;
}

// ============================================================================
// Sequential Thinking Tool Types
// ============================================================================

/**
 * Request for sequential thinking tool.
 */
export interface SequentialThinkingRequest {
  /** Current thought content */
  thought: string;
  /** Current thought number in sequence */
  thoughtNumber: number;
  /** Estimated total thoughts needed */
  totalThoughts: number;
  /** Whether next thought is needed */
  nextThoughtNeeded: boolean;
}

/**
 * Response from sequential thinking tool.
 */
export interface SequentialThinkingResponse {
  /** Whether another thought is needed */
  nextThoughtNeeded: boolean;
  /** Current thought content */
  thought: string;
  /** Updated thought number */
  thoughtNumber?: number;
  /** Updated total thoughts estimate */
  totalThoughts?: number;
  /** Additional properties */
  [key: string]: unknown;
}

// ============================================================================
// Mental Model Tool Types
// ============================================================================

/**
 * Request for mental model analysis.
 */
export interface MentalModelRequest {
  /** Mental model to apply */
  modelName: MentalModelType;
  /** Problem or situation to analyze */
  problem: string;
  /** Additional context */
  context?: string;
}

/**
 * Response from mental model analysis.
 */
export interface MentalModelResponse {
  /** Conclusion from the analysis */
  conclusion: string;
  /** Reasoning behind the conclusion */
  reasoning: string;
  /** Key insights discovered */
  insights?: string[];
  /** Additional properties */
  [key: string]: unknown;
}

// ============================================================================
// Design Pattern Tool Types
// ============================================================================

/**
 * Request for design pattern analysis.
 */
export interface DesignPatternRequest {
  /** Design pattern to apply */
  patternName: DesignPatternType;
  /** Context or problem domain */
  context: string;
  /** Specific requirements or constraints */
  constraints?: string[];
}

/**
 * Response from design pattern analysis.
 */
export interface DesignPatternResponse {
  /** Implementation steps */
  implementation: string[];
  /** Benefits of this pattern */
  benefits: string[];
  /** Trade-offs to consider */
  tradeOffs?: string[];
  /** Code structure suggestion */
  structure?: string;
  /** Additional properties */
  [key: string]: unknown;
}

// ============================================================================
// Programming Paradigm Tool Types
// ============================================================================

/**
 * Request for programming paradigm analysis.
 */
export interface ProgrammingParadigmRequest {
  /** Paradigm to analyze */
  paradigm: ProgrammingParadigmType;
  /** Problem context */
  context: string;
  /** Goals for the solution */
  goals?: string[];
}

/**
 * Response from programming paradigm analysis.
 */
export interface ProgrammingParadigmResponse {
  /** Recommended approach steps */
  approach: string[];
  /** Key principles to follow */
  principles: string[];
  /** Potential pitfalls */
  pitfalls?: string[];
  /** Additional properties */
  [key: string]: unknown;
}

// ============================================================================
// Decision Framework Tool Types
// ============================================================================

/**
 * Option for decision framework analysis.
 */
export interface DecisionOption {
  /** Option name */
  name: string;
  /** Option description */
  description: string;
  /** Option attributes (for weighted criteria) */
  attributes?: Record<string, number>;
}

/**
 * Request for decision framework analysis.
 */
export interface DecisionFrameworkRequest {
  /** Decision statement or question */
  decisionStatement: string;
  /** Available options */
  options: DecisionOption[];
  /** Type of analysis to perform */
  analysisType: DecisionFrameworkType;
  /** Criteria weights (for weighted-criteria) */
  criteriaWeights?: Record<string, number>;
}

/**
 * Response from decision framework analysis.
 */
export interface DecisionFrameworkResponse {
  /** Recommended option */
  recommendation: string;
  /** Rationale for the recommendation */
  rationale: string;
  /** Scores for each option (if applicable) */
  optionScores?: Record<string, number>;
  /** Sensitivity analysis results */
  sensitivity?: string;
  /** Additional properties */
  [key: string]: unknown;
}

// ============================================================================
// Metacognitive Monitoring Tool Types
// ============================================================================

/**
 * Request for metacognitive monitoring.
 */
export interface MetacognitiveMonitoringRequest {
  /** Current reasoning state */
  currentReasoning: string;
  /** Steps taken so far */
  stepsTaken: string[];
  /** Confidence level (0-1) */
  confidence: number;
  /** Goal or objective */
  goal: string;
}

/**
 * Response from metacognitive monitoring.
 */
export interface MetacognitiveMonitoringResponse {
  /** Assessment of current state */
  assessment: string;
  /** Identified uncertainty areas */
  uncertaintyAreas: string[];
  /** Recommended next steps */
  recommendedNextSteps: string[];
  /** Confidence adjustment suggestion */
  confidenceAdjustment?: number;
  /** Reasoning steps taken */
  reasoningSteps: string[];
  /** Additional properties */
  [key: string]: unknown;
}

// ============================================================================
// Structured Argumentation Tool Types
// ============================================================================

/**
 * Argument component for structured argumentation.
 */
export interface Argument {
  /** Claim being made */
  claim: string;
  /** Evidence supporting the claim */
  evidence: string[];
  /** Counter-arguments */
  counterArguments?: string[];
}

/**
 * Request for structured argumentation.
 */
export interface StructuredArgumentationRequest {
  /** Topic or proposition */
  topic: string;
  /** Arguments in favor */
  proArguments: Argument[];
  /** Arguments against */
  conArguments: Argument[];
}

/**
 * Response from structured argumentation.
 */
export interface StructuredArgumentationResponse {
  /** Synthesis of arguments */
  synthesis: string;
  /** Strongest argument identified */
  strongestArgument: Argument;
  /** Weakest points identified */
  weakPoints: string[];
  /** Suggested next steps for reasoning */
  suggestedNextTypes: string[];
  /** Overall conclusion */
  conclusion?: string;
  /** Additional properties */
  [key: string]: unknown;
}

// ============================================================================
// Visual Reasoning Tool Types
// ============================================================================

/**
 * Request for visual reasoning.
 */
export interface VisualReasoningRequest {
  /** System or architecture to visualize */
  system: string;
  /** Components and their relationships */
  components: Array<{ name: string; connections: string[] }>;
  /** Type of diagram requested */
  diagramType?: 'flowchart' | 'sequence' | 'component' | 'architecture';
}

/**
 * Response from visual reasoning.
 */
export interface VisualReasoningResponse {
  /** Textual representation of diagram */
  diagramDescription: string;
  /** Component flow */
  componentFlow: string[];
  /** Key architectural decisions */
  architecturalDecisions: string[];
  /** Potential bottlenecks */
  bottlenecks?: string[];
  /** Additional properties */
  [key: string]: unknown;
}

// ============================================================================
// Agent Types
// ============================================================================

/**
 * Output from any agent execution.
 */
export interface AgentOutput {
  /** Result data from agent */
  result: unknown;
  /** Reasoning behind the result */
  reasoning: string;
  /** Confidence score (0-1) */
  confidence: number;
  /** Agent name that produced this output */
  agentName: string;
  /** Timestamp of execution */
  timestamp: Date;
}

/**
 * Input for planning agent.
 */
export interface PlanningInput {
  /** Problem statement */
  problem: string;
  /** Goals to achieve */
  goals: string[];
  /** Constraints */
  constraints?: string[];
  /** Available resources */
  resources?: string[];
  /** Additional context */
  context?: string;
}

/**
 * Input for developer agent.
 */
export interface DevelopmentInput {
  /** Feature or component to implement */
  feature: string;
  /** Requirements */
  requirements: string[];
  /** Technical constraints */
  technicalConstraints?: string[];
  /** Integration points */
  integrations?: string[];
}

/**
 * Input for reviewer agent.
 */
export interface ReviewInput {
  /** Work product to review */
  workProduct: string;
  /** Quality criteria */
  criteria: string[];
  /** Previous agent outputs */
  previousOutputs: AgentOutput[];
}

// ============================================================================
// Pipeline Types
// ============================================================================

/**
 * Pipeline session status.
 */
export type PipelineStatus = 'pending' | 'running' | 'completed' | 'error' | 'cancelled';

/**
 * Agent step in pipeline.
 */
export interface AgentStep {
  /** Agent name */
  agentName: string;
  /** Step status */
  status: 'pending' | 'running' | 'completed' | 'error';
  /** Output from this agent */
  output?: AgentOutput;
  /** Error message if failed */
  error?: string;
  /** Start timestamp */
  startedAt?: Date;
  /** End timestamp */
  endedAt?: Date;
}

/**
 * Pipeline session representation.
 */
export interface PipelineSession {
  /** Unique session identifier */
  id: string;
  /** Current status */
  status: PipelineStatus;
  /** Input to the pipeline */
  input: unknown;
  /** Name of current agent being executed */
  currentAgent: string | null;
  /** All agent steps */
  steps: AgentStep[];
  /** Results from completed agents */
  results: AgentOutput[];
  /** Creation timestamp */
  createdAt: Date;
  /** Completion timestamp */
  completedAt?: Date;
  /** Error message if failed */
  error?: string;
}

/**
 * Pipeline configuration.
 */
export interface PipelineConfig {
  /** Agents in order of execution */
  agentOrder: string[];
  /** Whether to stop on first error */
  stopOnError: boolean;
  /** Timeout per agent in milliseconds */
  agentTimeout?: number;
  /** Whether to run agents in parallel */
  parallel: boolean;
}

/**
 * Pipeline execution request.
 */
export interface PipelineRequest {
  /** Input data for pipeline */
  input: unknown;
  /** Pipeline configuration */
  config?: Partial<PipelineConfig>;
  /** Context for the pipeline */
  context?: {
    projectName?: string;
    teamName?: string;
    industry?: string;
  };
}

/**
 * Pipeline execution response.
 */
export interface PipelineResponse {
  /** Session ID for tracking */
  sessionId: string;
  /** Initial status */
  status: PipelineStatus;
  /** Estimated completion time */
  estimatedCompletionTime?: number;
}

/**
 * Pipeline status response.
 */
export interface PipelineStatusResponse {
  /** Session ID */
  sessionId: string;
  /** Current status */
  status: PipelineStatus;
  /** Current agent being executed */
  currentAgent: string | null;
  /** Progress percentage (0-100) */
  progress: number;
  /** Completed steps */
  completedSteps: AgentStep[];
  /** Final results if completed */
  results: AgentOutput[];
  /** Error message if failed */
  error?: string;
}

// ============================================================================
// WebSocket Event Types
// ============================================================================

/**
 * Agent pipeline WebSocket message types.
 */
export type AgentWebSocketMessageType =
  | 'agent:iteration_start'
  | 'agent:complete'
  | 'pipeline:complete'
  | 'pipeline:error'
  | 'pipeline:cancelled';

/**
 * Agent pipeline WebSocket message structure.
 */
export interface AgentWebSocketMessage {
  /** Message type */
  type: AgentWebSocketMessageType;
  /** Session ID */
  sessionId: string;
  /** Agent name (if applicable) */
  agentName?: string;
  /** Message payload */
  payload?: unknown;
  /** Timestamp */
  timestamp: string;
  /** Error message (if applicable) */
  error?: string;
}
