/**
 * Type definitions for the Prioritization Frameworks Backend API.
 *
 * This module defines Data Transfer Objects (DTOs) and shared types
 * used for API requests and responses.
 *
 * @module types
 */

/**
 * Union type of all supported prioritization frameworks.
 */
export type FrameworkType =
  | 'RICE'           // Reach, Impact, Confidence, Effort
  | 'MoSCoW'         // Must have, Should have, Could have, Won't have
  | 'ValueEffort'    // Value vs Effort matrix
  | 'ICE'            // Impact, Confidence, Ease
  | 'Eisenhower'     // Urgent vs Important matrix
  | 'P0P4'           // Priority levels P0-P4
  | 'WSJF'           // Weighted Shortest Job First
  | 'Kano';          // Kano model for customer satisfaction

/**
 * Base interface for any item that can be prioritized.
 */
export interface PrioritizationItem {
  /** Unique identifier for the item */
  id: string;
  /** Title or name of the item */
  title: string;
  /** Detailed description of the item */
  description?: string;
  /** Category or grouping for the item */
  category?: string;
  /** Framework-specific input data */
  frameworkData?: Record<string, unknown>;
  /** Calculated priority score */
  priorityScore?: number;
  /** Rank position */
  rank?: number;
  /** Creation timestamp */
  createdAt: Date;
  /** Last update timestamp */
  updatedAt: Date;
  /** Optional metadata */
  metadata?: Record<string, unknown>;
}

/**
 * Session representation for session management.
 */
export interface Session {
  /** Unique session identifier */
  id: string;
  /** Session name */
  name: string;
  /** Project name associated with this session */
  projectName?: string;
  /** Project description */
  projectDescription?: string;
  /** Team name */
  teamName?: string;
  /** Industry context */
  industry?: string;
  /** Primary framework used in this session */
  framework: FrameworkType;
  /** Items to be prioritized */
  items: PrioritizationItem[];
  /** Creation timestamp */
  createdAt: Date;
  /** Last update timestamp */
  updatedAt: Date;
}

/**
 * DTO for creating a new session.
 */
export interface CreateSessionDTO {
  /** Session name */
  name: string;
  /** Project name (optional) */
  projectName?: string;
  /** Project description (optional) */
  projectDescription?: string;
  /** Team name (optional) */
  teamName?: string;
  /** Industry context (optional) */
  industry?: string;
  /** Primary framework for this session */
  framework: FrameworkType;
  /** Initial items (optional) */
  items?: Omit<PrioritizationItem, 'id' | 'createdAt' | 'updatedAt'>[];
}

/**
 * DTO for updating an existing session.
 */
export interface UpdateSessionDTO {
  /** Session name (optional) */
  name?: string;
  /** Project name (optional) */
  projectName?: string;
  /** Project description (optional) */
  projectDescription?: string;
  /** Team name (optional) */
  teamName?: string;
  /** Industry context (optional) */
  industry?: string;
  /** Framework (optional) */
  framework?: FrameworkType;
}

/**
 * DTO for prioritization calculation request.
 */
export interface PrioritizeRequestDTO {
  /** Framework to use for calculation */
  framework: FrameworkType;
  /** Item to prioritize */
  item: Omit<PrioritizationItem, 'id' | 'createdAt' | 'updatedAt'>;
  /** Context from other items for normalization (optional) */
  context?: PrioritizationItem[];
}

/**
 * DTO for bulk prioritization request.
 */
export interface BulkPrioritizeRequestDTO {
  /** Framework to use for calculation */
  framework: FrameworkType;
  /** Items to prioritize */
  items: Omit<PrioritizationItem, 'id' | 'createdAt' | 'updatedAt'>[];
}

/**
 * Framework suggestion for auto-fill.
 */
export interface FrameworkSuggestion {
  /** Field being suggested */
  field: string;
  /** Suggested value */
  suggestedValue: unknown;
  /** Confidence score (0-1) */
  confidence: number;
  /** Source of suggestion ('pattern' | 'metrics' | 'historical') */
  source?: 'pattern' | 'metrics' | 'historical';
  /** Reason for suggestion */
  reason: string;
}

/**
 * Collection of framework suggestions.
 */
export interface FrameworkSuggestions {
  /** Framework type */
  framework: FrameworkType;
  /** Array of suggestions */
  suggestions: FrameworkSuggestion[];
  /** Overall confidence (0-1) */
  overallConfidence: number;
}

/**
 * DTO for prioritization response.
 */
export interface PrioritizeResponseDTO {
  /** Calculated priority score */
  priorityScore: number;
  /** Rank position */
  rank?: number;
  /** Framework used */
  framework: FrameworkType;
  /** Details about the calculation */
  details: Record<string, unknown>;
  /** Auto-fill suggestions (if applicable) */
  suggestions?: FrameworkSuggestions;
}

/**
 * DTO for export request.
 */
export interface ExportRequestDTO {
  /** Session ID to export */
  sessionId: string;
  /** Format: 'csv' | 'json' */
  format: 'csv' | 'json';
  /** Fields to include (optional) */
  fields?: string[];
}

/**
 * Export result with file content.
 */
export interface ExportResultDTO {
  /** File name */
  fileName: string;
  /** MIME type */
  mimeType: string;
  /** File content */
  content: string;
  /** Download URL (if applicable) */
  downloadUrl?: string;
}

/**
 * Health check response.
 */
export interface HealthResponseDTO {
  /** Server status */
  status: 'healthy' | 'degraded' | 'unhealthy';
  /** Timestamp */
  timestamp: string;
  /** API version */
  version: string;
  /** Uptime in seconds */
  uptime: number;
  /** Active sessions count */
  activeSessions: number;
}

/**
 * Statistics response.
 */
export interface StatsResponseDTO {
  /** Total sessions */
  totalSessions: number;
  /** Total items across all sessions */
  totalItems: number;
  /** Average items per session */
  avgItemsPerSession: number;
  /** Most used framework */
  mostUsedFramework: FrameworkType;
  /** Sessions by framework */
  sessionsByFramework: Record<FrameworkType, number>;
  /** Server uptime in seconds */
  uptime: number;
}

/**
 * Standard API error response.
 */
export interface ErrorResponseDTO {
  /** Error code */
  error: string;
  /** Error message */
  message: string;
  /** Additional details */
  details?: Record<string, unknown>;
  /** Timestamp */
  timestamp: string;
  /** Request path */
  path: string;
}

/**
 * WebSocket message types.
 */
export type WebSocketMessageType =
  | 'session:created'
  | 'session:updated'
  | 'session:deleted'
  | 'item:added'
  | 'item:updated'
  | 'item:removed'
  | 'prioritization:complete'
  | 'welcome'
  | 'ping'
  | 'pong'
  | 'error';

/**
 * WebSocket message structure.
 */
export interface WebSocketMessage {
  /** Message type */
  type: WebSocketMessageType;
  /** Message payload */
  payload?: unknown;
  /** Message text (for welcome/error messages) */
  message?: string;
  /** Timestamp */
  timestamp: string;
  /** Error message (if applicable) */
  error?: string;
}

// ============================================================================
// Agent Pipeline Types (Phase 6)
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
  output?: AgentOutputDTO;
  /** Error message if failed */
  error?: string;
  /** Start timestamp */
  startedAt?: Date;
  /** End timestamp */
  endedAt?: Date;
}

/**
 * Agent output DTO.
 */
export interface AgentOutputDTO {
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
 * Pipeline session representation.
 */
export interface PipelineSessionDTO {
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
  results: AgentOutputDTO[];
  /** Creation timestamp */
  createdAt: Date;
  /** Completion timestamp */
  completedAt?: Date;
  /** Error message if failed */
  error?: string;
}

/**
 * Pipeline execution request.
 */
export interface PipelineRequestDTO {
  /** Input data for pipeline */
  input: unknown;
  /** Pipeline configuration */
  config?: {
    /** Agents in order of execution */
    agentOrder?: string[];
    /** Whether to stop on first error */
    stopOnError?: boolean;
    /** Timeout per agent in milliseconds */
    agentTimeout?: number;
    /** Whether to run agents in parallel */
    parallel?: boolean;
  };
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
export interface PipelineResponseDTO {
  /** Session ID for tracking */
  sessionId: string;
  /** Initial status */
  status: PipelineStatus;
  /** Estimated completion time in ms */
  estimatedCompletionTime?: number;
}

/**
 * Pipeline status response.
 */
export interface PipelineStatusResponseDTO {
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
  results: AgentOutputDTO[];
  /** Error message if failed */
  error?: string;
}
