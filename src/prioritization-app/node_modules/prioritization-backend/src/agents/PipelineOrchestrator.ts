/**
 * Pipeline Orchestrator for Agent Pipeline Management.
 *
 * This module orchestrates the execution of multiple agents in a pipeline,
 * managing sessions, progress tracking, and WebSocket event broadcasting.
 *
 * @module agents/PipelineOrchestrator
 */

import { WebSocket } from 'ws';
import { MCPClient } from '../mcp/MCPClient';
import { Agent } from '../agents/Agent';
import { PlanningAgent } from './PlanningAgent';
import { DeveloperAgent } from './DeveloperAgent';
import { ReviewerAgent } from './ReviewerAgent';
import {
  PipelineSession,
  AgentOutput,
  PipelineConfig,
  PipelineResponse,
  PipelineStatusResponse,
  AgentWebSocketMessage
} from '../agents/types';

/**
 * Default pipeline configuration.
 */
const DEFAULT_CONFIG: PipelineConfig = {
  agentOrder: ['PlanningAgent', 'DeveloperAgent', 'ReviewerAgent'],
  stopOnError: true,
  agentTimeout: 60000,
  parallel: false
};

/**
 * Pipeline Orchestrator that manages agent execution.
 *
 * Implements Chain of Responsibility pattern for sequential agent execution
 * with support for parallel execution and error handling.
 */
export class PipelineOrchestrator {
  /** Registered agents map */
  private readonly agents: Map<string, Agent>;

  /** Active pipeline sessions */
  private readonly sessions: Map<string, PipelineSession>;

  /** WebSocket clients for real-time updates */
  private readonly wsClients: Map<string, WebSocket>;

  /** MCP Client instance */
  private readonly mcp: MCPClient;

  /**
   * Create a new Pipeline Orchestrator.
   */
  constructor() {
    this.agents = new Map();
    this.sessions = new Map();
    this.wsClients = new Map();
    this.mcp = new MCPClient();

    // Initialize default agents
    this.initializeDefaultAgents();
  }

  /**
   * Initialize default agents with shared MCP client.
   */
  private initializeDefaultAgents(): void {
    const planningAgent = new PlanningAgent(this.mcp);
    const developerAgent = new DeveloperAgent(this.mcp);
    const reviewerAgent = new ReviewerAgent(this.mcp);

    this.registerAgent('PlanningAgent', planningAgent);
    this.registerAgent('DeveloperAgent', developerAgent);
    this.registerAgent('ReviewerAgent', reviewerAgent);

    console.log('[PipelineOrchestrator] Default agents initialized');
  }

  /**
   * Register an agent with the orchestrator.
   * @param name - Agent name
   * @param agent - Agent instance
   */
  registerAgent(name: string, agent: Agent): void {
    this.agents.set(name, agent);
    console.log(`[PipelineOrchestrator] Agent registered: ${name}`);
  }

  /**
   * Get a registered agent by name.
   * @param name - Agent name
   * @returns Agent instance or undefined
   */
  getAgent(name: string): Agent | undefined {
    return this.agents.get(name);
  }

  /**
   * Run the full agent pipeline.
   * @param input - Pipeline input data
   * @param config - Optional pipeline configuration
   * @returns Pipeline response with session ID
   */
  async runPipeline(input: unknown, config?: Partial<PipelineConfig>): Promise<PipelineResponse> {
    const sessionId = this.createSessionId();
    const mergedConfig = { ...DEFAULT_CONFIG, ...config };

    // Create session
    const session: PipelineSession = {
      id: sessionId,
      status: 'pending',
      input,
      currentAgent: null,
      steps: mergedConfig.agentOrder.map((agentName) => ({
        agentName,
        status: 'pending'
      })),
      results: [],
      createdAt: new Date()
    };

    this.sessions.set(sessionId, session);
    console.log(`[PipelineOrchestrator] Session created: ${sessionId}`);

    // Broadcast session start
    this.broadcast({
      type: 'agent:iteration_start',
      sessionId,
      agentName: 'PipelineOrchestrator',
      payload: { status: 'starting' },
      timestamp: new Date().toISOString()
    });

    // Start pipeline execution (async)
    this.executePipeline(sessionId, mergedConfig).catch((error) => {
      console.error('[PipelineOrchestrator] Pipeline error:', error);
    });

    return {
      sessionId,
      status: 'running',
      estimatedCompletionTime: mergedConfig.agentOrder.length * 5000 // ~5s per agent
    };
  }

  /**
   * Execute the pipeline for a session.
   * @param sessionId - Session ID
   * @param config - Pipeline configuration
   */
  private async executePipeline(
    sessionId: string,
    config: PipelineConfig
  ): Promise<void> {
    const session = this.sessions.get(sessionId);
    if (!session) {
      throw new Error(`Session not found: ${sessionId}`);
    }

    session.status = 'running';

    try {
      for (let i = 0; i < config.agentOrder.length; i++) {
        const agentName = config.agentOrder[i];

        // Check if session was cancelled or errored
        const currentStatus = session.status as string;
        if (currentStatus === 'cancelled' || currentStatus === 'error') {
          console.log(`[PipelineOrchestrator] Pipeline cancelled at agent: ${agentName}`);
          return;
        }

        // Get agent
        const agent = this.agents.get(agentName);
        if (!agent) {
          throw new Error(`Agent not found: ${agentName}`);
        }

        // Update session state
        session.currentAgent = agentName;
        const stepIndex = session.steps.findIndex((s) => s.agentName === agentName);
        if (stepIndex === -1) {
          throw new Error(`Step not found for agent: ${agentName}`);
        }

        const step = session.steps[stepIndex];
        step.status = 'running';
        step.startedAt = new Date();

        // Broadcast agent start
        this.broadcast({
          type: 'agent:iteration_start',
          sessionId,
          agentName,
          payload: { step: i + 1, total: config.agentOrder.length },
          timestamp: new Date().toISOString()
        });

        console.log(`[PipelineOrchestrator] Running agent: ${agentName}`);

        try {
          // Execute agent with timeout
          const agentOutput = await this.executeWithTimeout(agent, session.input, config.agentTimeout);

          // Store result
          step.output = agentOutput;
          step.status = 'completed';
          step.endedAt = new Date();
          session.results.push(agentOutput);

          // Broadcast agent completion
          this.broadcast({
            type: 'agent:complete',
            sessionId,
            agentName,
            payload: {
              confidence: agentOutput.confidence,
              reasoning: agentOutput.reasoning.substring(0, 200) + '...'
            },
            timestamp: new Date().toISOString()
          });

          console.log(`[PipelineOrchestrator] Agent completed: ${agentName}`);
        } catch (agentError) {
          const errorMessage = agentError instanceof Error ? agentError.message : 'Unknown error';
          step.status = 'error';
          step.error = errorMessage;
          step.endedAt = new Date();

          console.error(`[PipelineOrchestrator] Agent error: ${agentName} - ${errorMessage}`);

          if (config.stopOnError) {
            session.status = 'error';
            session.error = `Agent ${agentName} failed: ${errorMessage}`;
            session.completedAt = new Date();

            this.broadcast({
              type: 'pipeline:error',
              sessionId,
              agentName,
              error: errorMessage,
              timestamp: new Date().toISOString()
            });

            return;
          }
        }
      }

      // Pipeline completed successfully
      session.status = 'completed';
      session.completedAt = new Date();
      session.currentAgent = null;

      this.broadcast({
        type: 'pipeline:complete',
        sessionId,
        payload: {
          resultsCount: session.results.length,
          duration: session.completedAt.getTime() - session.createdAt.getTime()
        },
        timestamp: new Date().toISOString()
      });

      console.log(`[PipelineOrchestrator] Pipeline completed: ${sessionId}`);
    } catch (error) {
      session.status = 'error';
      session.error = error instanceof Error ? error.message : 'Unknown error';
      session.completedAt = new Date();

      this.broadcast({
        type: 'pipeline:error',
        sessionId,
        error: session.error,
        timestamp: new Date().toISOString()
      });

      console.error('[PipelineOrchestrator] Pipeline error:', error);
    }
  }

  /**
   * Execute agent with timeout.
   * @param agent - Agent to execute
   * @param input - Agent input
   * @param timeout - Timeout in milliseconds
   * @returns Agent output
   */
  private async executeWithTimeout(
    agent: Agent,
    input: unknown,
    timeout?: number
  ): Promise<AgentOutput> {
    const timeoutMs = timeout || 60000;

    const timeoutPromise = new Promise<AgentOutput>((_, reject) => {
      setTimeout(() => {
        reject(new Error(`Agent ${agent.getName()} timed out after ${timeoutMs}ms`));
      }, timeoutMs);
    });

    const executionPromise = agent.execute(input);

    return Promise.race([executionPromise, timeoutPromise]);
  }

  /**
   * Get session status.
   * @param sessionId - Session ID
   * @returns Pipeline status response or null
   */
  getSessionStatus(sessionId: string): PipelineStatusResponse | null {
    const session = this.sessions.get(sessionId);
    if (!session) {
      return null;
    }

    const completedSteps = session.steps.filter((s) => s.status === 'completed');
    const progress = session.steps.length > 0
      ? Math.round((completedSteps.length / session.steps.length) * 100)
      : 0;

    return {
      sessionId: session.id,
      status: session.status,
      currentAgent: session.currentAgent,
      progress,
      completedSteps,
      results: session.results
    };
  }

  /**
   * Get full session details.
   * @param sessionId - Session ID
   * @returns Pipeline session or null
   */
  getSession(sessionId: string): PipelineSession | null {
    return this.sessions.get(sessionId) || null;
  }

  /**
   * Cancel a running pipeline.
   * @param sessionId - Session ID
   * @returns True if cancelled successfully
   */
  cancelPipeline(sessionId: string): boolean {
    const session = this.sessions.get(sessionId);
    if (!session || session.status === 'completed' || session.status === 'cancelled') {
      return false;
    }

    session.status = 'cancelled';
    session.completedAt = new Date();

    this.broadcast({
      type: 'pipeline:cancelled',
      sessionId,
      timestamp: new Date().toISOString()
    });

    console.log(`[PipelineOrchestrator] Pipeline cancelled: ${sessionId}`);
    return true;
  }

  /**
   * Register a WebSocket client for real-time updates.
   * @param sessionId - Session ID to subscribe to
   * @param ws - WebSocket connection
   */
  registerWebSocket(sessionId: string, ws: WebSocket): void {
    this.wsClients.set(sessionId, ws);
    console.log(`[PipelineOrchestrator] WebSocket registered for session: ${sessionId}`);
  }

  /**
   * Unregister a WebSocket client.
   * @param sessionId - Session ID
   */
  unregisterWebSocket(sessionId: string): void {
    this.wsClients.delete(sessionId);
    console.log(`[PipelineOrchestrator] WebSocket unregistered for session: ${sessionId}`);
  }

  /**
   * Broadcast message to session WebSocket clients.
   * @param message - Message to broadcast
   */
  private broadcast(message: AgentWebSocketMessage): void {
    const ws = this.wsClients.get(message.sessionId);
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify(message));
    }

    // Also log to console
    console.log(`[PipelineOrchestrator] Broadcast: ${message.type} for ${message.sessionId}`);
  }

  /**
   * Stream progress to a specific WebSocket.
   * @param sessionId - Session ID
   * @param ws - WebSocket connection
   */
  streamProgress(sessionId: string, ws: WebSocket): void {
    const status = this.getSessionStatus(sessionId);
    if (status && ws.readyState === WebSocket.OPEN) {
      ws.send(
        JSON.stringify({
          type: 'agent:iteration_start',
          sessionId,
          payload: status,
          timestamp: new Date().toISOString()
        } as AgentWebSocketMessage)
      );
    }
  }

  /**
   * Create a unique session ID.
   * @returns Unique session identifier
   */
  private createSessionId(): string {
    return `pipeline_${Date.now()}_${Math.random().toString(36).substring(2, 9)}`;
  }

  /**
   * Get all active sessions.
   * @returns Array of active sessions
   */
  getActiveSessions(): PipelineSession[] {
    return Array.from(this.sessions.values()).filter(
      (s) => s.status === 'running' || s.status === 'pending'
    );
  }

  /**
   * Clean up completed sessions older than specified age.
   * @param maxAgeMs - Maximum age in milliseconds
   * @returns Number of sessions cleaned up
   */
  cleanupSessions(maxAgeMs: number = 3600000): number {
    const now = Date.now();
    let cleaned = 0;

    for (const [id, session] of this.sessions.entries()) {
      if (
        (session.status === 'completed' || session.status === 'error' || session.status === 'cancelled') &&
        session.completedAt &&
        now - session.completedAt.getTime() > maxAgeMs
      ) {
        this.sessions.delete(id);
        cleaned++;
      }
    }

    if (cleaned > 0) {
      console.log(`[PipelineOrchestrator] Cleaned up ${cleaned} sessions`);
    }

    return cleaned;
  }
}

export default PipelineOrchestrator;
