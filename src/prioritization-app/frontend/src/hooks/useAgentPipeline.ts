/**
 * useAgentPipeline Hook for Agent Pipeline Management.
 *
 * This hook provides React state management and API integration
 * for the agent pipeline execution system.
 *
 * @module hooks/useAgentPipeline
 */

import { useState, useCallback, useRef, useEffect } from 'react';

// ============================================================================
// Types
// ============================================================================

/** Pipeline status type */
type PipelineStatus = 'idle' | 'running' | 'complete' | 'error' | 'cancelled';

/** Agent step in pipeline */
interface AgentStep {
  /** Agent name */
  agentName: string;
  /** Step status */
  status: 'pending' | 'running' | 'completed' | 'error';
  /** Output from this agent */
  output?: {
    result: unknown;
    reasoning: string;
    confidence: number;
    agentName: string;
    timestamp: string;
  };
  /** Error message if failed */
  error?: string;
}

/** Agent output type */
interface AgentOutput {
  result: unknown;
  reasoning: string;
  confidence: number;
  agentName: string;
  timestamp: string;
}

/** Hook return type */
interface UseAgentPipelineReturn {
  /** Current session ID */
  sessionId: string | null;
  /** Pipeline status */
  status: PipelineStatus;
  /** Progress percentage (0-100) */
  progress: number;
  /** Current agent being executed */
  currentAgent: string | null;
  /** Results from completed agents */
  results: AgentOutput[];
  /** Completed steps */
  completedSteps: AgentStep[];
  /** Error message if any */
  error: string | null;
  /** Start a new pipeline execution */
  runPipeline: (input: unknown, config?: PipelineConfig) => Promise<string>;
  /** Cancel the running pipeline */
  cancelPipeline: () => Promise<void>;
  /** Get current status from server */
  refreshStatus: () => Promise<void>;
  /** Reset the hook state */
  reset: () => void;
  /** WebSocket connection status */
  isConnected: boolean;
}

/** Pipeline configuration */
interface PipelineConfig {
  /** Agents in order of execution */
  agentOrder?: string[];
  /** Whether to stop on first error */
  stopOnError?: boolean;
  /** Timeout per agent in milliseconds */
  agentTimeout?: number;
  /** Context for the pipeline */
  context?: {
    projectName?: string;
    teamName?: string;
    industry?: string;
  };
}

// ============================================================================
// Configuration
// ============================================================================

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:3001/api/v1';
const WS_BASE_URL = import.meta.env.VITE_WS_URL || 'ws://localhost:3001/ws';

// ============================================================================
// Hook Implementation
// ============================================================================

/**
 * Hook for managing agent pipeline execution.
 *
 * @returns Pipeline management interface
 *
 * @example
 * ```typescript
 * const {
 *   sessionId,
 *   status,
 *   progress,
 *   currentAgent,
 *   results,
 *   runPipeline,
 *   cancelPipeline,
 *   reset
 * } = useAgentPipeline();
 *
 * // Start pipeline
 * await runPipeline({
 *   problem: 'Implement user authentication',
 *   goals: ['Secure', 'Scalable']
 * });
 *
 * // Cancel pipeline
 * await cancelPipeline();
 *
 * // Reset state
 * reset();
 * ```
 */
export function useAgentPipeline(): UseAgentPipelineReturn {
  // State
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [status, setStatus] = useState<PipelineStatus>('idle');
  const [progress, setProgress] = useState<number>(0);
  const [currentAgent, setCurrentAgent] = useState<string | null>(null);
  const [results, setResults] = useState<AgentOutput[]>([]);
  const [completedSteps, setCompletedSteps] = useState<AgentStep[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [isConnected, setIsConnected] = useState<boolean>(false);

  // WebSocket ref
  const wsRef = useRef<WebSocket | null>(null);
  const statusPollInterval = useRef<NodeJS.Timeout | null>(null);

  // Connect to WebSocket for real-time updates
  useEffect(() => {
    const connectWebSocket = () => {
      try {
        wsRef.current = new WebSocket(WS_BASE_URL);

        wsRef.current.onopen = () => {
          console.log('[useAgentPipeline] WebSocket connected');
          setIsConnected(true);
        };

        wsRef.current.onmessage = (event) => {
          try {
            const message = JSON.parse(event.data);
            handleWebSocketMessage(message);
          } catch (err) {
            console.error('[useAgentPipeline] Error parsing WebSocket message:', err);
          }
        };

        wsRef.current.onclose = () => {
          console.log('[useAgentPipeline] WebSocket disconnected');
          setIsConnected(false);
          // Attempt reconnection after delay
          setTimeout(connectWebSocket, 3000);
        };

        wsRef.current.onerror = (err) => {
          console.error('[useAgentPipeline] WebSocket error:', err);
        };
      } catch (err) {
        console.error('[useAgentPipeline] Failed to connect WebSocket:', err);
      }
    };

    connectWebSocket();

    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
      if (statusPollInterval.current) {
        clearInterval(statusPollInterval.current);
      }
    };
  }, []);

  /**
   * Handle incoming WebSocket messages.
   */
  const handleWebSocketMessage = useCallback((message: {
    type: string;
    sessionId?: string;
    agentName?: string;
    payload?: unknown;
    error?: string;
  }) => {
    console.log('[useAgentPipeline] Received message:', message);

    switch (message.type) {
      case 'agent:iteration_start':
        if (message.agentName) {
          setCurrentAgent(message.agentName);
          setStatus('running');
        }
        break;

      case 'agent:complete':
        if (message.payload) {
          const payload = message.payload as { confidence?: number; reasoning?: string };
          setResults((prev) => [
            ...prev,
            {
              result: payload,
              reasoning: payload.reasoning || '',
              confidence: payload.confidence || 0,
              agentName: message.agentName || 'Unknown',
              timestamp: new Date().toISOString()
            }
          ]);
        }
        break;

      case 'pipeline:complete':
        setStatus('complete');
        setProgress(100);
        setCurrentAgent(null);
        if (statusPollInterval.current) {
          clearInterval(statusPollInterval.current);
        }
        break;

      case 'pipeline:error':
        setStatus('error');
        setError(message.error || 'Unknown pipeline error');
        setCurrentAgent(null);
        if (statusPollInterval.current) {
          clearInterval(statusPollInterval.current);
        }
        break;

      case 'pipeline:cancelled':
        setStatus('cancelled');
        setCurrentAgent(null);
        if (statusPollInterval.current) {
          clearInterval(statusPollInterval.current);
        }
        break;

      default:
        console.log('[useAgentPipeline] Unknown message type:', message.type);
    }
  }, []);

  /**
   * Start a new pipeline execution.
   * @param input - Input data for the pipeline
   * @param config - Optional pipeline configuration
   * @returns Session ID for tracking
   */
  const runPipeline = useCallback(async (
    input: unknown,
    config?: PipelineConfig
  ): Promise<string> => {
    // Reset state
    setStatus('running');
    setProgress(0);
    setCurrentAgent(null);
    setResults([]);
    setCompletedSteps([]);
    setError(null);

    try {
      const response = await fetch(`${API_BASE_URL}/agents/run`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          input,
          config,
          context: config?.context
        })
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.message || 'Failed to start pipeline');
      }

      const data = await response.json();
      const newSessionId = data.sessionId;

      setSessionId(newSessionId);
      setProgress(10);

      // Start polling for status updates
      startStatusPolling(newSessionId);

      // Register WebSocket for this session
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({
          type: 'subscribe',
          sessionId: newSessionId
        }));
      }

      return newSessionId;
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Unknown error';
      setStatus('error');
      setError(errorMessage);
      throw err;
    }
  }, []);

  /**
   * Cancel the running pipeline.
   */
  const cancelPipeline = useCallback(async () => {
    if (!sessionId) {
      console.warn('[useAgentPipeline] No session to cancel');
      return;
    }

    try {
      const response = await fetch(`${API_BASE_URL}/agents/cancel/${sessionId}`, {
        method: 'POST'
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.message || 'Failed to cancel pipeline');
      }

      setStatus('cancelled');
      setCurrentAgent(null);

      if (statusPollInterval.current) {
        clearInterval(statusPollInterval.current);
      }
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Unknown error';
      console.error('[useAgentPipeline] Cancel error:', errorMessage);
    }
  }, [sessionId]);

  /**
   * Get current status from server.
   */
  const refreshStatus = useCallback(async () => {
    if (!sessionId) return;

    try {
      const response = await fetch(`${API_BASE_URL}/agents/status/${sessionId}`);

      if (!response.ok) {
        if (response.status === 404) {
          throw new Error('Session not found');
        }
        throw new Error('Failed to fetch status');
      }

      const statusData = await response.json();

      setStatus(statusData.status);
      setProgress(statusData.progress);
      setCurrentAgent(statusData.currentAgent);
      setCompletedSteps(statusData.completedSteps || []);

      if (statusData.status === 'completed' || statusData.status === 'error' || statusData.status === 'cancelled') {
        setStatus(statusData.status);
        if (statusPollInterval.current) {
          clearInterval(statusPollInterval.current);
        }
      }

      if (statusData.error) {
        setError(statusData.error);
      }
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Unknown error';
      console.error('[useAgentPipeline] Refresh status error:', errorMessage);
    }
  }, [sessionId]);

  /**
   * Start polling for status updates.
   */
  const startStatusPolling = useCallback((_id: string) => {
    // Clear any existing interval
    if (statusPollInterval.current) {
      clearInterval(statusPollInterval.current);
    }

    // Poll every 2 seconds
    statusPollInterval.current = setInterval(() => {
      refreshStatus();
    }, 2000);
  }, [refreshStatus]);

  /**
   * Reset the hook state.
   */
  const reset = useCallback(() => {
    setSessionId(null);
    setStatus('idle');
    setProgress(0);
    setCurrentAgent(null);
    setResults([]);
    setCompletedSteps([]);
    setError(null);

    if (statusPollInterval.current) {
      clearInterval(statusPollInterval.current);
    }
  }, []);

  return {
    sessionId,
    status,
    progress,
    currentAgent,
    results,
    completedSteps,
    error,
    runPipeline,
    cancelPipeline,
    refreshStatus,
    reset,
    isConnected
  };
}

export default useAgentPipeline;
