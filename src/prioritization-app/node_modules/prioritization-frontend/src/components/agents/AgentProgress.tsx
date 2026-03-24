/**
 * AgentProgress Component for displaying agent pipeline progress.
 *
 * This component shows real-time progress of the agent pipeline execution
 * with visual indicators for each agent step.
 *
 * @module components/agents/AgentProgress
 */

import React, { useState, useCallback } from 'react';
import { useAgentPipeline } from '../../hooks/useAgentPipeline';

// ============================================================================
// Types
// ============================================================================

interface AgentProgressProps {
  /** Optional callback when pipeline completes */
  onComplete?: (results: unknown[]) => void;
  /** Optional custom title */
  title?: string;
  /** Whether to show detailed reasoning */
  showDetails?: boolean;
  /** Custom input data for pipeline */
  inputData?: unknown;
  /** Auto-start pipeline on mount */
  autoStart?: boolean;
}

interface AgentStepProps {
  step: {
    agentName: string;
    status: 'pending' | 'running' | 'completed' | 'error';
    output?: {
      reasoning: string;
      confidence: number;
    };
    error?: string;
  };
  isExpanded: boolean;
  onToggle: () => void;
}

// ============================================================================
// Agent Status Icon Component
// ============================================================================

/**
 * Display status icon for an agent step.
 */
const StatusIcon: React.FC<{ status: string }> = ({ status }) => {
  const icons: Record<string, React.ReactNode> = {
    pending: (
      <span className="agent-step-icon agent-step-icon--pending" title="Pending">
        <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
          <circle cx="8" cy="8" r="6" fill="none" stroke="currentColor" strokeWidth="2" />
        </svg>
      </span>
    ),
    running: (
      <span className="agent-step-icon agent-step-icon--running" title="Running">
        <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
          <circle cx="8" cy="8" r="6">
            <animateTransform
              attributeName="transform"
              type="rotate"
              from="0 8 8"
              to="360 8 8"
              dur="1s"
              repeatCount="indefinite"
            />
          </circle>
        </svg>
      </span>
    ),
    completed: (
      <span className="agent-step-icon agent-step-icon--completed" title="Completed">
        <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
          <path d="M8 0a8 8 0 1 1 0 16A8 8 0 0 1 8 0zm3.5 5.5l-4 4-2-2-1.5 1.5 3.5 3.5 5.5-5.5-1.5-1.5z" />
        </svg>
      </span>
    ),
    error: (
      <span className="agent-step-icon agent-step-icon--error" title="Error">
        <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
          <path d="M8 0a8 8 0 1 1 0 16A8 8 0 0 1 8 0zm0 12a1 1 0 1 0 0-2 1 1 0 0 0 0 2zm1-4V4H7v4h2z" />
        </svg>
      </span>
    )
  };

  return icons[status] || icons.pending;
};

// ============================================================================
// Agent Step Component
// ============================================================================

/**
 * Individual agent step display.
 */
const AgentStepItem: React.FC<AgentStepProps> = ({ step, isExpanded, onToggle }) => {
  const statusLabels: Record<string, string> = {
    pending: 'Pending',
    running: 'Running...',
    completed: 'Completed',
    error: 'Error'
  };

  const statusClasses: Record<string, string> = {
    pending: 'agent-step--pending',
    running: 'agent-step--running',
    completed: 'agent-step--completed',
    error: 'agent-step--error'
  };

  return (
    <div className={`agent-step ${statusClasses[step.status] || ''}`}>
      <div className="agent-step-header" onClick={onToggle}>
        <StatusIcon status={step.status} />
        <span className="agent-step-name">{step.agentName}</span>
        <span className="agent-step-status">{statusLabels[step.status]}</span>
        {step.output && (
          <span className="agent-step-confidence">
            Confidence: {Math.round((step.output.confidence || 0) * 100)}%
          </span>
        )}
        <span className="agent-step-toggle">
          {isExpanded ? '▼' : '▶'}
        </span>
      </div>

      {isExpanded && step.output && (
        <div className="agent-step-details">
          <div className="agent-step-reasoning">
            <h4>Reasoning:</h4>
            <pre>{step.output.reasoning}</pre>
          </div>
        </div>
      )}

      {step.error && (
        <div className="agent-step-error">
          <strong>Error:</strong> {step.error}
        </div>
      )}
    </div>
  );
};

// ============================================================================
// Main AgentProgress Component
// ============================================================================

/**
 * Agent pipeline progress display component.
 *
 * Shows real-time progress of agent pipeline execution with:
 * - Overall progress bar
 * - Individual agent step status
 * - Collapsible reasoning details
 * - Error display
 * - Control buttons
 */
export const AgentProgress: React.FC<AgentProgressProps> = ({
  onComplete,
  title = 'Agent Pipeline',
  showDetails = true,
  inputData,
  autoStart = false
}) => {
  const {
    sessionId,
    status,
    progress,
    currentAgent,
    results,
    completedSteps,
    error,
    runPipeline,
    cancelPipeline,
    reset,
    isConnected
  } = useAgentPipeline();

  const [expandedSteps, setExpandedSteps] = useState<Set<string>>(new Set(['current']));
  const [customInput, setCustomInput] = useState<string>('');

  // Handle pipeline completion
  React.useEffect(() => {
    if (status === 'complete' && onComplete && results.length > 0) {
      onComplete(results.map(r => r.result));
    }
  }, [status, onComplete, results]);

  // Auto-start pipeline
  React.useEffect(() => {
    if (autoStart && inputData) {
      runPipeline(inputData).catch(console.error);
    }
  }, [autoStart, inputData, runPipeline]);

  /**
   * Toggle step expansion.
   */
  const toggleStep = useCallback((agentName: string) => {
    setExpandedSteps((prev) => {
      const next = new Set(prev);
      if (next.has(agentName)) {
        next.delete(agentName);
      } else {
        next.add(agentName);
      }
      return next;
    });
  }, []);

  /**
   * Handle start pipeline with custom input.
   */
  const handleStart = useCallback(() => {
    const input = customInput.trim()
      ? { problem: customInput, goals: ['Efficient', 'Maintainable'] }
      : { problem: 'Implement feature', goals: ['Quality', 'Speed'] };

    runPipeline(input).catch(console.error);
  }, [customInput, runPipeline]);

  /**
   * Get status label.
   */
  const getStatusLabel = (): string => {
    const labels: Record<string, string> = {
      idle: 'Ready to start',
      running: currentAgent ? `Running: ${currentAgent}` : 'Processing...',
      complete: 'Pipeline completed',
      error: error || 'An error occurred',
      cancelled: 'Pipeline cancelled'
    };
    return labels[status] || status;
  };

  /**
   * Get status CSS class.
   */
  const getStatusClass = (): string => {
    const classes: Record<string, string> = {
      idle: 'agent-progress--idle',
      running: 'agent-progress--running',
      complete: 'agent-progress--complete',
      error: 'agent-progress--error',
      cancelled: 'agent-progress--cancelled'
    };
    return classes[status] || '';
  };

  // Default agents for display
  const defaultAgents = ['PlanningAgent', 'DeveloperAgent', 'ReviewerAgent'];

  // Build step list from completed steps and default agents
  const steps = completedSteps.length > 0
    ? completedSteps
    : defaultAgents.map((agentName) => ({
        agentName,
        status: 'pending' as const
      }));

  return (
    <div className={`agent-progress ${getStatusClass()}`}>
      {/* Header */}
      <div className="agent-progress-header">
        <h3 className="agent-progress-title">{title}</h3>
        <div className="agent-progress-status">
          <span className={`status-indicator status-indicator--${status}`}>
            {getStatusLabel()}
          </span>
          {sessionId && (
            <span className="session-id" title={sessionId}>
              ID: {sessionId.substring(0, 12)}...
            </span>
          )}
          <span className={`connection-indicator ${isConnected ? 'connected' : 'disconnected'}`}>
            {isConnected ? '●' : '○'}
          </span>
        </div>
      </div>

      {/* Progress Bar */}
      <div className="agent-progress-bar-container">
        <div
          className="agent-progress-bar"
          style={{ width: `${progress}%` }}
        />
        <span className="agent-progress-percentage">{progress}%</span>
      </div>

      {/* Agent Steps */}
      {showDetails && (
        <div className="agent-steps">
          {steps.map((step) => (
            <AgentStepItem
              key={step.agentName}
              step={step}
              isExpanded={
                expandedSteps.has(step.agentName) ||
                expandedSteps.has('current')
              }
              onToggle={() => toggleStep(step.agentName)}
            />
          ))}
        </div>
      )}

      {/* Results Summary */}
      {results.length > 0 && (
        <div className="agent-results-summary">
          <h4>Results Summary</h4>
          <ul className="results-list">
            {results.map((result, index) => (
              <li key={index} className="result-item">
                <strong>{result.agentName}</strong>
                <span>Confidence: {Math.round(result.confidence * 100)}%</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Input Field (when idle) */}
      {status === 'idle' && (
        <div className="agent-input-section">
          <textarea
            className="agent-input-textarea"
            placeholder="Enter problem statement..."
            value={customInput}
            onChange={(e) => setCustomInput(e.target.value)}
            rows={3}
          />
        </div>
      )}

      {/* Control Buttons */}
      <div className="agent-progress-controls">
        {status === 'idle' && (
          <button
            className="btn btn--primary"
            onClick={handleStart}
            disabled={!isConnected}
          >
            Start Pipeline
          </button>
        )}

        {status === 'running' && (
          <button
            className="btn btn--danger"
            onClick={cancelPipeline}
          >
            Cancel
          </button>
        )}

        {(status === 'complete' || status === 'error' || status === 'cancelled') && (
          <button
            className="btn btn--secondary"
            onClick={reset}
          >
            Reset
          </button>
        )}

        {!isConnected && (
          <span className="connection-warning">
            Connecting to server...
          </span>
        )}
      </div>

      {/* Error Display */}
      {error && (
        <div className="agent-error-display">
          <strong>Error:</strong> {error}
        </div>
      )}
    </div>
  );
};

export default AgentProgress;
