import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
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
// Agent Status Icon Component
// ============================================================================
/**
 * Display status icon for an agent step.
 */
const StatusIcon = ({ status }) => {
    const icons = {
        pending: (_jsx("span", { className: "agent-step-icon agent-step-icon--pending", title: "Pending", children: _jsx("svg", { width: "16", height: "16", viewBox: "0 0 16 16", fill: "currentColor", children: _jsx("circle", { cx: "8", cy: "8", r: "6", fill: "none", stroke: "currentColor", strokeWidth: "2" }) }) })),
        running: (_jsx("span", { className: "agent-step-icon agent-step-icon--running", title: "Running", children: _jsx("svg", { width: "16", height: "16", viewBox: "0 0 16 16", fill: "currentColor", children: _jsx("circle", { cx: "8", cy: "8", r: "6", children: _jsx("animateTransform", { attributeName: "transform", type: "rotate", from: "0 8 8", to: "360 8 8", dur: "1s", repeatCount: "indefinite" }) }) }) })),
        completed: (_jsx("span", { className: "agent-step-icon agent-step-icon--completed", title: "Completed", children: _jsx("svg", { width: "16", height: "16", viewBox: "0 0 16 16", fill: "currentColor", children: _jsx("path", { d: "M8 0a8 8 0 1 1 0 16A8 8 0 0 1 8 0zm3.5 5.5l-4 4-2-2-1.5 1.5 3.5 3.5 5.5-5.5-1.5-1.5z" }) }) })),
        error: (_jsx("span", { className: "agent-step-icon agent-step-icon--error", title: "Error", children: _jsx("svg", { width: "16", height: "16", viewBox: "0 0 16 16", fill: "currentColor", children: _jsx("path", { d: "M8 0a8 8 0 1 1 0 16A8 8 0 0 1 8 0zm0 12a1 1 0 1 0 0-2 1 1 0 0 0 0 2zm1-4V4H7v4h2z" }) }) }))
    };
    return icons[status] || icons.pending;
};
// ============================================================================
// Agent Step Component
// ============================================================================
/**
 * Individual agent step display.
 */
const AgentStepItem = ({ step, isExpanded, onToggle }) => {
    const statusLabels = {
        pending: 'Pending',
        running: 'Running...',
        completed: 'Completed',
        error: 'Error'
    };
    const statusClasses = {
        pending: 'agent-step--pending',
        running: 'agent-step--running',
        completed: 'agent-step--completed',
        error: 'agent-step--error'
    };
    return (_jsxs("div", { className: `agent-step ${statusClasses[step.status] || ''}`, children: [_jsxs("div", { className: "agent-step-header", onClick: onToggle, children: [_jsx(StatusIcon, { status: step.status }), _jsx("span", { className: "agent-step-name", children: step.agentName }), _jsx("span", { className: "agent-step-status", children: statusLabels[step.status] }), step.output && (_jsxs("span", { className: "agent-step-confidence", children: ["Confidence: ", Math.round((step.output.confidence || 0) * 100), "%"] })), _jsx("span", { className: "agent-step-toggle", children: isExpanded ? '▼' : '▶' })] }), isExpanded && step.output && (_jsx("div", { className: "agent-step-details", children: _jsxs("div", { className: "agent-step-reasoning", children: [_jsx("h4", { children: "Reasoning:" }), _jsx("pre", { children: step.output.reasoning })] }) })), step.error && (_jsxs("div", { className: "agent-step-error", children: [_jsx("strong", { children: "Error:" }), " ", step.error] }))] }));
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
export const AgentProgress = ({ onComplete, title = 'Agent Pipeline', showDetails = true, inputData, autoStart = false }) => {
    const { sessionId, status, progress, currentAgent, results, completedSteps, error, runPipeline, cancelPipeline, reset, isConnected } = useAgentPipeline();
    const [expandedSteps, setExpandedSteps] = useState(new Set(['current']));
    const [customInput, setCustomInput] = useState('');
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
    const toggleStep = useCallback((agentName) => {
        setExpandedSteps((prev) => {
            const next = new Set(prev);
            if (next.has(agentName)) {
                next.delete(agentName);
            }
            else {
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
    const getStatusLabel = () => {
        const labels = {
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
    const getStatusClass = () => {
        const classes = {
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
            status: 'pending'
        }));
    return (_jsxs("div", { className: `agent-progress ${getStatusClass()}`, children: [_jsxs("div", { className: "agent-progress-header", children: [_jsx("h3", { className: "agent-progress-title", children: title }), _jsxs("div", { className: "agent-progress-status", children: [_jsx("span", { className: `status-indicator status-indicator--${status}`, children: getStatusLabel() }), sessionId && (_jsxs("span", { className: "session-id", title: sessionId, children: ["ID: ", sessionId.substring(0, 12), "..."] })), _jsx("span", { className: `connection-indicator ${isConnected ? 'connected' : 'disconnected'}`, children: isConnected ? '●' : '○' })] })] }), _jsxs("div", { className: "agent-progress-bar-container", children: [_jsx("div", { className: "agent-progress-bar", style: { width: `${progress}%` } }), _jsxs("span", { className: "agent-progress-percentage", children: [progress, "%"] })] }), showDetails && (_jsx("div", { className: "agent-steps", children: steps.map((step) => (_jsx(AgentStepItem, { step: step, isExpanded: expandedSteps.has(step.agentName) ||
                        expandedSteps.has('current'), onToggle: () => toggleStep(step.agentName) }, step.agentName))) })), results.length > 0 && (_jsxs("div", { className: "agent-results-summary", children: [_jsx("h4", { children: "Results Summary" }), _jsx("ul", { className: "results-list", children: results.map((result, index) => (_jsxs("li", { className: "result-item", children: [_jsx("strong", { children: result.agentName }), _jsxs("span", { children: ["Confidence: ", Math.round(result.confidence * 100), "%"] })] }, index))) })] })), status === 'idle' && (_jsx("div", { className: "agent-input-section", children: _jsx("textarea", { className: "agent-input-textarea", placeholder: "Enter problem statement...", value: customInput, onChange: (e) => setCustomInput(e.target.value), rows: 3 }) })), _jsxs("div", { className: "agent-progress-controls", children: [status === 'idle' && (_jsx("button", { className: "btn btn--primary", onClick: handleStart, disabled: !isConnected, children: "Start Pipeline" })), status === 'running' && (_jsx("button", { className: "btn btn--danger", onClick: cancelPipeline, children: "Cancel" })), (status === 'complete' || status === 'error' || status === 'cancelled') && (_jsx("button", { className: "btn btn--secondary", onClick: reset, children: "Reset" })), !isConnected && (_jsx("span", { className: "connection-warning", children: "Connecting to server..." }))] }), error && (_jsxs("div", { className: "agent-error-display", children: [_jsx("strong", { children: "Error:" }), " ", error] }))] }));
};
export default AgentProgress;
