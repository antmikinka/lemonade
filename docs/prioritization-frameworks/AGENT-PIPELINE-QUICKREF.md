# Agent Pipeline Quick Reference (Web Application)

## Overview

This document provides a quick reference for implementing the recursive iterative agent pipeline using Clear Thought MCP tools for the Prioritization Frameworks **Web Application**. Agents run in a Node.js backend process and communicate with the React frontend via HTTP REST API and WebSocket connections.

---

## Pipeline Architecture (Web Context)

```
┌─────────────────────────────────────────────────────────────────┐
│                    FRONTEND (React SPA)                          │
│                                                                  │
│  ┌─────────────┐         ┌─────────────┐         ┌─────────┐   │
│  │   UI        │────────▶│  API        │────────▶│  WS     │   │
│  │   Components│         │  Client     │         │  Client │   │
│  └─────────────┘         └─────────────┘         └─────────┘   │
│         │                         │                    │        │
└─────────┼─────────────────────────┼────────────────────┼────────┘
          │                         │                    │
          │ HTTP/REST               │                    │ WebSocket
          │                         │                    │
┌─────────┼─────────────────────────┼────────────────────┼────────┐
│         ▼                         ▼                    ▼        │
│                    BACKEND (Node.js API)                        │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Express Server                                          │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │   │
│  │  │  REST API    │  │  WebSocket   │  │  Middleware  │   │   │
│  │  │  Routes      │  │  Server      │  │  (CORS, etc) │   │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘   │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Agent Pipeline Orchestrator                             │   │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌───┐ │   │
│  │  │Planning │ │Developer│ │Reviewer │ │ Writer  │ │...│ │   │
│  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘ └───┘ │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Clear Thought MCP Client                                │   │
│  │  (HTTP client to MCP server)                             │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Agent Roles and Responsibilities

### 1. Planning Analysis Strategist

**Purpose:** Creates implementation roadmap and technical strategy

**When to invoke:**
- Starting a new feature/component
- Breaking down complex requirements
- Creating technical specifications

**Clear Thought MCP Tools used:**
| Tool | Purpose | Example |
|------|---------|---------|
| `sequentialthinking` | Step-by-step analysis | Breaking down requirements into tasks |
| `mentalmodel` | First principles thinking | "What is the core problem we're solving?" |
| `decisionframework` | Choosing between options | "Should we use Strategy pattern or Factory pattern?" |
| `visualreasoning` | Architecture diagrams | Creating flowcharts and sequence diagrams |

**Input format (API):**
```json
{
  "task": "Create React component for RICE scoring",
  "requirements": ["Display reach input", "Show impact slider", "Confidence selector"],
  "constraints": ["Must use existing design system", "Mobile responsive"],
  "context": {
    "project": "prioritization-web-app",
    "existingComponents": ["FrameworkSelector", "BacklogList"]
  }
}
```

**Output:**
- Implementation plan with phases
- File structure
- Technical decisions with rationale
- Risk assessment

---

### 2. Senior Developer

**Purpose:** Implements code based on the plan

**When to invoke:**
- After planning agent creates roadmap
- Implementing specific components
- Fixing identified issues

**Clear Thought MCP Tools used:**
| Tool | Purpose | Example |
|------|---------|---------|
| `designpattern` | Applying patterns | "Use Strategy pattern for calculators" |
| `programmingparadigm` | Code structure | "Use functional approach for pure calculations" |
| `sequentialthinking` | Implementation steps | "Step 1: Create types, Step 2: Implement interface..." |

**Input format (API):**
```json
{
  "task": "Implement RICECalculator component",
  "specification": { "...": "planning doc output" },
  "files": ["frontend/src/components/scoring/RICECalculator.tsx"],
  "technicalConstraints": {
    "targetPlatform": "browser",
    "styleGuide": "existing CSS variables"
  }
}
```

**Output:**
- Working code implementation
- Unit tests
- Inline documentation

---

### 3. Quality Reviewer

**Purpose:** Reviews code/documentation for coherence and quality

**When to invoke:**
- After developer completes implementation
- Before merging changes
- When inconsistencies are detected

**Clear Thought MCP Tools used:**
| Tool | Purpose | Example |
|------|---------|---------|
| `metacognitivemonitoring` | Self-assessment of review | "Am I missing edge cases?" |
| `structuredargumentation` | Weighing trade-offs | "This approach is simpler but less flexible because..." |
| `debuggingapproach` | Finding issues | "Using divide and conquer to isolate the problem" |

**Input format (API):**
```json
{
  "review": "code",
  "artifacts": ["RICECalculator.tsx", "RICECalculator.test.tsx"],
  "criteria": ["Correctness", "Completeness", "Performance", "Accessibility"],
  "specificConcerns": ["Check confidence scale consistency"]
}
```

**Output:**
- Coherence analysis
- Technical accuracy review
- Inconsistency report
- Recommendations

---

### 4. Technical Writer Expert

**Purpose:** Creates comprehensive documentation

**When to invoke:**
- After code is implemented and reviewed
- Creating API documentation
- Writing user guides

**Clear Thought MCP Tools used:**
| Tool | Purpose | Example |
|------|---------|---------|
| `sequentialthinking` | Document structure | "Organize from overview to details" |
| `visualreasoning` | Diagrams | "Create sequence diagram for data flow" |

**Input format (API):**
```json
{
  "documentType": "API docs",
  "content": { "...": "implementation code" },
  "audience": "frontend developers",
  "format": "markdown"
}
```

**Output:**
- Well-structured markdown documentation
- Diagrams and visual aids
- Code examples
- Quick start guides

---

### 5. Enhanced Ecosystem Creator

**Purpose:** Creates supporting infrastructure and integrations

**When to invoke:**
- Creating new integrations (Jira, Linear)
- Setting up export functionality
- Generating component templates

**Clear Thought MCP Tools used:**
| Tool | Purpose | Example |
|------|---------|---------|
| `designpattern` | Integration patterns | "Use adapter pattern for Jira API" |
| `sequentialthinking` | Component generation | "Generate export handlers systematically" |

**Input format (API):**
```json
{
  "ecosystem": "Jira Integration",
  "domain": "issue tracking",
  "requiredComponents": ["JiraClient", "IssueMapper", "ExportHandler"]
}
```

**Output:**
- Integration components
- API adapters
- Export functionality

---

## Clear Thought MCP Tools Reference

### Tool Invocation via HTTP

```typescript
// Backend MCP Client Implementation
class MCPClient {
  private serverUrl: string;
  private sessionId: string | null;

  constructor(serverUrl: string) {
    this.serverUrl = serverUrl;
  }

  async connect(): Promise<void> {
    const response = await fetch(`${this.serverUrl}/api/session`, {
      method: 'POST'
    });
    const data = await response.json();
    this.sessionId = data.sessionId;
  }

  async invokeTool(
    toolName: string,
    parameters: Record<string, any>
  ): Promise<any> {
    const response = await fetch(
      `${this.serverUrl}/api/tools/${toolName}`,
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-Session-ID': this.sessionId!
        },
        body: JSON.stringify({ parameters })
      }
    );

    if (!response.ok) {
      throw new Error(`MCP tool failed: ${response.statusText}`);
    }

    return response.json();
  }
}
```

### Core Tools Quick Reference

| Tool | Best For | Parameters |
|------|----------|------------|
| `sequentialthinking` | Breaking down problems | `thought`, `thoughtNumber`, `totalThoughts` |
| `mentalmodel` | Architectural decisions | `model` (first_principles, pareto, etc.) |
| `designpattern` | Code structure | `pattern` (strategy, observer, etc.) |
| `programmingparadigm` | Paradigm selection | `paradigm` (functional, OOP, etc.) |
| `debuggingapproach` | Issue identification | `approach` (binary_search, divide_conquer) |
| `metacognitivemonitoring` | Quality assessment | `monitoringType`, `criteria` |
| `decisionframework` | Trade-off analysis | `analysisType` (pros-cons, weighted) |
| `visualreasoning` | Diagrams | `visualizationType` (flowchart, graph) |

---

## Pipeline API Usage

### Starting a Pipeline Run

```bash
# POST /api/v1/agents/run
curl -X POST http://localhost:3001/api/v1/agents/run \
  -H "Content-Type: application/json" \
  -d '{
    "task": "Create RICE calculator component",
    "context": {
      "project": "prioritization-web-app",
      "framework": "RICE"
    },
    "options": {
      "maxIterations": 3,
      "qualityThreshold": 80
    }
  }'
```

### Response

```json
{
  "pipelineId": "pipe_abc123",
  "status": "accepted",
  "estimatedTime": 30
}
```

### WebSocket Progress Updates

```typescript
// Frontend WebSocket client
const ws = new WebSocket('ws://localhost:3001/ws/agents/pipe_abc123');

ws.onmessage = (event) => {
  const message = JSON.parse(event.data);

  switch (message.type) {
    case 'iteration_start':
      console.log(`Starting iteration ${message.payload.iteration}`);
      break;
    case 'agent_complete':
      console.log(`${message.payload.agent} completed`);
      break;
    case 'pipeline_complete':
      console.log(`Pipeline done! Quality: ${message.payload.qualityScore}`);
      break;
    case 'error':
      console.error(`Error: ${message.payload.error}`);
      break;
  }
};
```

### Checking Pipeline Status

```bash
# GET /api/v1/agents/status/:pipelineId
curl http://localhost:3001/api/v1/agents/status/pipe_abc123
```

```json
{
  "pipelineId": "pipe_abc123",
  "status": "running",
  "currentAgent": "developer",
  "iteration": 2,
  "progress": 65,
  "artifacts": [
    { "type": "planning", "status": "complete" },
    { "type": "implementation", "status": "in_progress" }
  ]
}
```

---

## Pipeline Usage Examples

### Example 1: Creating a New Framework Calculator

**Step 1: Planning (API Call)**
```json
POST /api/v1/agents/run
{
  "task": "Create WSJF calculator for frontend",
  "context": {
    "project": "prioritization-web-app",
    "framework": "WSJF",
    "existingCalculators": ["RICE", "MoSCoW"]
  }
}
```

**Step 2: WebSocket Updates**
```
Iteration 1: Planning agent complete
Iteration 1: Developer agent complete
Iteration 1: Reviewer agent complete - Quality: 75 (needs iteration)
Iteration 2: Planning agent complete (with feedback)
Iteration 2: Developer agent complete (fixes applied)
Iteration 2: Reviewer agent complete - Quality: 88 (APPROVED)
Iteration 2: Writer agent complete
Iteration 2: Ecosystem agent complete
Pipeline complete!
```

**Step 3: Retrieve Results**
```json
GET /api/v1/agents/results/pipe_abc123

{
  "status": "completed",
  "iterations": 2,
  "qualityScore": 88,
  "artifacts": [
    { "type": "code", "file": "WSJFCalculator.ts" },
    { "type": "tests", "file": "WSJFCalculator.test.ts" },
    { "type": "docs", "file": "wsjf.md" }
  ]
}
```

---

## Orchestrator Implementation (Backend)

```typescript
// backend/src/agents/PipelineOrchestrator.ts

export class PipelineOrchestrator extends EventEmitter {
  private maxIterations = 3;
  private qualityThreshold = 80;

  async execute(input: PipelineInput): Promise<PipelineOutput> {
    let iteration = 0;
    let feedback = '';

    while (iteration < this.maxIterations) {
      iteration++;

      // Emit progress via WebSocket
      this.emit('iteration_start', { iteration, agent: 'planning' });

      // Phase 1: Planning
      const planningOutput = await this.runPlanningAgent({
        request: input.request,
        context: input.context,
        previousFeedback: feedback
      });
      this.emit('agent_complete', { iteration, agent: 'planning', output: planningOutput });

      // Phase 2: Development
      this.emit('iteration_start', { iteration, agent: 'developer' });
      const developerOutput = await this.runDeveloperAgent({
        planningBrief: planningOutput,
        codebase: input.codebase
      });
      this.emit('agent_complete', { iteration, agent: 'developer', output: developerOutput });

      // Phase 3: Review
      this.emit('iteration_start', { iteration, agent: 'reviewer' });
      const reviewOutput = await this.runReviewerAgent({
        implementation: developerOutput,
        criteria: planningOutput.qualityCriteria
      });
      this.emit('agent_complete', { iteration, agent: 'reviewer', output: reviewOutput });

      // Check quality threshold
      if (reviewOutput.qualityScore >= this.qualityThreshold && reviewOutput.criticalIssues === 0) {
        // Continue with documentation and ecosystem
        const writerOutput = await this.runWriterAgent({ implementation: developerOutput, reviewOutput });
        const ecosystemOutput = await this.runEcosystemAgent({ implementation: developerOutput, documentation: writerOutput });

        this.emit('pipeline_complete', {
          iterations: iteration,
          qualityScore: reviewOutput.qualityScore,
          artifacts: [developerOutput, writerOutput, ecosystemOutput]
        });

        return {
          implementation: developerOutput,
          documentation: writerOutput,
          ecosystem: ecosystemOutput,
          qualityScore: reviewOutput.qualityScore,
          iterations: iteration
        };
      }

      // Prepare for next iteration
      feedback = reviewOutput.feedbackForPlanning;
      this.emit('iteration_complete', {
        iteration,
        qualityScore: reviewOutput.qualityScore,
        passThreshold: false,
        feedback
      });
    }

    throw new Error(`Failed to meet quality threshold after ${iteration} iterations`);
  }
}
```

---

## Frontend Integration (React)

### Custom Hook for Pipeline

```typescript
// frontend/src/hooks/useAgentPipeline.ts

import { useState, useEffect, useCallback } from 'react';

export function useAgentPipeline() {
  const [status, setStatus] = useState<'idle' | 'running' | 'complete' | 'error'>('idle');
  const [progress, setProgress] = useState(0);
  const [currentAgent, setCurrentAgent] = useState<string | null>(null);
  const [results, setResults] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  const runPipeline = useCallback(async (task: string, context: any) => {
    setStatus('running');
    setError(null);
    setProgress(0);

    // Start pipeline
    const response = await fetch('/api/v1/agents/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ task, context })
    });

    const { pipelineId } = await response.json();

    // Connect WebSocket for real-time updates
    const ws = new WebSocket(`ws://localhost:3001/ws/agents/${pipelineId}`);

    ws.onmessage = (event) => {
      const message = JSON.parse(event.data);

      switch (message.type) {
        case 'iteration_start':
          setCurrentAgent(message.payload.agent);
          break;
        case 'agent_complete':
          // Update progress
          break;
        case 'pipeline_complete':
          setStatus('complete');
          setProgress(100);
          setResults(message.payload);
          ws.close();
          break;
        case 'error':
          setStatus('error');
          setError(message.payload.error);
          ws.close();
          break;
      }
    };

    return pipelineId;
  }, []);

  return { status, progress, currentAgent, results, error, runPipeline };
}
```

### Usage in Component

```tsx
import React from 'react';
import { useAgentPipeline } from '@hooks/useAgentPipeline';

function PipelineDemo() {
  const { status, progress, currentAgent, results, runPipeline } = useAgentPipeline();

  const handleRun = async () => {
    await runPipeline('Create RICE calculator', { framework: 'RICE' });
  };

  return (
    <div>
      <button onClick={handleRun} disabled={status === 'running'}>
        Run Pipeline
      </button>

      {status === 'running' && (
        <div>
          <p>Progress: {progress}%</p>
          <p>Current Agent: {currentAgent}</p>
        </div>
      )}

      {status === 'complete' && (
        <div>
          <p>Quality Score: {results.qualityScore}</p>
          <p>Iterations: {results.iterations}</p>
        </div>
      )}
    </div>
  );
}
```

---

## Best Practices

### 1. Agent Invocation (Web Context)

**DO:**
- Use async/await for API calls
- Handle WebSocket reconnection
- Show progress to users
- Implement retry logic

**DON'T:**
- Block UI during pipeline execution
- Ignore network errors
- Forget to clean up WebSocket connections

### 2. Iteration Management

**DO:**
- Set reasonable iteration limits (3-5)
- Track feedback between iterations
- Display iteration progress to users

**DON'T:**
- Allow infinite loops
- Hide iteration count from users
- Skip quality threshold checks

### 3. MCP Tool Usage

**DO:**
- Implement fallback for MCP unavailability
- Cache tool responses when possible
- Log tool invocations for debugging

**DON'T:**
- Block on MCP timeout (use fallback)
- Expose MCP errors directly to users
- Forget to handle tool failures

---

## Troubleshooting

| Issue | Symptom | Solution |
|-------|---------|----------|
| WebSocket disconnect | Updates stop | Implement reconnection logic |
| MCP timeout | Pipeline hangs | Add timeout + fallback strategy |
| Quality not improving | Same feedback repeated | Increase threshold or adjust criteria |
| CORS errors | Frontend can't reach backend | Configure CORS_ORIGIN in backend |
| Pipeline ID not found | 404 on status check | Check pipeline was started successfully |

---

## Metrics and Monitoring

| Metric | Target | Measurement |
|--------|--------|-------------|
| Pipeline success rate | 95%+ | Completed / Total runs |
| Average iterations | 2-3 | Iterations per task |
| WebSocket latency | < 100ms | Message delivery time |
| API response time (P95) | < 300ms | REST endpoint latency |
| MCP tool invocation | < 5s | Average tool response time |

---

## Conclusion

This web-based agent pipeline provides a robust framework for implementing the Prioritization Frameworks application. The key architectural difference from Electron is:

- **Frontend**: Pure React SPA (no IPC, no main process)
- **Backend**: Node.js API with WebSocket support
- **Communication**: HTTP REST + WebSocket (not IPC)
- **Deployment**: Separate frontend (Vercel) + backend (Railway)

For questions or clarifications, refer to the main documentation at `docs/prioritization-frameworks/`.
