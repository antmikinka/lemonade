# Agent Pipeline Specification

## Overview

This document defines the recursive iterative agent pipeline architecture for the Prioritization Frameworks web application. The pipeline coordinates five specialized agents that collaborate through clear handoff protocols, enhanced by Clear Thought MCP cognitive tools. Agents run in a Node.js backend process and communicate with the frontend via HTTP REST API and WebSocket connections.

## Pipeline Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Frontend (React SPA)                                  │
│                                                                          │
│  ┌─────────────┐         ┌─────────────┐         ┌─────────────────┐    │
│  │   UI        │────────▶│  API        │────────▶│  WebSocket      │    │
│  │   Components│         │  Client     │         │  Client         │    │
│  └─────────────┘         └─────────────┘         └─────────────────┘    │
│         │                         │                        │            │
│         │                         │ HTTP/REST              │ WS         │
│         │                         │                        │            │
└─────────┼─────────────────────────┼────────────────────────┼────────────┘
          │                         │                        │
          ▼                         ▼                        ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    Backend (Node.js API)                                 │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  Express/Fastify Server                                         │    │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │    │
│  │  │  REST API    │  │  WebSocket   │  │  Middleware          │   │    │
│  │  │  Routes      │  │  Server      │  │  (CORS, Auth, etc.)  │   │    │
│  │  └──────────────┘  └──────────────┘  └──────────────────────┘   │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                              │                                           │
│                              ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  Agent Pipeline Orchestrator                                    │    │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌───────────┐  │    │
│  │  │Planning │ │Developer│ │Reviewer │ │ Writer  │ │ Ecosystem │  │    │
│  │  │ Agent   │ │ Agent   │ │ Agent   │ │ Agent   │ │ Agent     │  │    │
│  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘ └───────────┘  │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                              │                                           │
│                              ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  Clear Thought MCP Client                                       │    │
│  │  - sequentialthinking    - decisionframework                    │    │
│  │  - mentalmodel           - visualreasoning                      │    │
│  │  - designpattern         - debuggingapproach                    │    │
│  │  - programmingparadigm   - metacognitivemonitoring              │    │
│  └─────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
```

### Agent Roles

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Agent Pipeline Orchestrator                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  1. Planning Analysis Strategist                                  │   │
│  │     - Creates implementation roadmap                              │   │
│  │     - Defines framework selection criteria                        │   │
│  │     - MCP Tools: sequentialthinking, mentalmodel, decisionframework│   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                              │                                           │
│                              ▼                                           │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  2. Senior Developer                                              │   │
│  │     - Implements code based on roadmap                            │   │
│  │     - Follows architectural patterns                              │   │
│  │     - MCP Tools: programmingparadigm, designpattern               │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                              │                                           │
│                              ▼                                           │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  3. Quality Reviewer                                              │   │
│  │     - Reviews for coherence and correctness                       │   │
│  │     - Identifies gaps and issues                                  │   │
│  │     - MCP Tools: debuggingapproach, metacognitivemonitoring       │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                              │                                           │
│              ┌───────────────┴───────────────┐                          │
│              │                               │                          │
│              ▼                               ▼                          │
│  ┌──────────────────┐            ┌──────────────────┐                   │
│  │ 4. Technical     │            │ 5. Ecosystem     │                   │
│  │    Writer        │            │    Creator       │                   │
│  │    - Docs        │            │    - Integrations│                   │
│  │    - Rationale   │            │    - Validation  │                   │
│  └──────────────────┘            └──────────────────┘                   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Pipeline Execution Flow                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Frontend                         Backend                                │
│     │                                │                                   │
│     │  POST /api/v1/agents/run       │                                   │
│     │  { task: "Create RICE calc",   │                                   │
│     │    context: {...} }            │                                   │
│     │───────────────────────────────▶│                                   │
│     │                                │                                   │
│     │  202 Accepted                  │                                   │
│     │  { pipelineId: "abc123" }      │                                   │
│     │◀───────────────────────────────│                                   │
│     │                                │                                   │
│     │  WebSocket Connect             │                                   │
│     │  /ws/agents/abc123             │                                   │
│     │════════════════════════════════▶│                                   │
│     │                                │                                   │
│     │  Iteration Updates             │                                   │
│     │  { iteration: 1,               │                                   │
│     │    agent: "planning",          │                                   │
│     │    status: "completed",        │                                   │
│     │    output: {...} }             │                                   │
│     │◀═══════════════════════════════│                                   │
│     │                                │                                   │
│     │  Final Result                  │                                   │
│     │  { status: "completed",        │                                   │
│     │    iterations: 2,              │                                   │
│     │    artifacts: [...] }          │                                   │
│     │◀───────────────────────────────│                                   │
│     │                                │                                   │
└─────────────────────────────────────────────────────────────────────────┘
```

## Agent Specifications

### 1. Planning Analysis Strategist

**Role:** Creates implementation roadmap and strategic direction

**Responsibilities:**
- Analyze user requirements and context
- Select appropriate prioritization frameworks
- Define implementation milestones
- Identify potential risks and dependencies
- Create strategic handoff brief for Developer

**Input:**
```typescript
interface PlanningInput {
  userRequest: string;
  context: {
    project: string;
    constraints: string[];
    preferences: Record<string, any>;
  };
  previousIterations: IterationResult[];
}
```

**Output:**
```typescript
interface PlanningOutput {
  roadmap: {
    phases: ImplementationPhase[];
    frameworkSelection: FrameworkType[];
    milestones: Milestone[];
  };
  risks: RiskAssessment[];
  developerBrief: string;
  qualityCriteria: QualityCriteria;
}
```

**MCP Tool Usage:**
| Tool | Purpose |
|------|---------|
| sequentialthinking | Step-by-step roadmap creation |
| mentalmodel | Framework selection reasoning |
| decisionframework | Priority and trade-off analysis |

---

### 2. Senior Developer

**Role:** Implements code based on strategic roadmap

**Responsibilities:**
- Review Planning brief and roadmap
- Implement TypeScript/React code
- Follow architectural patterns and best practices
- Create type definitions and interfaces
- Prepare code for review

**Input:**
```typescript
interface DeveloperInput {
  planningBrief: PlanningOutput;
  existingCodebase: CodebaseContext;
  technicalConstraints: {
    targetPlatform: string;
    dependencies: string[];
    styleGuide: string;
  };
}
```

**Output:**
```typescript
interface DeveloperOutput {
  implementation: {
    files: CodeFile[];
    types: TypeDefinition[];
    components: ComponentDefinition[];
  };
  technicalDecisions: TechnicalDecision[];
  reviewerBrief: string;
}
```

**MCP Tool Usage:**
| Tool | Purpose |
|------|---------|
| programmingparadigm | Code structure and patterns |
| designpattern | Architecture pattern selection |

---

### 3. Quality Reviewer

**Role:** Reviews implementation for coherence and correctness

**Responsibilities:**
- Review code against quality criteria
- Identify gaps, bugs, and inconsistencies
- Assess completeness against requirements
- Provide actionable feedback
- Calculate quality score

**Input:**
```typescript
interface ReviewerInput {
  implementation: DeveloperOutput;
  qualityCriteria: QualityCriteria;
  originalRequest: string;
}
```

**Output:**
```typescript
interface ReviewerOutput {
  qualityScore: number;  // 0-100
  issues: Issue[];
  suggestions: Suggestion[];
  passThreshold: boolean;
  feedbackForPlanning: string;
  writerBrief: string;
}
```

**MCP Tool Usage:**
| Tool | Purpose |
|------|---------|
| debuggingapproach | Systematic issue identification |
| metacognitivemonitoring | Quality threshold assessment |

---

### 4. Technical Writer

**Role:** Creates documentation and rationale

**Responsibilities:**
- Document implementation decisions
- Create user-facing documentation
- Write code comments and JSDoc
- Generate API documentation
- Create usage examples

**Input:**
```typescript
interface WriterInput {
  implementation: DeveloperOutput;
  reviewerOutput: ReviewerOutput;
  documentationRequirements: DocRequirements;
}
```

**Output:**
```typescript
interface WriterOutput {
  documentation: {
    readme: string;
    apiDocs: string;
    usageGuide: string;
    examples: Example[];
  };
  rationale: string;
  ecosystemBrief: string;
}
```

**MCP Tool Usage:**
| Tool | Purpose |
|------|---------|
| structuredargumentation | Clear documentation structure |
| visualreasoning | Diagram and chart generation |

---

### 5. Ecosystem Creator

**Role:** Creates integrations and validates completeness

**Responsibilities:**
- Create integration points
- Validate against original requirements
- Ensure cross-framework compatibility
- Generate export functionality
- Final validation pass

**Input:**
```typescript
interface EcosystemInput {
  implementation: DeveloperOutput;
  documentation: WriterOutput;
  integrationRequirements: IntegrationRequirements;
}
```

**Output:**
```typescript
interface EcosystemOutput {
  integrations: Integration[];
  exports: ExportFunction[];
  validationReport: ValidationReport;
  finalRecommendations: string;
}
```

**MCP Tool Usage:**
| Tool | Purpose |
|------|---------|
| collaborativereasoning | Integration compatibility |
| scientificmethod | Validation and testing |

---

## Iteration Protocol

### Quality Threshold System

```typescript
interface QualityThreshold {
  minimumScore: number;      // e.g., 80
  criticalIssues: number;    // e.g., 0 allowed
  majorIssues: number;       // e.g., 2 allowed
  maxIterations: number;     // e.g., 3
}

const DEFAULT_THRESHOLDS: QualityThreshold = {
  minimumScore: 80,
  criticalIssues: 0,
  majorIssues: 2,
  maxIterations: 3
};
```

### Feedback Loop Structure

```typescript
interface FeedbackLoop {
  iteration: number;
  reviewerFeedback: string;
  planningAdjustments: string[];
  developerChanges: string[];
  qualityImprovement: number; // Delta from previous
}

function processFeedback(feedback: FeedbackLoop): PlanningAdjustment {
  // Analyze reviewer feedback
  const issues = categorizeIssues(feedback.reviewerFeedback);

  // Determine necessary adjustments
  const adjustments = issues.map(issue => ({
    phase: issue.phase,
    action: issue.suggestedAction,
    priority: issue.severity
  }));

  return {
    iteration: feedback.iteration + 1,
    adjustments,
    focusAreas: identifyFocusAreas(issues)
  };
}
```

### Iteration Decision Tree

```
┌─────────────────────────────────────────────────────────────────┐
│              Iteration Decision Logic                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Quality Score >= 80?                                           │
│                                                                  │
│  YES ──▶ Check Critical Issues                                  │
│          │                                                       │
│          Critical == 0?                                          │
│          │                                                       │
│          YES ──▶ PASS ──▶ Finalize                              │
│          │                                                       │
│          NO ──▶ Continue Iteration                              │
│                                                                  │
│  NO ──▶ Iteration Count < Max?                                  │
│          │                                                       │
│          YES ──▶ Continue Iteration                             │
│          │                                                       │
│          NO ──▶ PASS WITH NOTES ──▶ Finalize                    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## API Integration

### REST API Endpoints

```typescript
// Pipeline execution endpoint
POST /api/v1/agents/run

// Request body
interface RunPipelineRequest {
  task: string;
  context?: Record<string, any>;
  options?: {
    maxIterations?: number;
    qualityThreshold?: number;
    agents?: AgentRole[]; // Run specific agents only
  };
}

// Response
interface RunPipelineResponse {
  pipelineId: string;
  status: 'accepted' | 'running' | 'completed' | 'failed';
  estimatedTime?: number; // seconds
}

// Status endpoint
GET /api/v1/agents/status/:pipelineId

// Response
interface PipelineStatus {
  pipelineId: string;
  status: 'running' | 'completed' | 'failed';
  currentAgent?: AgentRole;
  iteration: number;
  progress: number; // 0-100
  artifacts: ArtifactSummary[];
}
```

### WebSocket Protocol

```typescript
// Connection
ws://localhost:3001/ws/agents/:pipelineId

// Server -> Client messages
interface ServerMessage {
  type: 'iteration_start' | 'iteration_complete' | 'agent_complete' | 'pipeline_complete' | 'error';
  payload: {
    iteration?: number;
    agent?: AgentRole;
    status?: string;
    output?: any;
    error?: string;
  };
}

// Example message flow
{
  type: 'iteration_start',
  payload: { iteration: 1, agent: 'planning' }
}
{
  type: 'agent_complete',
  payload: { iteration: 1, agent: 'planning', output: {...} }
}
{
  type: 'iteration_complete',
  payload: { iteration: 1, qualityScore: 75, passThreshold: false }
}
{
  type: 'iteration_start',
  payload: { iteration: 2, agent: 'planning' }
}
{
  type: 'pipeline_complete',
  payload: { iterations: 2, finalQualityScore: 85, artifacts: [...] }
}
```

## Clear Thought MCP Integration

### Tool Invocation Patterns

```typescript
interface MCPToolInvocation {
  toolName: string;
  parameters: Record<string, any>;
  expectedOutput: string;
  fallbackStrategy: string;
}

const AGENT_MCP_MAPPINGS: Record<AgentRole, MCPToolInvocation[]> = {
  PLANNING: [
    {
      toolName: 'sequentialthinking',
      parameters: {
        thought: 'Analyze user requirements...',
        nextThoughtNeeded: true,
        thoughtNumber: 1,
        totalThoughts: 5
      },
      expectedOutput: 'Structured analysis',
      fallbackStrategy: 'Internal reasoning'
    },
    {
      toolName: 'mentalmodel',
      parameters: {
        model: 'Decision Matrix',
        context: 'Framework selection'
      },
      expectedOutput: 'Framework recommendation',
      fallbackStrategy: 'Rule-based selection'
    }
  ],
  DEVELOPER: [
    {
      toolName: 'programmingparadigm',
      parameters: {
        paradigm: 'Object-Oriented',
        language: 'TypeScript'
      },
      expectedOutput: 'Code structure guidance',
      fallbackStrategy: 'Standard patterns'
    },
    {
      toolName: 'designpattern',
      parameters: {
        pattern: 'Strategy',
        context: 'Framework calculators'
      },
      expectedOutput: 'Pattern implementation',
      fallbackStrategy: 'Direct implementation'
    }
  ],
  REVIEWER: [
    {
      toolName: 'debuggingapproach',
      parameters: {
        approach: 'Systematic Code Review',
        focus: ['Correctness', 'Completeness', 'Performance']
      },
      expectedOutput: 'Issue identification',
      fallbackStrategy: 'Checklist review'
    },
    {
      toolName: 'metacognitivemonitoring',
      parameters: {
        monitoringType: 'Quality Assessment',
        criteria: ['Correctness', 'Clarity', 'Completeness']
      },
      expectedOutput: 'Quality score',
      fallbackStrategy: 'Rubric-based scoring'
    }
  ],
  WRITER: [
    {
      toolName: 'structuredargumentation',
      parameters: {
        structure: 'Claim-Evidence-Warrant',
        topic: 'Implementation rationale'
      },
      expectedOutput: 'Structured documentation',
      fallbackStrategy: 'Standard documentation format'
    },
    {
      toolName: 'visualreasoning',
      parameters: {
        visualizationType: 'Architecture Diagram',
        content: 'Component relationships'
      },
      expectedOutput: 'Diagram specification',
      fallbackStrategy: 'Text description'
    }
  ],
  ECOSYSTEM: [
    {
      toolName: 'collaborativereasoning',
      parameters: {
        reasoningMode: 'Integration Validation',
        stakeholders: ['Developer', 'User', 'Maintainer']
      },
      expectedOutput: 'Integration assessment',
      fallbackStrategy: 'Compatibility checklist'
    },
    {
      toolName: 'scientificmethod',
      parameters: {
        hypothesis: 'Implementation meets requirements',
        testMethod: 'Validation against criteria'
      },
      expectedOutput: 'Validation report',
      fallbackStrategy: 'Manual validation'
    }
  ]
};
```

### MCP Client Implementation

```typescript
// backend/src/mcp/mcp-client.ts

import { EventEmitter } from 'events';

export class MCPClient extends EventEmitter {
  private serverUrl: string;
  private sessionId: string | null;

  constructor(serverUrl: string) {
    super();
    this.serverUrl = serverUrl;
    this.sessionId = null;
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
    if (!this.sessionId) {
      throw new Error('MCP client not connected');
    }

    const response = await fetch(`${this.serverUrl}/api/tools/${toolName}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-Session-ID': this.sessionId
      },
      body: JSON.stringify({ parameters })
    });

    if (!response.ok) {
      throw new Error(`MCP tool invocation failed: ${response.statusText}`);
    }

    return response.json();
  }

  async disconnect(): Promise<void> {
    if (this.sessionId) {
      await fetch(`${this.serverUrl}/api/session/${this.sessionId}`, {
        method: 'DELETE'
      });
      this.sessionId = null;
    }
  }
}
```

## Pipeline Orchestration

### Orchestrator Implementation

```typescript
// backend/src/agents/PipelineOrchestrator.ts

import { EventEmitter } from 'events';
import { MCPClient } from '../mcp/mcp-client';
import { PlanningAgent } from './agents/planning-agent';
import { DeveloperAgent } from './agents/developer-agent';
import { ReviewerAgent } from './agents/reviewer-agent';
import { WriterAgent } from './agents/writer-agent';
import { EcosystemAgent } from './agents/ecosystem-agent';

export class PipelineOrchestrator extends EventEmitter {
  private agents: Record<AgentRole, IAgent>;
  private mcpClient: MCPClient;
  private maxIterations = 3;
  private qualityThreshold = 80;

  constructor(mcpClient: MCPClient) {
    super();
    this.mcpClient = mcpClient;
    this.agents = {
      PLANNING: new PlanningAgent(mcpClient),
      DEVELOPER: new DeveloperAgent(mcpClient),
      REVIEWER: new ReviewerAgent(mcpClient),
      WRITER: new WriterAgent(mcpClient),
      ECOSYSTEM: new EcosystemAgent(mcpClient)
    };
  }

  async execute(input: PipelineInput): Promise<PipelineOutput> {
    let planningOutput: PlanningOutput;
    let developerOutput: DeveloperOutput;
    let reviewerOutput: ReviewerOutput;
    let iteration = 0;
    let feedback = '';

    do {
      iteration++;
      this.emit('iteration_start', { iteration, agent: 'planning' });

      // Phase 1: Planning
      planningOutput = await this.agents.PLANNING.execute({
        userRequest: input.request,
        context: input.context,
        previousFeedback: feedback
      });
      this.emit('agent_complete', { iteration, agent: 'planning', output: planningOutput });

      // Phase 2: Development
      this.emit('iteration_start', { iteration, agent: 'developer' });
      developerOutput = await this.agents.DEVELOPER.execute({
        planningBrief: planningOutput,
        existingCodebase: input.codebase,
        technicalConstraints: input.constraints
      });
      this.emit('agent_complete', { iteration, agent: 'developer', output: developerOutput });

      // Phase 3: Review
      this.emit('iteration_start', { iteration, agent: 'reviewer' });
      reviewerOutput = await this.agents.REVIEWER.execute({
        implementation: developerOutput,
        qualityCriteria: planningOutput.qualityCriteria,
        originalRequest: input.request
      });
      this.emit('agent_complete', { iteration, agent: 'reviewer', output: reviewerOutput });

      // Check if quality threshold met
      if (reviewerOutput.passThreshold) {
        // Phase 4: Documentation
        this.emit('iteration_start', { iteration, agent: 'writer' });
        const writerOutput = await this.agents.WRITER.execute({
          implementation: developerOutput,
          reviewerOutput,
          documentationRequirements: input.docRequirements
        });
        this.emit('agent_complete', { iteration, agent: 'writer', output: writerOutput });

        // Phase 5: Ecosystem
        this.emit('iteration_start', { iteration, agent: 'ecosystem' });
        const ecosystemOutput = await this.agents.ECOSYSTEM.execute({
          implementation: developerOutput,
          documentation: writerOutput,
          integrationRequirements: input.integrationRequirements
        });
        this.emit('agent_complete', { iteration, agent: 'ecosystem', output: ecosystemOutput });

        this.emit('pipeline_complete', {
          iterations: iteration,
          qualityScore: reviewerOutput.qualityScore,
          artifacts: [developerOutput, writerOutput, ecosystemOutput]
        });

        return {
          implementation: developerOutput,
          documentation: writerOutput,
          ecosystem: ecosystemOutput,
          qualityScore: reviewerOutput.qualityScore,
          iterations: iteration
        };
      }

      // Prepare feedback for next iteration
      feedback = reviewerOutput.feedbackForPlanning;
      this.emit('iteration_complete', {
        iteration,
        qualityScore: reviewerOutput.qualityScore,
        passThreshold: false,
        feedback
      });

    } while (iteration < this.maxIterations);

    // Max iterations reached, return best effort
    throw new Error(
      `Failed to meet quality threshold (${this.qualityThreshold}) after ${iteration} iterations. Best score: ${reviewerOutput.qualityScore}`
    );
  }
}
```

### API Routes Implementation

```typescript
// backend/src/api/routes.ts

import { Router } from 'express';
import { PipelineOrchestrator } from '../agents/PipelineOrchestrator';
import { WebSocketServer } from 'ws';

const router = Router();
const activePipelines = new Map<string, PipelineOrchestrator>();

// POST /api/v1/agents/run
router.post('/agents/run', async (req, res) => {
  const { task, context, options } = req.body;

  const pipelineId = generateId();
  const orchestrator = new PipelineOrchestrator(mcpClient);

  // Set up WebSocket for real-time updates
  const ws = wss.clients.has(pipelineId);

  // Emit events to WebSocket
  orchestrator.on('iteration_start', (data) => {
    ws?.send(JSON.stringify({ type: 'iteration_start', payload: data }));
  });
  orchestrator.on('agent_complete', (data) => {
    ws?.send(JSON.stringify({ type: 'agent_complete', payload: data }));
  });
  orchestrator.on('pipeline_complete', (data) => {
    ws?.send(JSON.stringify({ type: 'pipeline_complete', payload: data }));
  });

  activePipelines.set(pipelineId, orchestrator);

  // Start pipeline (don't await - run async)
  orchestrator.execute({ request: task, context, ...options })
    .then(result => {
      // Store result for retrieval
      storeResult(pipelineId, result);
    })
    .catch(error => {
      ws?.send(JSON.stringify({ type: 'error', payload: { error: error.message } }));
    })
    .finally(() => {
      activePipelines.delete(pipelineId);
    });

  res.status(202).json({
    pipelineId,
    status: 'accepted',
    estimatedTime: 30 // seconds
  });
});

// GET /api/v1/agents/status/:pipelineId
router.get('/agents/status/:pipelineId', (req, res) => {
  const { pipelineId } = req.params;
  const status = getPipelineStatus(pipelineId);
  res.json(status);
});

// WebSocket upgrade
wss.on('connection', (ws, req) => {
  const pipelineId = extractPipelineId(req.url);
  // Handle WebSocket messages
});

export default router;
```

## Error Handling

### Pipeline Error Types

```typescript
enum PipelineErrorType {
  AGENT_TIMEOUT = 'AGENT_TIMEOUT',
  AGENT_FAILURE = 'AGENT_FAILURE',
  QUALITY_DEGRADATION = 'QUALITY_DEGRADATION',
  MAX_ITERATIONS_EXCEEDED = 'MAX_ITERATIONS_EXCEEDED',
  HANDOFF_FAILURE = 'HANDOFF_FAILURE',
  MCP_UNAVAILABLE = 'MCP_UNAVAILABLE'
}

interface PipelineError {
  type: PipelineErrorType;
  agent: AgentRole;
  message: string;
  recoveryStrategy: RecoveryStrategy;
}
```

### Recovery Strategies

```typescript
interface RecoveryStrategy {
  action: 'RETRY' | 'SKIP_AGENT' | 'FALLBACK' | 'ABORT';
  maxRetries: number;
  fallbackAgent?: AgentRole;
  notificationRequired: boolean;
}

const ERROR_RECOVERY: Record<PipelineErrorType, RecoveryStrategy> = {
  [PipelineErrorType.AGENT_TIMEOUT]: {
    action: 'RETRY',
    maxRetries: 2,
    notificationRequired: false
  },
  [PipelineErrorType.AGENT_FAILURE]: {
    action: 'FALLBACK',
    maxRetries: 1,
    fallbackAgent: 'INTERNAL',
    notificationRequired: true
  },
  [PipelineErrorType.MCP_UNAVAILABLE]: {
    action: 'FALLBACK',
    maxRetries: 0,
    notificationRequired: true
  },
  [PipelineErrorType.QUALITY_DEGRADATION]: {
    action: 'SKIP_AGENT',
    maxRetries: 0,
    notificationRequired: true
  },
  [PipelineErrorType.MAX_ITERATIONS_EXCEEDED]: {
    action: 'ABORT',
    maxRetries: 0,
    notificationRequired: true
  },
  [PipelineErrorType.HANDOFF_FAILURE]: {
    action: 'RETRY',
    maxRetries: 3,
    notificationRequired: false
  }
};
```

## Performance Metrics

### Pipeline KPIs

```typescript
interface PipelineMetrics {
  averageIterations: number;
  averageQualityScore: number;
  averageExecutionTime: number; // milliseconds
  successRate: number;          // Percentage
  agentPerformance: Record<AgentRole, AgentMetrics>;
}

interface AgentMetrics {
  averageExecutionTime: number;
  failureRate: number;
  feedbackQuality: number;     // How actionable is feedback
  iterationContribution: number; // How much quality improves per iteration
}
```

### Performance Targets

| Metric | Target | Measurement |
|--------|--------|-------------|
| Pipeline Success Rate | > 95% | Passes threshold / Total runs |
| Average Iterations | < 2 | Mean iterations to pass |
| Average Quality Score | > 85 | Mean final quality score |
| P95 Execution Time | < 30s | 95th percentile latency |
| Agent Failure Rate | < 1% | Failures / Total agent calls |
| WebSocket Latency | < 100ms | Message delivery time |

---

**Related Documents:**
- [Architecture](./architecture.md) - System design overview
- [Implementation Guide](./implementation-guide.md) - Development instructions
