# Implementation Guide

## Overview

This step-by-step implementation guide provides detailed instructions for building the Prioritization Frameworks Web application. Follow these phases in order to create a complete, production-ready web application with React + Vite frontend and Node.js backend.

## Table of Contents

1. [Phase 1: Project Setup](#phase-1-project-setup)
2. [Phase 2: Core Type Definitions](#phase-2-core-type-definitions)
3. [Phase 3: Framework Calculators](#phase-3-framework-calculators)
4. [Phase 4: Auto-Fill Engine](#phase-4-auto-fill-engine)
5. [Phase 5: React Components](#phase-5-react-components)
6. [Phase 6: Backend API](#phase-6-backend-api)
7. [Phase 7: Agent Pipeline](#phase-7-agent-pipeline)
8. [Phase 8: Testing](#phase-8-testing)
9. [Phase 9: Build & Deployment](#phase-9-build--deployment)

---

## Phase 1: Project Setup

### Step 1.1: Initialize Project Structure

```bash
# Create project directory
mkdir -p src/prioritization-app
cd src/prioritization-app

# Create frontend and backend directories
mkdir -p frontend/src/{components,hooks,services,contexts,api,utils,styles}
mkdir -p backend/src/{api,agents,mcp,services,utils}

# Initialize frontend package.json
cd frontend
npm init -y

# Initialize backend package.json
cd ../backend
npm init -y
```

### Step 1.2: Install Frontend Dependencies

```bash
cd frontend

# Core React dependencies
npm install react@19 react-dom@19

# TypeScript
npm install typescript @types/react @types/react-dom --save-dev

# Vite build system
npm install vite@5 @vitejs/plugin-react --save-dev

# Visualization
npm install d3 @types/d3
npm install chart.js react-chartjs-2

# Utilities
npm install uuid
npm install @types/uuid --save-dev
npm install date-fns

# HTTP client
npm install axios

# Optional: React Query for API state management
npm install @tanstack/react-query
```

### Step 1.3: Install Backend Dependencies

```bash
cd backend

# Core Node.js dependencies
npm install express
npm install ws  # WebSocket server

# TypeScript
npm install typescript @types/node @types/express @types/ws --save-dev
npm install ts-node tsx --save-dev

# Utilities
npm install uuid zod
npm install @types/uuid --save-dev

# Optional: Fastify instead of Express
# npm install fastify @fastify/websocket

# Development
npm install nodemon --save-dev
```

### Step 1.4: Configure Frontend TypeScript

Create `frontend/tsconfig.json`:

```json
{
  "compilerOptions": {
    "target": "ES2020",
    "useDefineForClassFields": true,
    "lib": ["ES2020", "DOM", "DOM.Iterable"],
    "module": "ESNext",
    "skipLibCheck": true,
    "moduleResolution": "bundler",
    "allowImportingTsExtensions": true,
    "resolveJsonModule": true,
    "isolatedModules": true,
    "noEmit": true,
    "jsx": "react-jsx",
    "strict": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "noFallthroughCasesInSwitch": true,
    "baseUrl": ".",
    "paths": {
      "@/*": ["src/*"],
      "@components/*": ["src/components/*"],
      "@services/*": ["src/services/*"],
      "@hooks/*": ["src/hooks/*"],
      "@contexts/*": ["src/contexts/*"],
      "@api/*": ["src/api/*"],
      "@utils/*": ["src/utils/*"]
    }
  },
  "include": ["src"],
  "references": [{ "path": "./tsconfig.node.json" }]
}
```

Create `frontend/tsconfig.node.json`:

```json
{
  "compilerOptions": {
    "composite": true,
    "skipLibCheck": true,
    "module": "ESNext",
    "moduleResolution": "bundler",
    "allowSyntheticDefaultImports": true
  },
  "include": ["vite.config.ts"]
}
```

### Step 1.5: Configure Vite

Create `frontend/vite.config.ts`:

```typescript
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
      '@components': path.resolve(__dirname, './src/components'),
      '@services': path.resolve(__dirname, './src/services'),
      '@hooks': path.resolve(__dirname, './src/hooks'),
      '@contexts': path.resolve(__dirname, './src/contexts'),
      '@api': path.resolve(__dirname, './src/api'),
      '@utils': path.resolve(__dirname, './src/utils'),
    },
  },
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://localhost:3001',
        changeOrigin: true,
      },
      '/ws': {
        target: 'ws://localhost:3001',
        ws: true,
      },
    },
  },
  build: {
    outDir: 'dist',
    sourcemap: true,
    rollupOptions: {
      output: {
        manualChunks: {
          vendor: ['react', 'react-dom'],
          charts: ['d3', 'chart.js', 'react-chartjs-2'],
        },
      },
    },
  },
});
```

### Step 1.6: Configure Backend TypeScript

Create `backend/tsconfig.json`:

```json
{
  "compilerOptions": {
    "target": "ES2022",
    "module": "commonjs",
    "lib": ["ES2022"],
    "outDir": "./dist",
    "rootDir": "./src",
    "strict": true,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "forceConsistentCasingInFileNames": true,
    "resolveJsonModule": true,
    "declaration": true,
    "declarationMap": true,
    "sourceMap": true,
    "moduleResolution": "node"
  },
  "include": ["src/**/*"],
  "exclude": ["node_modules", "dist"]
}
```

### Step 1.7: Package.json Scripts

**Frontend `frontend/package.json`:**

```json
{
  "name": "prioritization-frontend",
  "version": "1.0.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "tsc && vite build",
    "preview": "vite preview",
    "lint": "eslint src --ext ts,tsx",
    "test": "vitest",
    "test:ui": "vitest --ui"
  },
  "dependencies": {
    "react": "^19.0.0",
    "react-dom": "^19.0.0",
    "axios": "^1.6.0",
    "d3": "^7.8.5",
    "chart.js": "^4.4.0",
    "react-chartjs-2": "^5.2.0",
    "@tanstack/react-query": "^5.0.0",
    "uuid": "^9.0.0",
    "date-fns": "^3.0.0"
  },
  "devDependencies": {
    "@types/react": "^19.0.0",
    "@types/react-dom": "^19.0.0",
    "@types/uuid": "^9.0.0",
    "@vitejs/plugin-react": "^4.2.0",
    "typescript": "^5.3.0",
    "vite": "^5.0.0",
    "vitest": "^1.0.0"
  }
}
```

**Backend `backend/package.json`:**

```json
{
  "name": "prioritization-backend",
  "version": "1.0.0",
  "scripts": {
    "dev": "tsx watch src/index.ts",
    "build": "tsc",
    "start": "node dist/index.js",
    "lint": "eslint src --ext ts",
    "test": "jest"
  },
  "dependencies": {
    "express": "^4.18.2",
    "ws": "^8.14.0",
    "uuid": "^9.0.0",
    "zod": "^3.22.0"
  },
  "devDependencies": {
    "@types/express": "^4.17.21",
    "@types/node": "^20.0.0",
    "@types/ws": "^8.5.10",
    "@types/uuid": "^9.0.0",
    "typescript": "^5.3.0",
    "tsx": "^4.6.0",
    "nodemon": "^3.0.0"
  }
}
```

### Step 1.8: Create Frontend Entry Points

Create `frontend/index.html`:

```html
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <link rel="icon" type="image/svg+xml" href="/vite.svg" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Prioritization Frameworks</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.tsx"></script>
  </body>
</html>
```

Create `frontend/src/main.tsx`:

```typescript
import React from 'react';
import ReactDOM from 'react-dom/client';
import { App } from './App';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import './styles/global.css';

const queryClient = new QueryClient();

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <QueryClientProvider client={queryClient}>
      <App />
    </QueryClientProvider>
  </React.StrictMode>
);
```

Create `frontend/src/styles/global.css`:

```css
:root {
  --bg-primary: #1a1a2e;
  --bg-secondary: #16213e;
  --bg-tertiary: #0f3460;
  --text-primary: #eee;
  --text-secondary: #ccc;
  --accent-primary: #e94560;
  --accent-secondary: #00adb5;
  --success: #00ff88;
  --warning: #ffaa00;
  --error: #ff4444;
}

* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
  background-color: var(--bg-primary);
  color: var(--text-primary);
  line-height: 1.6;
}

#root {
  min-height: 100vh;
}
```

### Step 1.9: Create Backend Entry Point

Create `backend/src/index.ts`:

```typescript
import express from 'express';
import { createServer } from 'http';
import { WebSocketServer } from 'ws';
import cors from 'cors';

const app = express();
const server = createServer(app);
const wss = new WebSocketServer({ server, path: '/ws' });

const PORT = process.env.PORT || 3001;

// Middleware
app.use(cors({
  origin: process.env.CORS_ORIGIN || 'http://localhost:5173',
  credentials: true,
}));
app.use(express.json());

// Health check
app.get('/health', (req, res) => {
  res.json({ status: 'ok', timestamp: new Date().toISOString() });
});

// API routes will be added here
// app.use('/api/v1', routes);

// WebSocket connection handling
wss.on('connection', (ws) => {
  console.log('WebSocket client connected');

  ws.on('message', (message) => {
    console.log('Received:', message.toString());
  });

  ws.on('close', () => {
    console.log('WebSocket client disconnected');
  });
});

server.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`);
  console.log(`WebSocket server running on ws://localhost:${PORT}/ws`);
});
```

---

## Phase 2: Core Type Definitions

### Step 2.1: Create Base Types

Create `frontend/src/services/prioritization/types.ts`:

```typescript
// Core entity types
export interface BacklogItem {
  id: string;
  title: string;
  description: string;
  metadata: BacklogMetadata;
  customFields: Record<string, any>;
  createdAt: Date;
  updatedAt: Date;
}

export interface BacklogMetadata {
  category?: string;
  epic?: string;
  assignee?: string;
  estimatedHours?: number;
  estimatedWeeks?: number;
  businessValue?: number;
  technicalRisk?: number;
  dependencies?: string[];
  deadline?: string;
  userSegment?: string;
  estimatedReach?: number;
  storyPoints?: number;
  dataAvailable?: boolean;
  userResearchConducted?: boolean;
  teamFamiliarity?: 'low' | 'medium' | 'high';
  technicalUncertainty?: 'low' | 'medium' | 'high';
  [key: string]: any;
}

// Framework types
export type FrameworkType =
  | 'MOSCOW'
  | 'WSJF'
  | 'RICE'
  | 'ICE'
  | 'KANO'
  | 'VALUE_EFFORT'
  | 'EISENHOWER'
  | 'P0P4';

export interface IFrameworkResult {
  itemId: string;
  frameworkType: FrameworkType;
  normalizedScore: number;
  rank: number;
  rationale: string;
  autoFillUsed: boolean;
  manualOverrides: ManualOverride[];
}

export interface ManualOverride {
  field: string;
  originalValue: number;
  newValue: number;
  reason: string;
  timestamp: Date;
}

// Session types
export interface PrioritizationSession {
  id: string;
  name: string;
  createdAt: Date;
  frameworkIds: FrameworkType[];
  items: BacklogItem[];
  results: FrameworkResult[];
  context: SessionContext;
}

export interface SessionContext {
  totalUserBase?: number;
  teamCapacity?: number;
  programIncrement?: string;
  strategicGoals?: string[];
  [key: string]: any;
}

// Utility types
export type FrameworkResult =
  | MoSCoWResult
  | WSJFResult
  | RICEResult
  | ICEResult
  | KanoResult
  | ValueEffortResult
  | EisenhowerResult
  | P0P4Result;
```

### Step 2.2: Create Framework-Specific Types

Append to `frontend/src/services/prioritization/types.ts`:

```typescript
// MoSCoW types
export type MoSCoWCategory = 'MUST_HAVE' | 'SHOULD_HAVE' | 'COULD_HAVE' | 'WONT_HAVE';

export interface MoSCoWResult extends IFrameworkResult {
  frameworkType: 'MOSCOW';
  category: MoSCoWCategory;
  categoryWeight: number;
  urgencyFactor: number;
  priorityScore: number;
  dependencies: string[];
}

// WSJF types
export interface WSJFResult extends IFrameworkResult {
  frameworkType: 'WSJF';
  userBusinessValue: number;
  timeCriticality: number;
  riskReductionOE: number;
  costOfDelay: number;
  jobSize: number;
  wsjfScore: number;
  confidence: number;
}

// RICE types
export interface RICEResult extends IFrameworkResult {
  frameworkType: 'RICE';
  reach: number;
  impact: number;
  confidence: number;
  effort: number;
  riceScore: number;
  confidenceLevel: 'High' | 'Medium' | 'Low';
}

// ICE types
export interface ICEResult extends IFrameworkResult {
  frameworkType: 'ICE';
  impact: number;
  confidence: number;
  ease: number;
  iceScore: number;
  category: 'Quick Win' | 'Major Project' | 'Hard Slog' | 'Money Pit';
}

// Kano types
export type KanoCategory = 'MUST_BE' | 'PERFORMANCE' | 'EXCITEMENT' | 'INDIFFERENT' | 'REVERSE' | 'QUESTIONABLE';

export interface KanoResult extends IFrameworkResult {
  frameworkType: 'KANO';
  category: KanoCategory;
  satisfactionCoefficient: number;
  dissatisfactionCoefficient: number;
  betterScore: number;
  worseScore: number;
  recommendedAction: string;
}

// Value vs Effort types
export type Quadrant = 'QUICK_WINS' | 'MAJOR_PROJECTS' | 'FILL_INS' | 'TIME_SINKS';

export interface ValueEffortResult extends IFrameworkResult {
  frameworkType: 'VALUE_EFFORT';
  valueScore: number;
  effortScore: number;
  quadrant: Quadrant;
  valueComponents: {
    businessValue: number;
    userValue: number;
    strategicValue: number;
  };
  effortComponents: {
    developmentEffort: number;
    complexity: number;
    dependencyCost: number;
  };
  recommendedAction: string;
}

// Eisenhower types
export type EisenhowerQuadrant = 'Q1_DO' | 'Q2_DECIDE' | 'Q3_DELEGATE' | 'Q4_DELETE';

export interface EisenhowerResult extends IFrameworkResult {
  frameworkType: 'EISENHOWER';
  urgencyScore: number;
  importanceScore: number;
  quadrant: EisenhowerQuadrant;
  urgencyFactors: {
    deadlineProximity: number;
    consequencesOfDelay: number;
    externalPressure: number;
  };
  importanceFactors: {
    alignmentWithGoals: number;
    longTermImpact: number;
    stakeholderValue: number;
  };
  actionTimeframe: string;
}

// P0-P4 types
export type PriorityLevel = 'P0' | 'P1' | 'P2' | 'P3' | 'P4';

export interface P0P4Result extends IFrameworkResult {
  frameworkType: 'P0P4';
  priorityLevel: PriorityLevel;
  priorityScore: number;
  severityFactors: {
    userImpact: number;
    dataRisk: number;
    revenueImpact: number;
    securityRisk: number;
    reputationRisk: number;
  };
  recommendedResponse: string;
  escalationPath: string[];
  slaDeadline: Date;
}
```

---

## Phase 3: Framework Calculators

### Step 3.1: Create Calculator Interface

Create `frontend/src/services/prioritization/calculators/IFrameworkCalculator.ts`:

```typescript
import { BacklogItem, FrameworkResult } from '../types';

export interface FrameworkContext {
  [key: string]: any;
}

export interface IFrameworkCalculator<T extends FrameworkResult> {
  frameworkType: T['frameworkType'];

  calculate(
    item: BacklogItem,
    context: FrameworkContext
  ): T;

  validate(
    item: BacklogItem,
    scores: Record<string, number>
  ): ValidationResult;

  normalize(score: number, allScores: number[]): number;

  rank(items: T[]): T[];
}

export interface ValidationResult {
  isValid: boolean;
  errors: string[];
  warnings: string[];
}
```

### Step 3.2: Implement RICE Calculator

Create `frontend/src/services/prioritization/calculators/rice-calculator.ts`:

```typescript
import { BacklogItem, RICEResult } from '../types';
import { IFrameworkCalculator, FrameworkContext, ValidationResult } from './IFrameworkCalculator';

export class RICECalculator implements IFrameworkCalculator<RICEResult> {
  readonly frameworkType = 'RICE';

  calculate(item: BacklogItem, context: FrameworkContext): RICEResult {
    const reach = item.metadata.estimatedReach || this.estimateReach(item, context);
    const impact = this.extractOrEstimate(item, 'impact', 1);
    const confidence = this.extractOrEstimate(item, 'confidence', 0.5);
    const effort = item.metadata.estimatedWeeks || this.estimateEffort(item);

    const riceScore = (reach * impact * confidence) / effort;
    const confidenceLevel = this.getConfidenceLevel(confidence);

    return {
      itemId: item.id,
      frameworkType: 'RICE',
      reach,
      impact,
      confidence,
      effort,
      riceScore,
      normalizedScore: 0,
      rank: 0,
      confidenceLevel,
      rationale: this.generateRationale(reach, impact, confidence, effort),
      autoFillUsed: false,
      manualOverrides: []
    };
  }

  private estimateReach(item: BacklogItem, context: FrameworkContext): number {
    const totalUsers = context.totalUserBase || 10000;

    if (item.metadata.userSegment === 'all_users') return totalUsers;
    if (item.metadata.userSegment === 'power_users') return totalUsers * 0.2;
    if (item.metadata.userSegment === 'new_users') return totalUsers * 0.1;

    return totalUsers * 0.5;
  }

  private extractOrEstimate(
    item: BacklogItem,
    field: string,
    defaultValue: number
  ): number {
    if (item.metadata[field] !== undefined) {
      return item.metadata[field];
    }

    const text = `${item.title} ${item.description}`.toLowerCase();

    if (field === 'impact') {
      if (/transformational|game changer/i.test(text)) return 3;
      if (/significant|major/i.test(text)) return 2;
      if (/moderate|improvement/i.test(text)) return 1;
      if (/minor|small/i.test(text)) return 0.5;
      return 0.25;
    }

    if (field === 'confidence') {
      let confidence = 0.5;
      if (item.metadata.dataAvailable) confidence += 0.2;
      if (item.metadata.userResearchConducted) confidence += 0.15;
      if (item.metadata.teamFamiliarity === 'high') confidence += 0.1;
      return Math.min(1.0, confidence);
    }

    return defaultValue;
  }

  private estimateEffort(item: BacklogItem): number {
    if (item.metadata.estimatedWeeks) return item.metadata.estimatedWeeks;
    if (item.metadata.storyPoints) return item.metadata.storyPoints * 0.5;

    let effort = 2;
    const text = `${item.title} ${item.description}`.toLowerCase();

    if (/simple|quick|minor/i.test(text)) effort -= 1;
    if (/complex|major|significant/i.test(text)) effort += 2;
    if (/integration|migration|refactor/i.test(text)) effort += 3;

    const depCount = item.metadata.dependencies?.length || 0;
    effort += depCount * 0.5;

    return Math.max(0.5, effort);
  }

  private getConfidenceLevel(confidence: number): 'High' | 'Medium' | 'Low' {
    if (confidence >= 0.81) return 'High';
    if (confidence >= 0.51) return 'Medium';
    return 'Low';
  }

  private generateRationale(
    reach: number,
    impact: number,
    confidence: number,
    effort: number
  ): string {
    const parts = [];

    if (reach > 5000) parts.push('broad reach');
    if (impact >= 2) parts.push('high impact');
    if (confidence < 0.5) parts.push('low confidence - validation recommended');
    if (effort > 8) parts.push('significant effort');

    return parts.length > 0
      ? `RICE: ${parts.join(', ')}`
      : 'Standard priority based on available data';
  }

  validate(item: BacklogItem, scores: Record<string, number>): ValidationResult {
    const errors: string[] = [];
    const warnings: string[] = [];

    if (scores.reach !== undefined && scores.reach < 0) {
      errors.push('Reach cannot be negative');
    }

    if (scores.impact !== undefined && (scores.impact < 0.25 || scores.impact > 3)) {
      errors.push('Impact must be between 0.25 and 3');
    }

    if (scores.confidence !== undefined && (scores.confidence < 0 || scores.confidence > 1)) {
      errors.push('Confidence must be between 0 and 1');
    }

    if (scores.effort !== undefined && scores.effort <= 0) {
      errors.push('Effort must be positive');
    }

    return {
      isValid: errors.length === 0,
      errors,
      warnings
    };
  }

  normalize(score: number, allScores: number[]): number {
    const maxScore = Math.max(...allScores);
    return maxScore > 0 ? (score / maxScore) * 100 : 0;
  }

  rank(items: RICEResult[]): RICEResult[] {
    return items
      .sort((a, b) => b.riceScore - a.riceScore)
      .map((item, index) => ({ ...item, rank: index + 1 }));
  }
}
```

---

## Phase 4: Auto-Fill Engine

### Step 4.1: Create Auto-Fill Engine

Create `frontend/src/services/prioritization/auto-fill/AutoFillEngine.ts`:

```typescript
import { BacklogItem, FrameworkResult, FrameworkType } from '../types';
import { FrameworkContext } from '../calculators/IFrameworkCalculator';
import { createCalculator } from '../calculators';

export interface AutoFillSuggestion {
  itemId: string;
  frameworkType: FrameworkType;
  scores: Record<string, number>;
  confidence: number;
  rationale: string;
}

export class AutoFillEngine {
  private context: FrameworkContext;

  constructor(context: FrameworkContext = {}) {
    this.context = context;
  }

  updateContext(context: Partial<FrameworkContext>): void {
    this.context = { ...this.context, ...context };
  }

  async suggestScores(
    item: BacklogItem,
    frameworkType: FrameworkType
  ): Promise<AutoFillSuggestion> {
    const calculator = createCalculator(frameworkType);
    const result = calculator.calculate(item, this.context);

    const scores = this.extractScores(result);
    const confidence = this.calculateConfidence(item, scores);
    const rationale = this.generateRationale(item, frameworkType, scores);

    return {
      itemId: item.id,
      frameworkType,
      scores,
      confidence,
      rationale
    };
  }

  private extractScores(result: FrameworkResult): Record<string, number> {
    const scores: Record<string, number> = {};

    for (const [key, value] of Object.entries(result)) {
      if (typeof value === 'number' &&
          key !== 'normalizedScore' &&
          key !== 'rank' &&
          key !== 'priorityScore') {
        scores[key] = value;
      }
    }

    return scores;
  }

  private calculateConfidence(
    item: BacklogItem,
    scores: Record<string, number>
  ): number {
    let confidence = 0.5;

    if (item.metadata.dataAvailable) confidence += 0.2;
    if (item.metadata.userResearchConducted) confidence += 0.15;

    if (item.metadata.technicalUncertainty === 'high') confidence -= 0.15;
    if (item.metadata.estimatedHours === undefined) confidence -= 0.1;

    return Math.min(1.0, Math.max(0.0, confidence));
  }

  private generateRationale(
    item: BacklogItem,
    frameworkType: FrameworkType,
    scores: Record<string, number>
  ): string {
    const parts = [`${frameworkType} auto-fill`];

    if (scores.reach && scores.reach > 10000) {
      parts.push('high reach detected');
    }
    if (scores.impact && scores.impact >= 2) {
      parts.push('high impact indicated');
    }
    if (scores.confidence && scores.confidence < 0.5) {
      parts.push('low confidence - review recommended');
    }

    return parts.join(': ');
  }
}
```

---

## Phase 5: React Components

### Step 5.1: Create Main App Component

Create `frontend/src/App.tsx`:

```tsx
import React, { useState } from 'react';
import { FrameworkSelector } from '@components/framework/FrameworkSelector';
import { BacklogInput } from '@components/backlog/BacklogInput';
import { ResultsDashboard } from '@components/ranking/ResultsDashboard';
import { PrioritizationService } from '@services/prioritization/PrioritizationService';

function App() {
  const [selectedFramework, setSelectedFramework] = useState<string | null>(null);
  const [backlogItems, setBacklogItems] = useState([]);
  const [results, setResults] = useState(null);

  const handleFrameworkSelect = (framework: string) => {
    setSelectedFramework(framework);
  };

  const handleBacklogUpdate = (items: any[]) => {
    setBacklogItems(items);
  };

  const handleCalculate = async () => {
    const service = new PrioritizationService();
    const calcResults = await service.calculate(backlogItems, selectedFramework);
    setResults(calcResults);
  };

  return (
    <div className="app">
      <header className="app-header">
        <h1>Prioritization Frameworks</h1>
        <p>Make data-driven prioritization decisions</p>
      </header>

      <main className="app-main">
        <section className="framework-section">
          <h2>Select Framework</h2>
          <FrameworkSelector
            selectedFramework={selectedFramework}
            onSelect={handleFrameworkSelect}
          />
        </section>

        <section className="backlog-section">
          <h2>Backlog Items</h2>
          <BacklogInput
            items={backlogItems}
            onUpdate={handleBacklogUpdate}
          />
        </section>

        <section className="actions-section">
          <button
            className="btn-primary"
            onClick={handleCalculate}
            disabled={!selectedFramework || backlogItems.length === 0}
          >
            Calculate Priorities
          </button>
        </section>

        {results && (
          <section className="results-section">
            <h2>Results</h2>
            <ResultsDashboard results={results} />
          </section>
        )}
      </main>
    </div>
  );
}

export default App;
```

### Step 5.2: Create Framework Selector

Create `frontend/src/components/framework/FrameworkSelector.tsx`:

```tsx
import React from 'react';

const FRAMEWORKS = [
  { id: 'MOSCOW', name: 'MoSCoW', description: 'Must/Should/Could/Won\'t have' },
  { id: 'RICE', name: 'RICE', description: 'Reach, Impact, Confidence, Effort' },
  { id: 'WSJF', name: 'WSJF', description: 'Weighted Shortest Job First' },
  { id: 'ICE', name: 'ICE', description: 'Impact, Confidence, Ease' },
  { id: 'VALUE_EFFORT', name: 'Value vs Effort', description: '2x2 Priority Matrix' },
  { id: 'EISENHOWER', name: 'Eisenhower', description: 'Urgent vs Important' },
  { id: 'KANO', name: 'Kano Model', description: 'Customer satisfaction categories' },
  { id: 'P0P4', name: 'P0-P4', description: 'Priority hierarchy (P0 critical)' },
];

interface FrameworkSelectorProps {
  selectedFramework: string | null;
  onSelect: (framework: string) => void;
}

export function FrameworkSelector({ selectedFramework, onSelect }: FrameworkSelectorProps) {
  return (
    <div className="framework-selector">
      {FRAMEWORKS.map((framework) => (
        <button
          key={framework.id}
          className={`framework-card ${selectedFramework === framework.id ? 'selected' : ''}`}
          onClick={() => onSelect(framework.id)}
        >
          <h3>{framework.name}</h3>
          <p>{framework.description}</p>
        </button>
      ))}
    </div>
  );
}
```

---

## Phase 6: Backend API

### Step 6.1: Create API Routes

Create `backend/src/api/routes.ts`:

```typescript
import { Router, Request, Response } from 'express';
import { PrioritizationService } from '../services/PrioritizationService';
import { SessionService } from '../services/SessionService';

const router = Router();
const prioritizationService = new PrioritizationService();
const sessionService = new SessionService();

// POST /api/v1/prioritize
router.post('/prioritize', async (req: Request, res: Response) => {
  try {
    const { items, framework } = req.body;
    const results = await prioritizationService.calculate(items, framework);
    res.json({ results });
  } catch (error) {
    res.status(400).json({ error: error.message });
  }
});

// POST /api/v1/prioritize/auto-fill
router.post('/auto-fill', async (req: Request, res: Response) => {
  try {
    const { item, framework } = req.body;
    const suggestion = await prioritizationService.getAutoFillSuggestion(item, framework);
    res.json({ suggestion });
  } catch (error) {
    res.status(400).json({ error: error.message });
  }
});

// Session endpoints
router.get('/sessions', async (req: Request, res: Response) => {
  const sessions = await sessionService.list();
  res.json({ sessions });
});

router.post('/sessions', async (req: Request, res: Response) => {
  const session = await sessionService.create(req.body);
  res.status(201).json({ session });
});

router.get('/sessions/:id', async (req: Request, res: Response) => {
  const session = await sessionService.get(req.params.id);
  if (!session) {
    return res.status(404).json({ error: 'Session not found' });
  }
  res.json({ session });
});

export default router;
```

---

## Phase 7: Agent Pipeline

### Step 7.1: Create Pipeline Orchestrator

Create `backend/src/agents/PipelineOrchestrator.ts`:

```typescript
import { EventEmitter } from 'events';

export interface PipelineInput {
  request: string;
  context: Record<string, any>;
  codebase?: any;
  constraints?: any;
}

export interface PipelineOutput {
  implementation: any;
  documentation: any;
  qualityScore: number;
  iterations: number;
}

export class PipelineOrchestrator extends EventEmitter {
  private maxIterations = 3;
  private qualityThreshold = 80;

  async execute(input: PipelineInput): Promise<PipelineOutput> {
    let iteration = 0;
    let feedback = '';

    while (iteration < this.maxIterations) {
      iteration++;

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

      // Check if quality threshold met
      if (reviewOutput.qualityScore >= this.qualityThreshold && reviewOutput.criticalIssues === 0) {
        // Phase 4: Documentation
        this.emit('iteration_start', { iteration, agent: 'writer' });
        const writerOutput = await this.runWriterAgent({
          implementation: developerOutput,
          reviewOutput
        });
        this.emit('agent_complete', { iteration, agent: 'writer', output: writerOutput });

        // Phase 5: Ecosystem
        this.emit('iteration_start', { iteration, agent: 'ecosystem' });
        const ecosystemOutput = await this.runEcosystemAgent({
          implementation: developerOutput,
          documentation: writerOutput
        });
        this.emit('agent_complete', { iteration, agent: 'ecosystem', output: ecosystemOutput });

        this.emit('pipeline_complete', {
          iterations: iteration,
          qualityScore: reviewOutput.qualityScore,
          artifacts: [developerOutput, writerOutput, ecosystemOutput]
        });

        return {
          implementation: developerOutput,
          documentation: writerOutput,
          qualityScore: reviewOutput.qualityScore,
          iterations: iteration
        };
      }

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

  private async runPlanningAgent(input: any): Promise<any> {
    // Implementation with MCP tool integration
    return {};
  }

  private async runDeveloperAgent(input: any): Promise<any> {
    return {};
  }

  private async runReviewerAgent(input: any): Promise<any> {
    return {};
  }

  private async runWriterAgent(input: any): Promise<any> {
    return {};
  }

  private async runEcosystemAgent(input: any): Promise<any> {
    return {};
  }
}
```

---

## Phase 8: Testing

### Step 8.1: Create Test Configuration

Create `frontend/vitest.config.ts`:

```typescript
import { defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react';
import path from 'path';

export default defineConfig({
  plugins: [react()],
  test: {
    globals: true,
    environment: 'jsdom',
    setupFiles: './src/test/setup.ts',
  },
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
});
```

### Step 8.2: Create Calculator Tests

Create `frontend/src/services/prioritization/calculators/__tests__/rice.test.ts`:

```typescript
import { describe, it, expect, beforeEach } from 'vitest';
import { RICECalculator } from '../rice-calculator';

describe('RICECalculator', () => {
  let calculator: RICECalculator;

  beforeEach(() => {
    calculator = new RICECalculator();
  });

  const createTestItem = (overrides = {}) => ({
    id: 'test-1',
    title: 'Test Feature',
    description: 'A test feature description',
    metadata: {
      estimatedReach: 10000,
      ...overrides
    },
    customFields: {},
    createdAt: new Date(),
    updatedAt: new Date()
  });

  describe('calculate', () => {
    it('should calculate RICE score correctly', () => {
      const item = createTestItem({
        estimatedReach: 10000,
        impact: 2,
        confidence: 0.8,
        estimatedWeeks: 4
      });

      const result = calculator.calculate(item, {});

      expect(result.riceScore).toBe(4000);
      expect(result.frameworkType).toBe('RICE');
    });

    it('should categorize confidence level correctly', () => {
      expect(calculator.calculate(createTestItem({ confidence: 0.85 }), {}).confidenceLevel).toBe('High');
      expect(calculator.calculate(createTestItem({ confidence: 0.6 }), {}).confidenceLevel).toBe('Medium');
      expect(calculator.calculate(createTestItem({ confidence: 0.4 }), {}).confidenceLevel).toBe('Low');
    });
  });

  describe('validate', () => {
    it('should return valid for correct scores', () => {
      const result = calculator.validate(createTestItem(), {
        reach: 1000,
        impact: 2,
        confidence: 0.8,
        effort: 4
      });

      expect(result.isValid).toBe(true);
    });

    it('should return invalid for negative reach', () => {
      const result = calculator.validate(createTestItem(), {
        reach: -100,
        impact: 2,
        confidence: 0.8,
        effort: 4
      });

      expect(result.isValid).toBe(false);
      expect(result.errors).toContain('Reach cannot be negative');
    });
  });
});
```

---

## Phase 9: Build & Deployment

### Step 9.1: Build Frontend

```bash
cd frontend

# Development build
npm run build

# Preview production build
npm run preview
```

### Step 9.2: Build Backend

```bash
cd backend

# Compile TypeScript
npm run build

# Start production server
npm start
```

### Step 9.3: Deploy to Vercel (Frontend)

1. Install Vercel CLI:
```bash
npm install -g vercel
```

2. Deploy:
```bash
cd frontend
vercel
```

3. Configure environment variables in Vercel dashboard:
```
VITE_API_URL=https://your-backend-url.railway.app/api
```

### Step 9.4: Deploy to Railway (Backend)

1. Install Railway CLI:
```bash
npm install -g @railway/cli
```

2. Deploy:
```bash
cd backend
railway login
railway init
railway up
```

3. Configure environment variables:
```bash
railway variables set PORT=3001
railway variables set CORS_ORIGIN=https://your-frontend.vercel.app
```

---

## Next Steps

1. **Complete remaining calculators** - WSJF, ICE, Kano, Value-Effort, Eisenhower, P0-P4
2. **Build UI components** - Visualization components for each framework
3. **Implement MCP integration** - Connect to Clear Thought MCP server
4. **Add export functionality** - CSV, JSON, Jira integration
5. **Create user documentation** - In-app help and tutorials

---

**Related Documents:**
- [Architecture](./architecture.md) - System design overview
- [Pipeline Specification](./pipeline.md) - Agent pipeline details
- [Framework Details](./frameworks/) - Individual framework specifications
