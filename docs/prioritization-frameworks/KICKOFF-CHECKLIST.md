# Prioritization Frameworks Web App - Project Kickoff Checklist

## Pre-Development Checklist

### Documentation Review
- [ ] Read `README.md` - Understand project overview
- [ ] Read `architecture.md` - Understand system design (Web App)
- [ ] Read `implementation-guide.md` - Understand implementation steps
- [ ] Read `IMPLEMENTATION-SUMMARY.md` - Understand current status and roadmap
- [ ] Read `AGENT-PIPELINE-QUICKREF.md` - Understand agent pipeline usage (Web context)
- [ ] Read all 8 framework docs - Understand each prioritization method

### Critical Decisions Made
- [x] **Web Application** - React + Vite frontend, Node.js backend
- [x] **8 frameworks** - MoSCoW, WSJF, RICE, ICE, Kano, Value-Effort, Eisenhower, P0-P4
- [x] **Agent pipeline** - Runs in Node.js backend with HTTP/WebSocket communication
- [x] **Clear Thought MCP** - Optional enhancement, not core dependency
- [x] **MVP scope** - MoSCoW, RICE, Value-Effort (recommended)
- [x] **Data persistence** - IndexedDB for client-side storage
- [ ] **Deployment targets** - TODO: Confirm Vercel + Railway

### Open Decisions (Resolve Before Phase 1)
- [ ] **MVP Framework Selection**
  - [ ] Option A: MoSCoW, RICE, Value-Effort (recommended)
  - [ ] Option B: RICE, ICE, Eisenhower
  - [ ] Other: _______________

- [ ] **Visualization Library**
  - [ ] D3.js only (more flexible, steeper learning curve)
  - [ ] Chart.js only (simpler, less flexible)
  - [ ] Both (D3 for matrices, Chart.js for bar charts)

- [ ] **Backend Hosting**
  - [ ] Railway (recommended - simple, GitHub integration)
  - [ ] Render (free tier available)
  - [ ] Fly.io (edge deployment)

---

## Phase 1: Project Setup (Week 1)

### 1.1 Create Project Structure
```bash
cd C:\Users\antmi\lemonade\src
mkdir -p prioritization-app/{frontend,backend}
mkdir -p frontend/src/{components,hooks,services,contexts,api,utils,styles}
mkdir -p backend/src/{api,agents,mcp,services,utils}
cd prioritization-app
```

- [ ] Directory structure created
- [ ] README.md in root with project overview

### 1.2 Initialize Frontend (Vite)
```bash
cd frontend
npm create vite@latest . -- --template react-ts
npm install
npm install axios @tanstack/react-query d3 chart.js react-chartjs-2 uuid date-fns
npm install -D vitest @testing-library/react @testing-library/jest-dom
```

- [ ] Vite project created
- [ ] Dependencies installed
- [ ] Verify dev server: `npm run dev`

### 1.3 Initialize Backend (Node.js)
```bash
cd ../backend
npm init -y
npm install express ws uuid zod cors
npm install -D typescript @types/node @types/express @types/ws @types/cors tsx nodemon
```

- [ ] package.json created
- [ ] Dependencies installed
- [ ] Verify server starts: `npm run dev`

### 1.4 Frontend TypeScript Configuration
```json
// frontend/tsconfig.json
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

- [ ] tsconfig.json created
- [ ] Path aliases configured

### 1.5 Vite Configuration
```typescript
// frontend/vite.config.ts
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
  },
});
```

- [ ] vite.config.ts created
- [ ] API proxy configured
- [ ] Test Vite dev server

### 1.6 Backend TypeScript Configuration
```json
// backend/tsconfig.json
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

- [ ] tsconfig.json created
- [ ] Test TypeScript compilation

### 1.7 Backend Entry Point
```typescript
// backend/src/index.ts
import express from 'express';
import { createServer } from 'http';
import { WebSocketServer } from 'ws';
import cors from 'cors';

const app = express();
const server = createServer(app);
const wss = new WebSocketServer({ server, path: '/ws' });

const PORT = process.env.PORT || 3001;

app.use(cors({
  origin: process.env.CORS_ORIGIN || 'http://localhost:5173',
  credentials: true,
}));
app.use(express.json());

app.get('/health', (req, res) => {
  res.json({ status: 'ok', timestamp: new Date().toISOString() });
});

server.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`);
  console.log(`WebSocket server running on ws://localhost:${PORT}/ws`);
});
```

- [ ] index.ts created
- [ ] CORS configured
- [ ] Health endpoint working
- [ ] Test server starts

### 1.8 Frontend Entry Points
```html
<!-- frontend/index.html -->
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

```typescript
// frontend/src/main.tsx
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

- [ ] index.html created
- [ ] main.tsx created
- [ ] global.css created
- [ ] Test React renders

### 1.9 Environment Configuration
```env
# backend/.env.example
PORT=3001
NODE_ENV=development
CORS_ORIGIN=http://localhost:5173
CLEAR_THOUGHT_MCP_URL=http://localhost:8080
```

```env
# frontend/.env.example
VITE_API_URL=http://localhost:3001/api
VITE_WS_URL=ws://localhost:3001/ws
```

- [ ] .env.example files created
- [ ] .env files created for local development

### Phase 1 Completion Criteria
- [ ] `npm run dev` (frontend) starts Vite dev server
- [ ] `npm run dev` (backend) starts Express server
- [ ] API proxy working (frontend -> backend)
- [ ] WebSocket connection working
- [ ] TypeScript compiles without errors
- [ ] Hot reload working for both frontend and backend

---

## Phase 2: Type Definitions (Week 1-2)

### 2.1 Core Types
- [ ] Create `frontend/src/services/prioritization/types.ts`
- [ ] Define `BacklogItem` interface
- [ ] Define `FrameworkType` union
- [ ] Define `IFrameworkResult` interface
- [ ] Define `NormalizedResult` interface
- [ ] Define `FrameworkContext` interface

### 2.2 Framework-Specific Types
- [ ] Define `MoSCoWResult` interface
- [ ] Define `RICEResult` interface
- [ ] Define `WSJFResult` interface
- [ ] Define `ICEResult` interface
- [ ] Define `KanoResult` interface
- [ ] Define `ValueEffortResult` interface
- [ ] Define `EisenhowerResult` interface
- [ ] Define `P0P4Result` interface

### 2.3 Calculator Interface
- [ ] Create `frontend/src/services/prioritization/calculators/IFrameworkCalculator.ts`
- [ ] Define `IFrameworkCalculator<T>` interface
- [ ] Define `ValidationResult` interface
- [ ] Define `FrameworkInput` interface
- [ ] Define `AutoFillSuggestion` interface

### 2.4 API Types
- [ ] Create `frontend/src/api/types.ts`
- [ ] Define API request types
- [ ] Define API response types
- [ ] Define WebSocket message types

### Phase 2 Completion Criteria
- [ ] All types compile without errors
- [ ] Type relationships are consistent
- [ ] JSDoc comments on all interfaces
- [ ] Example usage in test file

---

## Phase 3: Framework Calculators (Week 2-4)

### Priority Order (Complete in Order)

#### 3.1 MoSCoW Calculator (Template)
- [ ] Create `frontend/src/services/prioritization/calculators/moscow-calculator.ts`
- [ ] Implement `calculate()` method
- [ ] Implement `validate()` method
- [ ] Implement `normalize()` method
- [ ] Create unit tests

#### 3.2 RICE Calculator (Template)
- [ ] Create `frontend/src/services/prioritization/calculators/rice-calculator.ts`
- [ ] Implement `calculate()` method
- [ ] Implement `validate()` method
- [ ] Implement `normalize()` method
- [ ] Create unit tests

#### 3.3 Value vs Effort Calculator
- [ ] Create `frontend/src/services/prioritization/calculators/value-effort-calculator.ts`
- [ ] Implement quadrant logic
- [ ] Implement ROI calculation
- [ ] Implement normalization
- [ ] Create unit tests

#### 3.4 ICE Calculator
- [ ] Create `frontend/src/services/prioritization/calculators/ice-calculator.ts`
- [ ] Fix confidence scale (use 0-1, not 1-10)
- [ ] Implement calculation
- [ ] Implement normalization
- [ ] Create unit tests

#### 3.5 Eisenhower Calculator
- [ ] Create `frontend/src/services/prioritization/calculators/eisenhower-calculator.ts`
- [ ] Implement urgency/importance factors
- [ ] Implement quadrant logic
- [ ] Create unit tests

#### 3.6 P0-P4 Calculator
- [ ] Create `frontend/src/services/prioritization/calculators/p0p4-calculator.ts`
- [ ] Calibrate score thresholds
- [ ] Implement severity factors
- [ ] Implement SLA calculation
- [ ] Create unit tests

#### 3.7 WSJF Calculator
- [ ] Create `frontend/src/services/prioritization/calculators/wsjf-calculator.ts`
- [ ] Implement CD3 formula
- [ ] Implement job size scale
- [ ] Implement normalization
- [ ] Create unit tests

#### 3.8 Kano Calculator
- [ ] Create `frontend/src/services/prioritization/calculators/kano-calculator.ts`
- [ ] Add survey mode (required)
- [ ] Add proxy mode (with confidence penalty)
- [ ] Implement Better/Worse classification
- [ ] Implement coefficient calculation
- [ ] Add minimum sample size validation
- [ ] Create unit tests

### Phase 3 Completion Criteria
- [ ] All 8 calculators implemented
- [ ] All calculators pass unit tests
- [ ] 90%+ code coverage
- [ ] All calculators follow Strategy pattern
- [ ] Calculator factory function working

---

## Phase 4: Auto-Fill Engine (Week 3-4)

### 4.1 Auto-Fill Engine
- [ ] Create `frontend/src/services/prioritization/auto-fill/AutoFillEngine.ts`
- [ ] Implement context management
- [ ] Implement score suggestion logic
- [ ] Implement confidence calculation

### 4.2 Label Analyzer
- [ ] Create `frontend/src/services/prioritization/auto-fill/label-analyzer.ts`
- [ ] Implement label pattern recognition
- [ ] Create label-to-score mappings
- [ ] Implement multi-label handling

### 4.3 Metrics Extractor
- [ ] Create `frontend/src/services/prioritization/auto-fill/metrics-extractor.ts`
- [ ] Implement data extraction from items
- [ ] Implement metric calculations
- [ ] Implement cross-item analysis

### Phase 4 Completion Criteria
- [ ] Auto-fill engine provides suggestions
- [ ] Suggestions have confidence scores
- [ ] Suggestions are explainable
- [ ] Performance <100ms per suggestion

---

## Phase 5: React Components (Week 4-6)

### Component Priority

#### 5.1 Core Components
- [ ] FrameworkSelector component
- [ ] BacklogList component
- [ ] BacklogItem component
- [ ] BulkEditBar component

#### 5.2 Scoring Components
- [ ] ScoringPanel component
- [ ] ScoreInput component
- [ ] AutoFillSuggestion component
- [ ] ScoreHistory component

#### 5.3 Visualization Components
- [ ] Matrix2x2 component (for Value-Effort, Eisenhower)
- [ ] BarChart component (for MoSCoW, WSJF)
- [ ] ScatterPlot component (for RICE, ICE)
- [ ] KanoGrid component

#### 5.4 Ranking Components
- [ ] RankedTable component
- [ ] CrossFrameworkComparison component
- [ ] ExportReport component

#### 5.5 Session Components
- [ ] SessionList component
- [ ] SessionSave component
- [ ] ShareSession component

### Phase 5 Completion Criteria
- [ ] All components render correctly
- [ ] Components are responsive
- [ ] Keyboard navigation works
- [ ] Accessibility (ARIA labels)
- [ ] Component tests pass

---

## Phase 6: Backend API (Week 5-6)

### 6.1 API Routes
- [ ] Create `backend/src/api/routes.ts`
- [ ] POST `/api/v1/prioritize` endpoint
- [ ] POST `/api/v1/prioritize/auto-fill` endpoint
- [ ] Session CRUD endpoints
- [ ] Export endpoints (CSV, JSON)

### 6.2 WebSocket Server
- [ ] Create `backend/src/api/websocket.ts`
- [ ] Connection handling
- [ ] Agent pipeline progress events
- [ ] Error handling

### 6.3 Services
- [ ] Create `backend/src/services/PrioritizationService.ts`
- [ ] Create `backend/src/services/SessionService.ts`
- [ ] Create `backend/src/services/ExportService.ts`

### Phase 6 Completion Criteria
- [ ] All REST endpoints working
- [ ] WebSocket connections working
- [ ] API tests passing
- [ ] Error handling complete

---

## Phase 7: Agent Pipeline (Week 6-8)

### 7.1 Infrastructure
- [ ] Create `backend/src/agents/PipelineOrchestrator.ts`
- [ ] Create `backend/src/agents/agent-registry.ts`
- [ ] Create `backend/src/mcp/mcp-client.ts`
- [ ] Set up WebSocket event emission

### 7.2 Agent Implementations
- [ ] Planning agent wrapper
- [ ] Developer agent wrapper
- [ ] Reviewer agent wrapper
- [ ] Writer agent wrapper
- [ ] Ecosystem creator wrapper

### 7.3 Clear Thought MCP Integration
- [ ] Install MCP client
- [ ] Configure tool access
- [ ] Implement tool invocation
- [ ] Add error handling and fallbacks

### Phase 7 Completion Criteria
- [ ] Pipeline runs end-to-end
- [ ] Agents can use MCP tools
- [ ] Iteration loop works
- [ ] WebSocket progress streaming works
- [ ] Results are saved

---

## Phase 8: Testing (Week 7-9)

### Test Coverage

#### Unit Tests
- [ ] All calculator tests
- [ ] Auto-fill engine tests
- [ ] Normalization tests
- [ ] Type guard tests

#### Component Tests
- [ ] FrameworkSelector tests
- [ ] ScoringPanel tests
- [ ] Visualization tests
- [ ] RankingView tests

#### Integration Tests
- [ ] Calculator + Auto-fill
- [ ] Component + Service
- [ ] Session save/load

#### E2E Tests
- [ ] Full scoring flow
- [ ] Export flow
- [ ] Framework switching

### Phase 8 Completion Criteria
- [ ] 90%+ unit test coverage
- [ ] 80%+ component test coverage
- [ ] All critical paths covered by E2E
- [ ] All tests passing

---

## Phase 9: Build & Deployment (Week 9-10)

### 9.1 Frontend Build
```bash
cd frontend
npm run build  # Outputs to dist/
```

- [ ] Production build successful
- [ ] Source maps generated
- [ ] Bundle size optimized

### 9.2 Backend Build
```bash
cd backend
npm run build  # Compiles TypeScript to dist/
```

- [ ] TypeScript compilation successful
- [ ] All types resolved
- [ ] Production ready

### 9.3 Deploy Frontend (Vercel)
```bash
npm install -g vercel
cd frontend
vercel
```

- [ ] Vercel account set up
- [ ] Frontend deployed
- [ ] Environment variables configured
- [ ] Custom domain (optional)

### 9.4 Deploy Backend (Railway)
```bash
npm install -g @railway/cli
cd backend
railway login
railway init
railway up
```

- [ ] Railway account set up
- [ ] Backend deployed
- [ ] Environment variables configured
- [ ] CORS_ORIGIN updated

### Phase 9 Completion Criteria
- [ ] Frontend deployed to Vercel
- [ ] Backend deployed to Railway
- [ ] Production deployment working
- [ ] Monitoring configured

---

## Ongoing Tasks

### Documentation
- [ ] Update README with current status
- [ ] Maintain changelog
- [ ] Document breaking changes
- [ ] Add code comments

### Quality
- [ ] Run ESLint regularly
- [ ] Run Prettier formatting
- [ ] Review PRs before merging
- [ ] Keep test coverage high

### Risk Management
- [ ] Monitor iteration count in pipeline
- [ ] Watch for scope creep
- [ ] Regular backups
- [ ] Document decisions

---

## Sign-Off Checklist

Before considering project complete:

- [ ] All 8 frameworks implemented and working
- [ ] Auto-fill suggestions accurate (80%+ acceptance)
- [ ] Cross-framework comparison working
- [ ] Session persistence working
- [ ] Export to Jira/Linear/CSV working
- [ ] Documentation complete
- [ ] All tests passing
- [ ] Performance benchmarks met
- [ ] Accessibility audit passed
- [ ] User testing completed
- [ ] Production deployment live

---

## Contact and Resources

- **Project Root:** `src/prioritization-app/`
- **Frontend:** `src/prioritization-app/frontend/`
- **Backend:** `src/prioritization-app/backend/`
- **Documentation:** `docs/prioritization-frameworks/`
- **Main Issues:** Track in GitHub Issues
- **Agent Pipeline:** See `AGENT-PIPELINE-QUICKREF.md`

---

## Quick Start Commands

```bash
# Navigate to project
cd src/prioritization-app

# Frontend commands
cd frontend
npm install
npm run dev       # Development server (Vite)
npm run build     # Production build
npm test          # Run Vitest tests
npm run preview   # Preview production build

# Backend commands
cd ../backend
npm install
npm run dev       # Development server (tsx/nodemon)
npm run build     # Compile TypeScript
npm start         # Start production server
npm test          # Run Jest tests

# Deployment
cd frontend && vercel       # Deploy frontend to Vercel
cd ../backend && railway up # Deploy backend to Railway
```

---

**Last Updated:** 2026-03-18
**Version:** 0.1.0 (Web Application Architecture)
