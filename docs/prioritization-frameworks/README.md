# Prioritization Frameworks Web Application

## Overview

This standalone web application demonstrates eight industry-standard prioritization frameworks through an interactive, AI-powered interface. Built with a React + Vite frontend and Node.js backend for agent pipeline orchestration with Clear Thought MCP tools, it enables product managers, engineering leads, and stakeholders to make data-driven prioritization decisions.

## Quick Start

```bash
# Clone the repository
cd src/prioritization-app

# Install frontend dependencies
cd frontend && npm install

# Install backend dependencies
cd ../backend && npm install

# Start development servers
# Terminal 1 - Backend API
cd backend && npm run dev

# Terminal 2 - Frontend
cd frontend && npm run dev

# Build for production
npm run build    # Builds both frontend and backend
```

## Supported Frameworks

| Framework | Best For | Complexity |
|-----------|----------|------------|
| **MoSCoW** | Sprint planning, requirement categorization | Low |
| **WSJF** (Weighted Shortest Job First) | SAFe, portfolio prioritization | High |
| **RICE** (Reach, Impact, Confidence, Effort) | Product feature prioritization | Medium |
| **ICE** (Impact, Confidence, Ease) | Growth experiments, quick decisions | Low |
| **Kano Model** | Customer satisfaction analysis | Medium |
| **Value vs Effort Matrix** | Visual prioritization, stakeholder alignment | Low |
| **Eisenhower Matrix** | Task urgency/importance sorting | Low |
| **P0-P4 Priority Hierarchy** | Incident response, critical path planning | Low |

## Architecture Highlights

- **React SPA with Vite** - Fast development build system with HMR
- **Node.js Backend API** - RESTful API for agent pipeline and MCP integration
- **Clear Thought MCP Integration** - 11 cognitive tools enhance decision-making quality
- **Unified Scoring Engine** - Cross-framework comparison and normalization
- **Auto-fill Intelligence** - ML-assisted score suggestions based on item metadata
- **Real-time Visualization** - Interactive charts and matrices built with D3.js
- **IndexedDB Storage** - Client-side persistence for offline support

## Documentation Index

| Document | Description |
|----------|-------------|
| [Architecture](./architecture.md) | System design, component diagrams, data flow |
| [Pipeline Specification](./pipeline.md) | Agent pipeline workflow and coordination |
| [Implementation Guide](./implementation-guide.md) | Step-by-step development instructions |
| [Framework Details](./frameworks/) | Individual framework documentation |

## Key Features

### 1. Multi-Framework Analysis
Evaluate the same backlog items across multiple frameworks simultaneously to identify consistent priorities.

### 2. Auto-fill Templates
AI-powered suggestions for scores based on item descriptions, historical data, and team capacity.

### 3. Collaborative Reasoning
Multiple stakeholders can provide input, with the system aggregating and reconciling different perspectives.

### 4. Visualization Dashboard
Interactive charts, matrices, and heat maps for clear stakeholder communication.

### 5. Export & Integration
Export results to CSV, JSON, or integrate with Jira, Linear, and GitHub Projects.

## Technology Stack

| Layer | Technology |
|-------|------------|
| Frontend Build | Vite 5.x |
| Frontend Runtime | React 19 + TypeScript 5.3 |
| State Management | React Context API + Hooks |
| Styling | Pure CSS (Dark Theme) |
| Visualization | D3.js + Chart.js |
| Backend Runtime | Node.js 20+ |
| API Framework | Express.js / Fastify |
| MCP Tools | Clear Thought MCP Server |
| Frontend Deploy | Vercel / Netlify / GitHub Pages |
| Backend Deploy | Railway / Render / Fly.io |

## Project Structure

```
src/prioritization-app/
├── frontend/                    # React SPA (was src/)
│   ├── src/
│   │   ├── components/
│   │   │   ├── FrameworkSelector.tsx
│   │   │   ├── BacklogInput.tsx
│   │   │   ├── ScoreCard.tsx
│   │   │   └── Visualization/
│   │   ├── services/            # Calculators (unchanged)
│   │   │   └── prioritization/
│   │   │       ├── types.ts
│   │   │       ├── calculators/
│   │   │       └── auto-fill/
│   │   ├── hooks/
│   │   ├── contexts/
│   │   └── styles/
│   ├── index.html
│   ├── package.json
│   ├── tsconfig.json
│   └── vite.config.ts
├── backend/                     # NEW: Agent pipeline API
│   ├── src/
│   │   ├── agents/
│   │   │   ├── PipelineOrchestrator.ts
│   │   │   ├── PlanningAgent.ts
│   │   │   ├── DeveloperAgent.ts
│   │   │   ├── ReviewerAgent.ts
│   │   │   ├── WriterAgent.ts
│   │   │   └── EcosystemAgent.ts
│   │   ├── api/
│   │   │   └── routes.ts
│   │   ├── mcp/
│   │   │   └── mcp-client.ts
│   │   └── index.ts
│   ├── package.json
│   └── tsconfig.json
└── docs/
    └── prioritization-frameworks/
        ├── README.md
        ├── architecture.md
        ├── pipeline.md
        ├── implementation-guide.md
        └── frameworks/
            ├── moscow.md
            ├── wsjf.md
            ├── rice.md
            ├── ice.md
            ├── kano.md
            ├── value-effort.md
            ├── eisenhower.md
            └── p0p4.md
```

## Getting Started

### Prerequisites
- Node.js 20+
- npm or yarn
- Git

### Installation

1. Navigate to the application directory:
   ```bash
   cd src/prioritization-app
   ```

2. Install frontend dependencies:
   ```bash
   cd frontend && npm install
   ```

3. Install backend dependencies:
   ```bash
   cd ../backend && npm install
   ```

4. Configure Clear Thought MCP (optional):
   ```bash
   cd backend
   cp .env.example .env
   # Edit .env with your MCP server configuration
   ```

5. Start development:
   ```bash
   # Terminal 1 - Backend (port 3001)
   cd backend && npm run dev

   # Terminal 2 - Frontend (port 5173)
   cd frontend && npm run dev
   ```

6. Open browser to `http://localhost:5173`

### Building for Production

```bash
# Build frontend
cd frontend && npm run build

# Build backend
cd ../backend && npm run build

# Or build both from root
npm run build
```

### Deployment Options

#### Frontend (Static SPA)
- **Vercel** - Zero config deployment, automatic preview deployments
- **Netlify** - Similar to Vercel, great for JAMstack
- **GitHub Pages** - Free hosting for public repos
- **Cloudflare Pages** - Fast global CDN

#### Backend (Node.js API)
- **Railway** - Simple deployment with GitHub integration
- **Render** - Free tier available, automatic HTTPS
- **Fly.io** - Global edge deployment
- **AWS Lambda** - Serverless with API Gateway

## Usage Examples

### Single Framework Analysis

1. Select a framework (e.g., RICE)
2. Input backlog items with descriptions
3. Review auto-fill suggestions
4. Adjust scores as needed
5. View prioritized results

### Cross-Framework Comparison

1. Input backlog items once
2. Run multiple frameworks in parallel
3. Compare ranking correlations
4. Identify consensus priorities

### Collaborative Session

1. Share session URL with stakeholders
2. Each stakeholder provides input
3. System aggregates perspectives
4. Review consolidated recommendations

## Environment Variables

### Frontend (.env)
```env
VITE_API_URL=http://localhost:3001/api
VITE_WS_URL=ws://localhost:3001/ws
```

### Backend (.env)
```env
PORT=3001
NODE_ENV=development
CORS_ORIGIN=http://localhost:5173
CLEAR_THOUGHT_MCP_URL=http://localhost:8080
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/prioritize` | Calculate prioritization scores |
| POST | `/api/v1/auto-fill` | Get auto-fill suggestions |
| POST | `/api/v1/normalize` | Normalize scores across frameworks |
| GET | `/api/v1/sessions` | List saved sessions |
| POST | `/api/v1/sessions` | Create new session |
| GET | `/api/v1/sessions/:id` | Get session by ID |
| PUT | `/api/v1/sessions/:id` | Update session |
| DELETE | `/api/v1/sessions/:id` | Delete session |
| POST | `/api/v1/export` | Export results (CSV/JSON) |
| POST | `/api/v1/agents/run` | Run agent pipeline |

## License

MIT License - See LICENSE file for details.

## Contributing

1. Open an Issue to discuss major changes
2. Follow the implementation guide
3. Submit PR with tests and documentation

---

**Next Steps:**
- Read [Architecture](./architecture.md) for system design details
- Review [Pipeline Specification](./pipeline.md) for agent workflows
- Follow [Implementation Guide](./implementation-guide.md) for development
