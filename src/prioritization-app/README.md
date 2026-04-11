# Prioritization Frameworks Web Application

A web application that provides AI-powered prioritization frameworks to help teams make better decisions about projects and tasks.

## Overview

This application implements various prioritization methodologies including:
- **RICE Scoring Model** - Reach, Impact, Confidence, Effort
- **MoSCoW Method** - Must have, Should have, Could have, Won't have
- **Kano Model Analysis** - Customer satisfaction vs feature implementation
- **Value vs Effort Matrix** - Quick wins vs major projects
- **AI-Powered Insights** - Intelligent recommendations based on your data

## Architecture

The application follows a client-server architecture:

```
prioritization-app/
├── frontend/          # Vite + React 19 + TypeScript
│   ├── src/
│   │   ├── components/    # Reusable UI components
│   │   ├── hooks/         # Custom React hooks
│   │   ├── services/      # API and WebSocket services
│   │   ├── contexts/      # React context providers
│   │   ├── api/           # API client utilities
│   │   ├── utils/         # Pure utility functions
│   │   └── styles/        # Global and component styles
│   └── public/
└── backend/           # Node.js + Express + WebSocket
    └── src/
        ├── api/           # REST API endpoints
        ├── agents/        # AI agent implementations
        ├── mcp/           # MCP client integrations
        ├── services/      # Business logic services
        └── utils/         # Utility functions
```

## Prerequisites

- Node.js 18.0.0 or higher
- npm 9.0.0 or higher

## Installation

### Backend Setup

```bash
cd backend

# Install dependencies
npm install

# Copy environment example
cp .env.example .env

# Start development server
npm run dev
```

The backend will start on `http://localhost:3001`

### Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Copy environment example
cp .env.example .env

# Start development server
npm run dev
```

The frontend will start on `http://localhost:5173`

## Development Commands

### Backend

| Command | Description |
|---------|-------------|
| `npm run dev` | Start development server with hot reload |
| `npm run build` | Compile TypeScript to JavaScript |
| `npm run start` | Start production server |
| `npm run lint` | Run ESLint |
| `npm run test` | Run Jest tests |

### Frontend

| Command | Description |
|---------|-------------|
| `npm run dev` | Start Vite development server |
| `npm run build` | Build for production |
| `npm run preview` | Preview production build |
| `npm run lint` | Run ESLint |

## API Endpoints

### Health & Status

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Health check endpoint |
| `/api/version` | GET | API version information |
| `/` | GET | Server information |

### WebSocket

| Endpoint | Description |
|----------|-------------|
| `/ws` | WebSocket connection for real-time updates |

## Environment Variables

### Backend (.env)

```env
PORT=3001
HOST=localhost
NODE_ENV=development
LOG_LEVEL=debug
API_PREFIX=/api
```

### Frontend (.env)

```env
VITE_API_URL=http://localhost:3001
VITE_WS_URL=ws://localhost:3001/ws
VITE_NODE_ENV=development
VITE_ENABLE_DEBUG=false
```

## Features

### Phase 1 (Current)
- Project structure setup
- Express REST API server
- WebSocket real-time communication
- React 19 frontend with Vite
- Health check and status monitoring
- Dark theme UI

### Phase 2 (Planned)
- RICE Scoring implementation
- MoSCoW Method implementation
- Project/task management UI

### Phase 3 (Planned)
- Kano Model Analysis
- Value vs Effort Matrix
- AI agent integration

### Phase 4 (Planned)
- MCP client integration
- Advanced AI insights
- Reporting and export features

## Code Quality

- **TypeScript Strict Mode** - Full type safety
- **ESLint** - Code quality enforcement
- **JSDoc Comments** - Comprehensive documentation
- **Clean Code Principles** - Readable and maintainable code

## Contributing

1. Create a feature branch
2. Make your changes
3. Run linting and tests
4. Submit a pull request

## License

MIT
