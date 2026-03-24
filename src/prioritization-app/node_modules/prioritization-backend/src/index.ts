/**
 * Backend server entry point for Prioritization Frameworks Web Application.
 * Implements Express REST API and WebSocket server for real-time communication.
 *
 * @module index
 */

import express, { Request, Response } from 'express';
import { WebSocketServer, WebSocket } from 'ws';
import http from 'http';
import cors from 'cors';
import dotenv from 'dotenv';
import { routes } from './api/routes';
import { SessionService } from './services/SessionService';
import { WebSocketMessage, WebSocketMessageType } from './types';

// Load environment variables
dotenv.config();

const PORT = process.env.PORT ? parseInt(process.env.PORT, 10) : 3001;
const HOST = process.env.HOST || 'localhost';

/**
 * Store connected WebSocket clients.
 */
const clients: Set<WebSocket> = new Set();

/**
 * Broadcast message to all connected WebSocket clients.
 * @param message - Message to broadcast
 */
function broadcast(message: WebSocketMessage): void {
  const messageStr = JSON.stringify(message);
  clients.forEach(client => {
    if (client.readyState === WebSocket.OPEN) {
      client.send(messageStr);
    }
  });
}

/**
 * Send message to a specific WebSocket client.
 * @param client - Target WebSocket client
 * @param message - Message to send
 */
function sendToClient(client: WebSocket, message: WebSocketMessage): void {
  if (client.readyState === WebSocket.OPEN) {
    client.send(JSON.stringify(message));
  }
}

/**
 * Initialize Express application with CORS enabled.
 * @returns Configured Express application instance
 */
export function createApp(): express.Application {
  const app = express();

  // Middleware
  app.use(cors());
  app.use(express.json());
  app.use(express.urlencoded({ extended: true }));

  // API Routes (v1)
  app.use('/api/v1', routes);

  // Legacy health check endpoint for backwards compatibility
  app.get('/api/health', (_req: Request, res: Response) => {
    res.json({
      status: 'healthy',
      timestamp: new Date().toISOString(),
      version: '1.0.0'
    });
  });

  // API version endpoint
  app.get('/api/version', (_req: Request, res: Response) => {
    res.json({
      name: 'Prioritization Frameworks Backend',
      version: '1.0.0',
      apiVersion: 'v1',
      endpoints: {
        sessions: '/api/v1/sessions',
        prioritize: '/api/v1/prioritize',
        export: '/api/v1/export',
        health: '/api/v1/health',
        stats: '/api/v1/stats'
      }
    });
  });

  // Root endpoint
  app.get('/', (_req: Request, res: Response) => {
    res.json({
      message: 'Prioritization Frameworks API Server',
      version: '1.0.0',
      documentation: '/api/version',
      websocket: `ws://${HOST}:${PORT}/ws`
    });
  });

  return app;
}

/**
 * Express application instance for testing.
 * Exported for use in integration tests.
 */
export const app = createApp();

/**
 * Create HTTP server and attach WebSocket server.
 * @param appInstance - Express application instance
 * @returns Object containing HTTP server and WebSocket server
 */
export function createServers(appInstance: express.Application): { httpServer: http.Server; wsServer: WebSocketServer } {
  const httpServer = http.createServer(appInstance);

  const wsServer = new WebSocketServer({
    server: httpServer,
    path: '/ws'
  });

  /**
   * Handle WebSocket connections.
   * @param ws - WebSocket connection instance
   */
  wsServer.on('connection', (ws: WebSocket) => {
    console.log(`[${new Date().toISOString()}] WebSocket client connected`);

    // Add client to set
    clients.add(ws);

    // Send welcome message
    sendToClient(ws, {
      type: 'welcome',
      message: 'Connected to Prioritization Frameworks WebSocket',
      timestamp: new Date().toISOString()
    });

    // Handle incoming messages
    ws.on('message', (data: Buffer) => {
      try {
        const message = JSON.parse(data.toString());
        console.log(`[${new Date().toISOString()}] Received:`, message);

        // Echo response for health check
        if (message.type === 'ping') {
          sendToClient(ws, {
            type: 'pong',
            timestamp: new Date().toISOString()
          });
        }
      } catch (error) {
        console.error('Error parsing WebSocket message:', error);
      }
    });

    // Handle disconnection
    ws.on('close', () => {
      console.log(`[${new Date().toISOString()}] WebSocket client disconnected`);
      clients.delete(ws);
    });

    // Handle errors
    ws.on('error', (error: Error) => {
      console.error(`[${new Date().toISOString()}] WebSocket error:`, error);
      clients.delete(ws);
    });
  });

  /**
   * Subscribe SessionService events to WebSocket broadcasts.
   */
  SessionService.subscribe((event: string, data: unknown) => {
    const messageType = event as WebSocketMessageType;
    broadcast({
      type: messageType,
      payload: data,
      timestamp: new Date().toISOString()
    });
    console.log(`[${new Date().toISOString()}] Broadcast event:`, event, data);
  });

  return { httpServer, wsServer };
}

// Only start server if not in test mode
const isTestMode = process.env.NODE_ENV === 'test' || process.env.VITEST;

if (!isTestMode) {
  const appInstance = createApp();
  const { httpServer, wsServer } = createServers(appInstance);

  // Handle server errors
  httpServer.on('error', (error: NodeJS.ErrnoException) => {
    if (error.code === 'EADDRINUSE') {
      console.error(`Port ${PORT} is already in use. Please use a different port or stop the process using it.`);
      process.exit(1);
    }
    console.error('Server error:', error);
    process.exit(1);
  });

  // Start server
  httpServer.listen(PORT, () => {
    console.log(`
╔═══════════════════════════════════════════════════════════╗
║     Prioritization Frameworks Backend Server              ║
╠═══════════════════════════════════════════════════════════╣
║  HTTP Server:  http://${HOST}:${PORT}
║  WebSocket:    ws://${HOST}:${PORT}/ws
║  Health:       http://${HOST}:${PORT}/api/health
║  Version:      http://${HOST}:${PORT}/api/version
╚═══════════════════════════════════════════════════════════╝
    `);
  });

  // Graceful shutdown
  const shutdown = (signal: string): void => {
    console.log(`\n[${new Date().toISOString()}] Received ${signal}, shutting down gracefully...`);

    httpServer.close(() => {
      console.log('HTTP server closed');
      wsServer.close(() => {
        console.log('WebSocket server closed');
        process.exit(0);
      });
    });

    // Force shutdown after timeout
    setTimeout(() => {
      console.error('Forced shutdown after timeout');
      process.exit(1);
    }, 10000);
  };

  process.on('SIGTERM', () => shutdown('SIGTERM'));
  process.on('SIGINT', () => shutdown('SIGINT'));
}
