/**
 * API Routes for Prioritization Frameworks Backend.
 *
 * This module defines all REST API endpoints for session management,
 * prioritization calculations, and data export.
 *
 * @module api/routes
 */

import { Router, Request, Response } from 'express';
import { SessionService } from '../services/SessionService';
import { PrioritizationService } from '../services/PrioritizationService';
import {
  CreateSessionDTO,
  UpdateSessionDTO,
  PrioritizeRequestDTO,
  BulkPrioritizeRequestDTO,
  ExportRequestDTO,
  ErrorResponseDTO,
  PrioritizationItem,
  FrameworkType
} from '../types';

const router = Router();

// Start time for uptime calculation
const startTime = Date.now();

/**
 * Format error response.
 * @param res - Express response object
 * @param error - Error code
 * @param message - Error message
 * @param details - Additional error details
 * @param status - HTTP status code
 */
function formatErrorResponse(
  res: Response,
  error: string,
  message: string,
  details?: Record<string, unknown>,
  status: number = 400
): Response {
  const responseBody: ErrorResponseDTO = {
    error,
    message,
    details,
    timestamp: new Date().toISOString(),
    path: res.req.path
  };
  return res.status(status).json(responseBody);
}

/**
 * Validate request body has required fields.
 * @param body - Request body
 * @param requiredFields - Array of required field names
 * @returns Validation result with isValid and missing fields
 */
function validateRequiredFields(
  body: Record<string, unknown>,
  requiredFields: string[]
): { isValid: boolean; missing: string[] } {
  const missing = requiredFields.filter(field => !(field in body));
  return { isValid: missing.length === 0, missing };
}

// ============================================================================
// Session Management Routes
// ============================================================================

/**
 * POST /api/v1/sessions
 * Create a new session.
 *
 * Request body:
 * - name (required): Session name
 * - framework (required): Framework type
 * - projectName (optional): Project name
 * - projectDescription (optional): Project description
 * - teamName (optional): Team name
 * - industry (optional): Industry context
 * - items (optional): Initial items array
 */
router.post('/sessions', async (req: Request, res: Response) => {
  try {
    const validation = validateRequiredFields(req.body, ['name', 'framework']);
    if (!validation.isValid) {
      return formatErrorResponse(
        res,
        'ValidationError',
        `Missing required fields: ${validation.missing.join(', ')}`,
        { missingFields: validation.missing },
        400
      );
    }

    const createSessionDTO: CreateSessionDTO = {
      name: req.body.name,
      framework: req.body.framework,
      projectName: req.body.projectName,
      projectDescription: req.body.projectDescription,
      teamName: req.body.teamName,
      industry: req.body.industry,
      items: req.body.items
    };

    const session = await SessionService.create(createSessionDTO);
    return res.status(201).json(session);
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Unknown error';
    return formatErrorResponse(res, 'CreateError', message, undefined, 500);
  }
});

/**
 * GET /api/v1/sessions
 * List all sessions.
 */
router.get('/sessions', async (_req: Request, res: Response) => {
  try {
    const sessions = await SessionService.findAll();
    return res.json(sessions);
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Unknown error';
    return formatErrorResponse(res, 'FetchError', message, undefined, 500);
  }
});

/**
 * GET /api/v1/sessions/:id
 * Get session by ID.
 */
router.get('/sessions/:id', async (req: Request, res: Response) => {
  try {
    const session = await SessionService.findById(req.params.id);
    if (!session) {
      return formatErrorResponse(
        res,
        'NotFoundError',
        `Session not found: ${req.params.id}`,
        undefined,
        404
      );
    }
    return res.json(session);
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Unknown error';
    return formatErrorResponse(res, 'FetchError', message, undefined, 500);
  }
});

/**
 * PUT /api/v1/sessions/:id
 * Update session.
 *
 * Request body:
 * - name (optional): New session name
 * - projectName (optional): New project name
 * - projectDescription (optional): New project description
 * - teamName (optional): New team name
 * - industry (optional): New industry
 * - framework (optional): New framework
 */
router.put('/sessions/:id', async (req: Request, res: Response) => {
  try {
    const updateSessionDTO: UpdateSessionDTO = {
      name: req.body.name,
      projectName: req.body.projectName,
      projectDescription: req.body.projectDescription,
      teamName: req.body.teamName,
      industry: req.body.industry,
      framework: req.body.framework
    };

    // Remove undefined fields
    Object.keys(updateSessionDTO).forEach(key => {
      if (updateSessionDTO[key as keyof UpdateSessionDTO] === undefined) {
        delete updateSessionDTO[key as keyof UpdateSessionDTO];
      }
    });

    const session = await SessionService.update(req.params.id, updateSessionDTO);
    return res.json(session);
  } catch (error) {
    if (error instanceof Error && error.name === 'NotFoundError') {
      return formatErrorResponse(
        res,
        'NotFoundError',
        `Session not found: ${req.params.id}`,
        undefined,
        404
      );
    }
    const message = error instanceof Error ? error.message : 'Unknown error';
    return formatErrorResponse(res, 'UpdateError', message, undefined, 500);
  }
});

/**
 * DELETE /api/v1/sessions/:id
 * Delete session.
 */
router.delete('/sessions/:id', async (req: Request, res: Response) => {
  try {
    await SessionService.delete(req.params.id);
    return res.status(204).send();
  } catch (error) {
    if (error instanceof Error && error.name === 'NotFoundError') {
      return formatErrorResponse(
        res,
        'NotFoundError',
        `Session not found: ${req.params.id}`,
        undefined,
        404
      );
    }
    const message = error instanceof Error ? error.message : 'Unknown error';
    return formatErrorResponse(res, 'DeleteError', message, undefined, 500);
  }
});

/**
 * POST /api/v1/sessions/:id/items
 * Add item to session.
 *
 * Request body:
 * - title (required): Item title
 * - description (optional): Item description
 * - category (optional): Item category
 * - frameworkData (optional): Framework-specific data
 */
router.post('/sessions/:id/items', async (req: Request, res: Response) => {
  try {
    if (!req.body.title) {
      return formatErrorResponse(
        res,
        'ValidationError',
        'Missing required field: title',
        { missingFields: ['title'] },
        400
      );
    }

    const item: Omit<PrioritizationItem, 'id' | 'createdAt' | 'updatedAt'> = {
      title: req.body.title,
      description: req.body.description,
      category: req.body.category,
      frameworkData: req.body.frameworkData,
      metadata: req.body.metadata
    };

    const session = await SessionService.addItem(req.params.id, item);
    return res.status(201).json(session);
  } catch (error) {
    if (error instanceof Error && error.name === 'NotFoundError') {
      return formatErrorResponse(
        res,
        'NotFoundError',
        `Session not found: ${req.params.id}`,
        undefined,
        404
      );
    }
    const message = error instanceof Error ? error.message : 'Unknown error';
    return formatErrorResponse(res, 'CreateError', message, undefined, 500);
  }
});

/**
 * PUT /api/v1/sessions/:id/items/:itemId
 * Update item in session.
 *
 * Request body:
 * - title (optional): New title
 * - description (optional): New description
 * - category (optional): New category
 * - frameworkData (optional): New framework data
 * - priorityScore (optional): New priority score
 */
router.put(
  '/sessions/:id/items/:itemId',
  async (req: Request, res: Response) => {
    try {
      const updates: Partial<PrioritizationItem> = {
        title: req.body.title,
        description: req.body.description,
        category: req.body.category,
        frameworkData: req.body.frameworkData,
        priorityScore: req.body.priorityScore,
        rank: req.body.rank,
        metadata: req.body.metadata
      };

      // Remove undefined fields
      Object.keys(updates).forEach(key => {
        if (updates[key as keyof Partial<PrioritizationItem>] === undefined) {
          delete updates[key as keyof Partial<PrioritizationItem>];
        }
      });

      const session = await SessionService.updateItem(
        req.params.id,
        req.params.itemId,
        updates
      );
      return res.json(session);
    } catch (error) {
      if (error instanceof Error && error.name === 'NotFoundError') {
        return formatErrorResponse(
          res,
          'NotFoundError',
          `Item not found: ${req.params.itemId}`,
          undefined,
          404
        );
      }
      const message = error instanceof Error ? error.message : 'Unknown error';
      return formatErrorResponse(res, 'UpdateError', message, undefined, 500);
    }
  }
);

/**
 * DELETE /api/v1/sessions/:id/items/:itemId
 * Remove item from session.
 */
router.delete(
  '/sessions/:id/items/:itemId',
  async (req: Request, res: Response) => {
    try {
      const session = await SessionService.removeItem(
        req.params.id,
        req.params.itemId
      );
      return res.json(session);
    } catch (error) {
      if (error instanceof Error && error.name === 'NotFoundError') {
        return formatErrorResponse(
          res,
          'NotFoundError',
          `Item not found: ${req.params.itemId}`,
          undefined,
          404
        );
      }
      const message = error instanceof Error ? error.message : 'Unknown error';
      return formatErrorResponse(res, 'DeleteError', message, undefined, 500);
    }
  }
);

// ============================================================================
// Prioritization Routes
// ============================================================================

/**
 * POST /api/v1/prioritize
 * Calculate priority for single item.
 *
 * Request body:
 * - framework (required): Framework type
 * - item (required): Item to prioritize with frameworkData
 * - context (optional): Other items for normalization
 */
router.post('/prioritize', async (req: Request, res: Response) => {
  try {
    const validation = validateRequiredFields(req.body, ['framework', 'item']);
    if (!validation.isValid) {
      return formatErrorResponse(
        res,
        'ValidationError',
        `Missing required fields: ${validation.missing.join(', ')}`,
        { missingFields: validation.missing },
        400
      );
    }

    const request: PrioritizeRequestDTO = {
      framework: req.body.framework,
      item: req.body.item,
      context: req.body.context
    };

    const result = await PrioritizationService.prioritize(request);
    return res.json(result);
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Unknown error';
    const statusCode =
      error instanceof Error && error.name === 'ValidationError' ? 400 : 500;
    return formatErrorResponse(
      res,
      error instanceof Error ? error.name : 'CalculationError',
      message,
      undefined,
      statusCode
    );
  }
});

/**
 * POST /api/v1/prioritize/bulk
 * Calculate priority for multiple items.
 *
 * Request body:
 * - framework (required): Framework type
 * - items (required): Array of items to prioritize
 */
router.post('/prioritize/bulk', async (req: Request, res: Response) => {
  try {
    const validation = validateRequiredFields(req.body, ['framework', 'items']);
    if (!validation.isValid) {
      return formatErrorResponse(
        res,
        'ValidationError',
        `Missing required fields: ${validation.missing.join(', ')}`,
        { missingFields: validation.missing },
        400
      );
    }

    const request: BulkPrioritizeRequestDTO = {
      framework: req.body.framework,
      items: req.body.items
    };

    const results = await PrioritizationService.prioritizeBulk(request);
    return res.json(results);
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Unknown error';
    return formatErrorResponse(res, 'CalculationError', message, undefined, 500);
  }
});

// ============================================================================
// Export Routes
// ============================================================================

/**
 * POST /api/v1/export/csv
 * Export session data to CSV.
 *
 * Request body:
 * - sessionId (required): Session ID to export
 * - fields (optional): Fields to include
 */
router.post('/export/csv', async (req: Request, res: Response) => {
  try {
    const validation = validateRequiredFields(req.body, ['sessionId']);
    if (!validation.isValid) {
      return formatErrorResponse(
        res,
        'ValidationError',
        `Missing required fields: ${validation.missing.join(', ')}`,
        { missingFields: validation.missing },
        400
      );
    }

    const exportRequest: ExportRequestDTO = {
      sessionId: req.body.sessionId,
      format: 'csv',
      fields: req.body.fields
    };

    const session = await SessionService.findById(exportRequest.sessionId);
    if (!session) {
      return formatErrorResponse(
        res,
        'NotFoundError',
        `Session not found: ${exportRequest.sessionId}`,
        undefined,
        404
      );
    }

    // Generate CSV content
    const fields = exportRequest.fields || [
      'title',
      'description',
      'category',
      'priorityScore',
      'rank'
    ];

    const header = fields.join(',');
    const rows = session.items.map(item =>
      fields
        .map(field => {
          const value = item[field as keyof PrioritizationItem];
          if (typeof value === 'object' && value !== null) {
            return `"${JSON.stringify(value).replace(/"/g, '""')}"`;
          }
          return `"${String(value ?? '').replace(/"/g, '""')}"`;
        })
        .join(',')
    );

    const csvContent = [header, ...rows].join('\n');
    const fileName = `${session.name.replace(/\s+/g, '_')}_export.csv`;

    return res.json({
      fileName,
      mimeType: 'text/csv',
      content: csvContent
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Unknown error';
    return formatErrorResponse(res, 'ExportError', message, undefined, 500);
  }
});

/**
 * POST /api/v1/export/json
 * Export session data to JSON.
 *
 * Request body:
 * - sessionId (required): Session ID to export
 */
router.post('/export/json', async (req: Request, res: Response) => {
  try {
    const validation = validateRequiredFields(req.body, ['sessionId']);
    if (!validation.isValid) {
      return formatErrorResponse(
        res,
        'ValidationError',
        `Missing required fields: ${validation.missing.join(', ')}`,
        { missingFields: validation.missing },
        400
      );
    }

    const session = await SessionService.findById(req.body.sessionId);
    if (!session) {
      return formatErrorResponse(
        res,
        'NotFoundError',
        `Session not found: ${req.body.sessionId}`,
        undefined,
        404
      );
    }

    const jsonContent = JSON.stringify(session, null, 2);
    const fileName = `${session.name.replace(/\s+/g, '_')}_export.json`;

    return res.json({
      fileName,
      mimeType: 'application/json',
      content: jsonContent
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Unknown error';
    return formatErrorResponse(res, 'ExportError', message, undefined, 500);
  }
});

// ============================================================================
// Health & Status Routes
// ============================================================================

/**
 * GET /api/v1/health
 * Health check endpoint.
 */
router.get('/health', async (_req: Request, res: Response) => {
  try {
    const sessions = await SessionService.findAll();
    const uptime = Math.floor((Date.now() - startTime) / 1000);

    return res.json({
      status: 'healthy' as const,
      timestamp: new Date().toISOString(),
      version: '1.0.0',
      uptime,
      activeSessions: sessions.length
    });
  } catch (error) {
    return res.status(500).json({
      status: 'unhealthy' as const,
      timestamp: new Date().toISOString(),
      version: '1.0.0',
      uptime: 0,
      activeSessions: 0,
      error: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

/**
 * GET /api/v1/stats
 * Get statistics.
 */
router.get('/stats', async (_req: Request, res: Response) => {
  try {
    const stats = await SessionService.getStats();
    const uptime = Math.floor((Date.now() - startTime) / 1000);

    // Find most used framework
    let mostUsedFramework: string = 'RICE';
    let maxCount = 0;
    for (const [framework, count] of Object.entries(stats.sessionsByFramework)) {
      if (count > maxCount) {
        maxCount = count;
        mostUsedFramework = framework;
      }
    }

    return res.json({
      totalSessions: stats.totalSessions,
      totalItems: stats.totalItems,
      avgItemsPerSession: stats.avgItemsPerSession,
      mostUsedFramework: mostUsedFramework as FrameworkType,
      sessionsByFramework: stats.sessionsByFramework,
      uptime
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Unknown error';
    return formatErrorResponse(res, 'StatsError', message, undefined, 500);
  }
});

// ============================================================================
// Agent Pipeline Routes (Phase 6)
// ============================================================================

// Lazy import PipelineOrchestrator to avoid circular dependencies
let pipelineOrchestrator: import('../agents/PipelineOrchestrator').PipelineOrchestrator | undefined = undefined;

/**
 * Get or create the Pipeline Orchestrator singleton.
 * @returns Pipeline Orchestrator instance
 */
function getPipelineOrchestrator(): import('../agents/PipelineOrchestrator').PipelineOrchestrator {
  if (!pipelineOrchestrator) {
    const { PipelineOrchestrator } = require('../agents/PipelineOrchestrator');
    pipelineOrchestrator = new PipelineOrchestrator();
  }
  return pipelineOrchestrator!;
}

/**
 * POST /api/v1/agents/run
 * Start a new agent pipeline execution.
 *
 * Request body:
 * - input (required): Input data for the pipeline
 * - config (optional): Pipeline configuration
 *   - agentOrder: Array of agent names in execution order
 *   - stopOnError: Whether to stop on first error (default: true)
 *   - agentTimeout: Timeout per agent in ms (default: 60000)
 *   - parallel: Whether to run agents in parallel (default: false)
 * - context (optional): Pipeline context
 *   - projectName: Project name
 *   - teamName: Team name
 *   - industry: Industry context
 */
router.post('/agents/run', async (req: Request, res: Response) => {
  try {
    const validation = validateRequiredFields(req.body, ['input']);
    if (!validation.isValid) {
      return formatErrorResponse(
        res,
        'ValidationError',
        `Missing required fields: ${validation.missing.join(', ')}`,
        { missingFields: validation.missing },
        400
      );
    }

    const orchestrator = getPipelineOrchestrator();
    const response = await orchestrator.runPipeline(req.body.input, req.body.config);

    return res.status(202).json(response);
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Unknown error';
    return formatErrorResponse(res, 'PipelineError', message, undefined, 500);
  }
});

/**
 * GET /api/v1/agents/status/:id
 * Get pipeline session status.
 *
 * @param id - Pipeline session ID
 */
router.get('/agents/status/:id', async (req: Request, res: Response) => {
  try {
    const orchestrator = getPipelineOrchestrator();
    const status = orchestrator.getSessionStatus(req.params.id);

    if (!status) {
      return formatErrorResponse(
        res,
        'NotFoundError',
        `Pipeline session not found: ${req.params.id}`,
        undefined,
        404
      );
    }

    return res.json(status);
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Unknown error';
    return formatErrorResponse(res, 'FetchError', message, undefined, 500);
  }
});

/**
 * GET /api/v1/agents/session/:id
 * Get full pipeline session details.
 *
 * @param id - Pipeline session ID
 */
router.get('/agents/session/:id', async (req: Request, res: Response) => {
  try {
    const orchestrator = getPipelineOrchestrator();
    const session = orchestrator.getSession(req.params.id);

    if (!session) {
      return formatErrorResponse(
        res,
        'NotFoundError',
        `Pipeline session not found: ${req.params.id}`,
        undefined,
        404
      );
    }

    return res.json(session);
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Unknown error';
    return formatErrorResponse(res, 'FetchError', message, undefined, 500);
  }
});

/**
 * POST /api/v1/agents/cancel/:id
 * Cancel a running pipeline.
 *
 * @param id - Pipeline session ID
 */
router.post('/agents/cancel/:id', async (req: Request, res: Response) => {
  try {
    const orchestrator = getPipelineOrchestrator();
    const cancelled = orchestrator.cancelPipeline(req.params.id);

    if (!cancelled) {
      return formatErrorResponse(
        res,
        'BadRequestError',
        `Cannot cancel pipeline: ${req.params.id} may be completed or not found`,
        undefined,
        400
      );
    }

    return res.status(200).json({
      success: true,
      sessionId: req.params.id,
      status: 'cancelled',
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Unknown error';
    return formatErrorResponse(res, 'CancelError', message, undefined, 500);
  }
});

/**
 * GET /api/v1/agents/list
 * List all active pipeline sessions.
 */
router.get('/agents/list', async (_req: Request, res: Response) => {
  try {
    const orchestrator = getPipelineOrchestrator();
    const sessions = orchestrator.getActiveSessions();

    return res.json({
      activeSessions: sessions.length,
      sessions: sessions.map(s => ({
        id: s.id,
        status: s.status,
        currentAgent: s.currentAgent,
        createdAt: s.createdAt,
        resultsCount: s.results.length
      }))
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Unknown error';
    return formatErrorResponse(res, 'FetchError', message, undefined, 500);
  }
});

/**
 * GET /api/v1/agents/available
 * List available agents.
 */
router.get('/agents/available', async (_req: Request, res: Response) => {
  try {
    const agents = [
      { name: 'PlanningAgent', description: 'Analyzes problems and creates structured implementation plans' },
      { name: 'DeveloperAgent', description: 'Designs technical solutions and implementation strategies' },
      { name: 'ReviewerAgent', description: 'Reviews work products and identifies improvement opportunities' }
    ];

    return res.json({
      agents,
      defaultPipeline: ['PlanningAgent', 'DeveloperAgent', 'ReviewerAgent']
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Unknown error';
    return formatErrorResponse(res, 'FetchError', message, undefined, 500);
  }
});

export { router as routes };
export default router;
