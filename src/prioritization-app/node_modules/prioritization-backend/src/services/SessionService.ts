/**
 * Session Service for managing prioritization sessions.
 *
 * This service handles CRUD operations for sessions and their items.
 * Uses in-memory storage for Phase 5 (IndexedDB on frontend handles persistence).
 *
 * @module services/SessionService
 */

import {
  Session,
  CreateSessionDTO,
  UpdateSessionDTO,
  PrioritizationItem
} from '../types';

/**
 * In-memory session storage.
 * Map key is session ID, value is Session object.
 */
const sessionsStore: Map<string, Session> = new Map();

/**
 * Session listeners for WebSocket events.
 */
type SessionListener = (event: string, data: unknown) => void;
const listeners: SessionListener[] = [];

/**
 * Emit event to all registered listeners.
 * @param event - Event name
 * @param data - Event payload
 */
function emitEvent(event: string, data: unknown): void {
  const timestamp = new Date().toISOString();
  listeners.forEach(listener => {
    try {
      listener(event, { event, data, timestamp });
    } catch (error) {
      console.error('Error in session listener:', error);
    }
  });
}

/**
 * Generate a unique session ID.
 * @returns Unique session identifier
 */
function generateId(): string {
  return `session_${Date.now()}_${Math.random().toString(36).substring(2, 11)}`;
}

/**
 * Generate a unique item ID.
 * @returns Unique item identifier
 */
function generateItemId(): string {
  return `item_${Date.now()}_${Math.random().toString(36).substring(2, 9)}`;
}

/**
 * SessionService class for managing prioritization sessions.
 *
 * Provides methods for creating, reading, updating, and deleting sessions
 * and their associated prioritization items.
 */
export class SessionService {
  /**
   * Register a listener for session events.
   * @param listener - Callback function to receive events
   * @returns Unsubscribe function
   */
  public static subscribe(listener: SessionListener): () => void {
    listeners.push(listener);
    return () => {
      const index = listeners.indexOf(listener);
      if (index > -1) {
        listeners.splice(index, 1);
      }
    };
  }

  /**
   * Create a new session.
   *
   * @param sessionDto - Session creation data
   * @returns Created session object
   *
   * @example
   * ```typescript
   * const session = await SessionService.create({
   *   name: 'Q1 Features',
   *   framework: 'RICE',
   *   projectName: 'My Product'
   * });
   * ```
   */
  public static async create(sessionDto: CreateSessionDTO): Promise<Session> {
    const now = new Date();
    const id = generateId();

    const session: Session = {
      id,
      name: sessionDto.name,
      projectName: sessionDto.projectName,
      projectDescription: sessionDto.projectDescription,
      teamName: sessionDto.teamName,
      industry: sessionDto.industry,
      framework: sessionDto.framework,
      items: (sessionDto.items || []).map(item => ({
        ...item,
        id: generateItemId(),
        createdAt: now,
        updatedAt: now
      })),
      createdAt: now,
      updatedAt: now
    };

    sessionsStore.set(id, session);
    emitEvent('session:created', { session });

    return session;
  }

  /**
   * Retrieve all sessions.
   *
   * @returns Array of all sessions
   *
   * @example
   * ```typescript
   * const sessions = await SessionService.findAll();
   * ```
   */
  public static async findAll(): Promise<Session[]> {
    return Array.from(sessionsStore.values()).map(session => ({ ...session }));
  }

  /**
   * Retrieve a session by ID.
   *
   * @param id - Session identifier
   * @returns Session object or null if not found
   *
   * @example
   * ```typescript
   * const session = await SessionService.findById('session_123');
   * if (!session) {
   *   console.log('Session not found');
   * }
   * ```
   */
  public static async findById(id: string): Promise<Session | null> {
    const session = sessionsStore.get(id);
    return session ? { ...session } : null;
  }

  /**
   * Update an existing session.
   *
   * @param id - Session identifier
   * @param updates - Fields to update
   * @returns Updated session object
   * @throws {Error} If session not found
   *
   * @example
   * ```typescript
   * const session = await SessionService.update('session_123', {
   *   name: 'Updated Name'
   * });
   * ```
   */
  public static async update(id: string, updates: UpdateSessionDTO): Promise<Session> {
    const session = sessionsStore.get(id);
    if (!session) {
      const error = new Error(`Session not found: ${id}`);
      error.name = 'NotFoundError';
      throw error;
    }

    const updatedSession: Session = {
      ...session,
      ...updates,
      items: updates.framework && updates.framework !== session.framework
        ? session.items.map(item => ({ ...item, frameworkData: undefined }))
        : session.items,
      updatedAt: new Date()
    };

    sessionsStore.set(id, updatedSession);
    emitEvent('session:updated', { session: updatedSession });

    return updatedSession;
  }

  /**
   * Delete a session.
   *
   * @param id - Session identifier
   * @throws {Error} If session not found
   *
   * @example
   * ```typescript
   * await SessionService.delete('session_123');
   * ```
   */
  public static async delete(id: string): Promise<void> {
    const session = sessionsStore.get(id);
    if (!session) {
      const error = new Error(`Session not found: ${id}`);
      error.name = 'NotFoundError';
      throw error;
    }

    sessionsStore.delete(id);
    emitEvent('session:deleted', { sessionId: id });
  }

  /**
   * Add an item to a session.
   *
   * @param sessionId - Session identifier
   * @param item - Item to add (without id, createdAt, updatedAt)
   * @returns Updated session with new item
   * @throws {Error} If session not found
   *
   * @example
   * ```typescript
   * const session = await SessionService.addItem('session_123', {
   *   title: 'New Feature',
   *   description: 'Description here'
   * });
   * ```
   */
  public static async addItem(
    sessionId: string,
    item: Omit<PrioritizationItem, 'id' | 'createdAt' | 'updatedAt'>
  ): Promise<Session> {
    const session = sessionsStore.get(sessionId);
    if (!session) {
      const error = new Error(`Session not found: ${sessionId}`);
      error.name = 'NotFoundError';
      throw error;
    }

    const now = new Date();
    const newItem: PrioritizationItem = {
      ...item,
      id: generateItemId(),
      createdAt: now,
      updatedAt: now
    };

    const updatedSession: Session = {
      ...session,
      items: [...session.items, newItem],
      updatedAt: now
    };

    sessionsStore.set(sessionId, updatedSession);
    emitEvent('item:added', { sessionId, item: newItem });

    return updatedSession;
  }

  /**
   * Update an item within a session.
   *
   * @param sessionId - Session identifier
   * @param itemId - Item identifier
   * @param updates - Fields to update
   * @returns Updated session with modified item
   * @throws {Error} If session or item not found
   *
   * @example
   * ```typescript
   * const session = await SessionService.updateItem('session_123', 'item_456', {
   *   title: 'Updated Title'
   * });
   * ```
   */
  public static async updateItem(
    sessionId: string,
    itemId: string,
    updates: Partial<PrioritizationItem>
  ): Promise<Session> {
    const session = sessionsStore.get(sessionId);
    if (!session) {
      const error = new Error(`Session not found: ${sessionId}`);
      error.name = 'NotFoundError';
      throw error;
    }

    const itemIndex = session.items.findIndex(item => item.id === itemId);
    if (itemIndex === -1) {
      const error = new Error(`Item not found: ${itemId}`);
      error.name = 'NotFoundError';
      throw error;
    }

    const updatedItems = [...session.items];
    updatedItems[itemIndex] = {
      ...updatedItems[itemIndex],
      ...updates,
      id: itemId, // Ensure ID cannot be changed
      createdAt: updatedItems[itemIndex].createdAt, // Preserve creation date
      updatedAt: new Date()
    };

    const updatedSession: Session = {
      ...session,
      items: updatedItems,
      updatedAt: new Date()
    };

    sessionsStore.set(sessionId, updatedSession);
    emitEvent('item:updated', { sessionId, item: updatedItems[itemIndex] });

    return updatedSession;
  }

  /**
   * Remove an item from a session.
   *
   * @param sessionId - Session identifier
   * @param itemId - Item identifier
   * @returns Updated session without the item
   * @throws {Error} If session or item not found
   *
   * @example
   * ```typescript
   * const session = await SessionService.removeItem('session_123', 'item_456');
   * ```
   */
  public static async removeItem(sessionId: string, itemId: string): Promise<Session> {
    const session = sessionsStore.get(sessionId);
    if (!session) {
      const error = new Error(`Session not found: ${sessionId}`);
      error.name = 'NotFoundError';
      throw error;
    }

    const itemIndex = session.items.findIndex(item => item.id === itemId);
    if (itemIndex === -1) {
      const error = new Error(`Item not found: ${itemId}`);
      error.name = 'NotFoundError';
      throw error;
    }

    const updatedItems = [...session.items];
    updatedItems.splice(itemIndex, 1);

    const updatedSession: Session = {
      ...session,
      items: updatedItems,
      updatedAt: new Date()
    };

    sessionsStore.set(sessionId, updatedSession);
    emitEvent('item:removed', { sessionId, itemId });

    return updatedSession;
  }

  /**
   * Get session statistics.
   *
   * @returns Statistics about sessions and items
   */
  public static async getStats(): Promise<{
    totalSessions: number;
    totalItems: number;
    avgItemsPerSession: number;
    sessionsByFramework: Record<string, number>;
  }> {
    const sessions = Array.from(sessionsStore.values());
    const totalSessions = sessions.length;
    const totalItems = sessions.reduce((sum, s) => sum + s.items.length, 0);

    const sessionsByFramework: Record<string, number> = {};
    sessions.forEach(session => {
      sessionsByFramework[session.framework] = (sessionsByFramework[session.framework] || 0) + 1;
    });

    return {
      totalSessions,
      totalItems,
      avgItemsPerSession: totalSessions > 0 ? totalItems / totalSessions : 0,
      sessionsByFramework
    };
  }

  /**
   * Clear all sessions (useful for testing).
   *
   * @internal
   */
  public static async clearAll(): Promise<void> {
    sessionsStore.clear();
  }
}

export default SessionService;
