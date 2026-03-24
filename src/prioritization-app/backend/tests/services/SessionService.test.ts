/**
 * Unit tests for SessionService.
 *
 * Tests cover:
 * - Session creation
 * - Find all sessions
 * - Find session by ID
 * - Session updates
 * - Session deletion
 * - Add item to session
 * - Update item in session
 * - Remove item from session
 * - Get statistics
 * - Clear all sessions
 *
 * @module services/tests/SessionService.test
 */

import { describe, it, expect, beforeEach, afterAll } from 'vitest';
import { SessionService } from '../../src/services/SessionService';
import type { CreateSessionDTO } from '../../src/types';

describe('SessionService', () => {
  // Clean up before each test
  beforeEach(async () => {
    await SessionService.clearAll();
  });

  // Clean up after all tests
  afterAll(async () => {
    await SessionService.clearAll();
  });

  describe('create', () => {
    it('creates a new session with required fields', async () => {
      const sessionData: CreateSessionDTO = {
        name: 'Test Session',
        framework: 'RICE',
      };

      const session = await SessionService.create(sessionData);

      expect(session.id).toBeDefined();
      expect(session.name).toBe('Test Session');
      expect(session.framework).toBe('RICE');
      expect(session.createdAt).toBeInstanceOf(Date);
      expect(session.updatedAt).toBeInstanceOf(Date);
      expect(session.items).toEqual([]);
    });

    it('creates a session with all optional fields', async () => {
      const sessionData: CreateSessionDTO = {
        name: 'Complete Session',
        framework: 'RICE',
        projectName: 'Test Project',
        projectDescription: 'A test project',
        teamName: 'Team Alpha',
        industry: 'Technology',
        items: [
          {
            title: 'Initial Item',
            description: 'Description',
            category: 'Feature',
          },
        ],
      };

      const session = await SessionService.create(sessionData);

      expect(session.name).toBe('Complete Session');
      expect(session.projectName).toBe('Test Project');
      expect(session.projectDescription).toBe('A test project');
      expect(session.teamName).toBe('Team Alpha');
      expect(session.industry).toBe('Technology');
      expect(session.items).toHaveLength(1);
      expect(session.items[0].title).toBe('Initial Item');
      expect(session.items[0].id).toBeDefined();
    });

    it('generates unique IDs for multiple sessions', async () => {
      const session1 = await SessionService.create({
        name: 'Session 1',
        framework: 'RICE',
      });
      const session2 = await SessionService.create({
        name: 'Session 2',
        framework: 'RICE',
      });

      expect(session1.id).not.toBe(session2.id);
    });

    it('generates unique IDs for items in session', async () => {
      const session = await SessionService.create({
        name: 'Test Session',
        framework: 'RICE',
        items: [
          { title: 'Item 1' },
          { title: 'Item 2' },
        ],
      });

      expect(session.items[0].id).not.toBe(session.items[1].id);
    });
  });

  describe('findAll', () => {
    it('returns empty array when no sessions exist', async () => {
      const sessions = await SessionService.findAll();
      expect(sessions).toEqual([]);
    });

    it('returns all sessions', async () => {
      await SessionService.create({ name: 'Session 1', framework: 'RICE' });
      await SessionService.create({ name: 'Session 2', framework: 'MoSCoW' });
      await SessionService.create({ name: 'Session 3', framework: 'ICE' });

      const sessions = await SessionService.findAll();

      expect(sessions).toHaveLength(3);
      expect(sessions.map((s) => s.name)).toEqual(
        expect.arrayContaining(['Session 1', 'Session 2', 'Session 3'])
      );
    });

    it('returns copies of sessions (not references)', async () => {
      const session = await SessionService.create({
        name: 'Original',
        framework: 'RICE',
      });

      const sessions = await SessionService.findAll();
      sessions[0].name = 'Modified';

      const session2 = await SessionService.findById(session.id);
      expect(session2?.name).toBe('Original');
    });
  });

  describe('findById', () => {
    it('finds session by ID', async () => {
      const createdSession = await SessionService.create({
        name: 'Test Session',
        framework: 'RICE',
      });

      const foundSession = await SessionService.findById(createdSession.id);

      expect(foundSession).not.toBeNull();
      expect(foundSession?.id).toBe(createdSession.id);
      expect(foundSession?.name).toBe('Test Session');
    });

    it('returns null for non-existent session', async () => {
      const foundSession = await SessionService.findById('non-existent-id');
      expect(foundSession).toBeNull();
    });

    it('returns a copy of the session (not reference)', async () => {
      const createdSession = await SessionService.create({
        name: 'Original',
        framework: 'RICE',
      });

      const foundSession = await SessionService.findById(createdSession.id);
      if (foundSession) {
        foundSession.name = 'Modified';
      }

      const foundSession2 = await SessionService.findById(createdSession.id);
      expect(foundSession2?.name).toBe('Original');
    });
  });

  describe('update', () => {
    it('updates session name', async () => {
      const session = await SessionService.create({
        name: 'Original Name',
        framework: 'RICE',
      });

      const updatedSession = await SessionService.update(session.id, {
        name: 'Updated Name',
      });

      expect(updatedSession.name).toBe('Updated Name');
      expect(updatedSession.id).toBe(session.id);
      expect(updatedSession.updatedAt).not.toEqual(session.createdAt);
    });

    it('updates multiple fields', async () => {
      const session = await SessionService.create({
        name: 'Test Session',
        framework: 'RICE',
        projectName: 'Original Project',
      });

      const updatedSession = await SessionService.update(session.id, {
        name: 'New Name',
        projectName: 'New Project',
        teamName: 'New Team',
      });

      expect(updatedSession.name).toBe('New Name');
      expect(updatedSession.projectName).toBe('New Project');
      expect(updatedSession.teamName).toBe('New Team');
      expect(updatedSession.framework).toBe('RICE'); // Unchanged
    });

    it('clears framework data when framework changes', async () => {
      const session = await SessionService.create({
        name: 'Test Session',
        framework: 'RICE',
        items: [
          {
            title: 'Item with data',
            frameworkData: { reach: 100, impact: 2 },
          },
        ],
      });

      const updatedSession = await SessionService.update(session.id, {
        framework: 'MoSCoW',
      });

      expect(updatedSession.framework).toBe('MoSCoW');
      expect(updatedSession.items[0].frameworkData).toBeUndefined();
    });

    it('throws NotFoundError for non-existent session', async () => {
      await expect(
        SessionService.update('non-existent-id', { name: 'Updated' })
      ).rejects.toThrow('Session not found');
    });

    it('handles empty update body', async () => {
      const session = await SessionService.create({
        name: 'Test Session',
        framework: 'RICE',
      });

      const updatedSession = await SessionService.update(session.id, {});

      expect(updatedSession.name).toBe('Test Session');
      expect(updatedSession.framework).toBe('RICE');
    });
  });

  describe('delete', () => {
    it('deletes session successfully', async () => {
      const session = await SessionService.create({
        name: 'To Delete',
        framework: 'RICE',
      });

      await SessionService.delete(session.id);

      const foundSession = await SessionService.findById(session.id);
      expect(foundSession).toBeNull();
    });

    it('throws NotFoundError for non-existent session', async () => {
      await expect(
        SessionService.delete('non-existent-id')
      ).rejects.toThrow('Session not found');
    });
  });

  describe('addItem', () => {
    it('adds item to session', async () => {
      const session = await SessionService.create({
        name: 'Test Session',
        framework: 'RICE',
      });

      const updatedSession = await SessionService.addItem(session.id, {
        title: 'New Item',
        description: 'Item description',
        category: 'Feature',
      });

      expect(updatedSession.items).toHaveLength(1);
      expect(updatedSession.items[0].title).toBe('New Item');
      expect(updatedSession.items[0].description).toBe('Item description');
      expect(updatedSession.items[0].id).toBeDefined();
    });

    it('adds multiple items to session', async () => {
      const session = await SessionService.create({
        name: 'Test Session',
        framework: 'RICE',
      });

      let updatedSession = await SessionService.addItem(session.id, {
        title: 'Item 1',
      });
      updatedSession = await SessionService.addItem(updatedSession.id, {
        title: 'Item 2',
      });

      expect(updatedSession.items).toHaveLength(2);
      expect(updatedSession.items[0].title).toBe('Item 1');
      expect(updatedSession.items[1].title).toBe('Item 2');
    });

    it('throws NotFoundError for non-existent session', async () => {
      await expect(
        SessionService.addItem('non-existent-id', { title: 'Item' })
      ).rejects.toThrow('Session not found');
    });
  });

  describe('updateItem', () => {
    it('updates item title', async () => {
      const session = await SessionService.create({
        name: 'Test Session',
        framework: 'RICE',
      });

      const itemSession = await SessionService.addItem(session.id, {
        title: 'Original Title',
      });

      const itemId = itemSession.items[0].id;

      const updatedSession = await SessionService.updateItem(
        session.id,
        itemId,
        { title: 'Updated Title' }
      );

      expect(updatedSession.items[0].title).toBe('Updated Title');
    });

    it('updates item description', async () => {
      const session = await SessionService.create({
        name: 'Test Session',
        framework: 'RICE',
      });

      const itemSession = await SessionService.addItem(session.id, {
        title: 'Item',
        description: 'Original description',
      });

      const itemId = itemSession.items[0].id;

      const updatedSession = await SessionService.updateItem(
        session.id,
        itemId,
        { description: 'Updated description' }
      );

      expect(updatedSession.items[0].description).toBe('Updated description');
    });

    it('preserves item ID and createdAt', async () => {
      const session = await SessionService.create({
        name: 'Test Session',
        framework: 'RICE',
      });

      const itemSession = await SessionService.addItem(session.id, {
        title: 'Item',
      });

      const itemId = itemSession.items[0].id;
      const originalCreatedAt = itemSession.items[0].createdAt;

      const updatedSession = await SessionService.updateItem(
        session.id,
        itemId,
        { title: 'Updated' }
      );

      expect(updatedSession.items[0].id).toBe(itemId);
      expect(updatedSession.items[0].createdAt).toEqual(originalCreatedAt);
    });

    it('throws NotFoundError for non-existent session', async () => {
      await expect(
        SessionService.updateItem('non-existent-id', 'item-123', {
          title: 'Updated',
        })
      ).rejects.toThrow('Session not found');
    });

    it('throws NotFoundError for non-existent item', async () => {
      const session = await SessionService.create({
        name: 'Test Session',
        framework: 'RICE',
      });

      await expect(
        SessionService.updateItem(session.id, 'non-existent-item', {
          title: 'Updated',
        })
      ).rejects.toThrow('Item not found');
    });
  });

  describe('removeItem', () => {
    it('removes item from session', async () => {
      const session = await SessionService.create({
        name: 'Test Session',
        framework: 'RICE',
      });

      const itemSession = await SessionService.addItem(session.id, {
        title: 'To Remove',
      });

      const itemId = itemSession.items[0].id;

      const updatedSession = await SessionService.removeItem(
        session.id,
        itemId
      );

      expect(updatedSession.items).toHaveLength(0);
    });

    it('removes specific item when multiple items exist', async () => {
      const session = await SessionService.create({
        name: 'Test Session',
        framework: 'RICE',
      });

      let updatedSession = await SessionService.addItem(session.id, {
        title: 'Item 1',
      });
      updatedSession = await SessionService.addItem(updatedSession.id, {
        title: 'Item 2',
      });
      updatedSession = await SessionService.addItem(updatedSession.id, {
        title: 'Item 3',
      });

      const itemIdToRemove = updatedSession.items[1].id;

      const finalSession = await SessionService.removeItem(
        session.id,
        itemIdToRemove
      );

      expect(finalSession.items).toHaveLength(2);
      expect(finalSession.items.map((i) => i.title)).toEqual(
        expect.arrayContaining(['Item 1', 'Item 3'])
      );
    });

    it('throws NotFoundError for non-existent session', async () => {
      await expect(
        SessionService.removeItem('non-existent-id', 'item-123')
      ).rejects.toThrow('Session not found');
    });

    it('throws NotFoundError for non-existent item', async () => {
      const session = await SessionService.create({
        name: 'Test Session',
        framework: 'RICE',
      });

      await expect(
        SessionService.removeItem(session.id, 'non-existent-item')
      ).rejects.toThrow('Item not found');
    });
  });

  describe('getStats', () => {
    it('returns correct statistics', async () => {
      await SessionService.create({
        name: 'Session 1',
        framework: 'RICE',
        items: [{ title: 'Item 1' }, { title: 'Item 2' }],
      });
      await SessionService.create({
        name: 'Session 2',
        framework: 'MoSCoW',
        items: [{ title: 'Item 1' }],
      });
      await SessionService.create({
        name: 'Session 3',
        framework: 'RICE',
      });

      const stats = await SessionService.getStats();

      expect(stats.totalSessions).toBe(3);
      expect(stats.totalItems).toBe(3);
      expect(stats.avgItemsPerSession).toBe(1);
      expect(stats.sessionsByFramework.RICE).toBe(2);
      expect(stats.sessionsByFramework.MoSCoW).toBe(1);
    });

    it('returns zero stats when no sessions exist', async () => {
      const stats = await SessionService.getStats();

      expect(stats.totalSessions).toBe(0);
      expect(stats.totalItems).toBe(0);
      expect(stats.avgItemsPerSession).toBe(0);
      expect(stats.sessionsByFramework).toEqual({});
    });
  });

  describe('clearAll', () => {
    it('clears all sessions', async () => {
      await SessionService.create({
        name: 'Session 1',
        framework: 'RICE',
      });
      await SessionService.create({
        name: 'Session 2',
        framework: 'MoSCoW',
      });

      await SessionService.clearAll();

      const sessions = await SessionService.findAll();
      expect(sessions).toHaveLength(0);
    });
  });

  describe('subscribe and event emission', () => {
    it('emits session:created event', async () => {
      const listener = vi.fn();
      const unsubscribe = SessionService.subscribe(listener);

      await SessionService.create({
        name: 'Test Session',
        framework: 'RICE',
      });

      expect(listener).toHaveBeenCalledWith(
        'session:created',
        expect.objectContaining({
          event: 'session:created',
          data: expect.objectContaining({
            session: expect.objectContaining({
              name: 'Test Session',
            }),
          }),
        })
      );

      unsubscribe();
    });

    it('emits session:deleted event', async () => {
      const listener = vi.fn();
      const unsubscribe = SessionService.subscribe(listener);

      const session = await SessionService.create({
        name: 'To Delete',
        framework: 'RICE',
      });
      await SessionService.delete(session.id);

      expect(listener).toHaveBeenCalledWith(
        'session:deleted',
        expect.objectContaining({
          event: 'session:deleted',
          data: expect.objectContaining({
            sessionId: session.id,
          }),
        })
      );

      unsubscribe();
    });

    it('removes listener on unsubscribe', async () => {
      const listener = vi.fn();
      const unsubscribe = SessionService.subscribe(listener);
      unsubscribe();

      await SessionService.create({
        name: 'Test Session',
        framework: 'RICE',
      });

      expect(listener).not.toHaveBeenCalled();
    });
  });
});
