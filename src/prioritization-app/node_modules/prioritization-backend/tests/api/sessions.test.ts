/**
 * Integration tests for Sessions API endpoints.
 *
 * Tests cover:
 * - Session creation (POST /api/v1/sessions)
 * - List all sessions (GET /api/v1/sessions)
 * - Get session by ID (GET /api/v1/sessions/:id)
 * - Update session (PUT /api/v1/sessions/:id)
 * - Delete session (DELETE /api/v1/sessions/:id)
 * - Add item to session (POST /api/v1/sessions/:id/items)
 * - Update item (PUT /api/v1/sessions/:id/items/:itemId)
 * - Remove item (DELETE /api/v1/sessions/:id/items/:itemId)
 *
 * @module api/tests/sessions.test
 */

import request from 'supertest';
import { describe, it, expect, beforeEach, beforeAll, afterAll, vi } from 'vitest';
import { app } from '../../src/index';
import { SessionService } from '../../src/services/SessionService';

// Mock console to reduce noise in tests
vi.spyOn(console, 'log').mockImplementation(() => {});
vi.spyOn(console, 'error').mockImplementation(() => {});

describe('Sessions API', () => {
  // Clean up sessions before each test
  beforeEach(async () => {
    await SessionService.clearAll();
  });

  // Clean up after all tests
  afterAll(async () => {
    await SessionService.clearAll();
  });

  describe('POST /api/v1/sessions', () => {
    it('creates a new session with required fields', async () => {
      const response = await request(app)
        .post('/api/v1/sessions')
        .send({
          name: 'Test Session',
          framework: 'RICE',
        });

      expect(response.status).toBe(201);
      expect(response.body).toMatchObject({
        name: 'Test Session',
        framework: 'RICE',
      });
      expect(response.body.id).toBeDefined();
      expect(response.body.createdAt).toBeDefined();
      expect(response.body.updatedAt).toBeDefined();
    });

    it('creates a session with all optional fields', async () => {
      const sessionData = {
        name: 'Complete Session',
        framework: 'RICE' as const,
        projectName: 'Test Project',
        projectDescription: 'A test project description',
        teamName: 'Team Alpha',
        industry: 'Technology',
        items: [
          {
            title: 'Initial Item',
            description: 'An initial item',
            category: 'Feature',
          },
        ],
      };

      const response = await request(app)
        .post('/api/v1/sessions')
        .send(sessionData);

      expect(response.status).toBe(201);
      expect(response.body.name).toBe('Complete Session');
      expect(response.body.projectName).toBe('Test Project');
      expect(response.body.teamName).toBe('Team Alpha');
      expect(response.body.industry).toBe('Technology');
      expect(response.body.items).toHaveLength(1);
      expect(response.body.items[0].title).toBe('Initial Item');
    });

    it('returns 400 when name is missing', async () => {
      const response = await request(app)
        .post('/api/v1/sessions')
        .send({ framework: 'RICE' });

      expect(response.status).toBe(400);
      expect(response.body.error).toBe('ValidationError');
      expect(response.body.message).toContain('name');
    });

    it('returns 400 when framework is missing', async () => {
      const response = await request(app)
        .post('/api/v1/sessions')
        .send({ name: 'Test Session' });

      expect(response.status).toBe(400);
      expect(response.body.error).toBe('ValidationError');
      expect(response.body.message).toContain('framework');
    });

    it('returns 400 when both name and framework are missing', async () => {
      const response = await request(app)
        .post('/api/v1/sessions')
        .send({});

      expect(response.status).toBe(400);
      expect(response.body.error).toBe('ValidationError');
      expect(response.body.message).toContain('name');
      expect(response.body.message).toContain('framework');
    });
  });

  describe('GET /api/v1/sessions', () => {
    it('returns empty array when no sessions exist', async () => {
      const response = await request(app).get('/api/v1/sessions');

      expect(response.status).toBe(200);
      expect(response.body).toEqual([]);
    });

    it('returns all sessions', async () => {
      // Create multiple sessions
      await SessionService.create({ name: 'Session 1', framework: 'RICE' });
      await SessionService.create({ name: 'Session 2', framework: 'MoSCoW' });
      await SessionService.create({ name: 'Session 3', framework: 'ICE' });

      const response = await request(app).get('/api/v1/sessions');

      expect(response.status).toBe(200);
      expect(response.body).toHaveLength(3);
      expect(response.body.map((s: { name: string }) => s.name)).toEqual(
        expect.arrayContaining(['Session 1', 'Session 2', 'Session 3'])
      );
    });
  });

  describe('GET /api/v1/sessions/:id', () => {
    it('returns session by ID', async () => {
      const createdSession = await SessionService.create({
        name: 'Test Session',
        framework: 'RICE',
      });

      const response = await request(app).get(`/api/v1/sessions/${createdSession.id}`);

      expect(response.status).toBe(200);
      expect(response.body.id).toBe(createdSession.id);
      expect(response.body.name).toBe('Test Session');
      expect(response.body.framework).toBe('RICE');
    });

    it('returns 404 for non-existent session', async () => {
      const response = await request(app).get('/api/v1/sessions/non-existent-id');

      expect(response.status).toBe(404);
      expect(response.body.error).toBe('NotFoundError');
      expect(response.body.message).toContain('not found');
    });
  });

  describe('PUT /api/v1/sessions/:id', () => {
    it('updates session name', async () => {
      const createdSession = await SessionService.create({
        name: 'Original Name',
        framework: 'RICE',
      });

      const response = await request(app)
        .put(`/api/v1/sessions/${createdSession.id}`)
        .send({ name: 'Updated Name' });

      expect(response.status).toBe(200);
      expect(response.body.name).toBe('Updated Name');
      expect(response.body.id).toBe(createdSession.id);
    });

    it('updates multiple fields', async () => {
      const createdSession = await SessionService.create({
        name: 'Test Session',
        framework: 'RICE',
      });

      const response = await request(app)
        .put(`/api/v1/sessions/${createdSession.id}`)
        .send({
          name: 'New Name',
          projectName: 'New Project',
          teamName: 'New Team',
        });

      expect(response.status).toBe(200);
      expect(response.body.name).toBe('New Name');
      expect(response.body.projectName).toBe('New Project');
      expect(response.body.teamName).toBe('New Team');
    });

    it('returns 404 for non-existent session', async () => {
      const response = await request(app)
        .put('/api/v1/sessions/non-existent-id')
        .send({ name: 'Updated' });

      expect(response.status).toBe(404);
      expect(response.body.error).toBe('NotFoundError');
    });

    it('handles empty update body', async () => {
      const createdSession = await SessionService.create({
        name: 'Test Session',
        framework: 'RICE',
      });

      const response = await request(app)
        .put(`/api/v1/sessions/${createdSession.id}`)
        .send({});

      expect(response.status).toBe(200);
      // Should return the session unchanged
      expect(response.body.name).toBe('Test Session');
    });
  });

  describe('DELETE /api/v1/sessions/:id', () => {
    it('deletes session successfully', async () => {
      const createdSession = await SessionService.create({
        name: 'To Delete',
        framework: 'RICE',
      });

      const response = await request(app).delete(
        `/api/v1/sessions/${createdSession.id}`
      );

      expect(response.status).toBe(204);
      // Verify session is deleted
      const deletedSession = await SessionService.findById(createdSession.id);
      expect(deletedSession).toBeNull();
    });

    it('returns 404 for non-existent session', async () => {
      const response = await request(app).delete(
        '/api/v1/sessions/non-existent-id'
      );

      expect(response.status).toBe(404);
      expect(response.body.error).toBe('NotFoundError');
    });
  });

  describe('POST /api/v1/sessions/:id/items', () => {
    it('adds item to session', async () => {
      const session = await SessionService.create({
        name: 'Test Session',
        framework: 'RICE',
      });

      const response = await request(app)
        .post(`/api/v1/sessions/${session.id}/items`)
        .send({
          title: 'New Item',
          description: 'Item description',
          category: 'Feature',
        });

      expect(response.status).toBe(201);
      expect(response.body.items).toHaveLength(1);
      expect(response.body.items[0].title).toBe('New Item');
      expect(response.body.items[0].description).toBe('Item description');
    });

    it('adds multiple items to session', async () => {
      const session = await SessionService.create({
        name: 'Test Session',
        framework: 'RICE',
      });

      await request(app)
        .post(`/api/v1/sessions/${session.id}/items`)
        .send({ title: 'Item 1' });

      await request(app)
        .post(`/api/v1/sessions/${session.id}/items`)
        .send({ title: 'Item 2' });

      const updatedSession = await SessionService.findById(session.id);
      expect(updatedSession?.items).toHaveLength(2);
    });

    it('returns 400 when title is missing', async () => {
      const session = await SessionService.create({
        name: 'Test Session',
        framework: 'RICE',
      });

      const response = await request(app)
        .post(`/api/v1/sessions/${session.id}/items`)
        .send({ description: 'No title' });

      expect(response.status).toBe(400);
      expect(response.body.error).toBe('ValidationError');
      expect(response.body.message).toContain('title');
    });

    it('returns 404 for non-existent session', async () => {
      const response = await request(app)
        .post('/api/v1/sessions/non-existent-id/items')
        .send({ title: 'Item' });

      expect(response.status).toBe(404);
      expect(response.body.error).toBe('NotFoundError');
    });
  });

  describe('PUT /api/v1/sessions/:id/items/:itemId', () => {
    it('updates item title', async () => {
      const session = await SessionService.create({
        name: 'Test Session',
        framework: 'RICE',
      });

      const itemSession = await SessionService.addItem(session.id, {
        title: 'Original Title',
        description: 'Description',
      });

      const itemId = itemSession.items[0].id;

      const response = await request(app)
        .put(`/api/v1/sessions/${session.id}/items/${itemId}`)
        .send({ title: 'Updated Title' });

      expect(response.status).toBe(200);
      expect(response.body.items[0].title).toBe('Updated Title');
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

      const response = await request(app)
        .put(`/api/v1/sessions/${session.id}/items/${itemId}`)
        .send({ description: 'Updated description' });

      expect(response.status).toBe(200);
      expect(response.body.items[0].description).toBe('Updated description');
    });

    it('returns 404 for non-existent session', async () => {
      const response = await request(app)
        .put('/api/v1/sessions/non-existent-id/items/item-123')
        .send({ title: 'Updated' });

      expect(response.status).toBe(404);
    });

    it('returns 404 for non-existent item', async () => {
      const session = await SessionService.create({
        name: 'Test Session',
        framework: 'RICE',
      });

      const response = await request(app)
        .put(`/api/v1/sessions/${session.id}/items/non-existent-item`)
        .send({ title: 'Updated' });

      expect(response.status).toBe(404);
      expect(response.body.error).toBe('NotFoundError');
      expect(response.body.message).toContain('Item not found');
    });
  });

  describe('DELETE /api/v1/sessions/:id/items/:itemId', () => {
    it('removes item from session', async () => {
      const session = await SessionService.create({
        name: 'Test Session',
        framework: 'RICE',
      });

      const itemSession = await SessionService.addItem(session.id, {
        title: 'To Remove',
      });

      const itemId = itemSession.items[0].id;

      const response = await request(app).delete(
        `/api/v1/sessions/${session.id}/items/${itemId}`
      );

      expect(response.status).toBe(200);
      expect(response.body.items).toHaveLength(0);
    });

    it('returns 404 for non-existent session', async () => {
      const response = await request(app).delete(
        '/api/v1/sessions/non-existent-id/items/item-123'
      );

      expect(response.status).toBe(404);
    });

    it('returns 404 for non-existent item', async () => {
      const session = await SessionService.create({
        name: 'Test Session',
        framework: 'RICE',
      });

      const response = await request(app).delete(
        `/api/v1/sessions/${session.id}/items/non-existent-item`
      );

      expect(response.status).toBe(404);
      expect(response.body.error).toBe('NotFoundError');
    });
  });
});
