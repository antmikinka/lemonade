/**
 * Integration tests for Prioritization API endpoints.
 *
 * Tests cover:
 * - Single item prioritization (POST /api/v1/prioritize)
 * - Bulk prioritization (POST /api/v1/prioritize/bulk)
 * - All framework calculations
 * - Auto-fill suggestions
 * - Error handling for invalid inputs
 *
 * @module api/tests/prioritization.test
 */

import request from 'supertest';
import { describe, it, expect, beforeEach, afterAll, vi } from 'vitest';
import { app } from '../../src/index';
import { SessionService } from '../../src/services/SessionService';

// Mock console to reduce noise in tests
vi.spyOn(console, 'log').mockImplementation(() => {});
vi.spyOn(console, 'error').mockImplementation(() => {});

describe('Prioritization API', () => {
  beforeEach(async () => {
    await SessionService.clearAll();
  });

  afterAll(async () => {
    await SessionService.clearAll();
  });

  describe('POST /api/v1/prioritize', () => {
    describe('RICE Framework', () => {
      it('calculates RICE score correctly', async () => {
        const response = await request(app)
          .post('/api/v1/prioritize')
          .send({
            framework: 'RICE',
            item: {
              title: 'Test Feature',
              frameworkData: {
                reach: 100,
                impact: 2,
                confidence: 80,
                effort: 4,
              },
            },
          });

        expect(response.status).toBe(200);
        // Expected: (100 * 2 * 0.8) / 4 = 40
        expect(response.body.priorityScore).toBe(40);
        expect(response.body.details.reach).toBe(100);
        expect(response.body.details.impact).toBe(2);
        expect(response.body.details.confidence).toBe(80);
        expect(response.body.details.effort).toBe(4);
      });

      it('handles confidence as decimal (0-1)', async () => {
        const response = await request(app)
          .post('/api/v1/prioritize')
          .send({
            framework: 'RICE',
            item: {
              title: 'Test Feature',
              frameworkData: {
                reach: 100,
                impact: 2,
                confidence: 0.75,
                effort: 2,
              },
            },
          });

        expect(response.status).toBe(200);
        // Expected: (100 * 2 * 0.75) / 2 = 75
        expect(response.body.priorityScore).toBe(75);
      });

      it('returns suggestions when framework data is missing', async () => {
        const response = await request(app)
          .post('/api/v1/prioritize')
          .send({
            framework: 'RICE',
            item: {
              title: 'Add export feature for all users',
              description: 'High-demand feature for enterprise customers',
            },
          });

        expect(response.status).toBe(200);
        expect(response.body.suggestions).toBeDefined();
        expect(response.body.suggestions?.suggestions).toHaveLength(4);
      });
    });

    describe('MoSCoW Framework', () => {
      it('calculates Must have category', async () => {
        const response = await request(app)
          .post('/api/v1/prioritize')
          .send({
            framework: 'MoSCoW',
            item: {
              title: 'Legal compliance feature',
              frameworkData: {
                businessValue: 'critical',
                legalRequirement: true,
              },
            },
          });

        expect(response.status).toBe(200);
        expect(response.body.priorityScore).toBe(100);
        expect(response.body.details.category).toBe('Must have');
      });

      it('calculates Should have category', async () => {
        const response = await request(app)
          .post('/api/v1/prioritize')
          .send({
            framework: 'MoSCoW',
            item: {
              title: 'Customer requested feature',
              frameworkData: {
                businessValue: 'high',
                customerRequest: true,
              },
            },
          });

        expect(response.status).toBe(200);
        expect(response.body.priorityScore).toBe(75);
        expect(response.body.details.category).toBe('Should have');
      });

      it('calculates Could have category', async () => {
        const response = await request(app)
          .post('/api/v1/prioritize')
          .send({
            framework: 'MoSCoW',
            item: {
              title: 'Nice to have feature',
              frameworkData: {
                businessValue: 'medium',
              },
            },
          });

        expect(response.status).toBe(200);
        expect(response.body.priorityScore).toBe(50);
        expect(response.body.details.category).toBe('Could have');
      });

      it('calculates Won\'t have category', async () => {
        const response = await request(app)
          .post('/api/v1/prioritize')
          .send({
            framework: 'MoSCoW',
            item: {
              title: 'Low priority item',
              frameworkData: {
                businessValue: 'low',
              },
            },
          });

        expect(response.status).toBe(200);
        expect(response.body.priorityScore).toBe(25);
        expect(response.body.details.category).toBe("Won't have");
      });
    });

    describe('ICE Framework', () => {
      it('calculates ICE score correctly', async () => {
        const response = await request(app)
          .post('/api/v1/prioritize')
          .send({
            framework: 'ICE',
            item: {
              title: 'Test Feature',
              frameworkData: {
                impact: 8,
                confidence: 70,
                ease: 6,
              },
            },
          });

        expect(response.status).toBe(200);
        // Expected: 8 * 0.7 * 6 = 33.6
        expect(response.body.priorityScore).toBe(33.6);
      });
    });

    describe('Eisenhower Framework', () => {
      it('categorizes as Do First (urgent and important)', async () => {
        const response = await request(app)
          .post('/api/v1/prioritize')
          .send({
            framework: 'Eisenhower',
            item: {
              title: 'Critical production issue',
              frameworkData: {
                urgent: true,
                important: true,
              },
            },
          });

        expect(response.status).toBe(200);
        expect(response.body.priorityScore).toBe(100);
        expect(response.body.details.quadrant).toBe('DoFirst');
      });

      it('categorizes as Schedule (not urgent, important)', async () => {
        const response = await request(app)
          .post('/api/v1/prioritize')
          .send({
            framework: 'Eisenhower',
            item: {
              title: 'Strategic initiative',
              frameworkData: {
                urgent: false,
                important: true,
              },
            },
          });

        expect(response.status).toBe(200);
        expect(response.body.priorityScore).toBe(75);
        expect(response.body.details.quadrant).toBe('Schedule');
      });

      it('categorizes as Delegate (urgent, not important)', async () => {
        const response = await request(app)
          .post('/api/v1/prioritize')
          .send({
            framework: 'Eisenhower',
            item: {
              title: 'Routine urgent task',
              frameworkData: {
                urgent: true,
                important: false,
              },
            },
          });

        expect(response.status).toBe(200);
        expect(response.body.priorityScore).toBe(50);
        expect(response.body.details.quadrant).toBe('Delegate');
      });
    });

    describe('P0P4 Framework', () => {
      it('calculates P0 priority for critical issues', async () => {
        const response = await request(app)
          .post('/api/v1/prioritize')
          .send({
            framework: 'P0P4',
            item: {
              title: 'Critical security vulnerability',
              frameworkData: {
                baseSeverity: 5,
                severityFactors: {
                  usersAffected: 'all',
                  coreFunctionalityImpact: 'critical',
                  securityRisk: 'critical',
                  reputationalRisk: 'high',
                  revenueImpact: 'high',
                },
              },
            },
          });

        expect(response.status).toBe(200);
        expect(response.body.priorityScore).toBe(100);
        expect(response.body.details.priority).toBe('P0');
      });

      it('calculates P2 priority for moderate issues', async () => {
        const response = await request(app)
          .post('/api/v1/prioritize')
          .send({
            framework: 'P0P4',
            item: {
              title: 'Moderate bug fix',
              frameworkData: {
                baseSeverity: 3,
                severityFactors: {
                  usersAffected: 'some',
                  coreFunctionalityImpact: 'medium',
                  securityRisk: 'low',
                  reputationalRisk: 'low',
                  revenueImpact: 'low',
                },
              },
            },
          });

        expect(response.status).toBe(200);
        expect(response.body.details.priority).toBe('P2');
      });
    });

    describe('WSJF Framework', () => {
      it('calculates WSJF score correctly', async () => {
        const response = await request(app)
          .post('/api/v1/prioritize')
          .send({
            framework: 'WSJF',
            item: {
              title: 'High value feature',
              frameworkData: {
                userBusinessValue: 20,
                timeCriticality: 15,
                riskReductionOpportunity: 10,
                jobSize: 5,
              },
            },
          });

        expect(response.status).toBe(200);
        // Expected: (20 + 15 + 10) / 5 = 9
        expect(response.body.priorityScore).toBe(9);
        expect(response.body.details.costOfDelay).toBe(45);
      });
    });

    describe('ValueEffort Framework', () => {
      it('categorizes as QuickWin (high value, low effort)', async () => {
        const response = await request(app)
          .post('/api/v1/prioritize')
          .send({
            framework: 'ValueEffort',
            item: {
              title: 'Quick optimization',
              frameworkData: {
                value: 8,
                effort: 2,
              },
            },
          });

        expect(response.status).toBe(200);
        expect(response.body.details.quadrant).toBe('QuickWin');
      });

      it('categorizes as MajorProject (high value, high effort)', async () => {
        const response = await request(app)
          .post('/api/v1/prioritize')
          .send({
            framework: 'ValueEffort',
            item: {
              title: 'Major initiative',
              frameworkData: {
                value: 8,
                effort: 8,
              },
            },
          });

        expect(response.status).toBe(200);
        expect(response.body.details.quadrant).toBe('MajorProject');
      });

      it('categorizes as Avoid (low value, high effort)', async () => {
        const response = await request(app)
          .post('/api/v1/prioritize')
          .send({
            framework: 'ValueEffort',
            item: {
              title: 'Low value high effort task',
              frameworkData: {
                value: 2,
                effort: 8,
              },
            },
          });

        expect(response.status).toBe(200);
        expect(response.body.details.quadrant).toBe('Avoid');
      });
    });

    describe('Kano Framework', () => {
      it('categorizes as OneDimensional', async () => {
        const response = await request(app)
          .post('/api/v1/prioritize')
          .send({
            framework: 'Kano',
            item: {
              title: 'Performance improvement',
              frameworkData: {
                functionalScore: 5,
                dysfunctionalScore: 1,
              },
            },
          });

        expect(response.status).toBe(200);
        expect(response.body.details.category).toBe('OneDimensional');
        expect(response.body.priorityScore).toBe(90);
      });

      it('categorizes as Attractive', async () => {
        const response = await request(app)
          .post('/api/v1/prioritize')
          .send({
            framework: 'Kano',
            item: {
              title: 'Delightful feature',
              frameworkData: {
                functionalScore: 5,
                dysfunctionalScore: 2,
              },
            },
          });

        expect(response.status).toBe(200);
        expect(response.body.details.category).toBe('Attractive');
        expect(response.body.priorityScore).toBe(100);
      });

      it('categorizes as MustBe', async () => {
        const response = await request(app)
          .post('/api/v1/prioritize')
          .send({
            framework: 'Kano',
            item: {
              title: 'Basic requirement',
              frameworkData: {
                functionalScore: 4,
                dysfunctionalScore: 4,
              },
            },
          });

        expect(response.status).toBe(200);
        expect(response.body.details.category).toBe('MustBe');
        expect(response.body.priorityScore).toBe(80);
      });
    });

    describe('Error Handling', () => {
      it('returns 400 when framework is missing', async () => {
        const response = await request(app)
          .post('/api/v1/prioritize')
          .send({
            item: { title: 'Test' },
          });

        expect(response.status).toBe(400);
        expect(response.body.error).toBe('ValidationError');
      });

      it('returns 400 when item is missing', async () => {
        const response = await request(app)
          .post('/api/v1/prioritize')
          .send({
            framework: 'RICE',
          });

        expect(response.status).toBe(400);
        expect(response.body.error).toBe('ValidationError');
      });

      it('returns 500 for unsupported framework', async () => {
        const response = await request(app)
          .post('/api/v1/prioritize')
          .send({
            framework: 'InvalidFramework',
            item: { title: 'Test' },
          });

        expect(response.status).toBe(500);
      });
    });
  });

  describe('POST /api/v1/prioritize/bulk', () => {
    it('prioritizes multiple items with RICE', async () => {
      const items = [
        {
          title: 'Feature A',
          frameworkData: { reach: 100, impact: 2, confidence: 80, effort: 4 },
        },
        {
          title: 'Feature B',
          frameworkData: { reach: 200, impact: 3, confidence: 90, effort: 3 },
        },
        {
          title: 'Feature C',
          frameworkData: { reach: 50, impact: 1, confidence: 70, effort: 2 },
        },
      ];

      const response = await request(app)
        .post('/api/v1/prioritize/bulk')
        .send({
          framework: 'RICE',
          items,
        });

      expect(response.status).toBe(200);
      expect(response.body).toHaveLength(3);

      // Check ranks are assigned correctly
      // Feature B: (200 * 3 * 0.9) / 3 = 180 (rank 1)
      // Feature A: (100 * 2 * 0.8) / 4 = 40 (rank 2)
      // Feature C: (50 * 1 * 0.7) / 2 = 17.5 (rank 3)
      expect(response.body[0].rank).toBe(2);
      expect(response.body[1].rank).toBe(1);
      expect(response.body[2].rank).toBe(3);
    });

    it('returns 400 when framework is missing', async () => {
      const response = await request(app)
        .post('/api/v1/prioritize/bulk')
        .send({ items: [] });

      expect(response.status).toBe(400);
      expect(response.body.error).toBe('ValidationError');
    });

    it('returns 400 when items array is missing', async () => {
      const response = await request(app)
        .post('/api/v1/prioritize/bulk')
        .send({ framework: 'RICE' });

      expect(response.status).toBe(400);
      expect(response.body.error).toBe('ValidationError');
    });

    it('handles empty items array', async () => {
      const response = await request(app)
        .post('/api/v1/prioritize/bulk')
        .send({
          framework: 'RICE',
          items: [],
        });

      expect(response.status).toBe(200);
      expect(response.body).toHaveLength(0);
    });
  });
});
