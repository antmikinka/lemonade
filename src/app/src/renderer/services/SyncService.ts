/**
 * Sync Service
 * Orchestrates scheduled synchronization between GitHub and local state
 *
 * Features:
 * - Scheduled background sync (configurable interval)
 * - Incremental sync support (only fetch changes)
 * - Conflict resolution
 * - Offline mode support
 * - Rate limit awareness
 */

import type {
  GitHubConfig,
  GitHubServiceEvent,
  SyncState,
} from '../types/github';

import type { WorkItem, StrategicInitiative, DashboardMetrics } from '../types/workItem';

import { gitHubService, GitHubService } from './github/GitHubService';
import { developmentKPICalculator } from '../metrics/DevelopmentKPIs';
import { roiCalculator } from '../metrics/ROICalculator';

// ============================================================================
// Sync Configuration
// ============================================================================

export interface SyncConfig {
  githubConfig: GitHubConfig;
  syncInterval: number; // milliseconds (default: 5 minutes)
  enableIncrementalSync: boolean;
  enableGitAnalysis: boolean;
  maxConcurrentRequests: number;
}

export interface SyncResult {
  success: boolean;
  workItems: WorkItem[];
  initiatives: StrategicInitiative[];
  syncType: 'full' | 'incremental' | 'git-analysis';
  timestamp: Date;
  error?: string;
}

// ============================================================================
// Sync Orchestrator Class
// ============================================================================

export class SyncOrchestrator {
  private githubService: GitHubService;
  private config: SyncConfig | null = null;
  private syncIntervalId: NodeJS.Timeout | null = null;
  private syncInProgress = false;

  private state: SyncState = {
    status: 'idle',
    pendingChanges: 0,
  };

  private listeners: Set<(state: SyncState) => void> = new Set();
  private cachedWorkItems: WorkItem[] = [];
  private cachedInitiatives: StrategicInitiative[] = [];

  constructor(githubService?: GitHubService) {
    this.githubService = githubService || gitHubService;
  }

  /**
   * Initialize sync orchestrator with configuration
   */
  async initialize(config: SyncConfig): Promise<void> {
    this.config = config;

    // Initialize GitHub service
    await this.githubService.initialize(config.githubConfig);

    // Perform initial full sync
    await this.performFullSync();

    // Set up periodic incremental sync
    this.startPeriodicSync(config.syncInterval);
  }

  /**
   * Start periodic sync
   */
  startPeriodicSync(intervalMs: number): void {
    if (this.syncIntervalId) {
      clearInterval(this.syncIntervalId);
    }

    this.syncIntervalId = setInterval(() => {
      this.performIncrementalSync();
    }, intervalMs);

    this.updateState({
      nextScheduledSync: new Date(Date.now() + intervalMs).toISOString(),
    });
  }

  /**
   * Stop periodic sync
   */
  stopPeriodicSync(): void {
    if (this.syncIntervalId) {
      clearInterval(this.syncIntervalId);
      this.syncIntervalId = null;
    }

    this.updateState({ nextScheduledSync: undefined });
  }

  /**
   * Perform full sync - fetches all data
   */
  async performFullSync(): Promise<SyncResult> {
    if (this.syncInProgress) {
      return {
        success: false,
        workItems: [],
        initiatives: [],
        syncType: 'full',
        timestamp: new Date(),
        error: 'Sync already in progress',
      };
    }

    this.syncInProgress = true;
    this.updateState({ status: 'syncing' });

    try {
      // Fetch all data from GitHub
      const [issuesResult, prsResult, commitsResult] = await Promise.all([
        this.githubService.fetchIssues(),
        this.githubService.fetchPullRequests(),
        this.githubService.fetchCommits(),
      ]);

      // Link issues to PRs
      const linkMap = await this.githubService.linkIssuesToPRs(
        issuesResult.data,
        prsResult.data
      );

      // Convert to work items
      const workItems: WorkItem[] = [];

      // Add issues
      for (const issue of issuesResult.data) {
        const workItem = this.githubService.normalizeToWorkItem(issue);
        const linkedPRs = linkMap.get(issue.id.toString());
        if (linkedPRs) {
          workItem.linkedItems = linkedPRs.map((prId) => ({
            type: 'relates_to' as const,
            targetId: `gh-pr-${prId}`,
            source: 'github' as const,
          }));
        }
        workItems.push(workItem);
      }

      // Add PRs
      for (const pr of prsResult.data) {
        workItems.push(this.githubService.normalizeToWorkItem(pr));
      }

      // Add significant commits
      for (const commit of commitsResult.data) {
        if (this.isSignificantCommit(commit)) {
          workItems.push(this.githubService.normalizeToWorkItem(commit));
        }
      }

      // Calculate metrics
      const metrics = this.calculateMetrics(workItems);

      // Update cache
      this.cachedWorkItems = workItems;

      this.updateState({
        status: 'completed',
        lastFullSync: new Date().toISOString(),
        lastIncrementalSync: new Date().toISOString(),
        pendingChanges: workItems.length,
      });

      return {
        success: true,
        workItems,
        initiatives: this.cachedInitiatives,
        syncType: 'full',
        timestamp: new Date(),
      };
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';

      this.updateState({
        status: 'error',
        error: errorMessage,
      });

      return {
        success: false,
        workItems: [],
        initiatives: [],
        syncType: 'full',
        timestamp: new Date(),
        error: errorMessage,
      };
    } finally {
      this.syncInProgress = false;
    }
  }

  /**
   * Perform incremental sync - only fetches changes
   */
  async performIncrementalSync(): Promise<SyncResult> {
    if (this.syncInProgress) {
      return {
        success: false,
        workItems: [],
        initiatives: [],
        syncType: 'incremental',
        timestamp: new Date(),
        error: 'Sync already in progress',
      };
    }

    if (!this.state.lastIncrementalSync) {
      return this.performFullSync();
    }

    this.syncInProgress = true;
    this.updateState({ status: 'syncing' });

    try {
      const since = new Date(this.state.lastIncrementalSync);

      // Fetch only changes since last sync
      const [issuesResult, prsResult, commitsResult] = await Promise.all([
        this.githubService.fetchIssues({ since }),
        this.githubService.fetchPullRequests({ since }),
        this.githubService.fetchCommits({ since }),
      ]);

      // Merge with cached data
      const updatedWorkItems = this.mergeIncrementalData(
        issuesResult.data,
        prsResult.data,
        commitsResult.data
      );

      // Update cache
      this.cachedWorkItems = updatedWorkItems;

      this.updateState({
        status: 'completed',
        lastIncrementalSync: new Date().toISOString(),
        pendingChanges: updatedWorkItems.length - this.cachedWorkItems.length,
      });

      return {
        success: true,
        workItems: updatedWorkItems,
        initiatives: this.cachedInitiatives,
        syncType: 'incremental',
        timestamp: new Date(),
      };
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';

      this.updateState({
        status: 'error',
        error: errorMessage,
      });

      return {
        success: false,
        workItems: [],
        initiatives: [],
        syncType: 'incremental',
        timestamp: new Date(),
        error: errorMessage,
      };
    } finally {
      this.syncInProgress = false;
    }
  }

  /**
   * Get current sync state
   */
  getState(): SyncState {
    return { ...this.state };
  }

  /**
   * Get cached work items
   */
  getWorkItems(): WorkItem[] {
    return [...this.cachedWorkItems];
  }

  /**
   * Get cached initiatives
   */
  getInitiatives(): StrategicInitiative[] {
    return [...this.cachedInitiatives];
  }

  /**
   * Add state change listener
   */
  addStateListener(listener: (state: SyncState) => void): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  /**
   * Dispose of resources
   */
  dispose(): void {
    this.stopPeriodicSync();
    this.githubService.dispose();
    this.listeners.clear();
  }

  // ==========================================================================
  // Private Methods
  // ==========================================================================

  private updateState(partial: Partial<SyncState>): void {
    this.state = { ...this.state, ...partial };

    // Notify listeners
    for (const listener of this.listeners) {
      try {
        listener({ ...this.state });
      } catch (error) {
        console.error('Error in sync state listener:', error);
      }
    }
  }

  private mergeIncrementalData(
    issues: any[],
    prs: any[],
    commits: any[]
  ): WorkItem[] {
    const workItems = [...this.cachedWorkItems];
    const itemIds = new Map(workItems.map((item) => [item.id, item]));

    // Update or add issues
    for (const issue of issues) {
      const workItem = this.githubService.normalizeToWorkItem(issue);
      itemIds.set(workItem.id, workItem);
    }

    // Update or add PRs
    for (const pr of prs) {
      const workItem = this.githubService.normalizeToWorkItem(pr);
      itemIds.set(workItem.id, workItem);
    }

    // Update or add significant commits
    for (const commit of commits) {
      if (this.isSignificantCommit(commit)) {
        const workItem = this.githubService.normalizeToWorkItem(commit);
        itemIds.set(workItem.id, workItem);
      }
    }

    return Array.from(itemIds.values());
  }

  private isSignificantCommit(commit: any): boolean {
    // Only create work items for commits with substantial changes
    const totalChanges = (commit.stats?.total || 0);
    const filesChanged = (commit.files?.length || 0);

    return totalChanges > 50 || filesChanged > 5;
  }

  private calculateMetrics(workItems: WorkItem[]): DashboardMetrics {
    return {
      development: developmentKPICalculator.calculate(workItems),
      roi: roiCalculator.calculatePortfolio(workItems),
      relevance: {
        overall: 0,
        dimensions: {
          technicalInnovation: 0,
          marketAlignment: 0,
          competitiveParity: 0,
          futureProofing: 0,
          ecosystemIntegration: 0,
        },
        trends: {
          emerging: [],
          declining: [],
        },
        benchmarks: {
          industryAverage: 65,
          leaderAverage: 85,
          ourScore: 0,
        },
      },
      team: {
        workloadDistribution: [],
        collaboration: {
          averageReviewersPerPR: 0,
          crossTeamDependencies: 0,
        },
        capacity: {
          totalCapacity: 0,
          allocatedCapacity: 0,
          availableCapacity: 0,
          utilizationRate: 0,
        },
      },
    };
  }
}

// ============================================================================
// Export singleton instance
// ============================================================================

export const syncOrchestrator = new SyncOrchestrator();

// ============================================================================
// Token Storage Helper (for Electron main process IPC)
// ============================================================================

interface ElectronAPI {
  secureStorage: {
    set: (key: string, value: string) => Promise<void>;
    get: (key: string) => Promise<string | null>;
    delete: (key: string) => Promise<void>;
  };
}

interface WindowWithElectron extends Window {
  electronAPI?: ElectronAPI;
}

/**
 * Secure token storage using Electron's safe storage API
 * This MUST be called from the main process or via IPC.
 * In production, localStorage fallback is disabled for security.
 */
export const tokenStorage = {
  async saveToken(token: string): Promise<void> {
    if (typeof window !== 'undefined' && 'electronAPI' in window) {
      const electronWindow = window as WindowWithElectron;
      await electronWindow.electronAPI!.secureStorage.set('github-token', token);
    } else {
      // Production: Require Electron secure storage - no fallback
      throw new Error(
        'Secure storage unavailable. GitHub token storage requires the Electron app. ' +
        'Please run this application in Electron environment.'
      );
    }
  },

  async getToken(): Promise<string | null> {
    if (typeof window !== 'undefined' && 'electronAPI' in window) {
      const electronWindow = window as WindowWithElectron;
      return await electronWindow.electronAPI!.secureStorage.get('github-token');
    } else {
      // Production: Require Electron secure storage - no fallback
      throw new Error(
        'Secure storage unavailable. GitHub token retrieval requires the Electron app. ' +
        'Please run this application in Electron environment.'
      );
    }
  },

  async deleteToken(): Promise<void> {
    if (typeof window !== 'undefined' && 'electronAPI' in window) {
      const electronWindow = window as WindowWithElectron;
      await electronWindow.electronAPI!.secureStorage.delete('github-token');
    } else {
      // Production: Require Electron secure storage - no fallback
      throw new Error(
        'Secure storage unavailable. GitHub token deletion requires the Electron app. ' +
        'Please run this application in Electron environment.'
      );
    }
  },
};
