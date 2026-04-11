/**
 * Team Dashboard Context
 * Provides state management for the AI-Automated Team Tracking Dashboard
 *
 * This context integrates with:
 * - GitHub Service for auto-syncing
 * - Sync Orchestrator for scheduled updates
 * - Metrics calculators for KPIs, ROI, and relevance scoring
 * - AI Insights Engine for recommendations
 */

import React, {
  createContext,
  useContext,
  useReducer,
  useEffect,
  useCallback,
  ReactNode,
  useRef,
  useMemo,
} from 'react';

import type {
  WorkItem,
  StrategicInitiative,
  DashboardState,
  DashboardAction,
  DashboardMetrics,
  AIInsight,
  SyncState,
  DashboardFilters,
  Priority,
  StrategicTag,
  ROICategory,
  WorkItemStatus,
  TeamMetrics,
} from '../types/workItem';

import { syncOrchestrator, SyncConfig, tokenStorage } from '../services/SyncService';
import { developmentKPICalculator } from '../metrics/DevelopmentKPIs';
import { roiCalculator } from '../metrics/ROICalculator';
import { industryRelevanceScorer } from '../metrics/IndustryRelevance';
import { aiInsightsEngine } from '../services/AIInsightsEngine';

// ============================================================================
// Initial State
// ============================================================================

const initialState: DashboardState = {
  workItems: [],
  initiatives: [],
  teamMembers: [],
  metrics: {
    development: {
      velocity: { current: 0, average: 0, trend: 'stable', history: [] },
      cycleTime: {
        average: 0,
        median: 0,
        percentile90: 0,
        trend: 'stable',
        byType: { issue: 0, pr: 0, commit: 0, initiative: 0, epic: 0 },
      },
      pullRequest: {
        averageTimeToFirstReview: 0,
        averageTimeToMerge: 0,
        mergeRate: 0,
        reworkRate: 0,
      },
      codeQuality: {
        codeChurn: 0,
        averagePRSize: 0,
        reviewDepth: 0,
        defectRate: 0,
      },
      throughput: {
        itemsCompleted: 0,
        itemsStarted: 0,
        workInProgress: 0,
        throughputTrend: [],
      },
    },
    roi: {
      totalInvestment: { storyPoints: 0, hours: 0, cost: 0 },
      totalImpact: {
        revenueImpact: 0,
        costReduction: 0,
        riskReduction: 0,
        strategicValue: 0,
        developerProductivity: 0,
        total: 0,
      },
      roi: { ratio: 0, percentage: 0, paybackPeriod: 0, npv: 0 },
      byCategory: {
        'revenue-impact': { investment: 0, impact: 0, roi: 0, itemCount: 0 },
        'cost-reduction': { investment: 0, impact: 0, roi: 0, itemCount: 0 },
        'risk-mitigation': { investment: 0, impact: 0, roi: 0, itemCount: 0 },
        'strategic-capability': { investment: 0, impact: 0, roi: 0, itemCount: 0 },
        'developer-productivity': { investment: 0, impact: 0, roi: 0, itemCount: 0 },
        'customer-experience': { investment: 0, impact: 0, roi: 0, itemCount: 0 },
      },
      byInitiative: [],
    },
    relevance: {
      overall: 0,
      dimensions: {
        technicalInnovation: 0,
        marketAlignment: 0,
        competitiveParity: 0,
        futureProofing: 0,
        ecosystemIntegration: 0,
      },
      trends: { emerging: [], declining: [] },
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
  },
  insights: [],
  filters: {},
  viewMode: 'strategic',
  selectedWorkItem: undefined,
  selectedInitiative: undefined,
  syncState: {
    status: 'idle',
    lastFullSync: undefined,
    lastIncrementalSync: undefined,
    lastGitAnalysis: undefined,
    nextScheduledSync: undefined,
    pendingChanges: 0,
  },
  humanOverrides: {
    priorityOverrides: new Map(),
    impactOverrides: new Map(),
    claimOverrides: new Map(),
  },
  isCreatingInitiative: false,
  isConfiguringSync: false,
  isViewingInsights: false,
};

// ============================================================================
// Reducer
// ============================================================================

const dashboardReducer = (
  state: DashboardState,
  action: DashboardAction
): DashboardState => {
  switch (action.type) {
    case 'SET_WORK_ITEMS':
      return { ...state, workItems: action.payload };

    case 'ADD_WORK_ITEM':
      return { ...state, workItems: [...state.workItems, action.payload] };

    case 'UPDATE_WORK_ITEM':
      return {
        ...state,
        workItems: state.workItems.map((item) =>
          item.id === action.payload.id ? action.payload : item
        ),
      };

    case 'DELETE_WORK_ITEM':
      return {
        ...state,
        workItems: state.workItems.filter((item) => item.id !== action.payload),
        selectedWorkItem:
          state.selectedWorkItem?.id === action.payload ? undefined : state.selectedWorkItem,
      };

    case 'SET_INITIATIVES':
      return { ...state, initiatives: action.payload };

    case 'ADD_INITIATIVE':
      return { ...state, initiatives: [...state.initiatives, action.payload] };

    case 'UPDATE_INITIATIVE':
      return {
        ...state,
        initiatives: state.initiatives.map((init) =>
          init.id === action.payload.id ? action.payload : init
        ),
      };

    case 'DELETE_INITIATIVE':
      return {
        ...state,
        initiatives: state.initiatives.filter((init) => init.id !== action.payload),
        selectedInitiative:
          state.selectedInitiative?.id === action.payload
            ? undefined
            : state.selectedInitiative,
      };

    case 'LINK_WORK_TO_INITIATIVE':
      return {
        ...state,
        workItems: state.workItems.map((item) =>
          item.id === action.payload.workItemId
            ? { ...item, parentInitiative: action.payload.initiativeId }
            : item
        ),
      };

    case 'UNLINK_WORK_FROM_INITIATIVE':
      return {
        ...state,
        workItems: state.workItems.map((item) =>
          item.id === action.payload.workItemId
            ? { ...item, parentInitiative: undefined }
            : item
        ),
      };

    case 'SET_TEAM_MEMBERS':
      return { ...state, teamMembers: action.payload };

    case 'ADD_TEAM_MEMBER':
      return { ...state, teamMembers: [...state.teamMembers, action.payload] };

    case 'SELECT_WORK_ITEM':
      return { ...state, selectedWorkItem: action.payload };

    case 'SELECT_INITIATIVE':
      return { ...state, selectedInitiative: action.payload };

    case 'SET_FILTERS':
      return { ...state, filters: { ...state.filters, ...action.payload } };

    case 'RESET_FILTERS':
      return { ...state, filters: {} };

    case 'SET_VIEW_MODE':
      return { ...state, viewMode: action.payload };

    case 'SET_SYNC_STATE':
      return { ...state, syncState: { ...state.syncState, ...action.payload } };

    case 'SYNC_STARTED':
      return {
        ...state,
        syncState: { ...state.syncState, status: 'syncing', error: undefined },
      };

    case 'SYNC_COMPLETED':
      return {
        ...state,
        workItems: action.payload.workItems,
        initiatives: action.payload.initiatives,
        syncState: {
          ...state.syncState,
          status: 'completed',
          lastIncrementalSync: new Date().toISOString(),
        },
      };

    case 'SYNC_FAILED':
      return {
        ...state,
        syncState: {
          ...state.syncState,
          status: 'error',
          error: action.payload,
        },
      };

    case 'CLAIM_WORK_ITEM':
      return {
        ...state,
        workItems: state.workItems.map((item) =>
          item.id === action.payload.workItemId
            ? { ...item, claimedBy: action.payload.userId }
            : item
        ),
        humanOverrides: {
          ...state.humanOverrides,
          claimOverrides: new Map(state.humanOverrides.claimOverrides).set(
            action.payload.workItemId,
            {
              claimedBy: action.payload.userId,
              claimedAt: new Date().toISOString(),
            }
          ),
        },
      };

    case 'SET_PRIORITY_OVERRIDE':
      return {
        ...state,
        workItems: state.workItems.map((item) =>
          item.id === action.payload.workItemId
            ? {
                ...item,
                priority: action.payload.priority,
                priorityHistory: [
                  ...(item.priorityHistory || []),
                  {
                    from: item.priority,
                    to: action.payload.priority,
                    reason: 'manual_override' as const,
                    changedBy: 'current-user',
                    changedAt: new Date().toISOString(),
                    note: action.payload.reason,
                  },
                ],
              }
            : item
        ),
        humanOverrides: {
          ...state.humanOverrides,
          priorityOverrides: new Map(state.humanOverrides.priorityOverrides).set(
            action.payload.workItemId,
            {
              value: action.payload.priority,
              reason: action.payload.reason,
            }
          ),
        },
      };

    case 'SET_IMPACT_OVERRIDE':
      return {
        ...state,
        workItems: state.workItems.map((item) =>
          item.id === action.payload.workItemId
            ? { ...item, impactScore: action.payload.impactScore }
            : item
        ),
        humanOverrides: {
          ...state.humanOverrides,
          impactOverrides: new Map(state.humanOverrides.impactOverrides).set(
            action.payload.workItemId,
            {
              value: action.payload.impactScore,
              reason: action.payload.reason,
            }
          ),
        },
      };

    case 'ADD_STRATEGIC_TAG':
      return {
        ...state,
        workItems: state.workItems.map((item) =>
          item.id === action.payload.workItemId
            ? {
                ...item,
                strategicTags: [...(item.strategicTags || []), action.payload.tag],
              }
            : item
        ),
      };

    case 'REMOVE_STRATEGIC_TAG':
      return {
        ...state,
        workItems: state.workItems.map((item) =>
          item.id === action.payload.workItemId
            ? {
                ...item,
                strategicTags: item.strategicTags?.filter(
                  (tag) => tag.id !== action.payload.tagId
                ),
              }
            : item
        ),
      };

    case 'SET_ROI_CATEGORY':
      return {
        ...state,
        workItems: state.workItems.map((item) =>
          item.id === action.payload.workItemId
            ? { ...item, roiCategory: action.payload.roiCategory }
            : item
        ),
      };

    case 'UPDATE_METRICS':
      return { ...state, metrics: action.payload };

    case 'SET_INSIGHTS':
      return { ...state, insights: action.payload };

    case 'ACKNOWLEDGE_INSIGHT':
      return {
        ...state,
        insights: state.insights.map((insight) =>
          insight.id === action.payload.insightId
            ? {
                ...insight,
                acknowledgedBy: [
                  ...(insight.acknowledgedBy || []),
                  action.payload.userId,
                ],
              }
            : insight
        ),
      };

    case 'DISMISS_INSIGHT':
      return {
        ...state,
        insights: state.insights.map((insight) =>
          insight.id === action.payload.insightId
            ? {
                ...insight,
                dismissedBy: [...(insight.dismissedBy || []), action.payload.userId],
              }
            : insight
        ),
      };

    case 'SET_CREATING_INITIATIVE':
      return { ...state, isCreatingInitiative: action.payload };

    case 'SET_CONFIGURING_SYNC':
      return { ...state, isConfiguringSync: action.payload };

    case 'SET_VIEWING_INSIGHTS':
      return { ...state, isViewingInsights: action.payload };

    case 'RESET_DASHBOARD':
      return initialState;

    default:
      return state;
  }
};

// ============================================================================
// Context Type
// ============================================================================

export interface TeamDashboardContextType {
  state: DashboardState;
  dispatch: React.Dispatch<DashboardAction>;

  // Work item operations
  addWorkItem: (item: Omit<WorkItem, 'id' | 'createdAt' | 'updatedAt'>) => void;
  updateWorkItem: (item: WorkItem) => void;
  deleteWorkItem: (id: string) => void;
  claimWorkItem: (id: string) => void;
  linkWorkToInitiative: (workItemId: string, initiativeId: string) => void;

  // Legacy aliases for backward compatibility
  addIssue: (item: Omit<WorkItem, 'id' | 'createdAt' | 'updatedAt'>) => void;
  updateIssue: (item: WorkItem) => void;
  deleteIssue: (id: string) => void;
  moveIssue: (id: string, status: WorkItemStatus) => void;

  // Initiative operations
  addInitiative: (
    initiative: Omit<StrategicInitiative, 'id' | 'createdAt' | 'updatedAt'>
  ) => void;
  updateInitiative: (initiative: StrategicInitiative) => void;
  deleteInitiative: (id: string) => void;

  // Filter operations
  setFilters: (filters: Partial<DashboardFilters>) => void;
  resetFilters: () => void;
  setViewMode: (mode: DashboardState['viewMode']) => void;

  // Selection operations
  selectWorkItem: (item?: WorkItem) => void;
  selectInitiative: (initiative?: StrategicInitiative) => void;

  // Legacy alias for backward compatibility
  selectIssue: (item?: WorkItem) => void;

  // Override operations
  setPriorityOverride: (id: string, priority: Priority, reason: string) => void;
  setImpactOverride: (id: string, impactScore: number, reason: string) => void;
  addStrategicTag: (id: string, tag: StrategicTag) => void;
  removeStrategicTag: (id: string, tagId: string) => void;
  setROICategory: (id: string, roiCategory: ROICategory) => void;

  // Insight operations
  acknowledgeInsight: (id: string) => void;
  dismissInsight: (id: string) => void;

  // Modal operations
  showCreateInitiative: () => void;
  hideCreateInitiative: () => void;
  showSyncConfig: () => void;
  hideSyncConfig: () => void;
  toggleInsights: () => void;

  // Sync operations
  initializeSync: (config: SyncConfig) => Promise<void>;
  triggerSync: () => Promise<void>;
  configureSync: (config: Partial<SyncConfig>) => void;
}

const TeamDashboardContext = createContext<TeamDashboardContextType | undefined>(
  undefined
);

// ============================================================================
// Provider Component
// ============================================================================

interface TeamDashboardProviderProps {
  children: ReactNode;
}

export const TeamDashboardProvider: React.FC<TeamDashboardProviderProps> = ({
  children,
}) => {
  const [state, dispatch] = useReducer(dashboardReducer, initialState);
  const syncConfigRef = useRef<SyncConfig | null>(null);
  const previousWorkItemsRef = useRef<WorkItem[]>([]);

  // Memoize metrics calculations - only recalculate when work items actually change
  const calculatedMetrics = useMemo(() => {
    if (state.workItems.length === 0) {
      return null;
    }

    // Check if work items have actually changed (deep comparison for significant changes)
    const hasSignificantChange =
      state.workItems.length !== previousWorkItemsRef.current.length ||
      state.workItems.some((item, index) => {
        const prev = previousWorkItemsRef.current[index];
        return !prev || item.updatedAt !== prev.updatedAt || item.status !== prev.status;
      });

    if (!hasSignificantChange && previousWorkItemsRef.current.length > 0) {
      // Return null to skip recalculation if no significant changes
      return null;
    }

    // Update reference for next comparison
    previousWorkItemsRef.current = [...state.workItems];

    // Calculate development KPIs
    const devKPIs = developmentKPICalculator.calculate(state.workItems);

    // Calculate ROI
    const roi = roiCalculator.calculatePortfolio(state.workItems);

    // Calculate industry relevance
    const relevance = industryRelevanceScorer.calculateRelevance(state.workItems);

    return { devKPIs, roi, relevance };
  }, [state.workItems]);

  // Generate AI insights when metrics change
  const insights = useMemo(() => {
    if (!calculatedMetrics || state.initiatives.length === 0) {
      return [];
    }

    const teamMetrics: TeamMetrics = {
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
    };

    return aiInsightsEngine.generateInsights(
      state.workItems,
      calculatedMetrics.devKPIs,
      calculatedMetrics.roi,
      calculatedMetrics.relevance,
      teamMetrics,
      state.initiatives
    );
  }, [calculatedMetrics, state.workItems, state.initiatives]);

  // Update state only when metrics actually change
  useEffect(() => {
    if (!calculatedMetrics) return;

    // Update metrics in state
    dispatch({
      type: 'UPDATE_METRICS',
      payload: {
        development: calculatedMetrics.devKPIs,
        roi: calculatedMetrics.roi,
        relevance: calculatedMetrics.relevance,
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
        }, // Would be calculated separately with team data
      },
    });
  }, [calculatedMetrics]);

  // Update insights only when they change
  useEffect(() => {
    if (insights.length > 0 || state.insights.length > 0) {
      dispatch({ type: 'SET_INSIGHTS', payload: insights });
    }
  }, [insights]);

  // Set up sync state listener
  useEffect(() => {
    const unsubscribe = syncOrchestrator.addStateListener((syncState) => {
      dispatch({ type: 'SET_SYNC_STATE', payload: syncState });

      if (syncState.status === 'completed') {
        const workItems = syncOrchestrator.getWorkItems();
        const initiatives = syncOrchestrator.getInitiatives();
        dispatch({ type: 'SYNC_COMPLETED', payload: { workItems, initiatives } });
      } else if (syncState.status === 'error') {
        dispatch({ type: 'SYNC_FAILED', payload: syncState.error || 'Sync failed' });
      }
    });

    return unsubscribe;
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      syncOrchestrator.dispose();
    };
  }, []);

  // Work item operations
  const addWorkItem = useCallback(
    (item: Omit<WorkItem, 'id' | 'createdAt' | 'updatedAt'>) => {
      const now = new Date().toISOString();
      const newWorkItem: WorkItem = {
        ...item,
        id: `manual_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
        createdAt: now,
        updatedAt: now,
      };
      dispatch({ type: 'ADD_WORK_ITEM', payload: newWorkItem });
    },
    []
  );

  const updateWorkItem = useCallback((item: WorkItem) => {
    dispatch({
      type: 'UPDATE_WORK_ITEM',
      payload: { ...item, updatedAt: new Date().toISOString() },
    });
  }, []);

  const deleteWorkItem = useCallback((id: string) => {
    dispatch({ type: 'DELETE_WORK_ITEM', payload: id });
  }, []);

  const claimWorkItem = useCallback((id: string) => {
    dispatch({ type: 'CLAIM_WORK_ITEM', payload: { workItemId: id, userId: 'current-user' } });
  }, []);

  const linkWorkToInitiative = useCallback(
    (workItemId: string, initiativeId: string) => {
      dispatch({
        type: 'LINK_WORK_TO_INITIATIVE',
        payload: { workItemId, initiativeId },
      });
    },
    []
  );

  // Initiative operations
  const addInitiative = useCallback(
    (initiative: Omit<StrategicInitiative, 'id' | 'createdAt' | 'updatedAt'>) => {
      const now = new Date().toISOString();
      const newInitiative: StrategicInitiative = {
        ...initiative,
        id: `init_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
        createdAt: now,
        updatedAt: now,
      };
      dispatch({ type: 'ADD_INITIATIVE', payload: newInitiative });
    },
    []
  );

  const updateInitiative = useCallback((initiative: StrategicInitiative) => {
    dispatch({
      type: 'UPDATE_INITIATIVE',
      payload: { ...initiative, updatedAt: new Date().toISOString() },
    });
  }, []);

  const deleteInitiative = useCallback((id: string) => {
    dispatch({ type: 'DELETE_INITIATIVE', payload: id });
  }, []);

  // Filter operations
  const setFilters = useCallback((filters: Partial<DashboardFilters>) => {
    dispatch({ type: 'SET_FILTERS', payload: filters });
  }, []);

  const resetFilters = useCallback(() => {
    dispatch({ type: 'RESET_FILTERS' });
  }, []);

  const setViewMode = useCallback((mode: DashboardState['viewMode']) => {
    dispatch({ type: 'SET_VIEW_MODE', payload: mode });
  }, []);

  // Selection operations
  const selectWorkItem = useCallback((item?: WorkItem) => {
    dispatch({ type: 'SELECT_WORK_ITEM', payload: item });
  }, []);

  const selectInitiative = useCallback((initiative?: StrategicInitiative) => {
    dispatch({ type: 'SELECT_INITIATIVE', payload: initiative });
  }, []);

  // Override operations
  const setPriorityOverride = useCallback(
    (id: string, priority: Priority, reason: string) => {
      dispatch({
        type: 'SET_PRIORITY_OVERRIDE',
        payload: { workItemId: id, priority, reason },
      });
    },
    []
  );

  const setImpactOverride = useCallback(
    (id: string, impactScore: number, reason: string) => {
      dispatch({
        type: 'SET_IMPACT_OVERRIDE',
        payload: { workItemId: id, impactScore, reason },
      });
    },
    []
  );

  const addStrategicTag = useCallback(
    (id: string, tag: StrategicTag) => {
      dispatch({ type: 'ADD_STRATEGIC_TAG', payload: { workItemId: id, tag } });
    },
    []
  );

  const removeStrategicTag = useCallback((id: string, tagId: string) => {
    dispatch({ type: 'REMOVE_STRATEGIC_TAG', payload: { workItemId: id, tagId } });
  }, []);

  const setROICategory = useCallback((id: string, roiCategory: ROICategory) => {
    dispatch({ type: 'SET_ROI_CATEGORY', payload: { workItemId: id, roiCategory } });
  }, []);

  // Insight operations
  const acknowledgeInsight = useCallback((id: string) => {
    dispatch({ type: 'ACKNOWLEDGE_INSIGHT', payload: { insightId: id, userId: 'current-user' } });
  }, []);

  const dismissInsight = useCallback((id: string) => {
    dispatch({ type: 'DISMISS_INSIGHT', payload: { insightId: id, userId: 'current-user' } });
  }, []);

  // Modal operations
  const showCreateInitiative = useCallback(() => {
    dispatch({ type: 'SET_CREATING_INITIATIVE', payload: true });
  }, []);

  const hideCreateInitiative = useCallback(() => {
    dispatch({ type: 'SET_CREATING_INITIATIVE', payload: false });
  }, []);

  const showSyncConfig = useCallback(() => {
    dispatch({ type: 'SET_CONFIGURING_SYNC', payload: true });
  }, []);

  const hideSyncConfig = useCallback(() => {
    dispatch({ type: 'SET_CONFIGURING_SYNC', payload: false });
  }, []);

  const toggleInsights = useCallback(() => {
    dispatch({ type: 'SET_VIEWING_INSIGHTS', payload: !state.isViewingInsights });
  }, [state.isViewingInsights]);

  // Sync operations
  const initializeSync = useCallback(async (config: SyncConfig) => {
    syncConfigRef.current = config;

    // Store token securely
    await tokenStorage.saveToken(config.githubConfig.token);

    // Initialize sync orchestrator
    await syncOrchestrator.initialize(config);
  }, []);

  const triggerSync = useCallback(async () => {
    dispatch({ type: 'SYNC_STARTED' });
    await syncOrchestrator.performFullSync();
  }, []);

  const configureSync = useCallback((config: Partial<SyncConfig>) => {
    if (syncConfigRef.current) {
      syncConfigRef.current = { ...syncConfigRef.current, ...config };
    }
  }, []);

  // Legacy alias methods for backward compatibility
  const addIssue = useCallback(
    (item: Omit<WorkItem, 'id' | 'createdAt' | 'updatedAt'>) => {
      addWorkItem(item);
    },
    [addWorkItem]
  );

  const updateIssue = useCallback(
    (item: WorkItem) => {
      updateWorkItem(item);
    },
    [updateWorkItem]
  );

  const deleteIssue = useCallback(
    (id: string) => {
      deleteWorkItem(id);
    },
    [deleteWorkItem]
  );

  const moveIssue = useCallback(
    (id: string, status: WorkItemStatus) => {
      // Dispatch action to move issue to new status
      dispatch({
        type: 'UPDATE_WORK_ITEM',
        payload: {
          id,
          status,
          updatedAt: new Date().toISOString(),
        } as WorkItem,
      });
    },
    []
  );

  const selectIssue = useCallback(
    (item?: WorkItem) => {
      selectWorkItem(item);
    },
    [selectWorkItem]
  );

  const contextValue: TeamDashboardContextType = {
    state,
    dispatch,
    addWorkItem,
    updateWorkItem,
    deleteWorkItem,
    claimWorkItem,
    linkWorkToInitiative,
    addIssue,
    updateIssue,
    deleteIssue,
    moveIssue,
    addInitiative,
    updateInitiative,
    deleteInitiative,
    setFilters,
    resetFilters,
    setViewMode,
    selectWorkItem,
    selectInitiative,
    selectIssue,
    setPriorityOverride,
    setImpactOverride,
    addStrategicTag,
    removeStrategicTag,
    setROICategory,
    acknowledgeInsight,
    dismissInsight,
    showCreateInitiative,
    hideCreateInitiative,
    showSyncConfig,
    hideSyncConfig,
    toggleInsights,
    initializeSync,
    triggerSync,
    configureSync,
  };

  return (
    <TeamDashboardContext.Provider value={contextValue}>
      {children}
    </TeamDashboardContext.Provider>
  );
};

// ============================================================================
// Custom Hook
// ============================================================================

export const useTeamDashboard = (): TeamDashboardContextType => {
  const context = useContext(TeamDashboardContext);
  if (context === undefined) {
    throw new Error(
      'useTeamDashboard must be used within a TeamDashboardProvider'
    );
  }
  return context;
};

export default TeamDashboardContext;
