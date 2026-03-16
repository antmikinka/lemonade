/**
 * Sync Status Component
 * Displays GitHub sync status and provides refresh functionality
 */

import React from 'react';
import type { SyncState } from '../../../types/workItem';

interface SyncStatusProps {
  syncState: SyncState;
  onRefresh: () => void;
}

export const SyncStatus: React.FC<SyncStatusProps> = ({ syncState, onRefresh }) => {
  const statusConfig = {
    idle: {
      label: 'Ready',
      color: '#6b7280',
      icon: (
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <circle cx="12" cy="12" r="10" />
          <path d="M12 6v6l4 2" />
        </svg>
      ),
    },
    syncing: {
      label: 'Syncing...',
      color: '#3b82f6',
      icon: (
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <path d="M23 4v6h-6M1 20v-6h6" />
          <path d="M3.51 9a9 9 0 0114.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0020.49 15" />
        </svg>
      ),
    },
    completed: {
      label: 'Synced',
      color: '#10b981',
      icon: (
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <path d="M22 11.08V12a10 10 0 11-5.93-9.14" />
          <polyline points="22 4 12 14.01 9 11.01" />
        </svg>
      ),
    },
    error: {
      label: 'Error',
      color: '#ef4444',
      icon: (
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <circle cx="12" cy="12" r="10" />
          <line x1="12" y1="8" x2="12" y2="12" />
          <line x1="12" y1="16" x2="12.01" y2="16" />
        </svg>
      ),
    },
    rate_limited: {
      label: 'Rate Limited',
      color: '#f59e0b',
      icon: (
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <circle cx="12" cy="12" r="10" />
          <line x1="12" y1="8" x2="12" y2="12" />
          <line x1="12" y1="16" x2="12.01" y2="16" />
        </svg>
      ),
    },
  };

  const config = statusConfig[syncState.status] || statusConfig.idle;

  const formatLastSync = (dateString: string | null) => {
    if (!dateString) return 'Never';
    const date = new Date(dateString);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / 60000);

    if (diffMins < 1) return 'Just now';
    if (diffMins < 60) return `${diffMins}m ago`;
    const diffHours = Math.floor(diffMins / 60);
    if (diffHours < 24) return `${diffHours}h ago`;
    const diffDays = Math.floor(diffHours / 24);
    return `${diffDays}d ago`;
  };

  const formatResetTime = (resetAt: string) => {
    const date = new Date(resetAt);
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  return (
    <div className="sync-status" role="status" aria-live="polite">
      <div
        className="sync-status-indicator"
        style={{ color: config.color }}
        title={syncState.error || config.label}
      >
        {config.icon}
        <span className="sync-status-label">{config.label}</span>
      </div>

      {/* Last sync time */}
      {syncState.lastIncrementalSync && syncState.status !== 'syncing' && (
        <span className="sync-last-time" title={`Last synced: ${syncState.lastIncrementalSync}`}>
          Last sync: {formatLastSync(syncState.lastIncrementalSync)}
        </span>
      )}

      {/* Next scheduled sync */}
      {syncState.nextScheduledSync && syncState.status !== 'syncing' && (
        <span className="sync-next-time">
          Next: {formatLastSync(syncState.nextScheduledSync)}
        </span>
      )}

      {/* Rate limit info */}
      {syncState.rateLimitInfo && (
        <div className="sync-rate-limit">
          <span className="rate-limit-remaining">
            {syncState.rateLimitInfo.remaining} requests left
          </span>
          <span className="rate-limit-reset">
            Resets at {formatResetTime(syncState.rateLimitInfo.resetAt)}
          </span>
        </div>
      )}

      {/* Error message */}
      {syncState.error && syncState.status === 'error' && (
        <div className="sync-error" role="alert">
          {syncState.error}
        </div>
      )}

      {/* Rate limited message */}
      {syncState.status === 'rate_limited' && syncState.rateLimitInfo && (
        <div className="sync-rate-limited" role="alert">
          GitHub API rate limit exceeded. Next reset at{' '}
          {formatResetTime(syncState.rateLimitInfo.resetAt)}.
        </div>
      )}
    </div>
  );
};
