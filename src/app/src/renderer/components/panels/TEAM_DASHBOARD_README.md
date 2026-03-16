# Team Tracking Dashboard - Implementation Guide

## Overview

The Team Tracking Dashboard is a Kanban-style issue tracking system built into the Lemonade Electron application. It provides team-based task management with KPI tracking, sprint planning capabilities, and localStorage persistence.

## Files Created

### Components
```
src/app/src/renderer/
├── components/panels/
│   ├── TeamTrackingPanel.tsx          # Main panel component
│   └── TeamDashboard/
│       ├── IssueBoard.tsx             # Kanban board with drag-and-drop
│       ├── IssueCard.tsx              # Individual issue card display
│       ├── IssueDetailPanel.tsx       # Side panel for issue details/editing
│       ├── KPIDashboard.tsx           # KPI metrics dashboard
│       └── CreateIssueModal.tsx       # Modal for creating new issues
├── contexts/
│   └── TeamDashboardContext.tsx       # State management with useReducer
├── hooks/
│   └── useIssues.tsx                  # Issue management convenience hooks
├── types/
│   └── teamDashboard.ts               # TypeScript interfaces
├── utils/
│   └── issueStorage.ts                # localStorage persistence utilities
└── index.ts                           # Component exports
```

### Styles
Added to `src/app/styles.css` (~2000 lines of CSS for the dashboard)

## Features (MVP - Sprint 1-3)

1. **Issue Board (Kanban)**
   - Four columns: Backlog, In Progress, Review, Done
   - Drag-and-drop issue movement between columns
   - Visual priority indicators
   - Assignee avatars
   - Label support

2. **Issue Cards**
   - Title and description preview
   - Priority color coding
   - Assignee display
   - Story points
   - Due date with overdue indicator
   - Labels

3. **KPI Dashboard**
   - Quarterly metrics display
   - Issues completed
   - Average resolution time
   - Velocity (issues per sprint)
   - Completion rate percentage
   - Status distribution bar

4. **Issue Management**
   - Create new issues with full details
   - Edit issue properties
   - Delete issues
   - Filter by priority
   - Search issues
   - Board/List view toggle

5. **Persistence**
   - localStorage for all data
   - Automatic save on changes
   - Default team members pre-populated

6. **Accessibility**
   - Keyboard navigation (Tab, Enter, Escape)
   - Focus management
   - ARIA labels
   - WCAG AA color contrast

## TypeScript Interfaces

```typescript
interface Issue {
  id: string;
  title: string;
  description: string;
  status: 'backlog' | 'in_progress' | 'review' | 'done';
  priority: 'low' | 'medium' | 'high' | 'critical';
  assignee?: TeamMember;
  labels: string[];
  createdAt: string;
  updatedAt: string;
  resolvedAt?: string;
  dueDate?: string;
  storyPoints?: number;
}

interface TeamMember {
  id: string;
  name: string;
  avatar?: string;
  role?: string;
}

interface KPIMetrics {
  quarter: string;
  issuesCompleted: number;
  avgResolutionTime: number; // days
  velocity: number; // issues per sprint
  completionRate: number; // percentage
  totalIssues: number;
  issuesInBacklog: number;
  issuesInProgress: number;
  issuesInReview: number;
}
```

## Integration Instructions

### Option 1: Add as New Tab in ModelManager

1. Update `ModelManager.tsx`:
```typescript
// Add 'team' to the LeftPanelView type
export type LeftPanelView = 'models' | 'backends' | 'marketplace' | 'settings' | 'team';

// Import the TeamTrackingPanel
import { TeamTrackingPanel, TeamDashboardProvider } from './renderer';

// Add navigation item
const navItems = [
  // ... existing items
  { id: 'team', label: 'Team', icon: <TeamIcon /> },
];

// Add render case
{currentView === 'team' && (
  <TeamDashboardProvider>
    <TeamTrackingPanel isVisible={true} />
  </TeamDashboardProvider>
)}
```

### Option 2: Add as Top-Level Menu Item

1. Update `App.tsx` to include a new panel area
2. Add navigation in `TitleBar.tsx` or create new navigation component

### Option 3: Standalone Route

For future web version, add as a route:
```typescript
<Route path="/team" element={
  <TeamDashboardProvider>
    <TeamTrackingPanel isVisible={true} />
  </TeamDashboardProvider>
} />
```

## Usage Examples

### Using the Context Directly

```typescript
import { useTeamDashboard } from './contexts/TeamDashboardContext';

function MyComponent() {
  const { state, addIssue, moveIssue, selectIssue } = useTeamDashboard();

  return (
    <div>
      <p>Total issues: {state.issues.length}</p>
      <button onClick={() => addIssue({
        title: 'New Issue',
        description: 'Description',
        status: 'backlog',
        priority: 'medium',
        labels: ['feature'],
      })}>
        Add Issue
      </button>
    </div>
  );
}
```

### Using the useIssues Hook

```typescript
import { useIssues } from './hooks/useIssues';

function IssueManager() {
  const {
    issues,
    createIssue,
    transitionIssue,
    resolveIssue,
    getIssuesByStatus,
  } = useIssues();

  const inProgressIssues = getIssuesByStatus('in_progress');

  return (
    <div>
      {inProgressIssues.map(issue => (
        <div key={issue.id}>{issue.title}</div>
      ))}
    </div>
  );
}
```

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `Tab` | Navigate between interactive elements |
| `Enter` / `Space` | Activate focused button/card |
| `Escape` | Close modal/panel, cancel edit |
| `Ctrl+N` | New issue (future enhancement) |
| `Ctrl+F` | Focus search (future enhancement) |

## Data Persistence

Data is stored in localStorage under these keys:
- `team_dashboard_issues` - All issues
- `team_dashboard_members` - Team members
- `team_dashboard_settings` - View mode and preferences

## Future Enhancements (Sprint 4+)

1. **Backend Integration**
   - Sync with GitHub/GitLab issues
   - Jira integration
   - Real-time collaboration

2. **Advanced Features**
   - Sprint planning
   - Burndown charts
   - Custom workflows
   - Team workload view
   - Issue dependencies

3. **Notifications**
   - Assignment notifications
   - Due date reminders
   - Status change updates

## Code Style Compliance

- Pure CSS (no CSS-in-JS libraries)
- Dark theme matching existing app
- Functional components with hooks
- TypeScript strict mode
- WCAG AA accessibility
- No emojis (as per project conventions)

## Testing Checklist

- [ ] Drag-and-drop works between all columns
- [ ] Issue creation validates required fields
- [ ] KPI calculations are accurate
- [ ] localStorage persists across reloads
- [ ] Keyboard navigation works
- [ ] Screen reader announcements
- [ ] Responsive layout at all breakpoints
