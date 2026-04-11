# Eisenhower Matrix Framework

## Overview

The Eisenhower Matrix (also known as the Urgent-Important Matrix) is a time management and prioritization framework that helps categorize tasks based on their urgency and importance. Named after President Dwight D. Eisenhower, who was known for his exceptional time management skills, this matrix helps distinguish between what's truly important and what merely demands attention.

## Framework Details

### The Four Quadrants

```
┌─────────────────────────────────────────────────────────────────┐
│                  Eisenhower Matrix                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  URGENT │  ┌─────────────────┬─────────────────┐               │
│         │  │   DO            │   DECIDE        │               │
│         │  │   (Q1)          │   (Q2)          │               │
│         │  │   Crises        │   Planning      │               │
│         │  │   Deadlines     │   Prevention    │               │
│         │  ├─────────────────┼─────────────────┤               │
│         │  │   DELEGATE      │   DELETE        │               │
│         │  │   (Q3)          │   (Q4)          │               │
│         │  │   Interruptions │   Time Wasters  │               │
│         │  │   Some Meetings │   Busy Work     │               │
│  NOT    │  │   Some Emails   │   Trivia        │               │
│  URGENT │  └─────────────────┴─────────────────┘               │
│         │                                                       │
│         │  NOT IMPORTANT            IMPORTANT                   │
└─────────────────────────────────────────────────────────────────┘
```

### Quadrant Definitions

| Quadrant | Name | Characteristics | Strategy | Examples |
|----------|------|-----------------|----------|----------|
| **Q1: Do** | Urgent & Important | Crises, deadlines, emergencies | Handle immediately | Server outage, critical bug, due today |
| **Q2: Decide** | Not Urgent & Important | Planning, prevention, development | Schedule deliberately | Strategy, relationship building, learning |
| **Q3: Delegate** | Urgent & Not Important | Interruptions, some meetings | Minimize or delegate | Some emails, some calls, routine reports |
| **Q4: Delete** | Not Urgent & Not Important | Time wasters, trivia | Eliminate | Social media, busy work, gossip |

## Type Definitions

```typescript
type EisenhowerQuadrant = 'Q1_DO' | 'Q2_DECIDE' | 'Q3_DELEGATE' | 'Q4_DELETE';

interface EisenhowerResult {
  itemId: string;
  urgencyScore: number;      // 1-10
  importanceScore: number;   // 1-10
  quadrant: EisenhowerQuadrant;
  priorityRank: number;
  urgencyFactors: {
    deadlineProximity: number;     // 0 = no deadline, 10 = due now
    consequencesOfDelay: number;   // 1-10 severity
    externalPressure: number;      // 1-10 stakeholder pressure
  };
  importanceFactors: {
    alignmentWithGoals: number;    // 1-10 strategic alignment
    longTermImpact: number;        // 1-10 future value
    stakeholderValue: number;      // 1-10 importance to others
  };
  recommendedAction: string;
  actionTimeframe: string;
  rationale: string;
  autoFillUsed: boolean;
}

interface EisenhowerContext {
  currentDate: Date;
  workingHours: {
    start: number;  // 0-23
    end: number;    // 0-23
  };
  roleContext: 'individual' | 'manager' | 'executive';
  delegationOptions: string[];  // People/tasks can be delegated to
}

interface QuadrantDefinition {
  id: EisenhowerQuadrant;
  name: string;
  verb: string;
  description: string;
  color: string;
  icon: string;
  timeAllocation: string;  // Recommended time %
}

const QUADRANT_DEFINITIONS: Record<EisenhowerQuadrant, QuadrantDefinition> = {
  Q1_DO: {
    id: 'Q1_DO',
    name: 'Do',
    verb: 'DO NOW',
    description: 'Urgent and important - handle immediately',
    color: '#EF4444',
    icon: '🔴',
    timeAllocation: '20-30%'
  },
  Q2_DECIDE: {
    id: 'Q2_DECIDE',
    name: 'Decide',
    verb: 'SCHEDULE',
    description: 'Not urgent but important - plan deliberately',
    color: '#10B981',
    icon: '🟢',
    timeAllocation: '50-60%'
  },
  Q3_DELEGATE: {
    id: 'Q3_DELEGATE',
    name: 'Delegate',
    verb: 'DELEGATE',
    description: 'Urgent but not important - minimize or delegate',
    color: '#F59E0B',
    icon: '🟠',
    timeAllocation: '10-15%'
  },
  Q4_DELETE: {
    id: 'Q4_DELETE',
    name: 'Delete',
    verb: 'ELIMINATE',
    description: 'Not urgent and not important - eliminate',
    color: '#6B7280',
    icon: '⚫',
    timeAllocation: '<5%'
  }
};
```

## Scoring Guidelines

### Urgency Score (1-10)

Urgency measures how soon a task requires attention.

| Score | Description | Indicators |
|-------|-------------|------------|
| 9-10 | Critical urgency | Due now, immediate consequences |
| 7-8 | High urgency | Due within 24 hours |
| 5-6 | Moderate urgency | Due within a week |
| 3-4 | Low urgency | Due within a month |
| 1-2 | No urgency | No deadline, flexible timing |

```typescript
interface UrgencyFactors {
  deadlineProximity: number;     // 0-10 scale
  consequencesOfDelay: number;   // 1-10 scale
  externalPressure: number;      // 1-10 scale
}

function calculateUrgencyScore(factors: UrgencyFactors): number {
  const weights = {
    deadlineProximity: 0.5,
    consequencesOfDelay: 0.35,
    externalPressure: 0.15
  };

  const score =
    factors.deadlineProximity * weights.deadlineProximity +
    factors.consequencesOfDelay * weights.consequencesOfDelay +
    factors.externalPressure * weights.externalPressure;

  return Math.min(10, Math.max(1, Math.round(score * 10) / 10));
}

function calculateDeadlineProximity(deadline: Date | null, now: Date): number {
  if (!deadline) return 0;

  const hoursUntilDeadline = (deadline.getTime() - now.getTime()) / (1000 * 60 * 60);

  if (hoursUntilDeadline <= 0) return 10;       // Past due
  if (hoursUntilDeadline <= 2) return 9;        // Due in 2 hours
  if (hoursUntilDeadline <= 24) return 8;       // Due today
  if (hoursUntilDeadline <= 48) return 7;       // Due tomorrow
  if (hoursUntilDeadline <= 168) return 6;      // Due this week
  if (hoursUntilDeadline <= 336) return 5;      // Due in 2 weeks
  if (hoursUntilDeadline <= 720) return 3;      // Due this month
  return 1;                                      // No real deadline
}
```

### Importance Score (1-10)

Importance measures how much a task contributes to long-term goals and values.

| Score | Description | Indicators |
|-------|-------------|------------|
| 9-10 | Critical importance | Strategic, transformative impact |
| 7-8 | High importance | Significant goal alignment |
| 5-6 | Moderate importance | Some value contribution |
| 3-4 | Low importance | Minor contribution |
| 1-2 | Minimal importance | No real value |

```typescript
interface ImportanceFactors {
  alignmentWithGoals: number;    // 1-10 scale
  longTermImpact: number;        // 1-10 scale
  stakeholderValue: number;      // 1-10 scale
}

function calculateImportanceScore(factors: ImportanceFactors): number {
  const weights = {
    alignmentWithGoals: 0.4,
    longTermImpact: 0.4,
    stakeholderValue: 0.2
  };

  const score =
    factors.alignmentWithGoals * weights.alignmentWithGoals +
    factors.longTermImpact * weights.longTermImpact +
    factors.stakeholderValue * weights.stakeholderValue;

  return Math.min(10, Math.max(1, Math.round(score * 10) / 10));
}
```

## Auto-Fill Logic

### Complete Auto-Fill Engine

```typescript
class EisenhowerAutoFillEngine {
  constructor(private context: EisenhowerContext) {}

  async suggestClassification(item: BacklogItem): Promise<EisenhowerResult> {
    const urgencyScore = this.suggestUrgency(item);
    const importanceScore = this.suggestImportance(item);
    const quadrant = this.determineQuadrant(urgencyScore, importanceScore);

    return {
      itemId: item.id,
      urgencyScore,
      importanceScore,
      quadrant,
      priorityRank: 0,
      urgencyFactors: this.analyzeUrgencyFactors(item),
      importanceFactors: this.analyzeImportanceFactors(item),
      recommendedAction: QUADRANT_DEFINITIONS[quadrant].verb,
      actionTimeframe: this.getTimeframe(quadrant),
      rationale: this.generateRationale(item, quadrant, urgencyScore, importanceScore),
      autoFillUsed: true
    };
  }

  private suggestUrgency(item: BacklogItem): number {
    const text = `${item.title} ${item.description}`.toLowerCase();
    const now = this.context.currentDate;

    // Check for deadline in metadata
    let deadlineProximity = 0;
    if (item.metadata.deadline) {
      deadlineProximity = calculateDeadlineProximity(
        new Date(item.metadata.deadline),
        now
      );
    }

    // Analyze text for urgency indicators
    let urgencyIndicator = 3; // Default low-moderate

    if (/asap|urgent|emergency|critical|immediately|today/i.test(text)) {
      urgencyIndicator = 8;
    }
    if (/deadline|due|eod|cob|by end of/i.test(text)) {
      urgencyIndicator = Math.max(urgencyIndicator, 7);
    }
    if (/this week|this sprint|shortly/i.test(text)) {
      urgencyIndicator = Math.max(urgencyIndicator, 6);
    }
    if (/whenever|sometime|eventually|someday/i.test(text)) {
      urgencyIndicator = 2;
    }

    // Check for external pressure
    let externalPressure = 3;
    if (/ceo|executive|board|client|customer complaint/i.test(text)) {
      externalPressure = 8;
    }
    if (/manager|stakeholder|request/i.test(text)) {
      externalPressure = 5;
    }

    // Consequences of delay
    let consequences = 5;
    if (/blocking|stuck|cannot proceed/downtime/i.test(text)) {
      consequences = 9;
    }
    if (/affects revenue|customer impact/i.test(text)) {
      consequences = 7;
    }

    return calculateUrgencyScore({
      deadlineProximity,
      consequencesOfDelay: consequences,
      externalPressure
    });
  }

  private suggestImportance(item: BacklogItem): number {
    const text = `${item.title} ${item.description}`.toLowerCase();

    // Strategic alignment
    let alignment = 5;
    if (/strategic|roadmap|okrs|goals|vision/i.test(text)) {
      alignment = 8;
    }
    if (/revenue|growth|retention|acquisition/i.test(text)) {
      alignment = Math.max(alignment, 7);
    }
    if (/internal tool|maintenance|cleanup/i.test(text)) {
      alignment = Math.min(alignment, 5);
    }

    // Long-term impact
    let longTermImpact = 5;
    if (/foundation|infrastructure|platform|scalability/i.test(text)) {
      longTermImpact = 8;
    }
    if (/temporary|quick fix|band-aid/i.test(text)) {
      longTermImpact = 3;
    }
    if (/technical debt|refactor|sustainability/i.test(text)) {
      longTermImpact = 7;
    }

    // Stakeholder value
    let stakeholderValue = 5;
    if (/all users|every customer|enterprise/i.test(text)) {
      stakeholderValue = 8;
    }
    if (/power users|key accounts/i.test(text)) {
      stakeholderValue = 6;
    }
    if (/edge case|rarely used/niche/i.test(text)) {
      stakeholderValue = 3;
    }

    return calculateImportanceScore({
      alignmentWithGoals: alignment,
      longTermImpact,
      stakeholderValue
    });
  }

  private determineQuadrant(urgency: number, importance: number): EisenhowerQuadrant {
    const threshold = 5; // Midpoint threshold

    if (urgency >= threshold && importance >= threshold) {
      return 'Q1_DO';
    }
    if (urgency < threshold && importance >= threshold) {
      return 'Q2_DECIDE';
    }
    if (urgency >= threshold && importance < threshold) {
      return 'Q3_DELEGATE';
    }
    return 'Q4_DELETE';
  }

  private getTimeframe(quadrant: EisenhowerQuadrant): string {
    const timeframes = {
      Q1_DO: 'Immediately (today)',
      Q2_DECIDE: 'Schedule this week',
      Q3_DELEGATE: 'Delegate or handle within 48 hours',
      Q4_DELETE: 'Eliminate or do in spare time only'
    };
    return timeframes[quadrant];
  }

  private generateRationale(
    item: BacklogItem,
    quadrant: EisenhowerQuadrant,
    urgency: number,
    importance: number
  ): string {
    const quadName = QUADRANT_DEFINITIONS[quadrant].name;
    const verb = QUADRANT_DEFINITIONS[quadrant].verb;

    return `Classified as ${quadName}: ${verb}. Urgency: ${urgency}/10, Importance: ${importance}/10`;
  }
}
```

## Visualization

### Matrix View

```
┌─────────────────────────────────────────────────────────────────┐
│                  Eisenhower Matrix                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  URGENT │  ┌─────────────────┬─────────────────┐               │
│         │  │  DO NOW         │  SCHEDULE       │               │
│    10   │  │  ┌─────────┐    │  ┌───────────┐  │               │
│     8   │  │  │Critical │    │  │Strategic  │  │               │
│         │  │  │Bug Fix  │    │  │Planning   │  │               │
│     6   │  │  │Server   │    │  │Roadmap    │  │               │
│         │  │  │Outage   │    │  │Q2 Goals   │  │               │
│     4   │  │  │(9,8)    │    │  │(4,9)      │  │               │
│         │  ├─────────────────┼─────────────────┤               │
│         │  │  DELEGATE       │  ELIMINATE      │               │
│         │  │  ┌─────────┐    │  ┌───────────┐  │               │
│     2   │  │  │Routine  │    │  │Social     │  │               │
│         │  │  │Reports  │    │  │Scrolling  │  │               │
│         │  │  │Email    │    │  │Busy Work  │  │               │
│     0   │  │  │(7,3)    │    │  │(2,2)      │  │               │
│         │  └─────────────────┴─────────────────┘               │
│         │                                                       │
│         │  0    2    4    6    8    10                          │
│         │       NOT IMPORTANT       IMPORTANT                   │
└─────────────────────────────────────────────────────────────────┘
```

### Quadrant Distribution

```
┌─────────────────────────────────────────────────────────────────┐
│               Quadrant Time Distribution                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Q1: DO (Current: 35% | Target: 20-30%)                         │
│  ████████████████████████████████████  14 tasks                 │
│  ⚠️  Above recommended - risk of burnout                        │
│                                                                 │
│  Q2: DECIDE (Current: 40% | Target: 50-60%)                     │
│  ██████████████████████████████████████████  16 tasks           │
│  ✓  Good focus on important work                                 │
│                                                                 │
│  Q3: DELEGATE (Current: 20% | Target: 10-15%)                   │
│  ████████████████████  8 tasks                                  │
│  ⚠️  Consider delegating more                                   │
│                                                                 │
│  Q4: DELETE (Current: 5% | Target: <5%)                         │
│  █████  2 tasks                                                 │
│  ✓  Minimal time wasters                                         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Daily Planning View

```
┌─────────────────────────────────────────────────────────────────┐
│              Daily Eisenhower Planning                           │
│              Date: 2026-03-18                                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  🔴 DO NOW (Must complete today)                                │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ □ Critical Bug Fix - Payment Gateway (Q1)                 │  │
│  │ □ Server Outage Response (Q1)                             │  │
│  │ □ Client Presentation Prep (Q1)                           │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                 │
│  🟢 SCHEDULE (Important, plan time this week)                   │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ □ Q2 Strategic Planning - Tue 2-4pm                       │  │
│  │ □ Team Development Conversations - Wed                    │  │
│  │ □ Architecture Review - Thu                               │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                 │
│  🟠 DELEGATE (Urgent but not your best use of time)             │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ → Routine Status Report → Team Assistant                  │  │
│  │ → Meeting Invite Responses → EA                           │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                 │
│  ⚫ DELETE (Eliminate or do only in spare time)                  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ × Social Media Check - Limit to breaks                    │  │
│  │ × Inbox Zero - Not necessary daily                        │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## UI Component Specification

### EisenhowerMatrix Component

```typescript
interface EisenhowerMatrixProps {
  items: EisenhowerResult[];
  onSelectItem: (itemId: string) => void;
  onQuadrantFilter: (quadrant: EisenhowerQuadrant | null) => void;
  showCoordinates: boolean;
  showTimeAllocation: boolean;
}

interface EisenhowerCardProps {
  item: EisenhowerResult;
  onComplete: (itemId: string) => void;
  onDelegate: (itemId: string, assignee: string) => void;
  onSchedule: (itemId: string, scheduledTime: Date) => void;
  onDelete: (itemId: string) => void;
}

interface QuadrantSummaryProps {
  quadrant: EisenhowerQuadrant;
  itemCount: number;
  targetAllocation: string;
  actualAllocation: string;
  status: 'optimal' | 'warning' | 'critical';
}
```

### Visual Design

| Quadrant | Color | Icon | Alert State |
|----------|-------|------|-------------|
| Q1: Do | #EF4444 | 🔴 | Critical if >30% |
| Q2: Decide | #10B981 | 🟢 | Optimal at 50-60% |
| Q3: Delegate | #F59E0B | 🟠 | Warning if >15% |
| Q4: Delete | #6B7280 | ⚫ | Critical if >5% |

## Calculation Example

### Sample Item: Critical Bug Fix

```
Item: "Critical: Payment processing fails for 10% of transactions"

Urgency Analysis:
- Deadline Proximity: 10 ( happening now, affecting revenue)
- Consequences of Delay: 10 (direct revenue loss, customer impact)
- External Pressure: 9 (executive attention, customer complaints)

Urgency Score: (10 × 0.5) + (10 × 0.35) + (9 × 0.15) = 9.85

Importance Analysis:
- Alignment with Goals: 9 (revenue protection is strategic)
- Long-term Impact: 7 (prevents customer churn)
- Stakeholder Value: 9 (affects paying customers)

Importance Score: (9 × 0.4) + (7 × 0.4) + (9 × 0.2) = 8.2

Quadrant: Q1_DO (Urgency: 9.85, Importance: 8.2)
Recommended Action: DO NOW
Timeframe: Immediately (today)
```

### Comparison Table

| Item | Urgency | Importance | Quadrant | Action |
|------|---------|------------|----------|--------|
| Critical Bug | 9.85 | 8.2 | Q1 | DO NOW |
| Strategic Planning | 4 | 9 | Q2 | SCHEDULE |
| Status Report | 7 | 3 | Q3 | DELEGATE |
| Social Media | 2 | 2 | Q4 | DELETE |
| Client Deadline | 8 | 7 | Q1 | DO NOW |
| Team Development | 3 | 8 | Q2 | SCHEDULE |

## Best Practices

### Do's

1. **Focus on Q2** - This is where strategic work happens
2. **Minimize Q1 time** - Too much Q1 leads to burnout
3. **Delegate Q3 tasks** - Free up time for important work
4. **Eliminate Q4 activities** - They drain time without value
5. **Review daily** - Start each day with quadrant planning

### Don'ts

1. **Don't live in Q1** - Constant crisis mode is unsustainable
2. **Don't mistake urgency for importance** - Not all urgent matters are important
3. **Don't ignore Q2** - Neglecting Q2 creates future Q1 crises
4. **Don't feel guilty about Q4 elimination** - It's productive, not lazy

## Common Pitfalls

| Pitfall | Impact | Mitigation |
|---------|--------|------------|
| Urgency bias | Always in Q1, reactive mode | Schedule Q2 time blocks |
| Everything seems important | No Q4 elimination | Ask "What if I didn't do this?" |
| Not delegating | Overwhelmed with Q3 | Identify delegation candidates weekly |
| Q2 neglect | Future crises increase | Block Q2 time in calendar |

## Export Format

```json
{
  "framework": "Eisenhower Matrix",
  "generatedAt": "2026-03-18T10:00:00Z",
  "items": [
    {
      "id": "task-001",
      "title": "Critical Bug Fix",
      "scores": {
        "urgency": 9.85,
        "importance": 8.2
      },
      "quadrant": "Q1_DO",
      "recommendedAction": "DO NOW",
      "timeframe": "Immediately (today)"
    }
  ],
  "summary": {
    "q1Count": 4,
    "q2Count": 6,
    "q3Count": 3,
    "q4Count": 1,
    "timeDistribution": {
      "q1": 35,
      "q2": 40,
      "q3": 20,
      "q4": 5
    }
  }
}
```

---

**Related Frameworks:**
- [P0-P4 Priority Hierarchy](./p0p4.md) - Similar urgency-based approach
- [Value vs Effort Matrix](./value-effort.md) - Similar 2x2 structure
