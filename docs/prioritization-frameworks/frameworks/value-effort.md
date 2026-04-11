# Value vs Effort Matrix Framework

## Overview

The Value vs Effort Matrix (also known as the Impact vs Effort Matrix or Priority Matrix) is a simple yet powerful visual prioritization tool that plots initiatives on a 2x2 grid based on their expected value and required effort. It helps teams quickly identify quick wins, major projects, fill-ins, and time sinks.

## Framework Details

### The Four Quadrants

```
┌─────────────────────────────────────────────────────────────────┐
│                  Value vs Effort Matrix                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  HIGH │  ┌─────────────┬─────────────┐                          │
│  VALUE│  │   QUICK     │   MAJOR     │                          │
│       │  │   WINS      │   PROJECTS  │                          │
│       │  │   (High/Low)│   (High/High)                          │
│       │  │   DO FIRST  │   PLAN      │                          │
│       │  ├─────────────┼─────────────┤                          │
│       │  │   FILL-INS  │   TIME      │                          │
│       │  │   (Low/Low) │   SINKS     │                          │
│       │  │   DO LATER  │   AVOID     │                          │
│       │  │   (Low/High)│             │                          │
│  LOW  │  └─────────────┴─────────────┘                          │
│       │                                                         │
│       │  LOW          EFFORT           HIGH                     │
└─────────────────────────────────────────────────────────────────┘
```

### Quadrant Definitions

| Quadrant | Value | Effort | Strategy | Examples |
|----------|-------|--------|----------|----------|
| **Quick Wins** | High | Low | Do immediately | Bug fixes, copy changes, small optimizations |
| **Major Projects** | High | High | Plan carefully | New features, platform migrations |
| **Fill-Ins** | Low | Low | Do when capacity allows | Minor improvements, polish |
| **Time Sinks** | Low | High | Avoid or reframe | Legacy support, edge cases |

## Type Definitions

```typescript
type Quadrant = 'QUICK_WINS' | 'MAJOR_PROJECTS' | 'FILL_INS' | 'TIME_SINKS';

interface ValueEffortResult {
  itemId: string;
  valueScore: number;      // 1-10 or 1-100
  effortScore: number;     // 1-10 or 1-100
  quadrant: Quadrant;
  priorityRank: number;
  valueComponents: {
    businessValue: number;
    userValue: number;
    strategicValue: number;
  };
  effortComponents: {
    developmentEffort: number;
    complexity: number;
    dependencyCost: number;
  };
  rationale: string;
  recommendedAction: string;
  autoFillUsed: boolean;
}

interface ValueEffortContext {
  scoringScale: '1-10' | '1-100';
  quadrantThresholds: {
    valueThreshold: number;  // Above = High Value
    effortThreshold: number; // Above = High Effort
  };
  teamCapacity: number;
  strategicGoals: string[];
}

interface QuadrantDefinition {
  id: Quadrant;
  label: string;
  description: string;
  strategy: string;
  color: string;
  icon: string;
  actionVerb: string;
}

const QUADRANT_DEFINITIONS: Record<Quadrant, QuadrantDefinition> = {
  QUICK_WINS: {
    id: 'QUICK_WINS',
    label: 'Quick Wins',
    description: 'High value, low effort - maximum ROI',
    strategy: 'Execute immediately, these build momentum',
    color: '#10B981',
    icon: '⚡',
    actionVerb: 'DO FIRST'
  },
  MAJOR_PROJECTS: {
    id: 'MAJOR_PROJECTS',
    label: 'Major Projects',
    description: 'High value, high effort - strategic investments',
    strategy: 'Plan carefully, allocate dedicated resources',
    color: '#3B82F6',
    icon: '🎯',
    actionVerb: 'PLAN'
  },
  FILL_INS: {
    id: 'FILL_INS',
    label: 'Fill-Ins',
    description: 'Low value, low effort - nice to haves',
    strategy: 'Do when capacity allows, batch together',
    color: '#F59E0B',
    icon: '📋',
    actionVerb: 'SCHEDULE'
  },
  TIME_SINKS: {
    id: 'TIME_SINKS',
    label: 'Time Sinks',
    description: 'Low value, high effort - poor ROI',
    strategy: 'Avoid, automate, or reframe the approach',
    color: '#EF4444',
    icon: '⚠️',
    actionVerb: 'AVOID'
  }
};
```

## Scoring Guidelines

### Value Scoring (1-10 scale)

| Score | Description | Criteria |
|-------|-------------|----------|
| 9-10 | Exceptional value | Transformative business impact |
| 7-8 | High value | Significant metric improvement |
| 5-6 | Moderate value | Noticeable improvement |
| 3-4 | Low value | Minor improvement |
| 1-2 | Minimal value | Barely measurable impact |

```typescript
interface ValueCriteria {
  businessValue: {
    revenue: number;      // Expected revenue impact
    costSavings: number;  // Expected cost reduction
    retention: number;    // Impact on retention
  };
  userValue: {
    affectedUsers: number;     // Number of users impacted
    satisfactionImprovement: number; // NPS/CSAT impact
    painPointSeverity: number;   // Problem being solved
  };
  strategicValue: {
    alignment: number;      // Alignment with strategic goals
    competitiveAdvantage: number; // Differentiation potential
    enablingCapability: number;   // Enables other initiatives
  };
}

function calculateValueScore(criteria: ValueCriteria): number {
  // Weight the components
  const weights = {
    businessValue: 0.4,
    userValue: 0.35,
    strategicValue: 0.25
  };

  // Calculate sub-scores (each 1-10)
  const businessScore = Math.min(10,
    (criteria.businessValue.revenue / 100000) * 5 +
    (criteria.businessValue.costSavings / 50000) * 3 +
    (criteria.businessValue.retention * 10) * 2
  );

  const userScore = Math.min(10,
    (criteria.userValue.affectedUsers / 10000) * 4 +
    (criteria.userValue.satisfactionImprovement * 2) +
    (criteria.userValue.painPointSeverity / 10) * 4
  );

  const strategicScore = Math.min(10,
    (criteria.strategicValue.alignment / 10) * 4 +
    (criteria.strategicValue.competitiveAdvantage / 10) * 3 +
    (criteria.strategicValue.enablingCapability / 5) * 3
  );

  // Weighted average
  const totalScore =
    businessScore * weights.businessValue +
    userScore * weights.userValue +
    strategicScore * weights.strategicValue;

  return Math.round(totalScore * 10) / 10;
}
```

### Effort Scoring (1-10 scale)

Note: Higher effort score = MORE effort (harder to do)

| Score | Description | Criteria |
|-------|-------------|----------|
| 9-10 | Extreme effort | Months of work, large team |
| 7-8 | High effort | Weeks of work, multiple people |
| 5-6 | Moderate effort | Several days, some coordination |
| 3-4 | Low effort | 1-2 days, minimal coordination |
| 1-2 | Minimal effort | Hours, single person |

```typescript
interface EffortCriteria {
  developmentEffort: {
    estimatedHours: number;
    teamSize: number;
    skillRequirements: string[];
  };
  complexity: {
    technicalComplexity: number;    // 1-10
    integrationPoints: number;      // Number of systems
    dataMigration: boolean;
    legacyCodeInvolved: boolean;
  };
  dependencyCost: {
    crossTeamDependencies: number;
    externalDependencies: number;
    approvalRequirements: number;
    riskFactors: number;            // 1-10
  };
}

function calculateEffortScore(criteria: EffortCriteria): number {
  // Base effort from hours
  const hours = criteria.developmentEffort.estimatedHours;
  let baseEffort = Math.min(10, Math.log2(hours + 1) * 1.5);

  // Team size multiplier
  const teamMultiplier = 1 + (criteria.developmentEffort.teamSize - 1) * 0.1;
  baseEffort *= teamMultiplier;

  // Complexity additions
  if (criteria.complexity.technicalComplexity >= 7) baseEffort += 2;
  if (criteria.complexity.integrationPoints > 3) baseEffort += 1;
  if (criteria.complexity.dataMigration) baseEffort += 1.5;
  if (criteria.complexity.legacyCodeInvolved) baseEffort += 1;

  // Dependency additions
  baseEffort += criteria.dependencyCost.crossTeamDependencies * 0.5;
  baseEffort += criteria.dependencyCost.externalDependencies * 0.3;
  baseEffort += criteria.dependencyCost.approvalRequirements * 0.2;
  baseEffort += criteria.dependencyCost.riskFactors * 0.1;

  return Math.min(10, Math.max(1, Math.round(baseEffort * 10) / 10));
}
```

## Auto-Fill Logic

### Complete Auto-Fill Engine

```typescript
class ValueEffortAutoFill {
  constructor(private context: ValueEffortContext) {}

  async suggestScores(item: BacklogItem): Promise<ValueEffortResult> {
    const valueScore = this.suggestValue(item);
    const effortScore = this.suggestEffort(item);
    const quadrant = this.determineQuadrant(valueScore, effortScore);

    return {
      itemId: item.id,
      valueScore,
      effortScore,
      quadrant,
      priorityRank: 0, // Calculated after all items
      valueComponents: this.analyzeValueComponents(item),
      effortComponents: this.analyzeEffortComponents(item),
      rationale: this.generateRationale(item, valueScore, effortScore, quadrant),
      recommendedAction: QUADRANT_DEFINITIONS[quadrant].actionVerb,
      autoFillUsed: true
    };
  }

  private suggestValue(item: BacklogItem): number {
    const text = `${item.title} ${item.description}`.toLowerCase();
    let value = 5; // Default moderate

    // Revenue indicators
    if (/revenue|income|profit|monetize|paid/i.test(text)) value += 2;
    if (/customer acquisition|new market/i.test(text)) value += 1.5;

    // User impact
    if (/all users|every customer|majority/i.test(text)) value += 1.5;
    if (/power users|enterprise|vip/i.test(text)) value += 1;

    // Strategic alignment
    if (/strategic|roadmap|vision|okrs/i.test(text)) value += 1;
    if (/competitive|differentiator/i.test(text)) value += 1.5;

    // Pain point severity
    if (/critical|blocking|urgent|pain point/i.test(text)) value += 1;

    // Penalty for limited scope
    if (/internal|admin only|edge case|rare/i.test(text)) value -= 1;

    return Math.min(10, Math.max(1, Math.round(value * 10) / 10));
  }

  private suggestEffort(item: BacklogItem): number {
    const text = `${item.title} ${item.description}`.toLowerCase();
    let effort = 5; // Default moderate

    // Use estimated hours if available
    if (item.metadata.estimatedHours) {
      const hours = item.metadata.estimatedHours;
      effort = Math.min(10, Math.log2(hours + 1) * 1.5);
    }

    // Complexity indicators
    if (/simple|quick|minor|trivial/i.test(text)) effort -= 2;
    if (/complex|challenging|significant/i.test(text)) effort += 1;
    if (/integration|migration|refactor|rewrite/i.test(text)) effort += 2;

    // Scope indicators
    if (/cross-team|multiple systems|platform/i.test(text)) effort += 1.5;
    if (/single endpoint|isolated|standalone/i.test(text)) effort -= 1;

    // Add for dependencies
    const depCount = item.metadata.dependencies?.length || 0;
    effort += depCount * 0.5;

    return Math.min(10, Math.max(1, Math.round(effort * 10) / 10));
  }

  private determineQuadrant(value: number, effort: number): Quadrant {
    const threshold = this.context.quadrantThresholds.valueThreshold || 5;

    if (value >= threshold && effort < threshold) {
      return 'QUICK_WINS';
    }
    if (value >= threshold && effort >= threshold) {
      return 'MAJOR_PROJECTS';
    }
    if (value < threshold && effort < threshold) {
      return 'FILL_INS';
    }
    return 'TIME_SINKS';
  }

  private generateRationale(
    item: BacklogItem,
    value: number,
    effort: number,
    quadrant: Quadrant
  ): string {
    const valueDesc = value >= 7 ? 'high value' :
                      value >= 5 ? 'moderate value' : 'low value';

    const effortDesc = effort >= 7 ? 'significant effort' :
                       effort >= 5 ? 'moderate effort' : 'low effort';

    const action = QUADRANT_DEFINITIONS[quadrant].actionVerb;

    return `Auto-scored: ${valueDesc}, ${effortDesc} - ${action}`;
  }
}
```

## Visualization

### Matrix View with Items

```
┌─────────────────────────────────────────────────────────────────┐
│                  Value vs Effort Matrix                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  10 │  ┌─────────────────┬─────────────────┐                   │
│     │  │  ★ Login Opt.   │  ★ Mobile App   │                   │
│  8  │  │  ★ Bug Fixes    │  ★ Platform v2  │                   │
│     │  │  QUICK WINS     │  MAJOR PROJECTS │                   │
│  6  │  │  (4 items)      │  (3 items)      │                   │
│     │  ├─────────────────┼─────────────────┤                   │
│  4  │  │  ★ Copy Updates │  ★ Legacy Fix   │                   │
│     │  │  ★ Docs Refresh │  ★ Old Browser  │                   │
│  2  │  │  FILL-INS       │  TIME SINKS     │                   │
│     │  │  (5 items)      │  (2 items)      │                   │
│  0  │  └─────────────────┴─────────────────┘                   │
│     │                                                         │
│     │  0    2    4    6    8    10                            │
│     │           EFFORT                                        │
└─────────────────────────────────────────────────────────────────┘
```

### Quadrant Summary

```
┌─────────────────────────────────────────────────────────────────┐
│                  Quadrant Summary                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  QUICK WINS (Do First)              MAJOR PROJECTS (Plan)       │
│  ┌─────────────────────────────┐    ┌─────────────────────────┐ │
│  │ Item           V    E   ROI  │    │ Item        V    E   ROI│ │
│  │ Login Opt.     9    3   3.0  │    │ Mobile App  8    8   1.0│ │
│  │ Bug Fixes      8    2   4.0  │    │ Platform v2 9    7   1.3│ │
│  │ Search Fix     7    4   1.8  │    │ Analytics   7    6   1.2│ │
│  └─────────────────────────────┘    └─────────────────────────┘ │
│                                                                 │
│  FILL-INS (Schedule)                TIME SINKS (Avoid)          │
│  ┌─────────────────────────────┐    ┌─────────────────────────┐ │
│  │ Item           V    E   ROI  │    │ Item        V    E   ROI│ │
│  │ Copy Updates   4    2   2.0  │    │ Legacy Fix  3    7   0.4│ │
│  │ Docs Refresh   3    3   1.0  │    │ Old Browser 2    8   0.3│ │
│  └─────────────────────────────┘    └─────────────────────────┘ │
│                                                                 │
│  ROI = Value / Effort (higher = better return)                  │
└─────────────────────────────────────────────────────────────────┘
```

### ROI Ranking Chart

```
┌─────────────────────────────────────────────────────────────────┐
│                  ROI Ranking (Value/Effort)                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ROI                                                           │
│  5.0 │    ████                                                  │
│      │    ████  Bug Fixes (4.0)                                 │
│  4.0 │    ████                                                  │
│      │    ████                                                  │
│  3.0 │    ████    ████                                          │
│      │    ████    ████  Login Opt. (3.0)                        │
│  2.0 │    ████    ████    ████    ████                          │
│      │    ████    ████    ████    ████  Copy (2.0)              │
│  1.0 │    ████    ████    ████    ████  ████  ████  ████        │
│  0.0 │────████────████────████────████──████──████──████────    │
│        Bug     Login   Search   Copy    Docs   Mobile  Legacy   │
│        Fixes   Opt.    Fix             Refresh App    Fix      │
│                                                                 │
│  ████ = Quick Win    ████ = Major    ████ = Fill-in  ████ = Sink│
└─────────────────────────────────────────────────────────────────┘
```

## UI Component Specification

### MatrixBoard Component

```typescript
interface MatrixBoardProps {
  items: ValueEffortResult[];
  onSelectItem: (itemId: string) => void;
  onBulkMove: (itemIds: string[], targetQuadrant: Quadrant) => void;
  showGridlines: boolean;
  showLabels: boolean;
}

interface QuadrantPanelProps {
  quadrant: Quadrant;
  items: ValueEffortResult[];
  onItemSelect: (itemId: string) => void;
  onItemRemove: (itemId: string) => void;
}

interface ValueEffortSliderProps {
  itemType: 'value' | 'effort';
  value: number;
  onChange: (value: number) => void;
  min: number;
  max: number;
  markers: number[];
}
```

### Visual Design

| Quadrant | Background Color | Border Color |
|----------|-----------------|--------------|
| Quick Wins | #D1FAE5 | #10B981 |
| Major Projects | #DBEAFE | #3B82F6 |
| Fill-Ins | #FEF3C7 | #F59E0B |
| Time Sinks | #FEE2E2 | #EF4444 |

## Calculation Example

### Sample Item: Login Optimization

```
Value Analysis:
- Business Value: 8 (Reduces support tickets, improves conversion)
- User Value: 9 (Affects all users, reduces friction)
- Strategic Value: 7 (Aligns with retention goals)

Value Score: (8 × 0.4) + (9 × 0.35) + (7 × 0.25) = 8.1

Effort Analysis:
- Development Hours: 40 hours (1 week)
- Team Size: 2 developers
- Technical Complexity: 4/10
- Dependencies: 1 (auth service)

Effort Score: 3.5

Quadrant: QUICK WINS (High Value 8.1, Low Effort 3.5)
ROI: 8.1 / 3.5 = 2.31
Recommended Action: DO FIRST
```

### Comparison Table

| Item | Value | Effort | ROI | Quadrant |
|------|-------|--------|-----|----------|
| Bug Fixes | 8 | 2 | 4.0 | Quick Win |
| Login Optimization | 8.1 | 3.5 | 2.3 | Quick Win |
| Mobile App | 8 | 8 | 1.0 | Major Project |
| Platform v2 | 9 | 7 | 1.3 | Major Project |
| Copy Updates | 4 | 2 | 2.0 | Fill-In |
| Legacy Support | 3 | 7 | 0.4 | Time Sink |

## Best Practices

### Do's

1. **Start with Quick Wins** - Build momentum and credibility
2. **Plan Major Projects carefully** - They require resources and time
3. **Batch Fill-Ins** - Group low-effort items together
4. **Question Time Sinks** - Can they be reframed or automated?
5. **Revisit regularly** - Value and effort change over time

### Don'ts

1. **Don't ignore Quick Wins** - They fund Major Projects
2. **Don't start with Time Sinks** - Poor ROI drains resources
3. **Don't overestimate Value** - Be honest about impact
4. **Don't underestimate Effort** - Include all hidden costs

## Common Pitfalls

| Pitfall | Impact | Mitigation |
|---------|--------|------------|
| Optimism bias | Underestimated effort | Use historical data |
| Value inflation | Everything seems critical | Force-rank within quadrants |
| Ignoring dependencies | Hidden effort | Map dependencies first |
| Static analysis | Missed changes | Re-evaluate quarterly |
| Analysis paralysis | Too much discussion | Time-box decisions |

## Integration with Workflows

### Sprint Planning Template

```
┌─────────────────────────────────────────────────────────────────┐
│  Sprint ___ Planning              Date: ___________              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  QUICK WINS (Commit these first)                                │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ □ _________________  Value: __  Effort: __  ROI: __       │  │
│  │ □ _________________  Value: __  Effort: __  ROI: __       │  │
│  │ □ _________________  Value: __  Effort: __  ROI: __       │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                 │
│  MAJOR PROJECTS (Select 1 if capacity allows)                   │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ □ _________________  Value: __  Effort: __  ROI: __       │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                 │
│  Capacity: _____ points    Committed: _____ points              │
└─────────────────────────────────────────────────────────────────┘
```

## Export Format

```json
{
  "framework": "Value vs Effort",
  "generatedAt": "2026-03-18T10:00:00Z",
  "context": {
    "scoringScale": "1-10",
    "quadrantThresholds": {
      "valueThreshold": 5,
      "effortThreshold": 5
    }
  },
  "items": [
    {
      "id": "item-001",
      "title": "Login Optimization",
      "scores": {
        "value": 8.1,
        "effort": 3.5,
        "roi": 2.31
      },
      "quadrant": "QUICK_WINS",
      "priorityRank": 1,
      "recommendedAction": "DO FIRST"
    }
  ],
  "summary": {
    "quickWins": 4,
    "majorProjects": 3,
    "fillIns": 5,
    "timeSinks": 2,
    "avgROI": 1.8
  }
}
```

---

**Related Frameworks:**
- [ICE Framework](./ice.md) - Similar simplicity
- [Eisenhower Matrix](./eisenhower.md) - Similar 2x2 structure
