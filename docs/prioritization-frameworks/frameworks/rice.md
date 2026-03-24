# RICE Prioritization Framework

## Overview

RICE is a prioritization framework developed by Intercom that scores projects based on four factors: **R**each, **I**mpact, **C**onfidence, and **E**ffort. It provides a data-driven approach to prioritization that reduces bias and enables objective comparison of initiatives.

## Framework Details

### Core Formula

```
           Reach × Impact × Confidence
RICE Score = ───────────────────────────
                   Effort

Where higher RICE score = higher priority
```

### Component Definitions

| Component | Description | Typical Scale |
|-----------|-------------|---------------|
| **Reach** | How many people will this affect? | Absolute number (users/customers) |
| **Impact** | How much will this impact each person? | 0.25 - 3 (multiplier scale) |
| **Confidence** | How sure are we about our estimates? | 0% - 100% (percentage) |
| **Effort** | How much work will this take? | Person-weeks or person-months |

### Impact Scale

| Score | Description | Example |
|-------|-------------|---------|
| 3 | Massive impact | Core feature, game-changer |
| 2 | High impact | Significant improvement |
| 1 | Medium impact | Noticeable improvement |
| 0.5 | Low impact | Minor improvement |
| 0.25 | Minimal impact | Barely noticeable |

## Type Definitions

```typescript
interface RICEBacklogItem extends BacklogItem {
  reachMetric: 'users' | 'customers' | 'accounts' | 'transactions';
  timeFrame?: string; // e.g., "per month", "per quarter"
}

interface RICEResult {
  itemId: string;
  reach: number;           // Absolute count
  impact: number;          // 0.25 - 3
  confidence: number;      // 0 - 1 (0% - 100%)
  effort: number;          // Person-weeks
  riceScore: number;       // Final score
  normalizedScore: number; // 0-100 for comparison
  rank: number;
  confidenceLevel: 'High' | 'Medium' | 'Low';
  rationale: string;
  autoFillUsed: boolean;
  manualOverrides: Override[];
}

interface RICEContext {
  totalUserBase: number;
  measurementPeriod: string;
  teamCapacity: number; // Person-weeks per period
  historicalData?: {
    similarProjects: HistoricalProject[];
    actualReach: Map<string, number>;
    actualImpact: Map<string, number>;
  };
}

interface Override {
  field: 'reach' | 'impact' | 'confidence' | 'effort';
  originalValue: number;
  newValue: number;
  reason: string;
  timestamp: Date;
}
```

## Scoring Guidelines

### Reach

Reach should be quantified as an absolute number over a specific time period.

| Approach | Method |
|----------|--------|
| User-facing features | Number of users affected per month |
| Customer-facing | Number of customer accounts |
| Internal tools | Number of employees impacted |
| Platform changes | Percentage of traffic/usage |

```typescript
function calculateReach(item: BacklogItem, context: RICEContext): number {
  // Use provided reach if available
  if (item.metadata.estimatedReach) {
    return item.metadata.estimatedReach;
  }

  // Estimate based on user segment
  const segment = item.metadata.userSegment;
  switch (segment) {
    case 'all_users':
      return context.totalUserBase;
    case 'power_users':
      return context.totalUserBase * 0.2; // Assume 20% are power users
    case 'new_users':
      return context.totalUserBase * 0.1; // Assume 10% monthly new users
    case 'enterprise':
      return item.metadata.enterpriseAccounts || 0;
    default:
      return context.totalUserBase * 0.5; // Default to 50%
  }
}
```

### Impact

Impact uses a standardized multiplier scale to enable comparison.

```typescript
interface ImpactCriteria {
  massive: string[];    // Score 3
  high: string[];       // Score 2
  medium: string[];     // Score 1
  low: string[];        // Score 0.5
  minimal: string[];    // Score 0.25
}

const IMPACT_INDICATORS: ImpactCriteria = {
  massive: [
    /core feature/i, /game changer/i, /blockbuster/i,
    /revenue driver/i, /retention critical/i
  ],
  high: [
    /significant improvement/i, /major enhancement/i,
    /customer request/i, /high demand/i
  ],
  medium: [
    /improvement/i, /enhancement/i, /better/i,
    /optimize/i, /streamline/i
  ],
  low: [
    /minor improvement/i, /small change/i,
    /quality of life/i, /nice to have/i
  ],
  minimal: [
    /cosmetic/i, /polish/i, /edge case/i,
    /rarely used/i
  ]
};
```

### Confidence

Confidence acts as a risk discount on your estimates.

| Level | Percentage | Description |
|-------|------------|-------------|
| **High** | 81-100% | Strong data, proven pattern, low uncertainty |
| **Medium** | 51-80% | Some data, reasonable assumptions |
| **Low** | 0-50% | Guesswork, untested assumptions, high uncertainty |

```typescript
function calculateConfidence(item: BacklogItem, context: RICEContext): number {
  let confidence = 0.5; // Start at 50% (Medium)

  // Increase confidence with data backing
  if (item.metadata.dataAvailable) confidence += 0.2;
  if (item.metadata.userResearchConducted) confidence += 0.15;
  if (item.metadata.aBTestPlanned) confidence += 0.1;

  // Increase with team experience
  if (item.metadata.teamFamiliarity === 'high') confidence += 0.1;
  if (item.metadata.similarPastProjects?.length > 0) confidence += 0.1;

  // Decrease for uncertainty factors
  if (item.metadata.technicalUncertainty === 'high') confidence -= 0.15;
  if (item.metadata.marketUncertainty === 'high') confidence -= 0.15;
  if (item.metadata.dependencies?.length > 2) confidence -= 0.1;

  return Math.min(1.0, Math.max(0.0, confidence));
}
```

### Effort

Effort is measured in person-time (weeks or months).

```typescript
function estimateEffort(item: BacklogItem): number {
  // Use provided estimate if available
  if (item.metadata.estimatedWeeks) {
    return item.metadata.estimatedWeeks;
  }

  // Estimate based on story points
  if (item.metadata.storyPoints) {
    const points = item.metadata.storyPoints;
    // Rough conversion: 1 story point ≈ 0.5 person-weeks
    return points * 0.5;
  }

  // Analyze description for complexity
  let effort = 2; // Default 2 weeks
  const text = `${item.title} ${item.description}`.toLowerCase();

  if (/simple|quick|minor/i.test(text)) effort -= 1;
  if (/complex|major|significant/i.test(text)) effort += 2;
  if (/integration|migration|refactor/i.test(text)) effort += 3;
  if (/cross-team|multiple systems/i.test(text)) effort += 2;

  // Add time for each dependency
  const depCount = item.metadata.dependencies?.length || 0;
  effort += depCount * 0.5;

  return Math.max(0.5, effort);
}
```

## Auto-Fill Logic

### Complete Auto-Fill Engine

```typescript
class RICEAutoFillEngine {
  constructor(private context: RICEContext) {}

  async suggestScores(item: RICEBacklogItem): Promise<RICEResult> {
    const reach = this.suggestReach(item);
    const impact = this.suggestImpact(item);
    const confidence = this.suggestConfidence(item);
    const effort = this.suggestEffort(item);

    const riceScore = (reach * impact * confidence) / effort;

    return {
      itemId: item.id,
      reach,
      impact,
      confidence,
      effort,
      riceScore,
      normalizedScore: this.normalizeScore(riceScore),
      rank: 0, // Will be calculated after all items
      confidenceLevel: this.getConfidenceLevel(confidence),
      rationale: this.generateRationale(item, reach, impact, confidence, effort),
      autoFillUsed: true,
      manualOverrides: []
    };
  }

  private suggestReach(item: RICEBacklogItem): number {
    // Implementation as shown above
  }

  private suggestImpact(item: RICEBacklogItem): number {
    const text = `${item.title} ${item.description}`.toLowerCase();

    for (const pattern of IMPACT_INDICATORS.massive) {
      if (pattern.test(text)) return 3;
    }
    for (const pattern of IMPACT_INDICATORS.high) {
      if (pattern.test(text)) return 2;
    }
    for (const pattern of IMPACT_INDICATORS.medium) {
      if (pattern.test(text)) return 1;
    }
    for (const pattern of IMPACT_INDICATORS.low) {
      if (pattern.test(text)) return 0.5;
    }
    return 0.25;
  }

  private suggestConfidence(item: BacklogItem): number {
    // Implementation as shown above
  }

  private suggestEffort(item: BacklogItem): number {
    // Implementation as shown above
  }

  private normalizeScore(score: number, allScores: number[]): number {
    const maxScore = Math.max(...allScores);
    return (score / maxScore) * 100;
  }

  private getConfidenceLevel(confidence: number): 'High' | 'Medium' | 'Low' {
    if (confidence >= 0.81) return 'High';
    if (confidence >= 0.51) return 'Medium';
    return 'Low';
  }

  private generateRationale(
    item: BacklogItem,
    reach: number,
    impact: number,
    confidence: number,
    effort: number
  ): string {
    const parts = [];

    if (reach > this.context.totalUserBase * 0.5) {
      parts.push('affects majority of users');
    }
    if (impact >= 2) {
      parts.push('high individual impact');
    }
    if (confidence < 0.5) {
      parts.push('low confidence - validation recommended');
    }
    if (effort > 8) {
      parts.push('significant effort required');
    }

    return parts.length > 0
      ? `Auto-scored: ${parts.join(', ')}`
      : 'Standard priority based on available data';
  }
}
```

## Visualization

### RICE Score Comparison Chart

```
┌─────────────────────────────────────────────────────────────────┐
│                    RICE Score Comparison                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Score                                                         │
│   800 │                              ███                        │
│   600 │                    ███       ███                        │
│   400 │          ███       ███       ███       ███              │
│   200 │    ███   ███       ███   ███ ███   ███ ███              │
│     0 │────███───███───────███───███─███───███─███──────────    │
│         │ A/B    Notif    Search   Recs   Perf   Docs   Mobile  │
│         │ Test             Redesign                              │
└─────────────────────────────────────────────────────────────────┘
```

### Component Breakdown

```
┌─────────────────────────────────────────────────────────────────┐
│                  RICE Component Analysis                         │
├─────────────────────────────────────────────────────────────────┤
│  Item          Reach   Impact   Confidence   Effort   RICE      │
├─────────────────────────────────────────────────────────────────┤
│  A/B Testing   ████████   ███      ████       ██      480       │
│  Notifications ██████████ ████     ███        ███     320       │
│  Search        ██████     █████    ██         ████    187       │
│  Recommendations█████████ ██       █          █████   162       │
│  Performance   ████       ████     █████      ██████  160       │
│  Documentation ██         ██       ████       ██      80        │
│  Mobile App    █████      ██       ██         ██████  67        │
└─────────────────────────────────────────────────────────────────┘
```

### RICE Matrix (Reach vs Impact)

```
┌─────────────────────────────────────────────────────────────────┐
│                    RICE Priority Matrix                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  High Impact ─────────────────────────────────────────          │
│     │                                    │                      │
│     │    DO FIRST (High RICE)            │   MAJOR PROJECTS     │
│     │    High reach, high impact,        │    High impact,      │
│     │    low effort                      │    high effort       │
│     │                                    │                      │
│  ───┼────────────────────────────────────┼───                   │
│     │                                    │                      │
│     │    QUICK WINS                      │    HARD SLOGS        │
│     │    Medium metrics                  │    High effort,      │
│     │                                    │    lower impact      │
│     │                                    │                      │
│  Low Impact ────────────────────────────                       │
│                                                                 │
│     Low Reach                         High Reach                │
└─────────────────────────────────────────────────────────────────┘
```

## UI Component Specification

### RICECalculator Component

```typescript
interface RICECalculatorProps {
  item: RICEBacklogItem;
  context: RICEContext;
  result: RICEResult;
  onScoreChange: (field: keyof RICEResult, value: number) => void;
  onApplyAutoFill: () => void;
  autoFillLoading: boolean;
}

interface RICEInputProps {
  label: string;
  value: number;
  onChange: (value: number) => void;
  min: number;
  max: number;
  step: number;
  guidance: string;
  autoFillValue?: number;
  onApplyAutoFill: () => void;
}
```

### Visual Design

| Component | Color | Visual |
|-----------|-------|--------|
| Reach | #3B82F6 | Blue bar chart |
| Impact | #8B5CF6 | Purple scale indicator |
| Confidence | #10B981 | Green confidence meter |
| Effort | #F59E0B | Amber effort indicator |
| RICE Score | #EC4899 | Pink highlight for final score |

## Calculation Example

### Sample Feature: A/B Testing Platform

```
Reach:        5,000 users/month
              (Product managers and analysts using the platform)

Impact:       2 (High)
              (Significant improvement to experimentation velocity)

Confidence:   0.8 (80% - High)
              (Based on usage data from similar tools)

Effort:       20 person-weeks
              (2 developers × 10 weeks)

RICE Score:   (5000 × 2 × 0.8) / 20 = 400
```

### Comparison Table

| Feature | Reach | Impact | Confidence | Effort | RICE Score |
|---------|-------|--------|------------|--------|------------|
| A/B Testing | 5,000 | 2 | 0.8 | 20 | **400** |
| Push Notifications | 50,000 | 1 | 0.9 | 12 | **3,750** |
| Search Redesign | 30,000 | 1.5 | 0.6 | 16 | **1,687** |
| Recommendation Engine | 45,000 | 2 | 0.5 | 24 | **1,875** |
| Performance Optimization | 100,000 | 0.5 | 0.95 | 8 | **5,937** |

## Best Practices

### Do's

1. **Use real data for Reach** - Base on analytics, not guesses
2. **Calibrate Impact scores** - Compare to past projects
3. **Be honest about Confidence** - Low confidence means validate first
4. **Track actuals** - Compare estimates to real outcomes
5. **Review regularly** - RICE scores change as you learn

### Don'ts

1. **Don't manipulate scores** - Game the system = wrong priorities
2. **Don't ignore Confidence** - It's a critical risk factor
3. **Don't use for tiny tasks** - RICE is for significant initiatives
4. **Don't average team estimates** - Discuss and converge

## Common Pitfalls

| Pitfall | Impact | Mitigation |
|---------|--------|------------|
| Inflated Reach | Skews priorities | Use analytics data |
| Impact inflation | Everything becomes "massive" | Use reference projects |
| Overconfidence | Risk of failure | Require data for high confidence |
| Underestimated Effort | Missed deadlines | Use historical velocity |
| Ignoring Confidence | Risky projects prioritized | Weight confidence heavily |

## Integration with Product Workflows

### Sprint Planning Integration

```typescript
interface SprintPlanning {
  capacity: number; // Person-weeks
  backlog: RICEResult[];

  selectItemsForSprint(): RICEResult[] {
    const sorted = [...this.backlog].sort(
      (a, b) => b.riceScore - a.riceScore
    );

    const selected: RICEResult[] = [];
    let remainingCapacity = this.capacity;

    for (const item of sorted) {
      if (item.effort <= remainingCapacity) {
        selected.push(item);
        remainingCapacity -= item.effort;
      }
    }

    return selected;
  }
}
```

### Export Format

```json
{
  "framework": "RICE",
  "generatedAt": "2026-03-18T10:00:00Z",
  "context": {
    "totalUserBase": 100000,
    "measurementPeriod": "per month",
    "teamCapacity": 40
  },
  "items": [
    {
      "id": "feature-001",
      "title": "A/B Testing Platform",
      "scores": {
        "reach": 5000,
        "impact": 2,
        "confidence": 0.8,
        "effort": 20,
        "riceScore": 400,
        "normalizedScore": 67.5
      },
      "rank": 3,
      "confidenceLevel": "High",
      "rationale": "High-value tool for product team"
    }
  ]
}
```

---

**Related Frameworks:**
- [ICE Framework](./ice.md) - Simplified version of RICE
- [WSJF Framework](./wsjf.md) - Similar component-based approach
