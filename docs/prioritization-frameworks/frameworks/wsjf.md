# WSJF (Weighted Shortest Job First) Framework

## Overview

WSJF is a prioritization model used primarily in SAFe (Scaled Agile Framework) that prioritizes jobs (features, capabilities, epics) based on the **Cost of Delay** divided by **Job Size**. It maximizes economic benefit by delivering the highest value items in the shortest time.

## Framework Details

### Core Formula

```
        Cost of Delay (CD3)
WSJF = ─────────────────────
          Job Size

Where higher WSJF score = higher priority
```

### Cost of Delay Components

```
Cost of Delay = User-Business Value + Time Criticality + Risk Reduction & Opportunity Enablement
```

| Component | Description | Scale |
|-----------|-------------|-------|
| **User-Business Value** | Value to customer and business | 1-20 |
| **Time Criticality** | Urgency based on timing factors | 1-20 |
| **Risk Reduction & OE** | Risk mitigation and future opportunities | 1-20 |

### Job Size

| Factor | Description | Scale |
|--------|-------------|-------|
| **Job Size** | Relative effort/complexity | 1-20 (inverse) |

Note: Job size uses Fibonacci-like scale: 1, 2, 3, 5, 8, 13, 20

## Type Definitions

```typescript
interface WSJFResult {
  itemId: string;
  userBusinessValue: number;      // 1-20
  timeCriticality: number;         // 1-20
  riskReductionOE: number;         // 1-20
  costOfDelay: number;             // Sum of above (3-60)
  jobSize: number;                 // 1-20
  wsjfScore: number;               // CD3 / Job Size
  normalizedScore: number;         // 0-100 for comparison
  rank: number;
  rationale: string;
  confidence: number;
}

interface WSJFContext {
  programIncrement: string;
  teamCapacity: number;
  dependencies: Dependency[];
  marketWindow?: {
    opens: Date;
    closes: Date;
    peakValue: Date;
  };
}

interface WSJFCriteria {
  userBusinessValue: {
    high: string[];    // Criteria for 15-20 score
    medium: string[];  // Criteria for 8-14 score
    low: string[];     // Criteria for 1-7 score
  };
  timeCriticality: {
    fixedDeadline: boolean;
    seasonalFactor: number;
    competitivePressure: number;
  };
  riskReductionOE: {
    riskMitigation: number;
    opportunityEnablement: number;
    learningValue: number;
  };
}
```

## Scoring Guidelines

### User-Business Value Scale

| Score | Description | Indicators |
|-------|-------------|------------|
| 1-3 | Minimal value | Internal only, no customer impact |
| 4-7 | Low value | Minor improvement, limited users |
| 8-11 | Moderate value | Noticeable improvement, moderate usage |
| 12-15 | High value | Significant benefit, wide usage |
| 16-20 | Critical value | Revenue-critical, strategic importance |

### Time Criticality Scale

| Score | Description | Indicators |
|-------|-------------|------------|
| 1-3 | No urgency | Can be done anytime |
| 4-7 | Some urgency | Beneficial soon, no hard deadline |
| 8-11 | Time-sensitive | Should be done this PI |
| 12-15 | Urgent | Fixed deadline approaching |
| 16-20 | Critical | Market window closing, regulatory deadline |

### Risk Reduction & Opportunity Enablement Scale

| Score | Description | Indicators |
|-------|-------------|------------|
| 1-3 | Minimal RROE | No risk reduction, no new options |
| 4-7 | Low RROE | Minor learning, limited enablement |
| 8-11 | Moderate RROE | Some risk mitigation, enables 1-2 features |
| 12-15 | High RROE | Significant risk reduction, enables many features |
| 16-20 | Critical RROE | Removes major blocker, opens new markets |

### Job Size Scale

| Score | Story Points | Description |
|-------|-------------|-------------|
| 1 | 1-2 points | Trivial effort |
| 2 | 3-4 points | Small effort |
| 3 | 5-6 points | Moderate effort |
| 5 | 7-10 points | Significant effort |
| 8 | 11-16 points | Large effort |
| 13 | 17-25 points | Very large effort |
| 20 | 26+ points | Epic effort |

## Auto-Fill Logic

### Value Suggestion Algorithm

```typescript
function suggestUserBusinessValue(item: BacklogItem): number {
  let score = 10; // Default moderate value
  const text = `${item.title} ${item.description}`.toLowerCase();

  // Revenue indicators
  if (/revenue|income|profit|monetize/i.test(text)) score += 3;
  if (/customer acquisition|new user/i.test(text)) score += 2;

  // Strategic importance
  if (/strategic|roadmap|vision/i.test(text)) score += 2;
  if (/key differentiator|competitive advantage/i.test(text)) score += 3;

  // User impact
  if (/all users|every customer/i.test(text)) score += 2;
  if (/power users|enterprise/i.test(text)) score += 1;

  // Penalty for internal-only
  if (/internal tool|admin only/i.test(text)) score -= 2;

  return Math.min(20, Math.max(1, score));
}
```

### Time Criticality Suggestion

```typescript
function suggestTimeCriticality(item: BacklogItem, context: WSJFContext): number {
  let score = 10; // Default moderate urgency
  const text = `${item.title} ${item.description}`.toLowerCase();

  // Deadline indicators
  if (/deadline|must launch|go-live/i.test(text)) score += 4;
  if (/quarter|PI | program increment/i.test(text)) score += 2;

  // Seasonal factors
  if (/holiday|black friday|cyber monday/i.test(text)) score += 3;
  if (/back to school|summer|january/i.test(text)) score += 2;

  // Competitive pressure
  if (/competitor|market share|first to market/i.test(text)) score += 3;

  // Market window check
  if (context.marketWindow) {
    const daysToClose = differenceInDays(context.marketWindow.closes, new Date());
    if (daysToClose < 30) score += 5;
    else if (daysToClose < 60) score += 3;
  }

  return Math.min(20, Math.max(1, score));
}
```

### Job Size Estimation

```typescript
function suggestJobSize(item: BacklogItem): number {
  // Base on estimated hours if available
  if (item.metadata.estimatedHours) {
    const hours = item.metadata.estimatedHours;
    if (hours <= 4) return 1;
    if (hours <= 8) return 2;
    if (hours <= 16) return 3;
    if (hours <= 40) return 5;
    if (hours <= 80) return 8;
    if (hours <= 160) return 13;
    return 20;
  }

  // Analyze description for complexity indicators
  let size = 5; // Default
  const text = `${item.title} ${item.description}`.toLowerCase();

  // Complexity indicators
  if (/simple|quick|minor/i.test(text)) size -= 2;
  if (/complex|major|refactor/i.test(text)) size += 2;
  if (/integration|migration|rewrite/i.test(text)) size += 3;
  if (/multiple systems|cross-team/i.test(text)) size += 2;

  // Dependency impact
  const depCount = item.metadata.dependencies?.length || 0;
  size += depCount;

  return Math.min(20, Math.max(1, size));
}
```

## Visualization

### WSJF Priority Matrix

```
┌─────────────────────────────────────────────────────────────────┐
│                   WSJF Priority Quadrants                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  High Cost of Delay                                             │
│     │                                                           │
│     │   DO FIRST          DO NEXT                               │
│     │   (High WSJF)       (Medium WSJF)                         │
│     │   Quick wins        Important but larger                  │
│     │                                                           │
│  ───┼───────────────────────────────────────────────            │
│     │                                                           │
│     │   FILL IN           BACKLOG                               │
│     │   (Medium WSJF)     (Low WSJF)                            │
│     │   When capacity     Low priority                          │
│     │   available                                               │
│     │                                                           │
│  Low Cost of Delay                                              │
│                                                                 │
│     Small Job Size         Large Job Size                       │
└─────────────────────────────────────────────────────────────────┘
```

### WSJF Score Distribution

```
┌─────────────────────────────────────────────────────────────────┐
│                    WSJF Score Rankings                           │
├─────────────────────────────────────────────────────────────────┤
│  Rank  Item                          CD3   Size   WSJF          │
├─────────────────────────────────────────────────────────────────┤
│   1    Payment Gateway Integration   45    8      5.63          │
│   2    Mobile App Launch             42    13     3.23          │
│   3    API Rate Limiting             28    5      5.60          │
│   4    Dashboard Redesign            35    8      4.38          │
│   5    Documentation Update          15    3      5.00          │
│   6    Tech Debt Reduction           25    13     1.92          │
│   7    Analytics Dashboard           30    20     1.50          │
└─────────────────────────────────────────────────────────────────┘
```

## UI Component Specification

### WSJFCalculator Component

```typescript
interface WSJFCalculatorProps {
  item: BacklogItem;
  context: WSJFContext;
  onScoreChange: (scores: WSJFResult) => void;
  showAutoFill: boolean;
  onApplyAutoFill: (suggestions: Partial<WSJFResult>) => void;
}

interface WSJFSliderProps {
  label: string;
  value: number;
  min: number;
  max: number;
  markers: number[];
  guidance: string;
  onChange: (value: number) => void;
}
```

### Visual Design

| Element | Color | Description |
|---------|-------|-------------|
| High WSJF | #10B981 | Green - Do First |
| Medium WSJF | #F59E0B | Amber - Do Next |
| Low WSJF | #6B7280 | Gray - Backlog |
| Auto-fill suggestion | #3B82F6 | Blue badge |

## Calculation Example

### Sample Feature: Payment Gateway Integration

```
User-Business Value:     15 (Revenue-critical, all users)
Time Criticality:        18 (Q1 launch deadline, competitive pressure)
Risk Reduction & OE:     12 (Enables subscription model)
─────────────────────────────────────────────────
Cost of Delay (CD3):     45

Job Size:                8 (2 sprints, 3 developers)

WSJF Score:              45 / 8 = 5.63
```

### Comparison Table

| Feature | UBV | TC | RROE | CD3 | Size | WSJF |
|---------|-----|----|----|-----|----|----|
| Payment Gateway | 15 | 18 | 12 | 45 | 8 | 5.63 |
| Mobile App | 18 | 16 | 8 | 42 | 13 | 3.23 |
| API Rate Limiting | 12 | 10 | 6 | 28 | 5 | 5.60 |
| Dashboard v2 | 10 | 8 | 5 | 23 | 8 | 2.88 |

## SAFe Integration

### Program Increment Planning

```typescript
interface PIPlanning {
  increment: number;
  duration: weeks;
  features: Feature[];
  capacity: number; // Story points per team

  // WSJF-based selection
  selectFeatures(): Feature[] {
    return this.features
      .sort((a, b) => b.wsjfScore - a.wsjfScore)
      .reduce((selected, feature) => {
        if (selected.points + feature.jobSize <= this.capacity) {
          selected.features.push(feature);
          selected.points += feature.jobSize;
        }
        return selected;
      }, { features: [], points: 0 }).features;
  }
}
```

### WSJF in SAFe Events

| Event | WSJF Usage |
|-------|------------|
| PI Planning | Prioritize features for iteration |
| Sprint Planning | Select stories based on WSJF |
| Backlog Refinement | Re-evaluate WSJF scores |
| Inspect & Adapt | Review WSJF accuracy |

## Best Practices

### Do's

1. **Calibrate as a team** - Ensure shared understanding of scales
2. **Revisit scores regularly** - Market conditions change
3. **Consider dependencies** - High WSJF blocked items need attention
4. **Use relative sizing** - Compare to reference stories
5. **Document rationale** - Capture why scores were assigned

### Don'ts

1. **Don't average scores** - Use discussion to reach consensus
2. **Don't game the system** - Honest sizing is critical
3. **Don't ignore job size** - Quick wins matter
4. **Don't set and forget** - Re-evaluate each PI

## Common Pitfalls

| Pitfall | Impact | Mitigation |
|---------|--------|------------|
| Inflated urgency | Skews priorities | Use objective deadline criteria |
| Underestimated size | WSJF too high | Reference historical velocity |
| Ignoring dependencies | Blocked high-WSJF items | Map dependencies first |
| One-person scoring | Biased results | Team-based calibration |

## Export Format

```json
{
  "framework": "WSJF",
  "programIncrement": "PI 2026-Q1",
  "features": [
    {
      "id": "feat-001",
      "title": "Payment Gateway Integration",
      "scores": {
        "userBusinessValue": 15,
        "timeCriticality": 18,
        "riskReductionOE": 12,
        "costOfDelay": 45,
        "jobSize": 8,
        "wsjfScore": 5.63
      },
      "rank": 1,
      "rationale": "Revenue-critical with Q1 deadline"
    }
  ],
  "summary": {
    "totalFeatures": 12,
    "avgWSJF": 3.45,
    "topQuartileThreshold": 4.5
  }
}
```

---

**Related Frameworks:**
- [RICE Framework](./rice.md) - Similar component-based scoring
- [Value vs Effort Matrix](./value-effort.md) - Simpler value/effort comparison
