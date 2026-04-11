# ICE Framework

## Overview

ICE is a simplified prioritization framework that scores initiatives based on three factors: **I**mpact, **C**onfidence, and **E**ase. It's particularly useful for growth teams, rapid experimentation, and situations where quick decisions are needed with limited data.

## Framework Details

### Core Formula

```
ICE Score = Impact × Confidence × Ease

Where higher ICE score = higher priority
```

### Component Definitions

| Component | Description | Scale |
|-----------|-------------|-------|
| **Impact** | How much will this move the needle? | 1-10 |
| **Confidence** | How sure are we about our estimates? | 1-10 (or 0-100%) |
| **Ease** | How easy is this to implement? | 1-10 |

### Scale Comparison: RICE vs ICE

| Aspect | RICE | ICE |
|--------|------|-----|
| Complexity | High | Low |
| Data Required | Extensive | Minimal |
| Best For | Strategic planning | Rapid experiments |
| Effort Metric | Person-weeks | Relative ease |
| Reach | Explicit factor | Implicit in Impact |

## Type Definitions

```typescript
interface ICEResult {
  itemId: string;
  impact: number;          // 1-10
  confidence: number;      // 1-10
  ease: number;            // 1-10
  iceScore: number;        // Final score (1-1000)
  normalizedScore: number; // 0-100 for comparison
  rank: number;
  category: 'Quick Win' | 'Major Project' | 'Hard Slog' | 'Money Pit';
  rationale: string;
  autoFillUsed: boolean;
  manualOverrides: Override[];
}

interface ICEContext {
  primaryMetric: string;  // e.g., 'conversion_rate', 'retention', 'revenue'
  experimentVelocity: number; // Experiments per week capacity
  riskTolerance: 'Low' | 'Medium' | 'High';
}

interface ICECategory {
  quickWin: { minScore: number; description: string };
  majorProject: { minScore: number; description: string };
  hardSlog: { minScore: number; description: string };
  moneyPit: { minScore: number; description: string };
}

// Category thresholds
const ICE_CATEGORIES: ICECategory = {
  quickWin: { minScore: 500, description: 'High impact, easy to implement' },
  majorProject: { minScore: 200, description: 'High impact, significant effort' },
  hardSlog: { minScore: 50, description: 'Low impact, high effort' },
  moneyPit: { minScore: 0, description: 'Low impact, low confidence' }
};
```

## Scoring Guidelines

### Impact Scale (1-10)

| Score | Description | Indicators |
|-------|-------------|------------|
| 9-10 | Transformational | 10x improvement, game-changer |
| 7-8 | High impact | Significant metric movement |
| 5-6 | Moderate impact | Noticeable improvement |
| 3-4 | Low impact | Minor improvement |
| 1-2 | Minimal impact | Barely measurable |

```typescript
const IMPACT_CRITERIA = {
  transformational: [
    /10x/i, /game changer/i, /transformational/i,
    /order of magnitude/i, /breakthrough/i
  ],
  high: [
    /significant/i, /major improvement/i, /substantial/i,
    /key metric/i, /primary goal/i
  ],
  moderate: [
    /improvement/i, /enhancement/i, /better/i,
    /optimize/i, /increase/i
  ],
  low: [
    /minor/i, /small/i, /slight/i, /incremental/i
  ],
  minimal: [
    /negligible/i, /cosmetic/i, /edge case/i
  ]
};
```

### Confidence Scale (1-10)

| Score | Description | Data backing |
|-------|-------------|--------------|
| 9-10 | Near certainty | Multiple data sources, proven pattern |
| 7-8 | High confidence | Strong data, reasonable assumptions |
| 5-6 | Moderate confidence | Some data, educated guess |
| 3-4 | Low confidence | Limited data, hypothesis |
| 1-2 | Guesswork | No data, pure speculation |

```typescript
function calculateConfidence(item: BacklogItem): number {
  let confidence = 5; // Start at moderate (50%)

  // Data indicators
  if (item.metadata.dataAvailable) confidence += 2;
  if (item.metadata.userResearchConducted) confidence += 1;
  if (item.metadata.competitorAnalysis) confidence += 1;

  // Team experience
  if (item.metadata.teamFamiliarity === 'high') confidence += 1;
  if (item.metadata.similarPastProjects?.length > 2) confidence += 1;

  // Validation planned
  if (item.metadata.validationPlanned) confidence += 1;

  // Reduce for uncertainty
  if (item.metadata.technicalRisk === 'high') confidence -= 2;
  if (item.metadata.marketRisk === 'high') confidence -= 1;

  return Math.min(10, Math.max(1, confidence));
}
```

### Ease Scale (1-10)

Note: Higher ease = easier to implement (inverse of effort)

| Score | Description | Effort equivalent |
|-------|-------------|-------------------|
| 9-10 | Trivial | Minutes to hours, single person |
| 7-8 | Easy | Few hours, minimal coordination |
| 5-6 | Moderate | Days, some coordination |
| 3-4 | Challenging | 1-2 weeks, multiple people |
| 1-2 | Complex | Weeks to months, significant coordination |

```typescript
function calculateEase(item: BacklogItem): number {
  let ease = 5; // Start at moderate

  // Time estimates
  if (item.metadata.estimatedHours) {
    const hours = item.metadata.estimatedHours;
    if (hours <= 2) ease = 10;
    else if (hours <= 8) ease = 8;
    else if (hours <= 40) ease = 5;
    else if (hours <= 80) ease = 3;
    else ease = 1;
  }

  // Complexity factors
  const text = `${item.title} ${item.description}`.toLowerCase();

  if (/simple|quick|minor|trivial/i.test(text)) ease += 2;
  if (/complex|challenging|significant/i.test(text)) ease -= 1;
  if (/integration|migration|refactor/i.test(text)) ease -= 2;

  // Dependency impact
  const depCount = item.metadata.dependencies?.length || 0;
  ease -= depCount;

  // Team coordination
  if (item.metadata.requiresCrossTeam) ease -= 2;
  if (item.metadata.requiresApproval) ease -= 1;

  return Math.min(10, Math.max(1, ease));
}
```

## Auto-Fill Logic

### Complete Auto-Fill Implementation

```typescript
class ICEAutoFillEngine {
  constructor(private context: ICEContext) {}

  async suggestScores(item: BacklogItem): Promise<ICEResult> {
    const impact = this.suggestImpact(item);
    const confidence = this.suggestConfidence(item);
    const ease = this.suggestEase(item);

    const iceScore = impact * confidence * ease;
    const category = this.categorize(iceScore);

    return {
      itemId: item.id,
      impact,
      confidence,
      ease,
      iceScore,
      normalizedScore: (iceScore / 1000) * 100, // Max possible = 1000
      rank: 0,
      category,
      rationale: this.generateRationale(item, impact, confidence, ease, category),
      autoFillUsed: true,
      manualOverrides: []
    };
  }

  private suggestImpact(item: BacklogItem): number {
    const text = `${item.title} ${item.description}`.toLowerCase();
    let impact = 5; // Default moderate

    // Check for impact indicators
    for (const pattern of IMPACT_CRITERIA.transformational) {
      if (pattern.test(text)) return 10;
    }
    for (const pattern of IMPACT_CRITERIA.high) {
      if (pattern.test(text)) return 8;
    }
    for (const pattern of IMPACT_CRITERIA.moderate) {
      if (pattern.test(text)) return 5;
    }
    for (const pattern of IMPACT_CRITERIA.low) {
      if (pattern.test(text)) return 3;
    }

    // Adjust based on metadata
    if (item.metadata.businessValue >= 8) impact += 2;
    if (item.metadata.affectsRevenue) impact += 1;
    if (item.metadata.strategicInitiative) impact += 1;

    return Math.min(10, Math.max(1, impact));
  }

  private suggestConfidence(item: BacklogItem): number {
    // Implementation as shown above
  }

  private suggestEase(item: BacklogItem): number {
    // Implementation as shown above
  }

  private categorize(score: number): ICEResult['category'] {
    if (score >= 500) return 'Quick Win';
    if (score >= 200) return 'Major Project';
    if (score >= 50) return 'Hard Slog';
    return 'Money Pit';
  }

  private generateRationale(
    item: BacklogItem,
    impact: number,
    confidence: number,
    ease: number,
    category: string
  ): string {
    const factors = [];

    if (impact >= 8) factors.push('high potential impact');
    if (confidence >= 8) factors.push('strong confidence in estimates');
    if (ease >= 8) factors.push('quick to implement');

    if (category === 'Quick Win') {
      factors.push('excellent candidate for immediate execution');
    } else if (category === 'Money Pit') {
      factors.push('consider deprioritizing or reframing');
    }

    return factors.length > 0
      ? `Auto-scored: ${factors.join(', ')}`
      : 'Standard priority based on available data';
  }
}
```

## Visualization

### ICE Score Distribution

```
┌─────────────────────────────────────────────────────────────────┐
│                    ICE Score Distribution                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Quick Wins (500+)                                              │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Experiment A (729)  Experiment B (640)  Landing Page   │   │
│  │  Impact: 9           Impact: 8           (512)          │   │
│  │  Confidence: 9       Confidence: 8                      │   │
│  │  Ease: 9             Ease: 10                           │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  Major Projects (200-499)                                       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Feature X (384)     Email Campaign (245)               │   │
│  │  Impact: 8             Impact: 7                        │   │
│  │  Confidence: 8         Confidence: 7                    │   │
│  │  Ease: 6               Ease: 5                          │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  Hard Slogs (50-199)                                            │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Platform Migration (120)  SEO Optimization (75)        │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### ICE Matrix (Impact vs Ease)

```
┌─────────────────────────────────────────────────────────────────┐
│                      ICE Priority Matrix                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  High Impact ─────────────────────────────────────────          │
│     │                                    │                      │
│     │  ★ QUICK WINS ★                    │  MAJOR PROJECTS      │
│     │  High impact, easy                 │  High impact, hard   │
│     │  DO THESE FIRST!                   │  Plan carefully      │
│     │                                    │                      │
│     │  • Experiment A (729)              │  • Feature X (384)   │
│     │  • Landing Page (512)              │  • Redesign (280)    │
│     │                                    │                      │
│  ───┼────────────────────────────────────┼───                   │
│     │                                    │                      │
│     │  FILL-INS                          │  MONEY PITS          │
│     │  Low impact, easy                  │  Low impact, hard    │
│     │  Do when bored                     │  AVOID!              │
│     │                                    │                      │
│     │  • Copy tweaks (90)                │  • Legacy support    │
│     │  • Bug fixes (75)                  │    (45)              │
│     │                                    │                      │
│  Low Impact ────────────────────────────                       │
│                                                                 │
│     Easy                              Hard                      │
└─────────────────────────────────────────────────────────────────┘
```

### Experiment Velocity Chart

```
┌─────────────────────────────────────────────────────────────────┐
│              Experiment Velocity (ICE-Based)                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Experiments                                                     │
│    8 │                                                           │
│    6 │    ███                                                    │
│    4 │    ███       ███                                          │
│    2 │    ███   ███ ███   ███                                    │
│    0 │────███───███─███───███──────────────────────────────      │
│        Week1     Week2     Week3     Week4                       │
│                                                                 │
│  Legend: ███ Quick Wins  ███ Major Projects                     │
│                                                                 │
│  Target: 6 experiments/week (Quick Win velocity)                │
└─────────────────────────────────────────────────────────────────┘
```

## UI Component Specification

### ICECalculator Component

```typescript
interface ICECalculatorProps {
  item: BacklogItem;
  context: ICEContext;
  result: ICEResult;
  onScoreChange: (field: 'impact' | 'confidence' | 'ease', value: number) => void;
  onApplyAutoFill: () => void;
  autoFillLoading: boolean;
}

interface ICEScoreSliderProps {
  label: string;
  description: string;
  value: number;
  onChange: (value: number) => void;
  color: string;
  icon: React.ReactNode;
}
```

### Visual Design

| Component | Color | Visual |
|-----------|-------|--------|
| Impact | #EF4444 | Red gauge/meter |
| Confidence | #3B82F6 | Blue confidence bar |
| Ease | #10B981 | Green ease indicator |
| ICE Score | #8B5CF6 | Purple badge |
| Quick Win | #10B981 | Green highlight |
| Money Pit | #EF4444 | Red warning |

## Calculation Examples

### Example 1: Landing Page A/B Test

```
Impact:       8 (High - directly affects conversion)
Confidence:   8 (Strong data from previous tests)
Ease:         8 (Easy with A/B testing platform)

ICE Score:    8 × 8 × 8 = 512
Category:     Quick Win ✓
```

### Example 2: Platform Migration

```
Impact:       9 (Transformational for long-term)
Confidence:   6 (Moderate - some unknowns)
Ease:         2 (Very difficult, months of work)

ICE Score:    9 × 6 × 2 = 108
Category:     Hard Slog
```

### Comparison Table

| Experiment | Impact | Confidence | Ease | ICE Score | Category |
|------------|--------|------------|------|-----------|----------|
| Landing Page Test | 8 | 8 | 8 | **512** | Quick Win |
| Email Subject Test | 6 | 9 | 10 | **540** | Quick Win |
| Checkout Flow | 9 | 7 | 5 | **315** | Major Project |
| Platform Migration | 9 | 6 | 2 | **108** | Hard Slog |
| Tooltip Copy | 3 | 8 | 9 | **216** | Hard Slog |
| Legacy Browser Support | 2 | 9 | 3 | **54** | Money Pit |

## Growth Team Integration

### Experiment Pipeline

```typescript
interface ExperimentPipeline {
  weeklyCapacity: number; // Experiments per week
  backlog: ICEResult[];

  selectExperiments(): {
    quickWins: ICEResult[];
    majorProjects: ICEResult[];
  } {
    const sorted = [...this.backlog].sort(
      (a, b) => b.iceScore - a.iceScore
    );

    const quickWins: ICEResult[] = [];
    const majorProjects: ICEResult[] = [];

    // Prioritize quick wins for velocity
    for (const item of sorted) {
      if (item.category === 'Quick Win' && quickWins.length < this.weeklyCapacity) {
        quickWins.push(item);
      } else if (item.category === 'Major Project') {
        majorProjects.push(item);
      }
    }

    return { quickWins, majorProjects };
  }
}
```

### Sprint Planning Template

```
Week of: _______________

QUICK WINS (Target: 4-6 per week)
┌─────────────────────────────────────────────────────────────┐
│ 1. ____________________ ICE: ____ Impact: __ Conf: __ Ease: __│
│ 2. ____________________ ICE: ____ Impact: __ Conf: __ Ease: __│
│ 3. ____________________ ICE: ____ Impact: __ Conf: __ Ease: __│
│ 4. ____________________ ICE: ____ Impact: __ Conf: __ Ease: __│
└─────────────────────────────────────────────────────────────┘

MAJOR PROJECTS (Ongoing)
┌─────────────────────────────────────────────────────────────┐
│ 1. ____________________ ICE: ____                           │
│ 2. ____________________ ICE: ____                           │
└─────────────────────────────────────────────────────────────┘
```

## Best Practices

### Do's

1. **Use for rapid experimentation** - ICE excels at growth team workflows
2. **Prioritize Quick Wins** - Build velocity with easy wins
3. **Be honest about Ease** - Don't underestimate coordination costs
4. **Review and learn** - Track actual vs predicted impact
5. **Avoid Money Pits** - Low scores should be questioned

### Don'ts

1. **Don't overthink it** - ICE is meant to be fast and simple
2. **Don't use for strategic decisions** - Use RICE or WSJF for those
3. **Don't ignore Confidence** - It prevents wasted effort
4. **Don't game the system** - Honest scoring leads to better outcomes

## When to Use ICE vs RICE

| Factor | Use ICE | Use RICE |
|--------|---------|----------|
| Decision speed | Need decision in hours | Can spend days analyzing |
| Data availability | Limited data | Extensive data available |
| Scope | Experiments, small features | Strategic initiatives |
| Team | Growth/experimentation teams | Product/engineering teams |
| Impact measurement | Relative (1-10) | Absolute (user counts) |

## Export Format

```json
{
  "framework": "ICE",
  "generatedAt": "2026-03-18T10:00:00Z",
  "context": {
    "primaryMetric": "conversion_rate",
    "experimentVelocity": 6,
    "riskTolerance": "Medium"
  },
  "items": [
    {
      "id": "exp-001",
      "title": "Landing Page A/B Test",
      "scores": {
        "impact": 8,
        "confidence": 8,
        "ease": 8,
        "iceScore": 512,
        "normalizedScore": 51.2
      },
      "category": "Quick Win",
      "rank": 1,
      "rationale": "High impact, proven pattern, easy to execute"
    }
  ],
  "summary": {
    "quickWins": 4,
    "majorProjects": 3,
    "hardSlogs": 2,
    "moneyPits": 1
  }
}
```

---

**Related Frameworks:**
- [RICE Framework](./rice.md) - More detailed version with Reach
- [Value vs Effort Matrix](./value-effort.md) - Similar simplicity
