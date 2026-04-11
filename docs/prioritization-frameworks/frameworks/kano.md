# Kano Model Framework

## Overview

The Kano Model is a customer satisfaction framework that categorizes product features based on how customers perceive them. Developed by Professor Noriaki Kano in the 1980s, it helps teams understand which features drive satisfaction and which are simply expected.

## Framework Details

### Feature Categories

| Category | Symbol | Description | Satisfaction Impact |
|----------|--------|-------------|---------------------|
| **Must-be** | M | Basic expectations | Absence causes dissatisfaction; presence is neutral |
| **Performance** | P | More is better | Linear relationship between fulfillment and satisfaction |
| **Excitement** | E | Delighters | Absence is neutral; presence causes high satisfaction |
| **Indifferent** | I | Customers don't care | No impact on satisfaction |
| **Reverse** | R | Some prefer absence | Presence causes dissatisfaction |
| **Questionable** | Q | Invalid responses | Requires re-evaluation |

### The Kano Evaluation Matrix

```
┌─────────────────────────────────────────────────────────────────┐
│                    Kano Evaluation Matrix                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│                     Customer Reaction to Feature Presence       │
│                     Like    Expect   Neutral  Dislike           │
│            Like     Q        E        E       P                 │
│            Expect   R        I        I       M                 │
│ Customer   Neutral  R        I        I       M                 │
│ Reaction   Dislike  R        R        Q       Q                 │
│   to                                                          │
│ Feature                                                       │
│ Absence                                                     │
│                                                                 │
│  Q = Questionable, E = Excitement, P = Performance             │
│  M = Must-be, I = Indifferent, R = Reverse                     │
└─────────────────────────────────────────────────────────────────┘
```

## Type Definitions

```typescript
type KanoCategory = 'MUST_BE' | 'PERFORMANCE' | 'EXCITEMENT' | 'INDIFFERENT' | 'REVERSE' | 'QUESTIONABLE';

interface KanoQuestion {
  functionalForm: string;   // "How do you feel if feature IS present?"
  dysfunctionalForm: string; // "How do you feel if feature IS NOT present?"
  options: string[];        // Like, Expect, Neutral, Dislike, Live with it
}

interface KanoResult {
  itemId: string;
  category: KanoCategory;
  categoryScore: number;
  satisfactionCoefficient: number;
  dissatisfactionCoefficient: number;
  betterScore: number;      // Satisfaction increment
  worseScore: number;       // Dissatisfaction increment
  customerResponses: KanoResponse[];
  responseDistribution: Record<KanoCategory, number>;
  recommendedAction: string;
  rationale: string;
}

interface KanoResponse {
  customerId: string;
  functionalAnswer: number; // 1-5 scale
  dysfunctionalAnswer: number; // 1-5 scale
  importance?: number;      // Optional importance rating
  category: KanoCategory;
}

interface KanoCoefficients {
  better: number;  // (E + P) / (E + P + M + I)
  worse: number;   // -1 × (O + M) / (E + P + M + I)
  // Higher better = more satisfaction potential
  // Lower worse (more negative) = more dissatisfaction risk
}
```

## Kano Survey Method

### Question Structure

Each feature requires two questions:

**Functional Form (Positive):**
> "How would you feel if the product [has this feature]?"

**Dysfunctional Form (Negative):**
> "How would you feel if the product [does NOT have this feature]?"

### Response Options

1. **I like it** - Feature provides satisfaction
2. **I expect it** - Feature is expected/normal
3. **I am neutral** - No strong feeling
4. **I can live with it** - Acceptable but not preferred
5. **I dislike it** - Feature causes dissatisfaction

### Classification Logic

```typescript
const KANO_CLASSIFICATION: Record<string, Record<string, KanoCategory>> = {
  // Dysfunctional → Functional
  'Like':    { 'Like': 'Q', 'Expect': 'E', 'Neutral': 'E', 'Dislike': 'P' },
  'Expect':  { 'Like': 'R', 'Expect': 'I', 'Neutral': 'I', 'Dislike': 'M' },
  'Neutral': { 'Like': 'R', 'Expect': 'I', 'Neutral': 'I', 'Dislike': 'M' },
  'Dislike': { 'Like': 'R', 'Expect': 'R', 'Neutral': 'R', 'Dislike': 'Q' },
  'LiveWith':{ 'Like': 'R', 'Expect': 'I', 'Neutral': 'I', 'Dislike': 'M' }
};

function classifyResponse(functional: string, dysfunctional: string): KanoCategory {
  return KANO_CLASSIFICATION[dysfunctional]?.[functional] || 'QUESTIONABLE';
}
```

## Scoring Formulas

### Better/Worse Coefficients

```
Better (Satisfaction Increment) = (E + P) / (E + P + M + I)

Worse (Dissatisfaction Increment) = -1 × (O + M) / (E + P + M + I)

Where:
- E = Excitement responses
- P = Performance responses
- M = Must-be responses
- O = One-dimensional (same as Performance)
- I = Indifferent responses
```

### Interpretation

| Coefficient | Range | Interpretation |
|-------------|-------|----------------|
| Better | 0.5 - 1.0 | High satisfaction potential |
| Better | 0.3 - 0.5 | Moderate satisfaction potential |
| Better | 0.0 - 0.3 | Low satisfaction potential |
| Worse | -0.7 to -1.0 | High dissatisfaction risk |
| Worse | -0.4 to -0.7 | Moderate dissatisfaction risk |
| Worse | 0.0 to -0.4 | Low dissatisfaction risk |

### Category Scores

```typescript
interface KanoScores {
  // Percentage of responses in each category
  mustBePercent: number;
  performancePercent: number;
  excitementPercent: number;
  indifferentPercent: number;
  reversePercent: number;
  questionablePercent: number;

  // Dominant category (highest percentage)
  dominantCategory: KanoCategory;

  // Satisfaction coefficients
  better: number;
  worse: number;
}

function calculateKanoScores(responses: KanoResponse[]): KanoScores {
  const validResponses = responses.filter(r => r.category !== 'QUESTIONABLE');
  const total = validResponses.length;

  const counts = {
    M: validResponses.filter(r => r.category === 'MUST_BE').length,
    P: validResponses.filter(r => r.category === 'PERFORMANCE').length,
    E: validResponses.filter(r => r.category === 'EXCITEMENT').length,
    I: validResponses.filter(r => r.category === 'INDIFFERENT').length,
    R: validResponses.filter(r => r.category === 'REVERSE').length
  };

  return {
    mustBePercent: (counts.M / total) * 100,
    performancePercent: (counts.P / total) * 100,
    excitementPercent: (counts.E / total) * 100,
    indifferentPercent: (counts.I / total) * 100,
    reversePercent: (counts.R / total) * 100,
    questionablePercent: (responses.length - total) / responses.length * 100,
    dominantCategory: getDominantCategory(counts),
    better: (counts.E + counts.P) / total,
    worse: -1 * (counts.P + counts.M) / total
  };
}
```

## Auto-Fill Logic

### Category Suggestion Engine

```typescript
class KanoAutoFillEngine {
  async suggestCategory(item: BacklogItem, context?: KanoContext): Promise<KanoResult> {
    const text = `${item.title} ${item.description}`.toLowerCase();

    // Analyze feature type
    const featureType = this.analyzeFeatureType(text);

    // Suggest based on feature characteristics
    let suggestedCategory: KanoCategory;
    let rationale: string;

    switch (featureType) {
      case 'basic-requirement':
        suggestedCategory = 'MUST_BE';
        rationale = 'Fundamental feature customers expect as standard';
        break;

      case 'performance-enhancer':
        suggestedCategory = 'PERFORMANCE';
        rationale = 'Feature where more/better = higher satisfaction';
        break;

      case 'innovative-delighter':
        suggestedCategory = 'EXCITEMENT';
        rationale = 'Novel feature that exceeds expectations';
        break;

      case 'niche-request':
        suggestedCategory = 'INDIFFERENT';
        rationale = 'Feature with limited customer impact';
        break;

      default:
        suggestedCategory = 'PERFORMANCE';
        rationale = 'Default classification based on available data';
    }

    // Adjust based on metadata
    if (item.metadata.customerRequestCount > 100) {
      if (suggestedCategory === 'INDIFFERENT') {
        suggestedCategory = 'PERFORMANCE';
        rationale = 'Upgraded due to high customer request volume';
      }
    }

    return {
      itemId: item.id,
      category: suggestedCategory,
      categoryScore: this.getCategoryScore(suggestedCategory),
      satisfactionCoefficient: this.estimateSatisfaction(suggestedCategory),
      dissatisfactionCoefficient: this.estimateDissatisfaction(suggestedCategory),
      betterScore: suggestedCategory === 'EXCITEMENT' ? 0.9 :
                   suggestedCategory === 'PERFORMANCE' ? 0.6 :
                   suggestedCategory === 'MUST_BE' ? 0.3 : 0.1,
      worseScore: suggestedCategory === 'MUST_BE' ? -0.9 :
                  suggestedCategory === 'PERFORMANCE' ? -0.6 :
                  suggestedCategory === 'EXCITEMENT' ? -0.1 : -0.1,
      recommendedAction: this.getRecommendedAction(suggestedCategory),
      rationale
    };
  }

  private analyzeFeatureType(text: string): string {
    // Basic requirements
    if (/login|sign in|password|security|basic|fundamental/i.test(text)) {
      return 'basic-requirement';
    }

    // Performance features
    if (/faster|quicker|more|better|improved|enhanced|performance/i.test(text)) {
      return 'performance-enhancer';
    }

    // Excitement features
    if (/ai|machine learning|personalized|smart|innovative|first|novel/i.test(text)) {
      return 'innovative-delighter';
    }

    return 'niche-request';
  }

  private getCategoryScore(category: KanoCategory): number {
    const scores = {
      'MUST_BE': 60,
      'PERFORMANCE': 75,
      'EXCITEMENT': 90,
      'INDIFFERENT': 40,
      'REVERSE': 20,
      'QUESTIONABLE': 0
    };
    return scores[category];
  }

  private getRecommendedAction(category: KanoCategory): string {
    const actions = {
      'MUST_BE': 'Ensure this feature works flawlessly. It is expected.',
      'PERFORMANCE': 'Invest in continuous improvement. More is better.',
      'EXCITEMENT': 'Highlight in marketing. Creates competitive differentiation.',
      'INDIFFERENT': 'Consider deprioritizing. Low impact on satisfaction.',
      'REVERSE': 'Reconsider implementation. May cause dissatisfaction.',
      'QUESTIONABLE': 'Conduct additional customer research.'
    };
    return actions[category];
  }
}
```

## Visualization

### Kano Satisfaction Map

```
┌─────────────────────────────────────────────────────────────────┐
│                    Kano Satisfaction Map                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Satisfaction (Better)                                          │
│  1.0 │                        ★ Excitement Features            │
│      │                        (Delighters)                     │
│      │                    ★                                    │
│  0.5 │            ★ Performance Features                       │
│      │            (Linear)                                     │
│      │                                                         │
│  0.0 ─────★────────────────────────────────────────────        │
│      │ Must-be Features                                        │
│      │ (Basic Expectations)                                    │
│ -0.5 │                                                         │
│      │                                                         │
│ -1.0 │                                                         │
│      └─────────────────────────────────────────────────        │
│      0.0               0.5                1.0                  │
│                    Dissatisfaction (Worse)                      │
│                                                                 │
│  ★ = Feature positions based on coefficients                    │
└─────────────────────────────────────────────────────────────────┘
```

### Category Distribution

```
┌─────────────────────────────────────────────────────────────────┐
│                 Kano Category Distribution                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Must-be      ████████████████████  45%  (Basic expectations)  │
│  Performance  ██████████████        30%  (Linear satisfaction) │
│  Excitement   ████████              18%  (Delighters)          │
│  Indifferent  ███                    5%  (Low priority)        │
│  Reverse      █                      2%  (Reconsider)          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Feature Timeline View

```
┌─────────────────────────────────────────────────────────────────┐
│              Kano Feature Migration Over Time                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Feature          Launch    Year 1    Year 2    Year 3          │
│  ─────────────────────────────────────────────────────────      │
│  Touch ID         Excitement → Performance → Must-be            │
│  Face ID          Excitement → Performance → Must-be            │
│  Wireless Charge  Excitement → Performance                      │
│  USB-C Port       Performance  → Must-be                        │
│  Headphone Jack   Must-be    → (Removed)                        │
│                                                                 │
│  Note: Excitement features become Performance, then Must-be    │
│        as customer expectations evolve                          │
└─────────────────────────────────────────────────────────────────┘
```

## UI Component Specification

### KanoSurvey Component

```typescript
interface KanoSurveyProps {
  feature: BacklogItem;
  onComplete: (response: KanoResponse) => void;
}

interface KanoQuestionPairProps {
  feature: BacklogItem;
  functionalQuestion: string;
  dysfunctionalQuestion: string;
  options: string[];
  onFunctionalSelect: (answer: number) => void;
  onDysfunctionalSelect: (answer: number) => void;
}

interface KanoResultsChartProps {
  results: KanoResult[];
  onSelectFeature: (featureId: string) => void;
}
```

### Visual Design

| Category | Color | Symbol |
|----------|-------|--------|
| Must-be | #6B7280 | ⚫ Gray |
| Performance | #3B82F6 | 🔵 Blue |
| Excitement | #F59E0B | 🟡 Amber |
| Indifferent | #10B981 | 🟢 Green |
| Reverse | #EF4444 | 🔴 Red |
| Questionable | #8B5CF6 | 🟣 Purple |

## Calculation Example

### Sample Feature: Biometric Login

```
Survey Results (100 customers):
- Must-be: 45 responses
- Performance: 30 responses
- Excitement: 18 responses
- Indifferent: 5 responses
- Reverse: 2 responses

Coefficients:
Better = (E + P) / (E + P + M + I)
       = (18 + 30) / (18 + 30 + 45 + 5)
       = 48 / 98 = 0.49

Worse = -1 × (P + M) / (E + P + M + I)
      = -1 × (30 + 45) / 98
      = -75 / 98 = -0.77

Dominant Category: Must-be (45%)
Recommended Action: Ensure flawless implementation - customers expect this
```

## Product Lifecycle Integration

### Feature Evolution Pattern

```typescript
interface FeatureEvolution {
  feature: string;
  stages: {
    launch: KanoCategory;
    growth: KanoCategory;
    maturity: KanoCategory;
    decline: KanoCategory;
  };
}

// Typical evolution: Excitement → Performance → Must-be
const TYPICAL_EVOLUTION: FeatureEvolution = {
  feature: 'Fingerprint Authentication',
  stages: {
    launch: 'EXCITEMENT',     // Novel differentiator
    growth: 'PERFORMANCE',    // Expected but still valued
    maturity: 'MUST_BE',      // Standard expectation
    decline: 'MUST_BE'        // Still required
  }
};
```

### Strategic Implications

| Category | Investment Strategy | Marketing Focus |
|----------|---------------------|-----------------|
| Must-be | Maintain quality | Baseline messaging |
| Performance | Continuous improvement | Feature comparisons |
| Excitement | Protect IP, innovate | Differentiation campaigns |
| Indifferent | Minimize investment | Not highlighted |

## Best Practices

### Do's

1. **Survey real customers** - Don't guess their preferences
2. **Use both question forms** - Functional AND dysfunctional
3. **Analyze patterns** - Look for category migration over time
4. **Prioritize accordingly** - Must-be first, then Performance, then Excitement
5. **Track evolution** - Features change categories over time

### Don'ts

1. **Don't skip the survey** - Internal assumptions are often wrong
2. **Don't ignore Must-be** - They don't delight but their absence causes churn
3. **Don't over-invest in Indifferent** - Low ROI features
4. **Don't assume static categories** - Expectations evolve

## Common Pitfalls

| Pitfall | Impact | Mitigation |
|---------|--------|------------|
| Internal bias | Wrong categorization | Survey real customers |
| Small sample size | Unreliable results | Minimum 30-50 responses |
| Leading questions | Biased responses | Use standard Kano format |
| Ignoring Reverse | Unintended dissatisfaction | Act on Reverse feedback |
| Static analysis | Missed evolution | Re-survey periodically |

## Export Format

```json
{
  "framework": "Kano",
  "generatedAt": "2026-03-18T10:00:00Z",
  "surveyInfo": {
    "totalRespondents": 100,
    "dateRange": "2026-03-01 to 2026-03-15"
  },
  "features": [
    {
      "id": "feat-001",
      "title": "Biometric Login",
      "responses": {
        "mustBe": 45,
        "performance": 30,
        "excitement": 18,
        "indifferent": 5,
        "reverse": 2
      },
      "dominantCategory": "MUST_BE",
      "coefficients": {
        "better": 0.49,
        "worse": -0.77
      },
      "recommendedAction": "Ensure flawless implementation"
    }
  ]
}
```

---

**Related Frameworks:**
- [Value vs Effort Matrix](./value-effort.md) - Customer value focus
- [RICE Framework](./rice.md) - Impact consideration
