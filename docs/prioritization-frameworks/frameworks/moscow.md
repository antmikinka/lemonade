# MoSCoW Prioritization Framework

## Overview

MoSCoW is a prioritization technique that categorizes requirements into four buckets: **M**ust have, **S**hould have, **C**ould have, and **W**on't have (this time). It's widely used in Agile and Scrum methodologies for sprint planning and release management.

## Framework Details

### Categories

| Category | Symbol | Weight | Description |
|----------|--------|--------|-------------|
| **Must have** | M | 4 | Non-negotiable requirements. Without these, the project fails. |
| **Should have** | S | 3 | Important but not vital. Can be deferred if necessary. |
| **Could have** | C | 2 | Desirable but not necessary. Nice-to-have features. |
| **Won't have** | W | 1 | Lowest priority. Agreed to exclude from current scope. |

### Scoring Formula

MoSCoW uses **categorical assignment** rather than numeric scoring:

```
Priority Score = Category Weight × Urgency Factor

Where:
- Category Weight: M=4, S=3, C=2, W=1
- Urgency Factor: 1.0 (base), 1.5 (time-sensitive), 0.8 (flexible)
```

### Type Definitions

```typescript
interface MoSCoWResult {
  itemId: string;
  category: 'MUST_HAVE' | 'SHOULD_HAVE' | 'COULD_HAVE' | 'WONT_HAVE';
  categoryWeight: number;
  urgencyFactor: number;
  priorityScore: number;
  rationale: string;
  dependencies: string[];
  effortEstimate?: number;
}

interface MoSCoWCriteria {
  mustHave: string[];    // Criteria for Must Have classification
  shouldHave: string[];  // Criteria for Should Have classification
  couldHave: string[];   // Criteria for Could Have classification
  wontHave: string[];    // Criteria for Won't Have classification
}
```

## Decision Criteria

### Must Have (M)
- Legal or compliance requirement
- Critical safety concern
- Core functionality without which product is useless
- Blocking dependency for other features
- Revenue-critical feature

### Should Have (S)
- Significant business value
- Important to key stakeholders
- Improves usability significantly
- Competitive necessity
- Can workaround temporarily

### Could Have (C)
- Nice to have but not essential
- Low implementation effort
- Enhances user experience
- Requested by minority of users
- Can be added later easily

### Won't Have (W)
- Low business value
- High effort, low impact
- Out of scope for current iteration
- Better alternatives exist
- Insufficient resources

## Auto-Fill Logic

### Pattern Recognition

The auto-fill engine analyzes item descriptions for keywords and patterns:

```typescript
const MUST_HAVE_PATTERNS = [
  /compliance/i, /legal/i, /regulatory/i,
  /critical/i, /blocking/i, /security/i,
  /must/i, /required/i, /mandatory/i
];

const SHOULD_HAVE_PATTERNS = [
  /important/i, /significant/i, /key/i,
  /should/i, /priority/i, /stakeholder/i
];

const COULD_HAVE_PATTERNS = [
  /nice to have/i, /optional/i, /enhancement/i,
  /could/i, /desirable/i, /improve/i
];
```

### Scoring Suggestion Algorithm

```typescript
function suggestMoSCoWCategory(item: BacklogItem): MoSCoWResult {
  const text = `${item.title} ${item.description}`.toLowerCase();

  let category = 'COULD_HAVE'; // Default
  let confidence = 0.5;
  let rationale = 'Default classification based on insufficient data';

  // Check for Must Have indicators
  if (MUST_HAVE_PATTERNS.some(p => p.test(text))) {
    category = 'MUST_HAVE';
    confidence = 0.8;
    rationale = 'Contains critical/legal/security keywords';
  }
  // Check for Should Have indicators
  else if (SHOULD_HAVE_PATTERNS.some(p => p.test(text))) {
    category = 'SHOULD_HAVE';
    confidence = 0.7;
    rationale = 'Indicates significant business importance';
  }

  // Adjust based on metadata
  if (item.metadata.businessValue >= 8) {
    if (category !== 'MUST_HAVE') category = 'SHOULD_HAVE';
    confidence += 0.1;
  }

  return { category, confidence, rationale };
}
```

## Visualization

### Category Distribution Chart

```
┌─────────────────────────────────────────────────────────┐
│              MoSCoW Distribution                         │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Must Have     ████████████████████  8 items (32%)     │
│  Should Have   ██████████████        6 items (24%)     │
│  Could Have    ██████████████████    7 items (28%)     │
│  Won't Have    ████████              4 items (16%)     │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Priority Matrix View

```
┌─────────────────────────────────────────────────────────┐
│                    MoSCoW Matrix                         │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  High Impact ──────────────────────────────────         │
│     │         │              │              │           │
│     │    M    │      M       │      S       │           │
│     │  Must   │    Must      │    Should    │           │
│     │         │              │              │           │
│  ───┼─────────┼──────────────┼──────────────┼───        │
│     │         │              │              │           │
│     │    M    │      S       │      C       │           │
│     │  Must   │    Should    │    Could     │           │
│     │         │              │              │           │
│  ───┼─────────┼──────────────┼──────────────┼───        │
│     │         │              │              │           │
│     │    W    │      W       │      C       │           │
│     │  Won't  │    Won't     │    Could     │           │
│     │         │              │              │           │
│  Low Impact ──────────────────────────────────          │
│                                                         │
│     Low Effort                    High Effort           │
└─────────────────────────────────────────────────────────┘
```

## UI Component Specification

### MoSCoWBoard Component

```typescript
interface MoSCoWBoardProps {
  items: BacklogItem[];
  results: MoSCoWResult[];
  onRecategory: (itemId: string, newCategory: MoSCoWCategory) => void;
  onEditRationale: (itemId: string, rationale: string) => void;
}

interface MoSCoWColumnProps {
  category: MoSCoWCategory;
  items: Array<{ item: BacklogItem; result: MoSCoWResult }>;
  color: string;
  icon: React.ReactNode;
}
```

### Visual Design

| Category | Color | Icon |
|----------|-------|------|
| Must Have | #DC2626 (Red) | 🔴 |
| Should Have | #F59E0B (Amber) | 🟠 |
| Could Have | #10B981 (Green) | 🟢 |
| Won't Have | #6B7280 (Gray) | ⚪ |

## Usage Guidelines

### Best Practices

1. **Limit Must Haves**: Maximum 60% of total scope
2. **Time-box Won't Haves**: Specify when they might be reconsidered
3. **Stakeholder Alignment**: Get agreement on categories before starting
4. **Revisit Regularly**: Categories can change between sprints
5. **Document Rationale**: Always explain why an item is in its category

### Common Pitfalls

| Pitfall | Solution |
|---------|----------|
| Too many Must Haves | Force-rank within Must Haves |
| Vague Won't Haves | Specify conditions for reconsideration |
| Ignoring dependencies | Map dependencies before categorizing |
| One-person exercise | Involve all stakeholders |

## Integration Points

### Export Formats

```json
{
  "framework": "MoSCoW",
  "sessionId": "uuid",
  "generatedAt": "2026-03-18T10:00:00Z",
  "items": [
    {
      "id": "item-1",
      "title": "User Authentication",
      "category": "MUST_HAVE",
      "rationale": "Core functionality required for all other features"
    }
  ],
  "summary": {
    "mustHave": 5,
    "shouldHave": 8,
    "couldHave": 12,
    "wontHave": 3
  }
}
```

### Jira Integration

- Map MoSCoW categories to Jira Priority field
- Must Have → Highest
- Should Have → High
- Could Have → Medium
- Won't Have → Low (or remove from sprint)

## Example Session

### Sample Backlog

| Item | Category | Rationale |
|------|----------|-----------|
| User Login | Must Have | Core functionality |
| Password Reset | Should Have | Important but can workaround |
| Dark Mode | Could Have | Nice to have UX enhancement |
| AI Chatbot | Won't Have | Out of scope for MVP |

### Calculation Example

```
Item: User Login
- Category: MUST_HAVE
- Category Weight: 4
- Urgency Factor: 1.5 (blocking other features)
- Priority Score: 4 × 1.5 = 6.0

Item: Dark Mode
- Category: COULD_HAVE
- Category Weight: 2
- Urgency Factor: 1.0 (no time pressure)
- Priority Score: 2 × 1.0 = 2.0
```

---

**Related Frameworks:**
- [Value vs Effort Matrix](./value-effort.md) - For effort-based prioritization
- [P0-P4 Priority Hierarchy](./p0p4.md) - Similar hierarchical approach
