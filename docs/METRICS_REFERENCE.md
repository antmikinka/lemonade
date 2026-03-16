# Metrics Reference Guide

## Development KPIs

### Velocity

**Definition**: The rate at which the team completes work, measured in story points per sprint.

**Formula**:
```
Velocity = Sum of story points completed in sprint
Average Velocity = Sum of velocities / Number of sprints
```

**Interpretation**:
| Trend | Meaning | Action |
|-------|---------|--------|
| Increasing | Team productivity improving | Maintain current processes |
| Stable | Consistent delivery | Use for sprint planning |
| Decreasing | Potential blockers or burnout | Investigate root causes |

**Target Range**: 15-25% improvement quarter-over-quarter is healthy.

---

### Cycle Time

**Definition**: The time elapsed from when work starts on an item until it is delivered.

**Formula**:
```
Cycle Time = Completion Date - Start Date
```

**Components**:
- **Active Time**: Time spent actually working
- **Wait Time**: Time spent blocked or waiting for review

**Interpretation**:
| Value (days) | Assessment |
|--------------|------------|
| < 3 | Excellent |
| 3-7 | Good |
| 7-14 | Needs improvement |
| > 14 | Critical - investigate bottlenecks |

**Benchmarks**:
- Industry average: 7-10 days
- High-performing teams: 2-4 days

---

### PR Metrics

#### Time to First Review

**Definition**: Time from PR creation to first review action (comment, approval, or changes requested).

**Formula**:
```
Time to First Review = First Review Timestamp - PR Creation Timestamp
```

**Target**: < 24 hours

**Interpretation**:
| Value (hours) | Assessment |
|---------------|------------|
| < 8 | Excellent |
| 8-24 | Good |
| 24-48 | Needs improvement |
| > 48 | Critical - review bottleneck |

#### Time to Merge

**Definition**: Total time from PR creation to merge.

**Formula**:
```
Time to Merge = Merge Timestamp - PR Creation Timestamp
```

**Target**: < 48 hours

#### Merge Rate

**Definition**: Percentage of PRs that are successfully merged.

**Formula**:
```
Merge Rate = (Merged PRs / Total PRs) * 100
```

**Target**: > 85%

**Low merge rate indicates**:
- Too many experimental PRs
- Inadequate pre-PR discussion
- Changing requirements

#### Rework Rate

**Definition**: Percentage of PRs that have "changes requested" reviews.

**Formula**:
```
Rework Rate = (PRs with Changes Requested / Total PRs) * 100
```

**Target**: 20-40%

**Interpretation**:
- < 20%: May indicate insufficient review rigor
- 20-40%: Healthy review process
- > 40%: Communication or quality issues

---

### Code Quality Metrics

#### Code Churn

**Definition**: Percentage of code that is rewritten or significantly modified within 2 weeks of completion.

**Formula**:
```
Code Churn = (Items with Significant Follow-up Changes / Total Items) * 100
```

**Target**: < 15%

**High churn indicates**:
- Unclear requirements
- Technical debt
- Insufficient design review

#### Average PR Size

**Definition**: Average number of lines changed per PR.

**Formula**:
```
Average PR Size = Total Lines Changed / Number of PRs
```

**Target**: < 400 lines

**Interpretation**:
| Size (lines) | Assessment |
|--------------|------------|
| < 200 | Excellent - easy to review |
| 200-400 | Good |
| 400-800 | Large - consider splitting |
| > 800 | Too large - high risk |

#### Review Depth

**Definition**: Average number of review comments per PR.

**Formula**:
```
Review Depth = Total Review Comments / Number of PRs
```

**Target**: 3-8 comments per PR

**Interpretation**:
- < 2: May indicate superficial reviews
- 3-8: Healthy review engagement
- > 10: PR may be too complex or poorly prepared

#### Defect Rate

**Definition**: Percentage of issues that are reopened after being closed.

**Formula**:
```
Defect Rate = (Reopened Issues / Total Closed Issues) * 100
```

**Target**: < 5%

---

### Throughput Metrics

#### Work in Progress (WIP)

**Definition**: Number of items currently in progress or review.

**Target**: Limit to 2-3 items per developer

**Formula** (Little's Law):
```
Throughput = WIP / Cycle Time
```

**Interpretation**:
- High WIP with low throughput: Context switching overhead
- Low WIP with low throughput: Blockers or capacity issues
- Optimal: WIP matches team capacity

---

## ROI Metrics

### ROI Ratio

**Definition**: The ratio of net return to investment.

**Formula**:
```
ROI Ratio = (Total Impact - Total Investment) / Total Investment
```

**Interpretation**:
| Ratio | Category |
|-------|----------|
| > 2.0 | High ROI |
| 0.5-2.0 | Medium ROI |
| 0-0.5 | Low ROI |
| < 0 | Strategic Investment |

---

### Payback Period

**Definition**: Time required to recover the initial investment.

**Formula**:
```
Payback Period (days) = Total Investment / (Monthly Benefit / 30)
```

**Target**: < 180 days (6 months)

---

### Net Present Value (NPV)

**Definition**: Present value of future benefits minus initial investment.

**Formula**:
```
NPV = -Initial Investment + Σ (Annual Benefit / (1 + r)^t)
```
Where r = discount rate (10%), t = year (1-3)

**Target**: > 0

---

## Industry Relevance Metrics

### Overall Relevance Score

**Definition**: Composite score (0-100) measuring alignment with industry trends.

**Dimensions**:
1. **Technical Innovation** (25%): Use of cutting-edge technology
2. **Market Alignment** (25%): Addressing market demands
3. **Competitive Parity** (20%): Matching competitor capabilities
4. **Future Proofing** (20%): Sustainable technology choices
5. **Ecosystem Integration** (10%): Compatibility with standards

**Calculation**:
```
Overall = Σ (Dimension Score * Weight)
```

**Benchmarks**:
| Score | Assessment |
|-------|------------|
| 80-100 | Industry Leader |
| 65-79 | On Par |
| 50-64 | Lagging |
| < 50 | At Risk |

---

## Alert Thresholds

### Critical Alerts (Immediate Action)

| Metric | Threshold |
|--------|-----------|
| Velocity drop | > 30% from average |
| Cycle time increase | > 50% from baseline |
| PR review time | > 72 hours |
| Code churn | > 30% |
| Defect rate | > 15% |

### Warning Alerts (Monitor Closely)

| Metric | Threshold |
|--------|-----------|
| Velocity drop | 15-30% from average |
| Cycle time increase | 25-50% from baseline |
| PR review time | 48-72 hours |
| Code churn | 20-30% |
| WIP limit exceeded | > 3 items per developer |

---

## Data Collection Methods

### Automated Collection

| Metric | Data Source |
|--------|-------------|
| Cycle Time | GitHub issue/PR timestamps |
| PR Metrics | GitHub PR API |
| Code Churn | Git history analysis |
| Throughput | GitHub status changes |

### Manual Input Required

| Metric | Input Method |
|--------|--------------|
| Story Points | Team estimation or auto-estimated |
| ROI Category | Manual tagging on work items |
| Impact Score | Human adjustment of AI baseline |
| Strategic Tags | Manual assignment |

---

## Calculation Frequency

| Metric | Update Frequency |
|--------|------------------|
| Velocity | Per sprint (14 days) |
| Cycle Time | Real-time (per completion) |
| PR Metrics | Real-time (per PR event) |
| Code Quality | Daily batch |
| ROI | Per initiative milestone |
| Industry Relevance | Weekly |

---

## API Endpoints for Metrics

```typescript
// Get development KPIs
GET /api/v1/metrics/development?period=last_30_days

// Get ROI analysis
GET /api/v1/metrics/roi?initiativeId=xxx

// Get industry relevance
GET /api/v1/metrics/relevance

// Get AI insights
GET /api/v1/insights

// Trigger metrics recalculation
POST /api/v1/metrics/recalculate
```

---

## Metric Visualization Guidelines

### Chart Types

| Metric | Recommended Chart |
|--------|-------------------|
| Velocity trend | Line chart or bar chart |
| Cycle time distribution | Histogram or box plot |
| PR metrics | Gauge or radar chart |
| ROI by category | Stacked bar or treemap |
| Work item flow | Sankey or funnel |

### Color Coding

| Status | Color |
|--------|-------|
| Good/On Track | #10b981 (green) |
| Warning/At Risk | #f59e0b (amber) |
| Critical/Off Track | #ef4444 (red) |
| Neutral | #6b7280 (gray) |
