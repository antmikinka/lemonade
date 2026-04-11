# Prioritization Frameworks - Testing Suite

This directory contains the comprehensive testing suite for the Prioritization Frameworks Web Application (Phase 7).

## Overview

The testing suite includes:

1. **Frontend Unit Tests** (Vitest + React Testing Library)
2. **Backend Integration Tests** (Vitest + Supertest)
3. **E2E Tests** (Playwright)

## Directory Structure

```
prioritization-app/
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── common/
│   │   │   │   ├── Button.test.tsx
│   │   │   │   ├── Input.test.tsx
│   │   │   │   └── Select.test.tsx
│   │   │   ├── framework/
│   │   │   │   └── RICEInputForm.test.tsx
│   │   │   ├── results/
│   │   │   │   └── ResultsCard.test.tsx
│   │   │   └── backlog/
│   │   │       └── BacklogItem.test.tsx
│   │   └── tests/
│   │       └── setup.ts
│   ├── vite.config.ts
│   └── package.json
├── backend/
│   ├── src/
│   ├── tests/
│   │   ├── api/
│   │   │   ├── sessions.test.ts
│   │   │   └── prioritization.test.ts
│   │   ├── services/
│   │   │   └── SessionService.test.ts
│   │   └── setup.ts
│   ├── vitest.config.ts
│   └── package.json
├── e2e/
│   ├── tests/
│   │   ├── framework-selection.spec.ts
│   │   ├── prioritization-flow.spec.ts
│   │   ├── backlog-management.spec.ts
│   │   └── agent-pipeline.spec.ts
│   ├── playwright.config.ts
│   └── package.json
└── package.json
```

## Installation

### Prerequisites

- Node.js >= 18.0.0
- npm >= 9.0.0

### Install Dependencies

```bash
# Install all workspace dependencies
cd prioritization-app
npm install

# Install Playwright browsers
npm run e2e:install
```

## Running Tests

### All Tests

```bash
# Run all tests (frontend + backend + e2e)
npm run test:all
```

### Frontend Tests

```bash
# Run frontend tests in watch mode
npm run test:frontend

# Run frontend tests once
npm run test:run --workspace=frontend

# Run with coverage
npm run test:coverage --workspace=frontend

# Run with UI
npm run test:ui --workspace=frontend
```

### Backend Tests

```bash
# Run backend tests in watch mode
npm run test:backend

# Run backend tests once
npm run test:run --workspace=backend

# Run with coverage
npm run test:coverage --workspace=backend

# Run only API tests
npm run test:api --workspace=backend
```

### E2E Tests

```bash
# Run E2E tests on Chromium
npm run test:e2e:chromium

# Run E2E tests with UI
npm run test:e2e:ui

# Run E2E tests in debug mode
npm run test:e2e:debug

# Run E2E tests on specific browser
npm run test:e2e:firefox
npm run test:e2e:webkit

# Generate and view HTML report
npm run e2e:report
```

## Test Coverage

### Coverage Thresholds

The project enforces the following coverage thresholds:

| Metric      | Threshold |
|-------------|-----------|
| Statements  | 80%       |
| Branches    | 75%       |
| Functions   | 80%       |
| Lines       | 80%       |

### View Coverage Reports

```bash
# Run tests with coverage
npm run test:coverage

# Frontend coverage report (HTML)
open frontend/coverage/index.html

# Backend coverage report (HTML)
open backend/coverage/index.html
```

## Test Files Overview

### Frontend Component Tests

#### Button.test.tsx
Tests for the Button component:
- Rendering with children
- Click event handling
- Disabled state
- Variant classes (primary, secondary, danger, ghost)
- Size classes (sm, md, lg)
- Loading state
- Icon rendering (left, right)
- Accessibility attributes

#### Input.test.tsx
Tests for the Input component:
- Rendering with label
- Different input types (text, email, number, password, etc.)
- Error state display
- Helper text display
- Required field indication
- Addon rendering (left, right)
- Value changes and onChange handler
- Accessibility attributes

#### Select.test.tsx
Tests for the Select component:
- Options rendering
- Selection and onChange
- Disabled state
- Error state
- Helper text
- Grouped options (optgroup)
- Placeholder handling
- Accessibility attributes

#### RICEInputForm.test.tsx
Tests for the RICE Input Form:
- All inputs render (Reach, Impact, Confidence, Effort)
- Form submission
- Validation
- onChange handler
- Submit button state
- Loading state

#### ResultsCard.test.tsx
Tests for the Results Card:
- Score display
- Category badge rendering
- Framework name display
- Details breakdown
- Custom actions
- Different framework types

#### BacklogItem.test.tsx
Tests for the Backlog Item:
- Checkbox toggle
- Drag handle presence
- Content rendering
- Selection state
- Drag and drop handlers
- Category badge
- Custom actions

### Backend Tests

#### API Tests (sessions.test.ts)
Integration tests for session management endpoints:
- POST /api/v1/sessions (create)
- GET /api/v1/sessions (list all)
- GET /api/v1/sessions/:id (get by ID)
- PUT /api/v1/sessions/:id (update)
- DELETE /api/v1/sessions/:id (delete)
- POST /api/v1/sessions/:id/items (add item)
- PUT /api/v1/sessions/:id/items/:itemId (update item)
- DELETE /api/v1/sessions/:id/items/:itemId (remove item)

#### API Tests (prioritization.test.ts)
Integration tests for prioritization endpoints:
- POST /api/v1/prioritize (single item)
- POST /api/v1/prioritize/bulk (multiple items)
- All 8 frameworks (RICE, MoSCoW, ICE, Eisenhower, P0P4, WSJF, ValueEffort, Kano)
- Auto-fill suggestions
- Error handling

#### Service Tests (SessionService.test.ts)
Unit tests for SessionService:
- create()
- findAll()
- findById()
- update()
- delete()
- addItem()
- updateItem()
- removeItem()
- getStats()
- clearAll()
- Event subscription

### E2E Tests

#### framework-selection.spec.ts
Tests for framework selection functionality:
- All 8 frameworks display in selector
- Framework switching changes input form
- Framework-specific fields visibility
- Helper text display

#### prioritization-flow.spec.ts
Tests for complete prioritization workflow:
- RICE prioritization end-to-end
- Adding multiple items to backlog
- Priority ranking display
- Results validation
- Form validation

#### backlog-management.spec.ts
Tests for backlog management:
- Checkbox selection
- Bulk actions
- Drag and drop reordering
- Item deletion
- Empty state display

#### agent-pipeline.spec.ts
Tests for agent pipeline:
- Agent panel toggle
- Running pipeline
- Progress display
- Completion handling
- Available agents display

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: '18'

      - name: Install dependencies
        run: npm ci

      - name: Install Playwright browsers
        run: npx playwright install --with-deps chromium

      - name: Run frontend tests
        run: npm run test:run --workspace=frontend

      - name: Run backend tests
        run: npm run test:run --workspace=backend

      - name: Run E2E tests
        run: npm run test:e2e:chromium

      - name: Upload coverage reports
        uses: codecov/codecov-action@v3
        with:
          files: frontend/coverage/coverage-final.json,backend/coverage/coverage-final.json
```

## Troubleshooting

### Common Issues

#### Frontend Tests Fail
```bash
# Clear node modules and reinstall
rm -rf frontend/node_modules
npm install --workspace=frontend
```

#### Backend Tests Fail
```bash
# Check for port conflicts
lsof -i :3001

# Clear test cache
rm -rf backend/node_modules/.vitest
```

#### E2E Tests Fail
```bash
# Reinstall Playwright browsers
npx playwright install --with-deps

# Run in headed mode for debugging
npm run test:e2e:headed
```

### Test Debugging

#### Frontend
```bash
# Run with UI for interactive debugging
npm run test:ui --workspace=frontend
```

#### Backend
```bash
# Run with verbose output
npm run test:run --workspace=backend -- --reporter=verbose
```

#### E2E
```bash
# Run with Playwright debugger
npm run test:e2e:debug

# Run with browser visible
npm run test:e2e:headed
```

## Best Practices

### Writing Tests

1. **Use descriptive test names** - Follow the pattern: `should [expected behavior] when [condition]`
2. **Arrange-Act-Assert** - Structure tests clearly
3. **Test one thing per test** - Keep tests focused
4. **Use beforeEach for setup** - Avoid test interdependence
5. **Mock external dependencies** - Isolate units under test

### Test Data

1. **Use factories for test data** - Create helper functions
2. **Clean up after tests** - Use afterEach hooks
3. **Use realistic data** - Match production scenarios

### E2E Tests

1. **Use data-testid attributes** - More stable than CSS selectors
2. **Wait for elements properly** - Use Playwright's auto-wait
3. **Test user journeys** - Not just individual elements
4. **Run in multiple browsers** - Catch browser-specific issues

## Coverage Reports

After running `npm run test:coverage`, HTML reports are generated:

- Frontend: `frontend/coverage/index.html`
- Backend: `backend/coverage/index.html`

Open these in a browser to explore detailed coverage information.

## Performance

### Test Execution Time

Typical execution times:

| Test Suite | Time |
|------------|------|
| Frontend Unit | ~5s |
| Backend Integration | ~10s |
| E2E (Chromium) | ~30s |

### Optimizing Slow Tests

1. **Parallel execution** - Tests run in parallel by default
2. **Selective running** - Use `--grep` to run specific tests
3. **Shard tests** - Split tests across multiple runners in CI

## Contributing

When adding new features, ensure:

1. **Unit tests** cover component logic
2. **Integration tests** cover API endpoints
3. **E2E tests** cover critical user journeys
4. **Coverage thresholds** are maintained

Run all tests before submitting:

```bash
npm run test:all
```
