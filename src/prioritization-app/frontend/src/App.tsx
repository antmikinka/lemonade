/**
 * Main App component for Prioritization Frameworks Web Application.
 * Serves as the root component and layout container.
 *
 * Phase 4: Integrated React UI components for prioritization frameworks.
 */

import { useState, useCallback } from 'react';
import { FrameworkType } from './services/prioritization/types';

// Common components
import { Header, Card } from './components/common';

// Framework input forms
import {
  RICEInputForm,
  MoSCoWInputForm,
  KanoInputForm,
  ValueEffortInputForm,
  ICEInputForm,
  EisenhowerInputForm,
  P0P4InputForm,
  WSJFInputForm,
} from './components/framework';

// Results components
import {
  ResultsCard,
  FrameworkMatrix,
  ComparisonTable,
  PriorityRanking,
} from './components/results';

// Backlog components
import { BacklogList } from './components/backlog';

// Sample backlog items for demonstration
const sampleBacklogItems = [
  {
    id: '1',
    title: 'Implement User Authentication',
    description: 'Add OAuth2 authentication for secure user login',
    category: 'Security',
    createdAt: new Date('2024-01-15'),
  },
  {
    id: '2',
    title: 'Build Dashboard Analytics',
    description: 'Create real-time analytics dashboard for user metrics',
    category: 'Features',
    createdAt: new Date('2024-01-20'),
  },
  {
    id: '3',
    title: 'Mobile App Optimization',
    description: 'Improve performance on mobile devices',
    category: 'Performance',
    createdAt: new Date('2024-02-01'),
  },
  {
    id: '4',
    title: 'API Documentation',
    description: 'Create comprehensive API documentation for developers',
    category: 'Documentation',
    createdAt: new Date('2024-02-10'),
  },
];

/**
 * Main application component.
 * @returns Root React component for the Prioritization Frameworks application
 */
function App(): React.JSX.Element {
  // State management
  const [selectedFramework, setSelectedFramework] = useState<FrameworkType>('RICE');
  const [isDarkTheme, setIsDarkTheme] = useState<boolean>(true);
  const [selectedItems, setSelectedItems] = useState<string[]>([]);
  const [showResults, setShowResults] = useState<boolean>(false);

  // Sample results for demonstration
  const [sampleResults] = useState({
    RICE: {
      framework: 'RICE' as FrameworkType,
      score: 45.5,
      details: { reach: 100, impact: 2, confidence: 0.8, effort: 3 },
    },
  });

  // Handlers
  const handleFrameworkChange = useCallback((framework: FrameworkType) => {
    setSelectedFramework(framework);
    setShowResults(false);
  }, []);

  const handleThemeToggle = useCallback(() => {
    setIsDarkTheme((prev) => !prev);
  }, []);

  const handleItemSelect = useCallback((itemId: string) => {
    setSelectedItems((prev) =>
      prev.includes(itemId)
        ? prev.filter((id) => id !== itemId)
        : [...prev, itemId]
    );
  }, []);

  const handleAddItem = useCallback(() => {
    // Placeholder for add item functionality
    console.log('Add new item clicked');
  }, []);

  const handleCalculate = useCallback(() => {
    setShowResults(true);
  }, []);

  // Render the appropriate input form based on selected framework
  const renderFrameworkForm = () => {
    switch (selectedFramework) {
      case 'RICE':
        return <RICEInputForm onSubmit={handleCalculate} />;
      case 'MoSCoW':
        return <MoSCoWInputForm onSubmit={handleCalculate} />;
      case 'Kano':
        return <KanoInputForm onSubmit={handleCalculate} />;
      case 'ValueEffort':
        return <ValueEffortInputForm onSubmit={handleCalculate} />;
      case 'ICE':
        return <ICEInputForm onSubmit={handleCalculate} />;
      case 'Eisenhower':
        return <EisenhowerInputForm onSubmit={handleCalculate} />;
      case 'P0P4':
        return <P0P4InputForm onSubmit={handleCalculate} />;
      case 'WSJF':
        return <WSJFInputForm onSubmit={handleCalculate} />;
      default:
        return null;
    }
  };

  return (
    <div className="app">
      {/* Header with framework selector */}
      <Header
        selectedFramework={selectedFramework}
        onFrameworkChange={handleFrameworkChange}
        onThemeToggle={handleThemeToggle}
        isDarkTheme={isDarkTheme}
      />

      {/* Main Content Area */}
      <main className="app-main">
        {/* Two-column layout */}
        <div className="app-content-grid">
          {/* Left Column: Input Panel and Results */}
          <div className="app-column app-column-main">
            {/* Framework Input Panel */}
            <section className="app-section">
              {renderFrameworkForm()}
            </section>

            {/* Results Panel */}
            {showResults && (
              <section className="app-section">
                <ResultsCard
                  result={sampleResults[selectedFramework as keyof typeof sampleResults] || sampleResults.RICE}
                  itemTitle="Sample Feature"
                  itemDescription="This is a demonstration result"
                  showDetails
                />
              </section>
            )}
          </div>

          {/* Right Column: Backlog and Additional Panels */}
          <div className="app-column app-column-side">
            {/* Backlog List */}
            <section className="app-section">
              <BacklogList
                title="Backlog Items"
                items={sampleBacklogItems}
                selectedItems={selectedItems}
                onItemSelect={(item) => handleItemSelect(item.id)}
                onAddItem={handleAddItem}
                enableReorder
                showDescriptions
                showCategories
                limit={5}
              />
            </section>

            {/* Priority Ranking Preview */}
            <section className="app-section">
              <PriorityRanking
                title="Priority Ranking"
                framework={selectedFramework}
                items={[
                  {
                    id: '1',
                    title: 'User Authentication',
                    rank: 1,
                    score: 95.5,
                    category: 'P0',
                  },
                  {
                    id: '2',
                    title: 'Dashboard Analytics',
                    rank: 2,
                    score: 82.3,
                    category: 'P1',
                  },
                  {
                    id: '3',
                    title: 'Mobile Optimization',
                    rank: 3,
                    score: 71.8,
                    category: 'P2',
                  },
                ]}
                onItemSelected={(id) => console.log('Selected:', id)}
                highlightTop={3}
              />
            </section>
          </div>
        </div>

        {/* Full-width section for matrix visualization */}
        <section className="app-section app-section-full">
          <Card title="Value vs Effort Matrix" variant="default">
            <FrameworkMatrix
              type="value-effort"
              items={[
                { id: '1', title: 'Auth', x: 3, y: 8 },
                { id: '2', title: 'Dashboard', x: 7, y: 7 },
                { id: '3', title: 'Mobile', x: 5, y: 6 },
                { id: '4', title: 'Docs', x: 2, y: 4 },
              ]}
              onItemClick={(item) => console.log('Matrix item clicked:', item)}
            />
          </Card>
        </section>

        {/* Comparison Table Section */}
        <section className="app-section app-section-full">
          <ComparisonTable
            title="Feature Comparison"
            items={[
              {
                id: '1',
                title: 'User Authentication',
                description: 'OAuth2 implementation',
                results: { RICE: sampleResults.RICE },
              },
              {
                id: '2',
                title: 'Dashboard Analytics',
                description: 'Real-time metrics',
              },
              {
                id: '3',
                title: 'Mobile Optimization',
                description: 'Performance improvements',
              },
            ]}
            selectedItemId={selectedItems[0]}
            onItemSelected={(id) => handleItemSelect(id)}
          />
        </section>
      </main>

      {/* Footer */}
      <footer className="app-footer">
        <p>
          &copy; {new Date().getFullYear()} Prioritization Frameworks. All rights reserved.
        </p>
      </footer>
    </div>
  );
}

export default App;
