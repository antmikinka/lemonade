import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
/**
 * Main App component for Prioritization Frameworks Web Application.
 * Serves as the root component and layout container.
 *
 * Phase 4: Integrated React UI components for prioritization frameworks.
 */
import { useState, useCallback } from 'react';
// Common components
import { Header, Card } from './components/common';
// Framework input forms
import { RICEInputForm, MoSCoWInputForm, KanoInputForm, ValueEffortInputForm, ICEInputForm, EisenhowerInputForm, P0P4InputForm, WSJFInputForm, } from './components/framework';
// Results components
import { ResultsCard, FrameworkMatrix, ComparisonTable, PriorityRanking, } from './components/results';
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
function App() {
    // State management
    const [selectedFramework, setSelectedFramework] = useState('RICE');
    const [isDarkTheme, setIsDarkTheme] = useState(true);
    const [selectedItems, setSelectedItems] = useState([]);
    const [showResults, setShowResults] = useState(false);
    // Sample results for demonstration
    const [sampleResults] = useState({
        RICE: {
            framework: 'RICE',
            score: 45.5,
            details: { reach: 100, impact: 2, confidence: 0.8, effort: 3 },
        },
    });
    // Handlers
    const handleFrameworkChange = useCallback((framework) => {
        setSelectedFramework(framework);
        setShowResults(false);
    }, []);
    const handleThemeToggle = useCallback(() => {
        setIsDarkTheme((prev) => !prev);
    }, []);
    const handleItemSelect = useCallback((itemId) => {
        setSelectedItems((prev) => prev.includes(itemId)
            ? prev.filter((id) => id !== itemId)
            : [...prev, itemId]);
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
                return _jsx(RICEInputForm, { onSubmit: handleCalculate });
            case 'MoSCoW':
                return _jsx(MoSCoWInputForm, { onSubmit: handleCalculate });
            case 'Kano':
                return _jsx(KanoInputForm, { onSubmit: handleCalculate });
            case 'ValueEffort':
                return _jsx(ValueEffortInputForm, { onSubmit: handleCalculate });
            case 'ICE':
                return _jsx(ICEInputForm, { onSubmit: handleCalculate });
            case 'Eisenhower':
                return _jsx(EisenhowerInputForm, { onSubmit: handleCalculate });
            case 'P0P4':
                return _jsx(P0P4InputForm, { onSubmit: handleCalculate });
            case 'WSJF':
                return _jsx(WSJFInputForm, { onSubmit: handleCalculate });
            default:
                return null;
        }
    };
    return (_jsxs("div", { className: "app", children: [_jsx(Header, { selectedFramework: selectedFramework, onFrameworkChange: handleFrameworkChange, onThemeToggle: handleThemeToggle, isDarkTheme: isDarkTheme }), _jsxs("main", { className: "app-main", children: [_jsxs("div", { className: "app-content-grid", children: [_jsxs("div", { className: "app-column app-column-main", children: [_jsx("section", { className: "app-section", children: renderFrameworkForm() }), showResults && (_jsx("section", { className: "app-section", children: _jsx(ResultsCard, { result: sampleResults[selectedFramework] || sampleResults.RICE, itemTitle: "Sample Feature", itemDescription: "This is a demonstration result", showDetails: true }) }))] }), _jsxs("div", { className: "app-column app-column-side", children: [_jsx("section", { className: "app-section", children: _jsx(BacklogList, { title: "Backlog Items", items: sampleBacklogItems, selectedItems: selectedItems, onItemSelect: (item) => handleItemSelect(item.id), onAddItem: handleAddItem, enableReorder: true, showDescriptions: true, showCategories: true, limit: 5 }) }), _jsx("section", { className: "app-section", children: _jsx(PriorityRanking, { title: "Priority Ranking", framework: selectedFramework, items: [
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
                                            ], onItemSelected: (id) => console.log('Selected:', id), highlightTop: 3 }) })] })] }), _jsx("section", { className: "app-section app-section-full", children: _jsx(Card, { title: "Value vs Effort Matrix", variant: "default", children: _jsx(FrameworkMatrix, { type: "value-effort", items: [
                                    { id: '1', title: 'Auth', x: 3, y: 8 },
                                    { id: '2', title: 'Dashboard', x: 7, y: 7 },
                                    { id: '3', title: 'Mobile', x: 5, y: 6 },
                                    { id: '4', title: 'Docs', x: 2, y: 4 },
                                ], onItemClick: (item) => console.log('Matrix item clicked:', item) }) }) }), _jsx("section", { className: "app-section app-section-full", children: _jsx(ComparisonTable, { title: "Feature Comparison", items: [
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
                            ], selectedItemId: selectedItems[0], onItemSelected: (id) => handleItemSelect(id) }) })] }), _jsx("footer", { className: "app-footer", children: _jsxs("p", { children: ["\u00A9 ", new Date().getFullYear(), " Prioritization Frameworks. All rights reserved."] }) })] }));
}
export default App;
