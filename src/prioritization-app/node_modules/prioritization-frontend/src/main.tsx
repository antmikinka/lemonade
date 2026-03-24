/**
 * React application entry point.
 * Initializes the React DOM renderer and renders the root App component.
 */

import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import App from './App';
import './styles/global.css';

// Get root element and create React root
const rootElement = document.getElementById('root');

if (!rootElement) {
  throw new Error('Failed to find root element. Ensure index.html has a div with id="root"');
}

const root = createRoot(rootElement);

/**
 * Render the application in StrictMode for development benefits.
 * StrictMode helps identify potential problems in the application.
 */
root.render(
  <StrictMode>
    <App />
  </StrictMode>
);
