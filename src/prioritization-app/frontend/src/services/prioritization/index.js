/**
 * Prioritization Frameworks Module
 *
 * This module provides implementations of various prioritization frameworks
 * used in product management and software development. Each framework is
 * implemented as a calculator following the Strategy pattern.
 *
 * @module prioritization
 *
 * @example
 * ```typescript
 * import {
 *   RICECalculator,
 *   MoSCoWCalculator,
 *   type PrioritizationItem,
 *   type RICEInput
 * } from './services/prioritization';
 *
 * // Create calculator instances
 * const riceCalculator = new RICECalculator();
 * const moscowCalculator = new MoSCoWCalculator();
 *
 * // Calculate RICE score
 * const riceResult = riceCalculator.calculate({
 *   reach: 500,
 *   impact: 2,
 *   confidence: 80,
 *   effort: 3
 * });
 *
 * // Calculate MoSCoW category
 * const moscowResult = moscowCalculator.calculate({
 *   businessValue: 'critical',
 *   legalRequirement: true,
 *   customerRequest: false,
 *   riskIfNotDelivered: 'high'
 * });
 * ```
 */
// Core types
export * from './types';
// Calculators
export * from './calculators/RICECalculator';
export * from './calculators/MoSCoWCalculator';
// Calculator instances (factory exports)
export { createRICECalculator } from './calculators/RICECalculator';
export { createMoSCoWCalculator } from './calculators/MoSCoWCalculator';
