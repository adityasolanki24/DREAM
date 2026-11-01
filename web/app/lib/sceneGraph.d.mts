import type { GraphDataLite, GraphNode, HomeMemory, Snapshot } from '../types';

export const MAX_NODES: number;
export const NODE_TYPES: GraphNode['type'][];
export const LINK_TYPES: string[];
export function buildSceneGraphData(snapshot: Snapshot | null | undefined, home?: HomeMemory | null): GraphDataLite;
export function reuseNodes(cache: Map<string, GraphNode>, graph: GraphDataLite): GraphDataLite;
export function applyFilters(
  graph: GraphDataLite,
  filters?: { nodeTypes?: string[] | null; linkTypes?: string[] | null; labels?: string[] | null },
): GraphDataLite;
export function graphLabels(graph: GraphDataLite): string[];
export function neighbours(graph: GraphDataLite, id: string | null | undefined): Set<string>;
export function boundGraph(graph: GraphDataLite): GraphDataLite;
