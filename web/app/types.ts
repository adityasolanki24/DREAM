/** Shapes returned by DREAM's authenticated phone API (dream/runtime/phone_api.py). */
export type Task = {
  id: string;
  text: string;
  state: string;
  priority: number;
  created: number;
  updated: number;
  result?: string | null;
  steps: number;
  suspensions: number;
  resume: 'auto' | 'ask';
  // Which parser and model ran the task. Absent on tasks created before model choice existed.
  meta?: { provider?: string; model?: string | null };
};

/** One entry of snapshot.provider_choices. Older snapshots send only {id, label}; see normaliseProviderChoices. */
export type ProviderChoice = {
  id: string;
  label: string;
  available: boolean;
  reason: string | null;
  models: string[];
  default_model: string | null;
  supports: string[];
};

export type MapData = {
  cells: number[][];
  resolution_m: number;
  origin_m: number[];
  axes: string;
  odometry: string;
};

export type Candidate = {
  id: number;
  color?: string;
  position: number[];
  extent: number[];
  views: number;
};

export type HomeMemory = {
  available: boolean;
  source?: string;
  error?: string;
  objects?: {
    id: string;
    label: string;
    map_id: string;
    last_seen: number;
    position: number[];
    confidence_at_observation: number;
    observation_source: string;
  }[];
};

export type Snapshot = {
  provider: string;
  provider_label: string;
  provider_choices: ProviderChoice[];
  environment?: string;
  robot_time_s: number;
  odometry: number[];
  stop_reason: string | null;
  camera: { data_url?: string; time_s: number; transport?: string } | null;
  map: MapData | null;
  objects: Candidate[];
  persistent_memory: HomeMemory | null;
  task_graph: {
    kind: string;
    task_id: string;
    nodes: { id: string; label: string; args: Record<string, unknown> }[];
    edges: { source: string; target: string }[];
  } | null;
  outcome: { state: string; reason: string | null } | null;
  history: Task[];
};

export type Status = { running: string | null; robot_time_s?: number; stop_reason?: string | null };

export type Link = 'idle' | 'connecting' | 'online' | 'offline' | 'unauthorised';

declare global {
  interface Window {
    // Defined by the inline theme script in layout.tsx.
    dreamTheme?: { get(): 'system' | 'light' | 'dark'; set(mode: 'system' | 'light' | 'dark'): void };
  }
}
