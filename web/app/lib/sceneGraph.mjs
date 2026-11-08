/** DREAM scene/memory payloads → force-graph data. Pure functions, no React, no rendering.
 *
 * Deterministic and identity-stable: nodes carry ids that persist across snapshots
 * (`mem:<kind>:<id>` from home memory, `session:<candidate id>` for per-session spatial
 * memory), so the renderer can add/remove incrementally instead of rebuilding, and a
 * selection survives updates.
 *
 * Honesty rules, by design:
 * - Only relationships that are actually stored in home memory become links: room
 *   connectivity comes from the place_edges table and object→room links come from the
 *   place_id the runtime recorded with the observation. Session candidates have identity
 *   but NO stored relationships, so the session graph is nodes only — links are never
 *   inferred from layout, proximity or colour.
 * - Force-directed positions are layout, not physics. Real coordinates live in node meta
 *   (with their frame: persistent map frame vs per-session odometry frame) and are shown
 *   in the detail inspector, never used as node positions.
 * - The semantic graph (snapshot.semantic_graph) is drawn exactly as stated: one link per
 *   edge, typed by its status, with relation/provenance/evidence/score and score_kind in meta.
 *   Scores are heuristic weights, search-mass shares or counts, never probabilities.
 * - Nodes whose map frame matches the featured map are grouped; other maps are dropped
 *   with a note rather than mixed (different map frames are different coordinate systems).
 */

export const MAX_NODES = 400;  // bounded render: rooms/places first, then the best-evidence objects

export const NODE_TYPES = ['room', 'place', 'object', 'region', 'category', 'class', 'unexplored', 'memory'];
export const LINK_TYPES = [
  'connects', 'located-in',
  // semantic-graph edge statuses: each keeps a different kind of claim apart
  'observed', 'verified', 'inferred', 'hypothesis', 'knowledge', 'remembered',
];

const byId = (a, b) => String(a.id).localeCompare(String(b.id));
const linkOrder = (l) => `${l.source}\u2192${l.target}:${l.type}`;

/** One displayed link per (source, target, kind): stored rows can repeat, and both-way rows resolve to one edge. */
function dedupeLinks(links) {
  const out = new Map();
  for (const l of links) {
    const key = `${l.source}\u0000${l.target}\u0000${l.type}`;
    if (!out.has(key)) out.set(key, l);
  }
  return [...out.values()].sort((a, b) => linkOrder(a).localeCompare(linkOrder(b)));
}

/** Stored connectivity edges are written both ways; one displayed edge per physical connection. */
function inputLinks(edges, keyOf) {
  const out = new Map();
  for (const e of edges ?? []) {
    if (!e || !keyOf.has(e.a) || !keyOf.has(e.b) || e.a === e.b) continue;
    const [s, t] = String(e.a) <= String(e.b) ? [e.a, e.b] : [e.b, e.a];
    const key = `${s}\u0000${t}\u0000${e.kind}`;
    if (!out.has(key)) out.set(key, { source: keyOf.get(s), target: keyOf.get(t), type: e.kind });
  }
  return [...out.values()];
}

function homeNodes(home, mapId) {
  const nodeMap = new Map();  // defensive: duplicate rows cannot duplicate entities
  const places = (home.places ?? []).filter((p) => p && p.map_id === mapId);
  for (const p of places) {
    nodeMap.set(`mem:place:${p.id}`, {
      id: `mem:place:${p.id}`,
      type: p.kind === 'room' ? 'room' : 'place',
      label: p.name || p.id,
      value: p.kind === 'room' ? 5 : 2.5,
      meta: { placeId: p.id, parent: p.parent || null, x: p.x, y: p.y, mapId },
    });
  }
  const keyOf = new Map(places.map((p) => [p.id, `mem:place:${p.id}`]));
  const objects = (home.objects ?? []).filter((o) => o && o.map_id === mapId);
  for (const o of objects) {
    nodeMap.set(`mem:object:${o.id}`, {
      id: `mem:object:${o.id}`,
      type: 'object',
      label: o.label || o.id,  // unknown labels fall back to the identity string
      value: 1.2 + Math.min(2.3, Math.max(0.2, (o.sightings ?? 1) * 0.4)),
      meta: {
        objectId: o.id, label: o.label ? o.label : null, mapId: o.map_id, lastSeen: o.last_seen,
        confidence: o.confidence_at_observation, source: o.observation_source,
        sightings: o.sightings ?? null, placeId: o.place_id ?? null,
        position: Array.isArray(o.position) ? o.position : null,
        history: Array.isArray(o.history) ? o.history : [],
        affordances: Array.isArray(o.affordances) ? o.affordances : [],
      },
    });
  }
  const links = dedupeLinks([
    ...inputLinks(home.edges, keyOf).filter((l) => l.type === 'connects'),
    ...objects.flatMap((o) => {
      const placeKey = o.place_id && keyOf.get(o.place_id);
      return placeKey
        ? [{ source: `mem:object:${o.id}`, target: placeKey, type: 'located-in' }]
        : [];
    }),
  ]);
  return { nodes: [...nodeMap.values()], links };
}

/** Which persistent map to display: the one with the most remembered objects (ties: first by name). */
function featuredMap(home) {
  const counts = new Map();
  for (const o of home.objects ?? []) if (o?.map_id) counts.set(o.map_id, (counts.get(o.map_id) ?? 0) + 1);
  if (!counts.size) return null;
  return [...counts].sort((a, b) => (b[1] - a[1]) || String(a[0]).localeCompare(String(b[0])))[0][0];
}

function fromHome(home) {
  const mapId = featuredMap(home);
  if (mapId == null) {
    return { source: 'home-empty', mapId: null, nodes: [], links: [], notes: [] };
  }
  const { nodes, links } = homeNodes(home, mapId);
  const elsewhere = (home.objects ?? []).filter((o) => o.map_id !== mapId).length;
  const notes = [];
  if (elsewhere) notes.push(`${elsewhere} remembered items live in other map frames and are not mixed into this graph.`);
  if (nodes.length) notes.push('Layout is force-directed; coordinates are metadata (inspector), not physics.');
  if ((home.objects ?? []).length && !Array.isArray(home.places)) {
    notes.push('This backend snapshot has no places payload; entity relationships would be closer to the runtime with an updated backend.');
  }
  return { source: 'home-memory', mapId, nodes, links, notes };
}

/** Per-session candidates: identity and geometry only. No links — nothing is stored to draw. */
function fromSession(objects) {
  const nodeMap = new Map();
  for (const o of objects) {
    nodeMap.set(`session:${o.id}`, {
      id: `session:${o.id}`,
      type: 'object',
      label: `candidate ${o.id}`,
      value: 2,
      meta: {
        color: o.color ?? null,
        position: Array.isArray(o.position) ? o.position : null,
        extent: Array.isArray(o.extent) ? o.extent : null,
        views: o.views ?? null,
        frame: 'session odometry frame (not persistent)',
      },
    });
  }
  const notes = [
    'Session observations only: no identity in persistent memory and no stored relationships, so the graph has no edges by design.',
  ];
  return { source: 'session-candidates', mapId: null, nodes: [...nodeMap.values()], links: [], notes };
}

const prettyId = (id) => String(id).replace(/[:_]/g, ' ');

/** The label with the most views on a tracked object (ties: by name); null when none recorded. */
function topLabel(labels) {
  const rows = Object.entries(labels && typeof labels === 'object' ? labels : {})
    .filter(([, v]) => typeof v === 'number');
  rows.sort((a, b) => (b[1] - a[1]) || a[0].localeCompare(b[0]));
  return rows.length ? rows[0][0] : null;
}

/** The robot's per-episode semantic graph, converted one-to-one (see header honesty rules). */
function fromSemantic(sg) {
  const nodeMap = new Map();
  for (const n of sg.nodes ?? []) {
    if (!n || n.id == null) continue;
    const attrs = n.attrs && typeof n.attrs === 'object' ? n.attrs : {};
    const kind = String(n.kind);
    const best = kind === 'object' ? topLabel(attrs.labels) : null;
    const name = String(n.id).includes(':') ? String(n.id).split(':').slice(1).join(':') : String(n.id);
    let label;
    if (kind === 'object') label = best ? `${best} ${name}` : `object ${name}`;
    else if (kind === 'category' || kind === 'class') label = name;
    else label = prettyId(n.id);
    nodeMap.set(`sem:${n.id}`, {
      id: `sem:${n.id}`,
      type: kind,
      label,
      value: kind === 'region' ? 5 : kind === 'object' ? 2.5 : 2,
      meta: {
        ...attrs,
        label: best,
        semanticId: n.id,
        frame: attrs.frame ?? sg.frame ?? null,
      },
    });
  }
  const links = [];
  for (const e of sg.edges ?? []) {
    const source = `sem:${e?.source}`;
    const target = `sem:${e?.target}`;
    if (!nodeMap.has(source) || !nodeMap.has(target)) continue;  // never draw a dangling claim
    links.push({
      source, target, type: String(e.status),
      meta: {
        relation: e.relation, provenance: e.provenance ?? null,
        evidence: Array.isArray(e.evidence) ? e.evidence : [],
        score: typeof e.score === 'number' ? e.score : null, score_kind: e.score_kind ?? null,
        time: e.time ?? null, frame: e.frame ?? null,
      },
    });
  }
  return {
    source: 'semantic-graph', mapId: null, nodes: [...nodeMap.values()], links,
    notes: [
      'Semantic graph of this episode. Link type is the claim status; scores carry their own kind (heuristic weight, share of search mass, view count) and are not probabilities.',
      'Layout is force-directed; coordinates are metadata (inspector), not physics.',
    ],
  };
}

/**
 * Build the graph from one snapshot:
 * the robot's semantic graph when it has one, else
 * home memory (rooms + stored relationships) when connected, else session candidates.
 */
export function buildSceneGraphData(snapshot, home = snapshot?.persistent_memory ?? null) {
  if (snapshot?.semantic_graph?.nodes?.length) return fromSemantic(snapshot.semantic_graph);
  if (home?.available && (home.objects?.length || home.places?.length)) return fromHome(home);
  if (home?.available) {
    return { source: 'home-empty', mapId: null, nodes: [], links: [], notes: [] };
  }
  const session = snapshot?.objects ?? [];
  if (session.length) return fromSession(session);
  return {
    source: 'none', mapId: null, nodes: [], links: [],
    notes: snapshot
      ? ['The robot has not recorded objects in this session yet.']
      : [],
  };
}

/**
 * Feed the renderer reused node objects so the force engine keeps layout state across snapshots
 * (d3 stores x/y/z on the node objects themselves). ``cache`` is a caller-owned Map<id, node>:
 * the same object is carried forward and only adapter-owned fields (id/type/label/value/meta)
 * are refreshed with ``Object.assign`` — the adapter never emits x/y/z, so engine-set
 * coordinates and velocities survive untouched. Removed ids are evicted, so switching sources
 * (home ↔ session) does not leak stale objects.
 */
export function reuseNodes(cache, graph) {
  const seen = new Set();
  const nodes = graph.nodes.map((n) => {
    seen.add(n.id);
    const cached = cache.get(n.id);
    if (cached) {
      Object.assign(cached, n);  // mutate in place: the engine may already hold this object
      cache.set(n.id, cached);
      return cached;
    }
    cache.set(n.id, n);
    return n;
  });
  for (const id of [...cache.keys()]) if (!seen.has(id)) cache.delete(id);
  return { ...graph, nodes };
}

/** Filter by entity/relationship types and label; endpoints removed, links follow. */
export function applyFilters(graph, { nodeTypes = null, linkTypes = null, labels = null } = {}) {
  const keep = new Set(
    graph.nodes.filter((n) => {
      if (nodeTypes && !nodeTypes.includes(n.type)) return false;
      if (labels && n.type === 'object' && n.meta?.label != null && !labels.includes(n.meta.label)) return false;
      return true;
    }).map((n) => n.id),
  );
  return {
    ...graph,
    nodes: graph.nodes.filter((n) => keep.has(n.id)),
    links: graph.links.filter((l) => keep.has(l.source) && keep.has(l.target) && (!linkTypes || linkTypes.includes(l.type))),
  };
}

/** Distinct object labels in the graph, sorted — for the filter dropdown. */
export function graphLabels(graph) {
  return [...new Set(graph.nodes.filter((n) => n.type === 'object' && n.meta?.label).map((n) => n.meta.label))].sort();
}

/** Direct neighbours of a node id (undirected adjacency over links). */
export function neighbours(graph, id) {
  if (id == null) return new Set();
  const other = new Set();
  for (const l of graph.links) {
    if (l.source === id) other.add(l.target);
    else if (l.target === id) other.add(l.source);
  }
  return other;
}

/** Deterministic bound for very large inputs: places/rooms survive whole, objects by best evidence. */
export function boundGraph(graph) {
  const rooms = graph.nodes.filter((n) => n.type !== 'object');
  const objects = graph.nodes.filter((n) => n.type === 'object')
    .sort((a, b) => (b.meta?.confidence ?? 0) - (a.meta?.confidence ?? 0) || byId(a, b));
  if (rooms.length + objects.length <= MAX_NODES) return graph;
  const keep = new Set([...rooms, ...objects.slice(0, MAX_NODES - rooms.length)].map((n) => n.id));
  const hidden = objects.length - Math.max(0, MAX_NODES - rooms.length);
  return {
    ...graph,
    nodes: graph.nodes.filter((n) => keep.has(n.id)),
    links: graph.links.filter((l) => keep.has(l.source) && keep.has(l.target)),
    notes: [...graph.notes, `Render bounded to ${MAX_NODES} nodes; ${hidden} low-evidence objects hidden.`],
  };
}
