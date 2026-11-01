import { test } from 'node:test';
import { deepStrictEqual, strictEqual, ok } from 'node:assert';
import {
  applyFilters,
  boundGraph,
  buildSceneGraphData,
  graphLabels,
  MAX_NODES,
  neighbours,
  reuseNodes,
} from './app/lib/sceneGraph.mjs';

const demoHome = () => ({
  available: true,
  source: 'home memory: last recorded sightings; current state not assessed',
  place_source: 'places and stored connectivity edges from the same home memory',
  history_per_object: 6,
  objects: [
    { id: 'cabinet_handle-aa', label: 'cabinet_handle', map_id: 'demo-home', last_seen: 1000, position: [0.5, 0.8, 0.9],
      confidence_at_observation: 0.92, observation_source: 'demo: inspector', place_id: 'room:kitchen', sightings: 3, history: [], affordances: [] },
    { id: 'charger-bb', label: 'charger', map_id: 'demo-home', last_seen: 900, position: [6.2, 0.4, 0.5],
      confidence_at_observation: 0.8, observation_source: 'demo: head_rgbd', place_id: null, sightings: 1, history: [], affordances: [] },
    { id: 'crane-cc', label: 'crane', map_id: 'other-map', last_seen: 800, position: [9, 9, 1],
      confidence_at_observation: 0.5, observation_source: 'demo: x', place_id: null, sightings: 1, history: [], affordances: [] },
  ],
  places: [
    { id: 'room:kitchen', name: 'kitchen', kind: 'room', x: 0, y: 0, map_id: 'demo-home', parent: null },
    { id: 'room:hall', name: 'hall', kind: 'room', x: 3, y: 3, map_id: 'demo-home', parent: null },
    { id: 'room:bedroom', name: 'bedroom', kind: 'room', x: 6, y: 0, map_id: 'demo-home', parent: null },
    { id: 'place:counter', name: 'counter', kind: 'place', x: 0.5, y: 0.8, map_id: 'demo-home', parent: 'room:kitchen' },
  ],
  edges: [
    { a: 'room:kitchen', b: 'room:hall', kind: 'connects' },
    { a: 'room:hall', b: 'room:kitchen', kind: 'connects' },  // stored both ways; one physical edge
    { a: 'room:hall', b: 'room:bedroom', kind: 'connects' },
  ],
});

const snap = (home, objects = []) => ({ odometry: [0, 0, 0], objects, persistent_memory: home });
const idSet = (g) => new Set(g.nodes.map((n) => n.id));
const linkKeys = (g) => g.links.map((l) => [l.source, l.target, l.type].sort().join('\u0000'));

const demoGraph = () => buildSceneGraphData(snap(demoHome()));

test('home memory converts to nodes with stable mem: ids and honest types', () => {
  const g = demoGraph();
  deepStrictEqual(idSet(g), new Set([
    'mem:place:room:kitchen', 'mem:place:room:hall', 'mem:place:room:bedroom', 'mem:place:place:counter',
    'mem:object:cabinet_handle-aa', 'mem:object:charger-bb',
  ]));
  const kitchen = g.nodes.find((n) => n.label === 'kitchen');
  strictEqual(kitchen.type, 'room');
  const counter = g.nodes.find((n) => n.label === 'counter');
  strictEqual(counter.type, 'place');
  ok(counter.meta.parent === 'room:kitchen');
  const handle = g.nodes.find((n) => n.label === 'cabinet_handle');
  strictEqual(handle.meta.sightings, 3);
  strictEqual(handle.meta.source, 'demo: inspector');
});

test('only stored relationships become links; direction and kind are preserved', () => {
  const g = demoGraph();
  // two room connections (the both-way pair dedupes) plus one object→place link
  strictEqual(g.links.length, 3);
  deepStrictEqual(
    new Set(g.links.filter((l) => l.type === 'connects').map((l) => `${l.source}\u0000${l.target}`)),
    new Set(['mem:place:room:bedroom\u0000mem:place:room:hall', 'mem:place:room:hall\u0000mem:place:room:kitchen']),
  );
  const located = g.links.find((l) => l.type === 'located-in');
  ok(located, 'the handle has a stored place, so a located-in link must exist');
  strictEqual(located.source, 'mem:object:cabinet_handle-aa');  // object → place, as recorded
  strictEqual(located.target, 'mem:place:room:kitchen');
  // the charger was never assigned a place: the graph must not attach one
  strictEqual(
    g.links.filter((l) => l.source === 'mem:object:charger-bb' || l.target === 'mem:object:charger-bb').length,
    0,
  );
});

test('force positions never carry physical coordinates; real ones stay in metadata', () => {
  const g = demoGraph();
  ok(g.nodes.every((n) => n.x === undefined && n.y === undefined && n.z === undefined));
  const kitchen = g.nodes.find((n) => n.label === 'kitchen');
  deepStrictEqual([kitchen.meta.x, kitchen.meta.y], [0, 0]);
  const handle = g.nodes.find((n) => n.label === 'cabinet_handle');
  deepStrictEqual(handle.meta.position, [0.5, 0.8, 0.9]);
  ok(g.notes.some((n) => n.includes('coordinates are metadata')), 'the view must say positions are not physics');
});

test('identity is stable across snapshots via reuseNodes; engine state survives, data refreshes', () => {
  const cache = new Map();
  const home1 = demoHome();
  const g1 = reuseNodes(cache, buildSceneGraphData(snap(home1)));
  const node1 = g1.nodes.find((n) => n.id === 'mem:object:cabinet_handle-aa');
  node1.x = 1.5;  // simulate the force engine having placed the node
  node1.y = -2;

  const home2 = structuredClone(home1);
  home2.objects[0].sightings = 4;
  home2.objects[0].position = [0.51, 0.79, 0.9];
  home2.objects = home2.objects.slice(0, 1);  // charger no longer in the latest snapshot
  const g2 = reuseNodes(cache, buildSceneGraphData(snap(home2)));
  const node2 = g2.nodes.find((n) => n.id === 'mem:object:cabinet_handle-aa');
  strictEqual(node2, node1, 'same node object so the engine keeps its state');
  strictEqual(node2.x, 1.5);
  strictEqual(node2.meta.sightings, 4, 'adapter-owned fields are refreshed');
  deepStrictEqual(node2.meta.position, [0.51, 0.79, 0.9]);
  ok(!cache.has('mem:object:charger-bb'), 'removed ids are evicted from the cache');
});

test('duplicate rows collapse: repeated and both-way edges, repeated entities', () => {
  const home = demoHome();
  home.edges = [...home.edges, { a: 'room:kitchen', b: 'room:hall', kind: 'connects' }];
  home.objects = [...home.objects, ...structuredClone(home.objects).slice(0, 1)];
  const g = buildSceneGraphData(snap(home));
  strictEqual(g.nodes.filter((n) => n.id === 'mem:object:cabinet_handle-aa').length, 1);
  strictEqual(g.links.filter((l) => l.type === 'connects').length, 2, 'three stored rows, two physical connections');
});

test('empty and incomplete data produce honest, quiet graphs', () => {
  deepStrictEqual(buildSceneGraphData(undefined).nodes, []);
  const none = buildSceneGraphData(snap(null));
  strictEqual(none.source, 'none');
  ok(none.notes.length > 0, 'a connected session with nothing observed is explained');

  const empty = buildSceneGraphData(snap({ ...demoHome(), objects: [] }));
  strictEqual(empty.source, 'home-empty');
  strictEqual(empty.nodes.length, 0);

  // older backend shape: sightings without the places payload → entities only, no invented edges
  const { places, edges, ...placesIdle } = demoHome();
  const bare = buildSceneGraphData(snap(placesIdle));
  strictEqual(bare.nodes.length, 2);
  strictEqual(bare.links.length, 0);
  ok(bare.notes.some((n) => n.includes('places payload')));

  // unusable home memory falls back to the session observations
  const brokenGraph = buildSceneGraphData(snap({ available: false, error: 'home memory could not be read' },
    [{ id: 7, color: '#fff', position: [1, 2, 0.3], extent: [0.2, 0.2, 0.4], views: 3 }]));
  strictEqual(brokenGraph.source, 'session-candidates');
});

test('missing labels fall back to identity and stay reachable through filters', () => {
  const home = demoHome();
  home.objects = [
    ...home.objects,
    { id: 'nameless-99', label: '', map_id: 'demo-home', last_seen: 100, position: [1, 1, 1],
      confidence_at_observation: 0.4, observation_source: 'demo: x', place_id: null, sightings: 1, history: [], affordances: [] },
  ];
  const g = buildSceneGraphData(snap(home));
  const nameless = g.nodes.find((n) => n.id === 'mem:object:nameless-99');
  strictEqual(nameless.label, 'nameless-99', 'unknown labels fall back to the stored identity');
  ok(nameless.meta.label === null);
  deepStrictEqual(graphLabels(g), ['cabinet_handle', 'charger'], 'unlabeled objects are not filter choices');

  const onlyCharger = applyFilters(g, { labels: ['charger'] });
  strictEqual(onlyCharger.nodes.length, 6, 'rooms/places (no label) stay as context, plus charger and the unlabeled object');
  ok(onlyCharger.nodes.some((n) => n.id === 'mem:object:nameless-99'), 'unlabeled items stay visible under a label filter');
  ok(onlyCharger.nodes.every((n) => n.label !== 'cabinet_handle'));
});

test('entity and relationship filters, and nobody invents edges to bridge removed types', () => {
  const g = demoGraph();
  const objectsOnly = applyFilters(g, { nodeTypes: ['object'] });
  deepStrictEqual(idSet(objectsOnly), new Set(['mem:object:cabinet_handle-aa', 'mem:object:charger-bb']));
  strictEqual(objectsOnly.links.length, 0, 'located-in links leave with their place endpoints');

  const rooms = applyFilters(g, { nodeTypes: ['room', 'place'], linkTypes: ['connects'] });
  ok(rooms.links.every((l) => l.type === 'connects'));
  ok(!rooms.links.some((l) => l.type === 'located-in'));
  strictEqual(rooms.nodes.length, 4);
});

test('neighbour sets cover both link directions and reject unknown ids', () => {
  const g = demoGraph();
  deepStrictEqual(
    [...neighbours(g, 'mem:object:cabinet_handle-aa')],
    ['mem:place:room:kitchen'],
  );
  deepStrictEqual(
    new Set([
      ...neighbours(g, 'mem:place:room:hall'),
    ]),
    new Set(['mem:place:room:kitchen', 'mem:place:room:bedroom']),
  );
  strictEqual(neighbours(g, 'no-such-node').size, 0);
  strictEqual(neighbours(g, null).size, 0);
});

test('session candidates are identity only: nodes, no edges, odometry frame declared', () => {
  const g = buildSceneGraphData(snap(null, [
    { id: 7, color: '#88aaff', position: [1, 2, 0.3], extent: [0.2, 0.2, 0.4], views: 3 },
    { id: 9, color: '#ffffff', position: [4, 5, 0.2], extent: [0.3, 0.3, 0.3], views: 1 },
  ]));
  strictEqual(g.source, 'session-candidates');
  deepStrictEqual([...g.nodes].sort((a, b) => a.id.localeCompare(b.id)).map((n) => n.id), ['session:7', 'session:9']);
  deepStrictEqual(g.links, [], 'no stored relationships exist, so the graph draws none');
  const node = g.nodes[0];
  ok(node.meta.frame.includes('odometry'), 'the coordinate frame is stated, never silently map-frame');
  ok(node.meta.position !== undefined);
  deepStrictEqual(applyFilters(g, {}).links, []);
  ok(g.notes.some((n) => n.includes('no edges by design')));
});

test('oversized graphs keep structure first, best-evidence objects second, and say so', () => {
  const base = demoHome();
  const filler = Array.from({ length: 420 }, (_, i) => ({
    id: `o${String(i).padStart(3, '0')}`, label: `thing${i}`, map_id: 'demo-home', last_seen: i,
    position: [i, 0, 0], confidence_at_observation: i / 1000, observation_source: 'gen',
    place_id: null, sightings: 1, history: [], affordances: [],
  }));
  const g = buildSceneGraphData(snap({ ...base, objects: [...base.objects, ...filler] }));
  const bounded = boundGraph(g);
  strictEqual(bounded.nodes.length, MAX_NODES);
  strictEqual(bounded.nodes.filter((n) => n.type === 'room' || n.type === 'place').length, 4, 'structures survive whole');
  const expected = new Set(bounded.nodes.filter((n) => n.type === 'object').map((n) => n.id));
  // best evidence = highest confidence; the handle (0.92) must beat every filler
  ok(expected.has('mem:object:cabinet_handle-aa'));
  ok(!expected.has('mem:object:o000'), 'the lowest-confidence filler is the first hidden');
  ok(bounded.notes.some((n) => n.includes('bounded')), 'the bound is disclosed, not silent');
});

test('mixed map frames are not folded together; the exclusion is stated', () => {
  const g = demoGraph();
  ok(!idSet(g).has('mem:object:crane-cc'), 'entities from other map frames stay out');
  ok(g.notes.some((n) => n.includes('other map frames')));
  strictEqual(g.mapId, 'demo-home');
  ok(linkKeys(g).every((k) => !k.includes('crane')));
});
