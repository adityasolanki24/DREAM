'use client';
import { Network, Scan, X } from 'lucide-react';
import dynamic from 'next/dynamic';
import { useEffect, useMemo, useRef, useState } from 'react';
import { relativeAge, wallTime } from '../../lib/format';
import { usePref } from '../../lib/prefs';
import { useSession } from '../../lib/session';
import type { GraphLink, GraphNode } from '../../types';
import { Quiet } from '../ui';
import {
  applyFilters,
  boundGraph,
  buildSceneGraphData,
  graphLabels,
  LINK_TYPES,
  NODE_TYPES,
  reuseNodes,
} from '../../lib/sceneGraph.mjs';

/** ThreeJS renderer is heavy: dynamic import with SSR off, mounted only while this tab is visible. */
const Canvas = dynamic(() => import('./SceneGraphCanvas'), {
  ssr: false,
  loading: () => <div className="sgraph-loading faint small">Loading renderer…</div>,
});

const ENTITY_NAME: Record<string, string> = {
  room: 'Rooms', place: 'Places', object: 'Objects', region: 'Regions', category: 'Categories',
  class: 'Classes', unexplored: 'Unexplored', memory: 'Memories',
};
const ENTITY_KIND: Record<string, string> = {
  room: 'Room', place: 'Place', object: 'Object', region: 'Region', category: 'Category',
  class: 'Class', unexplored: 'Unexplored', memory: 'Memory',
};
/** Every filter key the view persists; absent = enabled, unknown = rejected by the validator. */
const ALL_FILTERS = [...NODE_TYPES, ...LINK_TYPES];

const SOURCE_LABEL: Record<string, string> = {
  'semantic-graph': 'Semantic graph',
  'home-memory': 'Home memory',
  'session-candidates': 'Session observations',
  'home-empty': 'Home memory (empty)',
  none: 'No observations yet',
};

const Row = ({ k, v }: { k: string; v: React.ReactNode }) => (
  <div className="diag-row"><dt>{k}</dt><dd>{v}</dd></div>
);

const num = (v: unknown): number | null => (typeof v === 'number' && Number.isFinite(v) ? v : null);
const str = (v: unknown): string | null => (typeof v === 'string' && v.length ? v : null);
const vec = (v: unknown): number[] | null =>
  (Array.isArray(v) && v.every((x) => typeof x === 'number') ? (v as number[]) : null);

/** The inspector for one selected entity: what the robot actually recorded, with frames and provenance. */
function Details({ node, links, byId, onClose }: {
  node: GraphNode; links: GraphLink[]; byId: Map<string, GraphNode>; onClose: () => void;
}) {
  const claims = links.filter((l) => l.meta && (l.source === node.id || l.target === node.id));
  const m = (node.meta ?? {}) as Record<string, unknown>;
  const history = Array.isArray(m.history) ? (m.history as Record<string, unknown>[]) : [];
  const affordances = Array.isArray(m.affordances) ? (m.affordances as Record<string, unknown>[]) : [];
  const lastSeen = num(m.lastSeen);
  const mapId = str(m.mapId);
  const parent = str(m.parent);
  const placeId = str(m.placeId);
  const source = str(m.source);
  const frame = str(m.frame);
  const colour = str(m.color);
  const confidence = num(m.confidence);
  const sightings = num(m.sightings);
  const views = num(m.views);
  const position = vec(m.position);
  const extent = vec(m.extent);
  return (
    <div className="sgraph-details">
      <h3 className="rail-label">
        {node.label}
        <span className="meta-pill">{ENTITY_KIND[node.type] ?? node.type}</span>
        <button className="icon-btn sm" onClick={onClose} aria-label="Close details"><X size={13} aria-hidden /></button>
      </h3>
      <dl>
        <Row k="Id" v={<span className="mono">{node.id}</span>} />
        {mapId && <Row k="Map frame" v={<span className="mono">{mapId}</span>} />}
        {parent && <Row k="Parent" v={<span className="mono">{parent}</span>} />}
        {node.type === 'object' && placeId && <Row k="Stored place" v={<span className="mono">{placeId}</span>} />}
        {confidence != null && <Row k="Confidence" v={`${Math.round(confidence * 100)}%`} />}
        {sightings != null && <Row k="Sightings" v={String(sightings)} />}
        {source && <Row k="Recorded by" v={source} />}
        {lastSeen != null && (
          <Row k="Last seen" v={<span title={wallTime(lastSeen)}>{relativeAge(lastSeen)}</span>} />
        )}
        {position && (
          <Row k="Position" v={<span className="mono">{position.map((v) => v.toFixed(2)).join(', ')} m</span>} />
        )}
        {frame && <Row k="Coordinates" v={frame} />}
        {views != null && <Row k="Views" v={String(views)} />}
        {extent && (
          <Row k="Extent" v={<span className="mono">{extent.map((v) => v.toFixed(2)).join(' × ')} m</span>} />
        )}
        {colour && (
          <Row k="Colour" v={<><span className="swatch" style={{ background: colour }} aria-hidden /> <span className="mono">{colour}</span></>} />
        )}
      </dl>
      {claims.length > 0 && (
        <section className="sgraph-history">
          <div className="rail-label">Claims <span className="faint">· status, relation, score kind</span></div>
          <ul>
            {claims.map((l, i) => {
              const cm = l.meta as Record<string, unknown>;
              const score = num(cm.score);
              const evidence = Array.isArray(cm.evidence) ? (cm.evidence as unknown[]).map(String) : [];
              const outgoing = l.source === node.id;
              const other = byId.get(outgoing ? l.target : l.source);
              return (
                <li key={i}>
                  <b>{l.type}</b>
                  <span>
                    {outgoing ? '' : `${other?.label ?? l.source} `}{str(cm.relation) ?? '—'}{outgoing ? ` ${other?.label ?? l.target}` : ''}
                  </span>
                  {score != null && <span className="mono">{score} · {str(cm.score_kind) ?? 'score kind not stated'}</span>}
                  {str(cm.provenance) && <span className="faint">{str(cm.provenance)}</span>}
                  {evidence.length > 0 && <span className="faint mono">{evidence.join(', ')}</span>}
                </li>
              );
            })}
          </ul>
        </section>
      )}
      {affordances.length > 0 && (
        <section className="sgraph-history">
          <div className="rail-label">Affordances <span className="faint">· stored skill outcomes</span></div>
          <ul>
            {affordances.map((a, i) => {
              const t = num(a.time);
              return (
                <li key={i}>
                  <b>{str(a.skill) ?? '—'}</b>
                  <span className={str(a.outcome) === 'succeeded' ? 'tone-good' : str(a.outcome) === 'failed' ? 'tone-bad' : ''}>
                    {str(a.outcome) ?? '—'}
                  </span>
                  {t != null && <span className="mono" title={wallTime(t)}>{relativeAge(t)}</span>}
                </li>
              );
            })}
          </ul>
        </section>
      )}
      {history.length > 0 && (
        <section className="sgraph-history">
          <div className="rail-label">Observation history <span className="faint">· newest first</span></div>
          <ul>
            {history.map((h, i) => {
              const t = num(h.time);
              const p = vec(h.position);
              return (
                <li key={i}>
                  {t != null && <span className="mono" title={wallTime(t)}>{relativeAge(t)}</span>}
                  <span className="mono">{p ? `${p.map((v) => v.toFixed(2)).join(', ')} m` : 'position unknown'}</span>
                  <span className="faint">{str(h.source) ?? ''}</span>
                </li>
              );
            })}
          </ul>
        </section>
      )}
    </div>
  );
}

/**
 * The semantic scene graph: entities DREAM actually remembers (rooms, places, objects,
 * session candidates) and only the relationships its memory stores (room connectivity,
 * object→place). Force-graph layout, WorldMap physics and React Flow task execution
 * stay three distinct views.
 */
export function SceneGraph() {
  const { snapshot } = useSession();
  const [selected, setSelected] = useState<string | null>(null);
  const [enabled, setEnabled] = usePref<string[]>(
    'research-graph',
    ALL_FILTERS,
    (v) => Array.isArray(v) && v.every((k) => ALL_FILTERS.includes(k)),
  );
  const [labelSel, setLabelSel] = useState<string[]>([]);
  const fit = useRef<(() => void) | null>(null);

  const base = useMemo(() => buildSceneGraphData(snapshot), [snapshot]);
  const cache = useRef(new Map());
  const stable = useMemo(() => reuseNodes(cache.current, base), [base]);
  const labels = useMemo(() => graphLabels(stable), [stable]);

  // Rendering derivations are cheap at bounded size; no memo, so they always match render state.
  const nodeTypes = NODE_TYPES.filter((t) => enabled.includes(t));
  const linkTypes = LINK_TYPES.filter((t) => enabled.includes(t));
  const filtered = boundGraph(
    applyFilters(stable, { nodeTypes, linkTypes, labels: labelSel.length ? labelSel : null }),
  );

  // Selection lives in the full graph (not the filtered one), so it survives filter changes.
  const byId = useMemo(() => new Map(stable.nodes.map((n) => [n.id, n])), [stable]);
  const selNode = selected != null ? byId.get(selected) ?? null : null;
  useEffect(() => {
    if (selected != null && !byId.has(selected)) setSelected(null);
  }, [byId, selected]);

  const toggle = (key: string) =>
    setEnabled((cur) => {
      const next = cur.includes(key) ? cur.filter((k) => k !== key) : [...cur, key];
      return ALL_FILTERS.filter((k) => next.includes(k));  // canonical order
    });
  const toggleLabel = (lb: string) =>
    setLabelSel((cur) => (cur.includes(lb) ? cur.filter((k) => k !== lb) : [...cur, lb]));
  const resetFilters = () => {
    setEnabled(ALL_FILTERS);
    setLabelSel([]);
  };

  const stableCount = (t: string) => stable.nodes.filter((n) => n.type === t).length;
  const linkCount = (t: string) => stable.links.filter((l) => l.type === t).length;
  const shrunk = filtered.nodes.length !== stable.nodes.length || filtered.links.length !== stable.links.length;

  return (
    <div className="sgraph">
      <div className="sgraph-head">
        <div className="sgraph-source">
          <span className="meta-pill">{SOURCE_LABEL[base.source] ?? base.source}</span>
          {base.mapId && <span className="meta-pill mono">map frame {base.mapId}</span>}
          <span className="faint small mono">{stable.nodes.length} entities · {stable.links.length} stored links</span>
          {shrunk && <span className="faint small mono">showing {filtered.nodes.length} · {filtered.links.length}</span>}
        </div>
        {filtered.notes.length > 0 && (
          <ul className="sgraph-notes">
            {filtered.notes.map((n, i) => <li key={i}>{n}</li>)}
          </ul>
        )}
      </div>

      <div className="sgraph-filters">
        <div className="chiprow" role="group" aria-label="Entity types">
          {NODE_TYPES.map((t) => (
            <button
              key={t}
              className="chip"
              aria-pressed={enabled.includes(t)}
              disabled={stableCount(t) === 0}
              title={stableCount(t) === 0 ? `No ${ENTITY_NAME[t].toLowerCase()} in this graph` : undefined}
              onClick={() => toggle(t)}
            >
              {ENTITY_NAME[t]}<span className="chip-n mono">{stableCount(t)}</span>
            </button>
          ))}
        </div>
        {stable.links.length > 0 && (
          <div className="chiprow" role="group" aria-label="Relationship types">
            {LINK_TYPES.filter((t) => linkCount(t) > 0).map((t) => (
              <button key={t} className="chip" aria-pressed={enabled.includes(t)} onClick={() => toggle(t)}>
                {t}<span className="chip-n mono">{linkCount(t)}</span>
              </button>
            ))}
          </div>
        )}
        {labels.length > 1 && (
          <div className="chiprow" role="group" aria-label="Labels">
            {labels.map((lb) => (
              <button key={lb} className={`chip ${lb.length > 16 ? 'chip-ellipsis' : ''}`} aria-pressed={labelSel.includes(lb)} onClick={() => toggleLabel(lb)}>
                {lb}
              </button>
            ))}
          </div>
        )}
      </div>

      {filtered.nodes.length > 0 ? (
        <div className="sgraph-stage">
          <Canvas
            graph={filtered}
            selected={selected}
            onSelect={setSelected}
            onFit={(api) => { fit.current = api.fit; }}
          />
          <div className="sgraph-tools">
            <button className="icon-btn sm" onClick={() => fit.current?.()} aria-label="Fit graph in view" title="Fit graph in view">
              <Scan size={14} aria-hidden />
            </button>
          </div>
        </div>
      ) : stable.nodes.length === 0 ? (
        <Quiet icon={<Network size={16} />} title={sourceEmptyTitle(base.source, snapshot)}>
          {sourceEmptyBody(base.source, snapshot)}
        </Quiet>
      ) : (
        <Quiet title="Nothing matches the filters" action={<button className="chip" onClick={resetFilters}>Reset filters</button>}>
          Every entity or relationship is filtered out.
        </Quiet>
      )}

      {stable.nodes.length > 0 && (
        selNode
          ? <Details node={selNode} links={stable.links} byId={byId} onClose={() => setSelected(null)} />
          : <p className="faint small sgraph-hint">Select a node to inspect what the robot recorded about it. Layout positions are semantic, not physical.</p>
      )}
    </div>
  );
}

function sourceEmptyTitle(source: string, snapshot: unknown): string {
  if (source === 'home-empty') return 'Home memory is connected but empty';
  return snapshot ? 'Nothing observed yet' : 'Not connected';
}

function sourceEmptyBody(source: string, snapshot: unknown): React.ReactNode {
  switch (source) {
    case 'home-empty':
      return 'Home memory is connected but stores no entities yet. They appear once the robot records sightings.';
    case 'session-candidates':
      return 'Session observations exist but produced no graph nodes (unexpected — please report).';
    default:
      return snapshot
        ? 'The robot has not recorded entities this session yet. Home memory is not connected: it loads with --memory <home-memory.sqlite> on the backend (read-only).'
        : 'Connect to the robot session to see what it has observed.';
  }
}
