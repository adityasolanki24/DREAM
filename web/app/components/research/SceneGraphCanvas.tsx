'use client';
import type { ForceGraph3DInstance } from '3d-force-graph';
import { useEffect, useMemo, useRef, useState } from 'react';
// Only the 3D renderer: the umbrella react-force-graph package also loads its VR/AR variants, which need the A-Frame global.
import ForceGraph3D from 'react-force-graph-3d';
import type { GraphDataLite } from '../../types';
import { useThemeIsLight } from '../Strands';

type Props = {
  graph: GraphDataLite;
  selected: string | null;
  onSelect: (id: string | null) => void;
  /** Called after mount with the view controls; flows through next/dynamic where refs would not. */
  onFit?: (api: { fit: () => void }) => void;
};

/** What the renderer guarantees per node/link at runtime; every field optional because the engine diffs objects it owns. */
type SceneNode = {
  id?: string | number;
  label?: string;
  type?: 'room' | 'place' | 'object';
  value?: number;
  meta?: Record<string, unknown>;
  x?: number;
  y?: number;
  z?: number;
};
type SceneLink = { source?: string | number | SceneNode; target?: string | number | SceneNode; type?: string };

const escape = (s: string) =>
  s.replace(/[&<>"']/g, (ch) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[ch] as string));

const idOf = (v: string | number | SceneNode | undefined) =>
  v == null ? '' : typeof v === 'object' ? String(v.id ?? '') : String(v);

/** Frees the GPU-side resources three-forcegraph leaves behind on unmount; disposals are idempotent. */
function teardown(g: ForceGraph3DInstance) {
  try {
    g.scene().traverse((obj: unknown) => {
      const o = obj as { geometry?: { dispose(): void }; material?: { dispose(): void } | Array<{ dispose(): void }> };
      o.geometry?.dispose();
      if (Array.isArray(o.material)) o.material.forEach((m) => m.dispose());
      else o.material?.dispose();
    });
    const r = g.renderer();
    r.dispose();
    r.forceContextLoss();
  } catch {
    /* instance already dismantled by the engine's own cleanup */
  }
}

/**
 * The ThreeJS/WebGL renderer for the semantic scene graph. Imported dynamically with SSR off by
 * `SceneGraph.tsx` and mounted only while its tab is visible. Colours come from the app's theme
 * tokens (re-read when the theme flips) over a transparent background so it reads as part of the
 * interface. Node positions here are the force layout — the robot's coordinates live in node
 * metadata and the inspector, never as scene positions.
 */
const SceneGraphCanvas = function SceneGraphCanvas({ graph, selected, onSelect, onFit }: Props) {
  const container = useRef<HTMLDivElement>(null);
  const box = useRef<ForceGraph3DInstance | undefined>(undefined);
  const inst = useRef<ForceGraph3DInstance | null>(null);
  const fitted = useRef(false);
  const [size, setSize] = useState({ w: 0, h: 0 });
  const light = useThemeIsLight();

  useEffect(() => {
    if (!onFit) return;
    onFit({ fit: () => inst.current?.zoomToFit(500, 36) });
  }, [onFit]);

  // Size from the container (resizable panel / focus layer / mobile), not the window.
  useEffect(() => {
    const el = container.current;
    if (!el) return;
    const ro = new ResizeObserver((entries) => {
      const r = entries[0]?.contentRect;
      if (r) setSize((s) => (Math.abs(r.width - s.w) > 1 || Math.abs(r.height - s.h) > 1 ? { w: Math.round(r.width), h: Math.round(r.height) } : s));
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  // Keep the last attached instance: React may detach object refs before unmount cleanups run.
  useEffect(() => {
    if (box.current) inst.current = box.current;
  });

  // Unmount: stop rendering and free the WebGL context explicitly.
  useEffect(
    () => () => {
      const g = inst.current;
      inst.current = null;
      fitted.current = false;
      if (g) teardown(g);
    },
    [],
  );

  // Fit once: engine settle or first valid size, whichever comes first. Viewport is then owned by the user.
  const autoFit = () => {
    if (fitted.current || !inst.current || size.w < 40) return;
    fitted.current = true;
    inst.current.zoomToFit(450, 36);
  };
  useEffect(() => {
    if (fitted.current || size.w < 40) return;
    const t = window.setTimeout(autoFit, 700);
    return () => window.clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [size.w]);

  const tokens = useMemo(() => {
    const css = getComputedStyle(document.documentElement);
    const pick = (n: string) => {
      const v = css.getPropertyValue(n).trim();
      return v.length ? v : '#888888';
    };
    return {
      room: pick('--accent'),
      place: pick('--map-robot'),
      object: pick('--amber'),
      selected: pick('--focus-ring'),
      dim: pick('--text-disabled'),
      link: pick('--graph-edge'),
    };
    // tokens are read fresh whenever the theme flips (data-theme attribute change)
  }, [light]);

  const highlight = useMemo(() => {
    if (selected == null) return null;
    const set = new Set<string>();
    for (const l of graph.links) {
      if (l.source === selected) set.add(l.target);
      else if (l.target === selected) set.add(l.source);
    }
    return set;
  }, [graph.links, selected]);

  const nodeColor = (n: SceneNode) => {
    const id = idOf(n.id);
    if (selected != null) {
      if (id === selected) return tokens.selected;
      if (highlight && !highlight.has(id)) return tokens.dim;
    }
    return tokens[n.type === 'room' ? 'room' : n.type === 'place' ? 'place' : 'object'];
  };
  const linkColor = (l: SceneLink) => {
    if (selected != null && (idOf(l.source) === selected || idOf(l.target) === selected)) return tokens.selected;
    return tokens.link;
  };
  const linkWidth = (l: SceneLink) =>
    selected != null && (idOf(l.source) === selected || idOf(l.target) === selected) ? 2.4 : 1;
  const nodeLabel = (n: SceneNode) => {
    const conf = n.meta && typeof n.meta.confidence === 'number' ? Math.round((n.meta.confidence as number) * 100) : null;
    return `<b>${escape(n.label ?? '')}</b><span class="sgraph-tip">${escape(n.type ?? '')}${
      conf != null ? ` · ${conf}% confidence` : ''
    }</span>`;
  };

  return (
    <div className="sgraph-canvas" ref={container}>
      {size.w > 40 && size.h > 40 && (
        <ForceGraph3D
          ref={box}
          graphData={graph}
          width={size.w}
          height={size.h}
          showNavInfo={false}
          backgroundColor="rgba(0,0,0,0)"
          rendererConfig={{ antialias: true, alpha: true }}
          nodeRelSize={3.6}
          nodeVal={(n: SceneNode) => n.value ?? 1}
          nodeColor={nodeColor}
          nodeLabel={nodeLabel}
          enableNodeDrag={false}
          onNodeClick={(n: SceneNode) => onSelect(String(n.id))}
          onBackgroundClick={() => onSelect(null)}
          linkColor={linkColor}
          linkOpacity={0.55}
          linkWidth={linkWidth}
          linkDirectionalArrowLength={(l: SceneLink) => (l.type === 'located-in' ? 3.2 : 0)}
          linkDirectionalArrowRelPos={1}
          onEngineStop={autoFit}
        />
      )}
    </div>
  );
};

export default SceneGraphCanvas;
