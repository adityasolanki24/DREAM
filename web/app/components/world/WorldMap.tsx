'use client';
import { Compass } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import { mapPixels, worldToPixel } from '../../map.mjs';
import type { Snapshot } from '../../types';
import { Quiet } from '../ui';

/** Bumps whenever <html data-theme> changes so canvas drawing can re-read the colour tokens. */
function useThemeTick(): number {
  const [tick, setTick] = useState(0);
  useEffect(() => {
    const observer = new MutationObserver(() => setTick((n) => n + 1));
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
    return () => observer.disconnect();
  }, []);
  return tick;
}

/** Top-down occupancy map with object candidates and the robot's pose, scaled to fit its container. */
export function WorldMap({ snapshot, legend = true, compact = false }: { snapshot: Snapshot | null; legend?: boolean; compact?: boolean }) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const map = snapshot?.map;
  const themeTick = useThemeTick();
  useEffect(() => {
    const c = canvas.current;
    if (!c || !map?.cells?.length) return;
    const ctx = c.getContext('2d');
    if (!ctx) return;
    const css = getComputedStyle(document.documentElement);
    const token = (name: string) => css.getPropertyValue(name).trim();
    const cols = map.cells.length;
    const rows = map.cells[0].length;
    const S = Math.max(2, Math.min(10, Math.floor(1600 / Math.max(cols, rows)))); // canvas px per map cell
    c.width = cols * S;
    c.height = rows * S;
    ctx.fillStyle = token('--map-unknown');
    ctx.fillRect(0, 0, c.width, c.height);
    const fills = [token('--map-unknown'), token('--map-free'), token('--map-occupied')];
    for (const { x, y, value } of mapPixels(map.cells)) {
      ctx.fillStyle = fills[value] ?? fills[0];
      ctx.fillRect(x * S, y * S, S, S);
    }
    // Marker scale: keep it readable, but never let it dwarf a small canvas (compact inset) — capped at a tenth of the short side.
    const unit = Math.max(S * 1.6, Math.min(Math.max(c.width, c.height) / 48, Math.min(c.width, c.height) / 10));
    for (const o of snapshot?.objects ?? []) {
      const p = worldToPixel(map, o.position[0], o.position[1]);
      if (!p) continue;
      ctx.strokeStyle = token('--amber');
      ctx.lineWidth = Math.max(1.5, unit / 5);
      ctx.beginPath();
      ctx.arc(p.px * S, p.py * S, unit * 0.55, 0, Math.PI * 2);
      ctx.stroke();
    }
    // No pose history exists in the snapshot, so there is deliberately no trail (--map-trail unused).
    const [x, y, yaw] = snapshot?.odometry ?? [];
    const r = x == null ? null : worldToPixel(map, x, y);
    if (r) {
      ctx.save();
      ctx.translate(r.px * S, r.py * S);
      ctx.fillStyle = token('--map-robot');
      ctx.globalAlpha = 0.22;
      ctx.beginPath();
      ctx.arc(0, 0, unit * 1.1, 0, Math.PI * 2);
      ctx.fill();
      ctx.globalAlpha = 1;
      ctx.rotate(-(yaw ?? 0)); // canvas y points down, map y points up
      ctx.beginPath();
      ctx.moveTo(unit * 1.0, 0);
      ctx.lineTo(-unit * 0.65, unit * 0.62);
      ctx.lineTo(-unit * 0.3, 0);
      ctx.lineTo(-unit * 0.65, -unit * 0.62);
      ctx.closePath();
      ctx.fill();
      ctx.lineJoin = 'round';
      ctx.lineWidth = Math.max(1.5, unit / 7);
      ctx.strokeStyle = token('--card');
      ctx.stroke();
      ctx.restore();
    }
  }, [map, snapshot?.objects, snapshot?.odometry, themeTick]);

  if (!map?.cells?.length) {
    return compact
      ? <div className="map-empty-compact"><Compass size={14} aria-hidden /> No map yet</div>
      : <Quiet icon={<Compass size={18} />} title="No map yet">The occupancy map is built from what the robot has observed in this session.</Quiet>;
  }
  return (
    <div className={`world-map ${compact ? 'compact' : ''}`}>
      <div className="world-canvas">
        <canvas ref={canvas} role="img" aria-label="Occupancy map with robot position and heading" />
      </div>
      {legend && (
        <div className="legend" aria-label="Map legend">
          <span><i className="k-free" /> free</span>
          <span><i className="k-unk" /> unknown</span>
          <span><i className="k-occ" /> occupied</span>
          <span><i className="k-obj" /> object candidate</span>
          <span><i className="k-bot" /> robot · arrow is heading</span>
          <span className="legend-scale mono">{map.resolution_m} m/cell · odometry {map.odometry}</span>
        </div>
      )}
    </div>
  );
}
