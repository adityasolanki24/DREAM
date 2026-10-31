'use client';
import { Box } from 'lucide-react';
import { useCameraState } from '../../lib/camera';
import { clock, relativeAge } from '../../lib/format';
import { useSession } from '../../lib/session';
import { ageText } from '../camera/CameraFrame';
import { Quiet } from '../ui';

function Row({ k, v, tone }: { k: string; v: React.ReactNode; tone?: string }) {
  return <div className="diag-row"><dt>{k}</dt><dd className={tone}>{v}</dd></div>;
}

/** Simulation, session and streaming facts exactly as the backend and the browser measured them. */
export function Diagnostics() {
  const { snapshot, link, stream, status, models, fixture } = useSession();
  const cam = useCameraState();
  const odo = snapshot?.odometry;
  return (
    <div className="diagnostics">
      <section>
        <h3 className="rail-label">Simulation</h3>
        <dl>
          <Row k="Environment" v={snapshot?.environment ?? 'not labelled'} />
          <Row k="Robot clock" v={snapshot ? clock(snapshot.robot_time_s) : '—'} />
          <Row k="Odometry" v={odo ? odo.map((x) => x.toFixed(3)).join(', ') : '—'} />
          <Row k="Odometry source" v={snapshot?.map?.odometry ?? '—'} />
          <Row k="Safety stop" v={snapshot ? (snapshot.stop_reason ?? 'none') : '—'} tone={snapshot?.stop_reason ? 'bad' : ''} />
          <Row k="Running task" v={status?.running ?? 'none'} />
        </dl>
      </section>
      <section>
        <h3 className="rail-label">Camera stream</h3>
        <dl>
          <Row k="Shown source" v={cam.transportLabel} />
          <Row k="WebRTC" v={cam.rtc.live ? `${cam.rtc.fps} fps · dropped ${cam.rtc.dropped} · jitter-buffer ${Math.round(cam.rtc.jitterMs)} ms (not end-to-end)` : 'not connected'} />
          <Row k="JPEG" v={cam.jpeg.active ? `${cam.jpeg.status} · ${cam.jpeg.fps} fps · dropped ${cam.jpeg.dropped}` : cam.rtc.live ? 'idle (WebRTC active)' : cam.jpeg.status} />
          {cam.jpeg.active && <Row k="JPEG timing" v={`age ≈${Math.round(cam.jpeg.ageMs)} ms · transport ${Math.round(cam.jpeg.transportMs)} · decode ${Math.round(cam.jpeg.decodeMs)} · encode ${Math.round(cam.jpeg.encodeMs)} ms`} />}
          <Row k="Frame age" v={ageText(cam.ageS)} />
          <Row k="State" v={cam.disconnected ? 'disconnected' : cam.frozen ? 'paused or not sensing' : cam.source === 'none' ? 'no frame' : 'receiving'} tone={cam.disconnected || cam.frozen ? 'amber' : ''} />
        </dl>
      </section>
      <section>
        <h3 className="rail-label">Session</h3>
        <dl>
          <Row k="Link" v={link} tone={link === 'online' ? 'good' : 'amber'} />
          <Row k="Event stream" v={stream.id ? `${stream.id.slice(0, 12)} · cursor ${stream.cursor}` : '—'} />
          <Row k="Default provider" v={snapshot?.provider_label ?? '—'} />
          <Row k="Providers" v={models.choices.map((c) => `${c.id}${c.available ? '' : ' (unavailable)'}`).join(', ') || '—'} />
          {fixture && <Row k="Data" v={`interface-preview fixture “${fixture.name}” · not robot data`} tone="amber" />}
        </dl>
      </section>
    </div>
  );
}

/** Session object candidates and the read-only home memory. */
export function MemoryView() {
  const { snapshot } = useSession();
  const home = snapshot?.persistent_memory;
  const objects = snapshot?.objects ?? [];
  return (
    <div className="memory">
      <section>
        <h3 className="rail-label">Seen this session <span className="faint">· {objects.length} candidates · not persistent</span></h3>
        {objects.length ? (
          <table className="data-table">
            <thead><tr><th>Id</th><th>Colour</th><th className="r">x</th><th className="r">y</th><th className="r">Views</th></tr></thead>
            <tbody>
              {objects.map((o) => (
                <tr key={o.id}>
                  <td className="mono">{o.id}</td>
                  <td><span className="swatch" style={o.color ? { background: o.color } : undefined} aria-hidden /> {o.color ?? '—'}</td>
                  <td className="r mono">{o.position[0].toFixed(2)}</td>
                  <td className="r mono">{o.position[1].toFixed(2)}</td>
                  <td className="r mono">{o.views}</td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : <p className="faint small pad-y">Objects appear after the robot looks around.</p>}
      </section>
      <section>
        <h3 className="rail-label">Home memory <span className="faint">· read-only</span></h3>
        {home?.available ? (
          home.objects?.length ? (
            <>
              <p className="faint small">{home.source}</p>
              <ul className="mem-list">
                {home.objects.map((o) => (
                  <li key={o.id}><Box size={13} aria-hidden /><div><b>{o.label}</b><span>{o.map_id} · ({o.position.slice(0, 2).map((v) => v.toFixed(2)).join(', ')}) m · {o.observation_source} · seen {relativeAge(o.last_seen)}</span></div></li>
                ))}
              </ul>
            </>
          ) : <p className="faint small pad-y">Home memory is empty. {home.source}</p>
        ) : (
          <Quiet title="Not connected">{home?.error ?? 'Start the backend with --memory <home-memory.sqlite> to show stored sightings (read-only).'}</Quiet>
        )}
      </section>
    </div>
  );
}
