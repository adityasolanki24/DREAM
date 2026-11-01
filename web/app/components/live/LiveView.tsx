'use client';
import { motion } from 'motion/react';
import { Camera, CheckCircle2, Circle, Expand, Loader2, Map as MapIcon, Maximize, MessageSquare, Minimize, PanelRightClose, ScanEye, Shrink, XCircle } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import { Group, Panel, Separator, useDefaultLayout } from 'react-resizable-panels';
import { useCameraState } from '../../lib/camera';
import { useDraft } from '../../lib/draft';
import { clock } from '../../lib/format';
import { presenceFor } from '../../lib/presence';
import { layoutStorage, useMediaQuery } from '../../lib/prefs';
import { useSession } from '../../lib/session';
import { useUi } from '../../lib/ui';
import { CameraFrame, ageText } from '../camera/CameraFrame';
import { Composer, Suggestions, fillExample, useExamples } from '../chat/Composer';
import { ThreadView } from '../chat/Conversation';
import { DreamStrands } from '../Strands';
import { PresenceLine } from '../PresenceLine';
import { TaskControls } from '../task/TaskControls';
import { Segmented, Tip } from '../ui';
import { WorldMap } from '../world/WorldMap';

function Check({ state, label, value }: { state: 'ok' | 'wait' | 'off' | 'bad'; label: string; value: string }) {
  const Icon = state === 'ok' ? CheckCircle2 : state === 'wait' ? Loader2 : state === 'bad' ? XCircle : Circle;
  return (
    <li className={`check check-${state}`}>
      <Icon size={13} aria-hidden className={state === 'wait' ? 'spin' : ''} />
      <span className="check-label">{label}</span>
      <span className="check-value">{value}</span>
    </li>
  );
}

/** What the stage shows before any frame exists: honest connection facts and a way to make a frame appear. */
function StageEmpty({ onOpenChat }: { onOpenChat: () => void }) {
  const { link, snapshot } = useSession();
  const { setStageView, openSettings } = useUi();
  const cam = useCameraState();
  const { setDraft, focus } = useDraft();
  const examples = useExamples();
  const online = link === 'online';
  const title = link === 'connecting' ? 'Connecting to the session' : !online ? 'The robot backend is unreachable' : 'Waiting for the first frame';
  const body = !online
    ? 'No frame without a live session; the interface keeps retrying in the background.'
    : 'The head camera has not produced a frame yet. Frames stream when the media server runs, or arrive with each observation a task makes.';
  return (
    <div className="stage-empty">
      <div className="stage-floor" aria-hidden />
      <div className="stage-empty-body">
        <span className="stage-kicker"><ScanEye size={14} aria-hidden /> Head camera</span>
        <h2>{title}</h2>
        <p>{body}</p>
        <ul className="checks" aria-label="Camera connection">
          <Check state={online ? 'ok' : link === 'connecting' ? 'wait' : 'bad'} label="Session" value={online ? 'connected' : link === 'connecting' ? 'connecting' : 'offline'} />
          <Check state={cam.rtc.live ? 'ok' : 'off'} label="WebRTC stream" value={cam.rtc.live ? 'live' : online ? 'not connected · retrying every 3 s' : '—'} />
          <Check state={cam.jpeg.active ? 'ok' : online ? 'wait' : 'off'} label="JPEG stream" value={online ? cam.jpeg.status : '—'} />
          <Check state={snapshot?.camera ? 'ok' : 'off'} label="Observation frames" value={snapshot?.camera ? 'available' : 'appear when a tool observes'} />
        </ul>
        <div className="stage-empty-actions">
          {online && examples.map((x) => (
            <button key={x} className="btn sm stage-btn" onClick={() => { setDraft(fillExample(x)); onOpenChat(); requestAnimationFrame(focus); }}>
              Try “{x}”
            </button>
          ))}
          {online && snapshot?.map && <button className="btn sm stage-btn" onClick={() => setStageView('map')}><MapIcon size={13} aria-hidden /> Show the map</button>}
          {!online && <button className="btn sm stage-btn" onClick={openSettings}>Check access token</button>}
        </div>
        <p className="stage-env mono">{snapshot?.environment ?? 'Environment not labelled'}{snapshot ? ` · robot clock ${clock(snapshot.robot_time_s)}` : ''}</p>
      </div>
    </div>
  );
}

/** What DREAM is doing, and the way to stop it. */
function ActivityBar() {
  const { shown, feed, snapshot, link } = useSession();
  const { listening } = useUi();
  const presence = presenceFor(shown, shown ? feed.threads[shown.id] : undefined, snapshot?.stop_reason ?? null, link, listening);
  const runnable = shown && ['running', 'suspended', 'queued'].includes(shown.state);
  return (
    <div className={`activity-hud state-${presence.state}`} role="status" aria-live="polite">
      <DreamStrands state={presence.state} size={34} />
      <div className="activity-copy">
        <b>{presence.title}</b>
        <PresenceLine presence={presence} task={runnable ? shown : undefined} />
      </div>
      {shown && <TaskControls task={shown} size="sm" tone="stage" />}
    </div>
  );
}

function Stage({ chatOpen, setChatOpen, compact }: { chatOpen: boolean; setChatOpen: (v: boolean) => void; compact: boolean }) {
  const { snapshot, link } = useSession();
  const { stageView, setStageView, focusMode, setFocusMode } = useUi();
  const cam = useCameraState();
  const stageRef = useRef<HTMLDivElement>(null);
  const [fullscreen, setFullscreen] = useState(false);
  useEffect(() => {
    const onChange = () => setFullscreen(document.fullscreenElement === stageRef.current);
    document.addEventListener('fullscreenchange', onChange);
    return () => document.removeEventListener('fullscreenchange', onChange);
  }, []);
  const hasFrame = cam.source !== 'none';
  const hasMap = Boolean(snapshot?.map?.cells?.length);
  const odo = snapshot?.odometry;
  const view = stageView;

  return (
    <div className="stage" ref={stageRef} data-view={view} data-frame={hasFrame} data-stopped={Boolean(snapshot?.stop_reason)}>
      <div className={`stage-layer layer-camera ${view === 'camera' ? 'is-main' : 'is-inset'}`} hidden={view !== 'camera' && !hasFrame}
        onClick={view !== 'camera' ? () => setStageView('camera') : undefined} role={view !== 'camera' ? 'button' : undefined} aria-label={view !== 'camera' ? 'Show the camera full size' : undefined}>
        {hasFrame ? <CameraFrame className="stage-frame" badges={view === 'camera'} /> : <StageEmpty onOpenChat={() => setChatOpen(true)} />}
      </div>
      <div className={`stage-layer layer-map ${view === 'map' ? 'is-main' : 'is-inset'}`} hidden={view !== 'map' && (!hasMap || compact)}
        onClick={view !== 'map' ? () => setStageView('map') : undefined} role={view !== 'map' ? 'button' : undefined} aria-label={view !== 'map' ? 'Show the map full size' : undefined}>
        <WorldMap snapshot={snapshot} legend={view === 'map'} compact={view !== 'map'} />
      </div>

      {/* top-left: source and freshness */}
      <div className="hud hud-tl">
        <span className="hud-chip">
          <span className={`dot ${cam.disconnected ? 'tone-warn' : cam.frozen ? 'tone-warn' : hasFrame ? 'tone-good pulse' : ''}`} aria-hidden />
          <span>{view === 'camera' ? cam.transportLabel : 'Spatial map'}</span>
          {view === 'camera' && hasFrame && <span className="hud-faint">{cam.statusLabel}</span>}
          {view === 'camera' && hasFrame && cam.source !== 'webrtc' && <span className="hud-faint num">· {ageText(cam.ageS)} old</span>}
        </span>
        {cam.testPattern && <span className="hud-chip hud-warn">Preview test pattern · not a camera frame</span>}
      </div>

      {/* top-right: perspective, focus, fullscreen, chat */}
      <div className="hud hud-tr">
        <Segmented label="Stage view" value={view} onChange={setStageView} size="sm" options={[
          { id: 'camera', label: 'Camera', icon: <Camera size={13} aria-hidden /> },
          { id: 'map', label: 'Map', icon: <MapIcon size={13} aria-hidden /> },
        ]} />
        {!compact && <Tip label={focusMode ? 'Exit focus' : 'Focus on the stage'} kbd="F"><button className="icon-btn stage-icon" aria-pressed={focusMode} aria-label={focusMode ? 'Exit focus mode' : 'Focus mode'} onClick={() => setFocusMode((v) => !v)}>{focusMode ? <Shrink size={15} /> : <Expand size={15} />}</button></Tip>}
        <Tip label={fullscreen ? 'Exit full screen' : 'Full screen'}><button className="icon-btn stage-icon" aria-label={fullscreen ? 'Exit full screen' : 'Full screen'} onClick={() => (fullscreen ? document.exitFullscreen() : stageRef.current?.requestFullscreen())}>{fullscreen ? <Minimize size={15} /> : <Maximize size={15} />}</button></Tip>
        {!compact && !chatOpen && <Tip label="Open conversation" kbd="C"><button className="icon-btn stage-icon" aria-label="Open conversation" onClick={() => setChatOpen(true)}><MessageSquare size={15} /></button></Tip>}
      </div>

      {/* bottom-left: what DREAM is doing, and the way to stop it */}
      {!compact && <div className="hud hud-bl"><ActivityBar /></div>}

      {/* bottom-right: lightweight real telemetry */}
      {snapshot && (
        <dl className="hud hud-br telemetry mono">
          <div><dt>t</dt><dd>{clock(snapshot.robot_time_s)}</dd></div>
          {odo && odo.length >= 2 && <div><dt>x y</dt><dd>{odo[0].toFixed(2)} {odo[1].toFixed(2)} m</dd></div>}
          {odo && odo.length >= 3 && <div><dt>yaw</dt><dd>{odo[2].toFixed(2)} rad</dd></div>}
          <div><dt>stop</dt><dd className={snapshot.stop_reason ? 'bad' : ''}>{snapshot.stop_reason ?? 'none'}</dd></div>
        </dl>
      )}
    </div>
  );
}

function ChatPanel({ onClose }: { onClose?: () => void }) {
  const { tasks, feed, stopping, models, status } = useSession();
  return (
    <section className="live-chat" aria-label="DREAM conversation">
      <header className="panel-bar">
        <h2>Conversation</h2>
        <span className="faint small num">{tasks.length || ''}</span>
        <span className="conv-spacer" />
        {onClose && <Tip label="Hide conversation" kbd="C"><button className="icon-btn sm" aria-label="Hide conversation" onClick={onClose}><PanelRightClose size={15} /></button></Tip>}
      </header>
      <ThreadView tasks={tasks} threads={feed.threads} stopping={stopping} choices={models.choices} running={status?.running ?? null}
        empty={<p className="thread-empty">Goals you send appear here with every tool DREAM runs and the runtime&apos;s reported outcome.</p>} />
      <div className="live-composer">
        <Suggestions />
        <Composer variant="panel" />
      </div>
    </section>
  );
}

export function LiveView() {
  const { liveChat, setLiveChat, focusMode } = useUi();
  const compact = useMediaQuery('(max-width: 860px)');
  const layout = useDefaultLayout({ id: 'live-split', storage: layoutStorage, panelIds: ['stage', 'chat'] });
  const chatOpen = liveChat && !focusMode;

  if (compact) {
    return (
      <div className="live is-compact">
        <Stage chatOpen compact setChatOpen={() => {}} />
        <div className="compact-activity"><ActivityBar /></div>
        <ChatPanel />
      </div>
    );
  }
  return (
    <div className="live">
      <Group orientation="horizontal" className="live-group" defaultLayout={chatOpen ? layout.defaultLayout : undefined} onLayoutChanged={chatOpen ? layout.onLayoutChanged : undefined}>
        <Panel id="stage" minSize="45%" className="live-stage-panel">
          <Stage chatOpen={chatOpen} setChatOpen={setLiveChat} compact={false} />
        </Panel>
        {chatOpen && <>
            <Separator className="resize-handle" />
            <Panel id="chat" defaultSize={380} minSize={300} maxSize={600} className="live-chat-panel">
              <motion.div className="fill" initial={{ opacity: 0, x: 12 }} animate={{ opacity: 1, x: 0 }} transition={{ duration: 0.22, ease: [0.22, 1, 0.36, 1] }}>
                <ChatPanel onClose={() => setLiveChat(false)} />
              </motion.div>
            </Panel>
        </>}
      </Group>
    </div>
  );
}
