'use client';
import { AnimatePresence, LayoutGroup, motion } from 'motion/react';
import { ArrowUpRight, MessagesSquare, Minimize2 } from 'lucide-react';
import { describeMeta } from '../../../modelChoice.mjs';
import { useCameraState } from '../../lib/camera';
import { RUNNABLE, clock, relativeAge, shortModelLabel, stateTone } from '../../lib/format';
import { presenceFor, type Presence } from '../../lib/presence';
import { useMediaQuery } from '../../lib/prefs';
import { useSession } from '../../lib/session';
import { useUi } from '../../lib/ui';
import { CameraFrame } from '../camera/CameraFrame';
import { Composer, Suggestions } from '../chat/Composer';
import { AgentTurn, ThreadView } from '../chat/Conversation';
import { DreamStrands } from '../Strands';
import { PresenceLine } from '../PresenceLine';
import { TaskControls } from '../task/TaskControls';
import { Tip } from '../ui';

const spring = { type: 'spring', duration: 0.5, bounce: 0.1 } as const;

function headline(p: Presence, online: boolean): string {
  if (!online) return p.title;
  if (p.state === 'idle') return 'Ready when you are';
  return p.title;
}

function SessionFacts() {
  const { snapshot, models, link } = useSession();
  const provider = models.choices.find((c) => c.id === models.choice?.provider);
  return (
    <dl className="facts-line">
      <div><dt>Environment</dt><dd>{snapshot?.environment ?? 'not labelled'}</dd></div>
      <div><dt>Robot clock</dt><dd className="num">{snapshot ? clock(snapshot.robot_time_s) : '—'}</dd></div>
      <div><dt>Safety stop</dt><dd className={snapshot?.stop_reason ? 'bad' : ''}>{link !== 'online' ? '—' : snapshot?.stop_reason ?? 'none'}</dd></div>
      <div className="facts-provider"><dt>Parser</dt><dd title={provider?.label}>{shortModelLabel(models.choices, models.choice?.provider, models.choice?.model) ?? '—'}</dd></div>
    </dl>
  );
}

function Recent({ excludeId, onOpen }: { excludeId?: string; onOpen: () => void }) {
  const { tasks } = useSession();
  const recent = [...tasks].filter((t) => t.id !== excludeId).sort((a, b) => b.updated - a.updated).slice(0, 3);
  if (!recent.length) return null;
  return (
    <div className="recent">
      <div className="recent-head"><span className="rail-label">Earlier</span><button className="text-btn" onClick={onOpen}>All activity <ArrowUpRight size={12} aria-hidden /></button></div>
      <ul>
        {recent.map((t) => (
          <li key={t.id}>
            <button onClick={onOpen}>
              <span className={`dot tone-${stateTone(t.state)}`} aria-hidden />
              <span className="recent-text">{t.text}</span>
              <span className="recent-state">{t.state}</span>
              <time className="recent-time">{relativeAge(t.updated)}</time>
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
}

function CameraGlance() {
  const cam = useCameraState();
  const { setMode } = useUi();
  if (cam.source === 'none') return null;
  return (
    <button className="glance" onClick={() => setMode('live')} aria-label="Open the live camera">
      <CameraFrame className="glance-frame" badges={false} />
      <span className="glance-label"><span className={`dot ${cam.disconnected ? 'tone-warn' : 'tone-good'}`} /> {cam.transportLabel} <ArrowUpRight size={12} aria-hidden /></span>
    </button>
  );
}

export function CompanionView() {
  const { shown, feed, snapshot, link, tasks, stopping, models, status } = useSession();
  const { listening, companionExpanded, setCompanionExpanded } = useUi();
  const narrow = useMediaQuery('(max-width: 860px)');
  const online = link === 'online';
  const thread = shown ? feed.threads[shown.id] : undefined;
  const presence = presenceFor(shown, thread, snapshot?.stop_reason ?? null, link, listening);
  const expanded = companionExpanded && tasks.length > 0;
  // The latest exchange earns the space only while it is live or just ended; otherwise it joins the history list.
  const featured = shown && (RUNNABLE.includes(shown.state) || Date.now() / 1000 - shown.updated < 90) ? shown : undefined;

  return (
    <LayoutGroup>
      <div className={`companion ${expanded ? 'is-expanded' : 'is-home'}`}>
        <AnimatePresence mode="popLayout" initial={false}>
          {!expanded ? (
            <motion.section key="home" className="companion-home scroll" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} transition={{ duration: 0.2 }}>
              <div className="home-column">
                <div className="home-spacer top" aria-hidden />
                <motion.div layoutId="presence-strands" transition={spring} className="home-strands">
                  <DreamStrands state={presence.state} size={narrow ? 140 : 184} />
                </motion.div>
                <div className="presence-copy" role="status" aria-live="polite">
                  <motion.h1 layoutId="presence-title" transition={spring} className="presence-title">{headline(presence, online)}</motion.h1>
                  <PresenceLine className="presence-detail" presence={presence} task={shown} />
                </div>
                {shown && <div className="home-controls"><TaskControls task={shown} size="sm" /></div>}
                <motion.div layoutId="presence-composer" transition={spring} className="home-composer">
                  <Composer variant="hero" autoFocus={!narrow} />
                </motion.div>
                <Suggestions className="home-suggestions" />
                {featured && shown && (
                  <div className="latest">
                    <div className="latest-head">
                      <span className="rail-label">Latest</span>
                      <button className="text-btn" onClick={() => setCompanionExpanded(true)}><MessagesSquare size={12} aria-hidden /> Open conversation</button>
                    </div>
                    <div className="latest-body">
                      <p className="latest-goal">{shown.text}</p>
                      <AgentTurn task={shown} thread={thread} stopping={stopping.has(shown.id)} modelLabel={shortModelLabel(models.choices, shown.meta?.provider, shown.meta?.model)} modelTitle={describeMeta(models.choices, shown.meta)} compact />
                    </div>
                  </div>
                )}
                <Recent excludeId={featured?.id} onOpen={() => setCompanionExpanded(true)} />
                <div className="home-spacer bottom" aria-hidden />
                <SessionFacts />
              </div>
            </motion.section>
          ) : (
            <motion.section key="conv" className="companion-conv" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} transition={{ duration: 0.2 }}>
              <aside className="presence-rail">
                <motion.div layoutId="presence-strands" transition={spring} className="rail-strands">
                  <DreamStrands state={presence.state} size={narrow ? 40 : 128} />
                </motion.div>
                <div className="rail-copy" role="status" aria-live="polite">
                  <motion.h1 layoutId="presence-title" transition={spring} className="rail-title">{presence.title}</motion.h1>
                  <PresenceLine className="rail-detail" presence={presence} task={shown} />
                </div>
                {shown && <TaskControls task={shown} size="sm" />}
                {narrow && <button className="icon-btn sm" aria-label="Collapse conversation" onClick={() => setCompanionExpanded(false)}><Minimize2 size={14} /></button>}
                {!narrow && <CameraGlance />}
                {!narrow && <SessionFacts />}
              </aside>
              <div className="conv-main">
                <header className="conv-head">
                  <h2>Conversation</h2>
                  <span className="faint small num">{tasks.length} {tasks.length === 1 ? 'task' : 'tasks'}</span>
                  <span className="conv-spacer" />
                  <Tip label="Back to the companion home" kbd="E"><button className="btn ghost sm" onClick={() => setCompanionExpanded(false)}><Minimize2 size={13} aria-hidden /> Compact</button></Tip>
                </header>
                <ThreadView tasks={tasks} threads={feed.threads} stopping={stopping} choices={models.choices} running={status?.running ?? null} />
                <motion.div layoutId="presence-composer" transition={spring} className="conv-composer">
                  <Suggestions />
                  <Composer variant="docked" autoFocus={!narrow} />
                </motion.div>
              </div>
            </motion.section>
          )}
        </AnimatePresence>
      </div>
    </LayoutGroup>
  );
}
