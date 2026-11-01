'use client';
import GradientWaves from './reactbits/GradientWaves';
import LightRays from './reactbits/LightRays';
import { useThemeIsLight } from './Strands';

/**
 * The one background layer for the whole interface — Companion, Live, Research, open chats and the
 * connect screen. GradientWaves rolls in from below and LightRays beams down from top-centre. Mounted
 * a single time in AppShell inside .app so every tab sees the identical layer; pointer gimmicks (mouse
 * light, cursor shadow, click swell) are off so the backdrop reacts to nothing, and it never carries
 * live state.
 */
export function AmbientBackdrop() {
  const light = useThemeIsLight();
  return (
    <div className="ambient" aria-hidden>
      <div className="ambient-waves">
        <GradientWaves
          horizonColor={light ? '#C8D4E2' : '#131C46'}
          waveColor={light ? '#DFE8F2' : '#0A0E1E'}
          crestColor={light ? '#7395BF' : '#6E8DFF'}
          speed={0.75}
          opacity={light ? 0.35 : 0.5}
          dpr={1.25}
          mouseInteraction={false}
          swellOnClick={false}
        />
      </div>
      <div className="ambient-rays">
        <LightRays
          raysOrigin="top-center"
          raysColor={light ? '#88A8CF' : '#9AD8FF'}
          raysSpeed={0.35}
          opacity={light ? 0.2 : 0.28}
          followMouse={false}
          cursorShadow={false}
          clickPulse={false}
          maxDevicePixelRatio={1.25}
        />
      </div>
    </div>
  );
}
