These components are vendored from React Bits (https://reactbits.dev, repository DavidHDev/react-bits),
retrieved 2026-11-01 from the package registry endpoints (`reactbits.dev/r/<Name>-TS-CSS`):

- `Strands.tsx` + `Strands.css` — originally unmodified; on 2026-11-04 a `roll` prop and `uRoll` uniform
  were added so the presence orb can be spun by pointer (the strands' sample space rotates inside the
  glass sphere; sphere shading stays view-fixed; a slow idle roll integrates in the animation loop).
  The change is documented in a comment at the top of `Strands.tsx`
- `ThoughtLine.tsx` + `ThoughtLine.css` — the three @hugeicons icons are swapped for lucide-react
  equivalents (SparklesIcon→Sparkles, ArrowDown01Icon→ChevronDown, Tick02Icon→Check) so no extra icon
  package is added; the swap is documented in a comment at the top of `ThoughtLine.tsx`
- `StarBorder.tsx` + `StarBorder.css` — unmodified
- `GradientWaves.tsx` + `GradientWaves.css` — unmodified
- `LightRays.tsx` + `LightRays.css` — unmodified

React Bits is MIT licensed: https://github.com/DavidHDev/react-bits/blob/main/LICENSE
