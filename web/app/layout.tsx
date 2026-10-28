import type { Metadata, Viewport } from 'next';
import './style.css';

export const metadata: Metadata = {
  title: 'DREAM | Mission control',
  description: 'Chat with DREAM, watch the robot work, and stop it safely.',
};
export const viewport: Viewport = { width: 'device-width', initialScale: 1 };

// Runs before first paint (also in the static export). 'dream-theme' holds system | light | dark and is not the
// access token. In system mode the page follows prefers-color-scheme, including live changes.
const themeScript = `(function(){
  var KEY='dream-theme', root=document.documentElement, mq=window.matchMedia('(prefers-color-scheme: dark)');
  function stored(){try{var v=localStorage.getItem(KEY);return v==='light'||v==='dark'?v:'system';}catch(e){return 'system';}}
  function apply(){var m=stored();var t=m==='system'?(mq.matches?'dark':'light'):m;root.setAttribute('data-theme',t);root.setAttribute('data-theme-mode',m);}
  apply();
  var onChange=function(){if(stored()==='system')apply();};
  if(mq.addEventListener)mq.addEventListener('change',onChange);else if(mq.addListener)mq.addListener(onChange);
  window.dreamTheme={get:stored,set:function(m){try{if(m==='system')localStorage.removeItem(KEY);else localStorage.setItem(KEY,m);}catch(e){}apply();}};
})();`;

export default function Layout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" data-theme="light" suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: themeScript }} />
      </head>
      <body>{children}</body>
    </html>
  );
}
