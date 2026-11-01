import type { Metadata, Viewport } from 'next';
import { GeistMono } from 'geist/font/mono';
import { GeistSans } from 'geist/font/sans';
import './style.css';
import './styles/shell.css';
import './styles/strands.css';
import './styles/chat.css';
import './styles/companion.css';
import './styles/live.css';
import './styles/research.css';

export const metadata: Metadata = {
  title: 'DREAM',
  description: 'Chat with DREAM, watch the robot work, and stop it safely.',
};
export const viewport: Viewport = { width: 'device-width', initialScale: 1, themeColor: '#0d0e10' };

// Runs before first paint (also in the static export). 'dream-theme' holds system | light | dark and is not the
// access token. DREAM is dark-first: with nothing stored it is dark. In system mode the page follows
// prefers-color-scheme, including live changes.
const themeScript = `(function(){
  var KEY='dream-theme', root=document.documentElement, mq=window.matchMedia('(prefers-color-scheme: dark)');
  function stored(){try{var v=localStorage.getItem(KEY);return v==='light'||v==='system'?v:'dark';}catch(e){return 'dark';}}
  function apply(){var m=stored();var t=m==='system'?(mq.matches?'dark':'light'):m;root.setAttribute('data-theme',t);root.setAttribute('data-theme-mode',m);}
  apply();
  var onChange=function(){if(stored()==='system')apply();};
  if(mq.addEventListener)mq.addEventListener('change',onChange);else if(mq.addListener)mq.addListener(onChange);
  window.dreamTheme={get:stored,set:function(m){try{if(m==='dark')localStorage.removeItem(KEY);else localStorage.setItem(KEY,m);}catch(e){}apply();}};
})();`;

export default function Layout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" data-theme="dark" className={`${GeistSans.variable} ${GeistMono.variable}`} suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: themeScript }} />
      </head>
      <body>{children}</body>
    </html>
  );
}
