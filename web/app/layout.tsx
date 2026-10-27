import type { Metadata, Viewport } from 'next';
import './style.css';

export const metadata: Metadata = {
  title: 'DREAM | Mission control',
  description: 'Chat with DREAM, watch the robot work, and stop it safely.',
};
export const viewport: Viewport = { width: 'device-width', initialScale: 1, themeColor: '#111111' };

export default function Layout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
