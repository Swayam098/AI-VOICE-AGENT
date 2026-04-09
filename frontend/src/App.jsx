import React, { useState } from 'react';
import InteractionPanel from './components/InteractionPanel';
import IntelligenceDashboard from './components/IntelligenceDashboard';
import { PhoneCall } from 'lucide-react';
import './index.css';

function App() {
  const [logs, setLogs] = useState([]);
  const [isProcessing, setIsProcessing] = useState(false);

  const handleNewLog = (log) => {
    if (log.type === 'user') setIsProcessing(true);
    if (log.type === 'agent' || log.type === 'error') setIsProcessing(false);
    
    setLogs((prev) => [log, ...prev]);
  };

  return (
    <div style={{ padding: '2rem', maxWidth: '1400px', margin: '0 auto', height: '100vh', display: 'flex', flexDirection: 'column' }}>
      <header style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '2rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
          <div className="glass-panel" style={{ padding: '0.75rem', borderRadius: '12px' }}>
            <PhoneCall size={28} color="var(--accent-blue)" />
          </div>
          <div>
            <h1 style={{ margin: 0, fontSize: '1.75rem', fontWeight: 700, letterSpacing: '-0.5px' }}>Nexus Voice AI</h1>
            <span style={{ color: 'var(--text-secondary)', fontSize: '0.9rem' }}>Multilingual Intelligence Agent</span>
          </div>
        </div>
        <div className="glass-pill" style={{ borderColor: 'rgba(16, 185, 129, 0.3)', color: 'var(--accent-green)' }}>
          <span style={{ width: 8, height: 8, borderRadius: '50%', background: 'var(--accent-green)', display: 'inline-block' }}></span>
          System Online
        </div>
      </header>

      <main style={{ display: 'grid', gridTemplateColumns: 'minmax(400px, 1fr) minmax(400px, 1.5fr)', gap: '1.5rem', flex: 1, minHeight: 0 }}>
        <section style={{ height: '100%' }}>
          <InteractionPanel onNewLog={handleNewLog} isProcessing={isProcessing} />
        </section>
        
        <section style={{ height: '100%' }}>
          <IntelligenceDashboard logs={logs} />
        </section>
      </main>
    </div>
  );
}

export default App;
