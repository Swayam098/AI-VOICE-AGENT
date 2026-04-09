import React from 'react';
import { Cpu, Globe, MessageCircle, AlertTriangle } from 'lucide-react';

const IntelligenceDashboard = ({ logs }) => {
  return (
    <div className="glass-panel" style={{ display: 'flex', flexDirection: 'column', height: '100%', padding: '1.5rem', overflow: 'hidden' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '1.5rem' }}>
        <div style={{ padding: '0.75rem', background: 'rgba(139, 92, 246, 0.1)', borderRadius: '12px' }}>
          <Cpu size={24} color="var(--accent-purple)" />
        </div>
        <div>
          <h2 style={{ margin: 0, fontSize: '1.25rem', fontWeight: 600 }}>Intelligence Logs</h2>
          <span style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>Real-time telemetry & routing</span>
        </div>
      </div>

      <div style={{ flex: 1, overflowY: 'auto', paddingRight: '0.5rem', display: 'flex', flexDirection: 'column', gap: '1rem' }}>
        {logs.length === 0 ? (
          <div style={{ display: 'flex', height: '100%', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)' }}>
            Awaiting conversation data...
          </div>
        ) : (
          logs.map((log, index) => (
            <div key={index} style={{ 
              background: 'rgba(0, 0, 0, 0.2)', 
              borderRadius: '12px', 
              padding: '1rem',
              borderLeft: `4px solid ${log.type === 'user' ? 'var(--accent-blue)' : log.type === 'error' ? 'var(--danger)' : 'var(--accent-purple)'}`
            }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem', fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                <span style={{ textTransform: 'uppercase', fontWeight: 600, letterSpacing: '0.5px' }}>{log.type}</span>
                <span>{log.timestamp}</span>
              </div>
              
              <div style={{ marginBottom: log.metadata ? '1rem' : '0', color: 'var(--text-primary)', lineHeight: 1.5 }}>
                {log.text}
              </div>

              {log.metadata && (
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem', marginTop: '0.75rem' }}>
                  {log.metadata.detected_language && (
                    <span className="glass-pill" style={{ color: 'var(--accent-blue)' }}>
                      <Globe size={12} /> {log.metadata.detected_language}
                    </span>
                  )}
                  {log.metadata.intent && (
                    <span className="glass-pill" style={{ color: 'var(--accent-pink)' }}>
                      <MessageCircle size={12} /> {log.metadata.intent}
                    </span>
                  )}
                  {log.metadata.routing && (
                    <span className="glass-pill" style={{ color: 'var(--accent-green)' }}>
                      <Cpu size={12} /> Model: {log.metadata.routing.model}
                    </span>
                  )}
                  {log.metadata.sentiment?.urgency_score > 50 && (
                    <span className="glass-pill" style={{ color: 'var(--danger)', borderColor: 'rgba(239, 68, 68, 0.3)' }}>
                      <AlertTriangle size={12} /> Urgent
                    </span>
                  )}
                </div>
              )}
            </div>
          ))
        )}
      </div>
    </div>
  );
};

export default IntelligenceDashboard;
