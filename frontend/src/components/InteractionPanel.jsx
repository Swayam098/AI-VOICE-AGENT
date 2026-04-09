import React, { useState } from 'react';
import { Send, Mic, Activity } from 'lucide-react';

const InteractionPanel = ({ onNewLog, isProcessing }) => {
  const [inputText, setInputText] = useState('');

  const handleSend = async () => {
    if (!inputText.trim() || isProcessing) return;
    
    const textToProcess = inputText;
    setInputText('');
    
    try {
      const response = await fetch('http://127.0.0.1:8000/api/test/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text: textToProcess,
          call_sid: 'dashboard-test'
        })
      });
      
      const data = await response.json();
      onNewLog({ type: 'user', text: textToProcess, timestamp: new Date().toLocaleTimeString() });
      onNewLog({ 
        type: 'agent', 
        text: data.response, 
        timestamp: new Date().toLocaleTimeString(),
        metadata: data
      });
    } catch (error) {
      console.error("Failed to communicate with API", error);
      onNewLog({ type: 'error', text: "Failed to reach AI Backend.", timestamp: new Date().toLocaleTimeString() });
    }
  };

  return (
    <div className="glass-panel" style={{ display: 'flex', flexDirection: 'column', height: '100%', padding: '1.5rem' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '2rem' }}>
        <div style={{ padding: '0.75rem', background: 'rgba(59, 130, 246, 0.1)', borderRadius: '12px' }}>
          <Activity size={24} color="var(--accent-blue)" />
        </div>
        <div>
          <h2 style={{ margin: 0, fontSize: '1.25rem', fontWeight: 600 }}>Live Interaction</h2>
          <span style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>Test language routing via text</span>
        </div>
      </div>

      {isProcessing && (
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: '1rem' }}>
          <div className="waveform-container">
            <div className="wave-bar"></div>
            <div className="wave-bar"></div>
            <div className="wave-bar"></div>
            <div className="wave-bar"></div>
            <div className="wave-bar"></div>
          </div>
          <span style={{ color: 'var(--accent-purple)', fontSize: '0.9rem', fontWeight: 500, letterSpacing: '1px' }}>
            AGENT IS PROCESSING...
          </span>
        </div>
      )}

      {!isProcessing && (
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)' }}>
          <Mic size={48} style={{ opacity: 0.2, marginBottom: '1rem' }} />
          <p>Ready for input</p>
        </div>
      )}

      <div style={{ display: 'flex', gap: '0.75rem', marginTop: 'auto' }}>
        <input 
          type="text" 
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && handleSend()}
          placeholder="Say something in English, Hindi, or Hinglish..."
          className="glass-input"
          disabled={isProcessing}
        />
        <button 
          onClick={handleSend} 
          className="glass-button" 
          disabled={!inputText.trim() || isProcessing}
          style={{ padding: '0.75rem 1rem' }}
        >
          <Send size={20} />
        </button>
      </div>
    </div>
  );
};

export default InteractionPanel;
