 // src/components/VoiceAssistant.js
import React, { useState, useRef } from "react";
import toast from 'react-hot-toast';

// Animated waveform SVG (simple placeholder)
const Waveform = () => (
  <svg width="180" height="90" viewBox="0 0 180 90" fill="none">
    <defs>
      <linearGradient id="wave" x1="0" y1="0" x2="180" y2="90" gradientUnits="userSpaceOnUse">
        <stop stopColor="#7F5FFF" />
        <stop offset="1" stopColor="#5DF2E6" />
      </linearGradient>
    </defs>
    <path
      d="M0 45 Q30 10 60 45 T120 45 T180 45"
      stroke="url(#wave)" strokeWidth="4" fill="none" opacity="0.7">
      <animate attributeName="d" values="M0 45 Q30 10 60 45 T120 45 T180 45;M0 45 Q30 80 60 45 T120 45 T180 45;M0 45 Q30 10 60 45 T120 45 T180 45" dur="2s" repeatCount="indefinite" />
    </path>
  </svg>
);

const VoiceAssistant = () => {
  const [open, setOpen] = useState(false);
  const [listening, setListening] = useState(false);
  const [input, setInput] = useState("");
  const [aiResponse, setAiResponse] = useState("");
  const [loading, setLoading] = useState(false);
  const [notified, setNotified] = useState(false);
  const audioRef = useRef(null);
  const recognitionRef = useRef(null);

  // Start/stop browser voice recognition (Web Speech API)
  const handleVoiceInput = () => {
    if (!('webkitSpeechRecognition' in window)) {
      alert('Voice recognition not supported in this browser.');
      return;
    }
    if (!listening) {
      const recognition = new window.webkitSpeechRecognition();
      recognition.lang = 'en-US';
      recognition.interimResults = false;
      recognition.maxAlternatives = 1;
      recognition.onresult = (event) => {
        const transcript = event.results[0][0].transcript;
        setInput(transcript);
        setListening(false);
      };
      recognition.onend = () => setListening(false);
      recognition.onerror = () => setListening(false);
      recognition.start();
      recognitionRef.current = recognition;
      setListening(true);
    } else {
      recognitionRef.current && recognitionRef.current.stop();
      setListening(false);
    }
  };

  // Send message to RAG backend and display answer, then play TTS
  const handleSend = async () => {
    if (!input.trim()) return;
    setLoading(true);
    setAiResponse("");
    try {
      const response = await fetch("/api/rag-assistant", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: input })
      });
      const data = await response.json();
      if (data.answer) {
        setAiResponse(data.answer);
        // TTS: Only for successful answers
        try {
          const ttsResp = await fetch("/api/tts", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text: data.answer })
          });
          const ttsData = await ttsResp.json();
          if (ttsData.audio && audioRef.current) {
            audioRef.current.src = `data:audio/mp3;base64,${ttsData.audio}`;
            audioRef.current.play();
          }
        } catch (ttsErr) {
          toast.error("TTS failed");
        }
      } else if (data.error) setAiResponse(data.error);
      else setAiResponse("Sorry, no answer returned.");
    } catch (err) {
      setAiResponse("Sorry, there was an error connecting to the backend.");
    } finally {
      setLoading(false);
      setInput("");
    }
  };

  // Reset notification on modal open
  const handleOpen = () => {
    setOpen(true);
    setNotified(false);
  };

  // UI text
  const userName = "Doctor"; // Replace with dynamic name if available
  const greeting = `Hi, ${userName}`;
  const prompt = "How can I help you ?";

  return (
    <>
      {/* Floating Voice Button */}
      <button
        className="fixed z-50 bottom-6 right-6 bg-gradient-to-br from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 text-white rounded-full shadow-lg w-16 h-16 flex items-center justify-center focus:outline-none transition-all duration-200"
        onClick={handleOpen}
        aria-label="Open Voice Assistant"
      >
        <svg className="w-8 h-8" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" d="M12 18v2m0 0c-3.314 0-6-2.686-6-6v-2a6 6 0 1112 0v2c0 3.314-2.686 6-6 6z" />
        </svg>
      </button>

      {/* Modal - Redesigned Voice Search UI */}
      {open && (
        <div className="fixed inset-0 z-50 flex items-center justify-center">
          {/* Abstract lines background */}
          <div className="absolute inset-0 bg-gradient-to-br from-[#181A2A] to-[#2B2250]">
            <svg className="absolute top-0 left-0 w-full h-full opacity-20" viewBox="0 0 400 400" fill="none">
              <path d="M0 100 Q200 200 400 100" stroke="#7F5FFF" strokeWidth="2" fill="none" />
              <path d="M0 200 Q200 300 400 200" stroke="#5DF2E6" strokeWidth="2" fill="none" />
              <path d="M0 300 Q200 400 400 300" stroke="#7F5FFF" strokeWidth="2" fill="none" />
            </svg>
          </div>
          {/* Main Card */}
          <div className="relative z-10 w-full max-w-md mx-auto rounded-3xl shadow-2xl overflow-hidden bg-gradient-to-br from-[#23244A] to-[#2B2250] border border-blue-900">
            {/* Top Bar */}
            <div className="flex items-center justify-between px-4 py-3 border-b border-blue-800 bg-transparent">
              <button onClick={() => setOpen(false)} className="text-blue-200 hover:text-white p-1 rounded-full">
                <svg className="w-7 h-7" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
              <span className="text-blue-100 font-semibold text-lg">Voice Search</span>
              <button className="text-blue-200 hover:text-white p-1 rounded-full">
                <svg className="w-6 h-6" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
                  <circle cx="12" cy="12" r="2" />
                  <circle cx="19" cy="12" r="2" />
                  <circle cx="5" cy="12" r="2" />
                </svg>
              </button>
            </div>
            {/* Greeting & Prompt */}
            <div className="flex flex-col items-center justify-center py-6">
              <span className="text-blue-200 text-base mb-1">{greeting}</span>
              <span className="text-white text-2xl font-bold mb-2">{prompt}</span>
            </div>
            {/* Animated Waveform in Glassy Sphere */}
            <div className="flex items-center justify-center mb-6">
              <div className="w-44 h-44 rounded-full bg-gradient-to-br from-[#7F5FFF33] to-[#5DF2E633] flex items-center justify-center shadow-2xl relative">
                <div className="absolute inset-0 rounded-full bg-gradient-to-br from-[#7F5FFF22] to-[#5DF2E622] blur-xl" />
                <Waveform />
              </div>
            </div>
            {/* AI Response */}
            {aiResponse && (
              <div className="flex items-center justify-center pb-4">
                <span className="text-blue-100 text-base text-center px-6 font-medium">
                  {aiResponse}
                </span>
              </div>
            )}
            {/* Input Area */}
            <div className="flex items-center justify-center gap-2 pb-8 px-6">
              <input
                type="text"
                className="flex-1 border rounded-full px-4 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-400 bg-[#23244A] text-white placeholder-blue-200"
                placeholder="Type your question..."
                value={input}
                onChange={e => setInput(e.target.value)}
                onKeyDown={e => e.key === 'Enter' && handleSend()}
                disabled={loading}
              />
              <button
                onClick={handleSend}
                className="bg-blue-600 hover:bg-blue-700 text-white rounded-full px-4 py-2 text-sm font-semibold shadow"
                disabled={!input.trim() || loading}
              >
                {loading ? '...' : 'Send'}
              </button>
              <button
                onClick={handleVoiceInput}
                className={`rounded-full p-3 bg-gradient-to-br from-blue-600 to-purple-600 shadow-lg focus:outline-none ${listening ? "animate-pulse" : ""}`}
                aria-label="Voice Input"
                disabled={loading}
              >
                <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" d="M12 18v2m0 0c-3.314 0-6-2.686-6-6v-2a6 6 0 1112 0v2c0 3.314-2.686 6-6 6z" />
                </svg>
              </button>
            </div>
          </div>
        </div>
      )}
      {/* Hidden audio element for TTS playback */}
      <audio ref={audioRef} style={{ display: 'none' }} />
    </>
  );
};

export default VoiceAssistant;