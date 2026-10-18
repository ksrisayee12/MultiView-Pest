import React, { useState, useEffect, useRef } from 'react';
import {
  MessageSquare,
  Send,
  X,
  Sparkles,
  Bot,
  User,
  ShieldCheck,
  AlertCircle,
  HelpCircle,
  Loader2,
  Sprout,
  Leaf,
} from 'lucide-react';
import { FormattedAiResponse } from './FormattedAiResponse';

interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  guardrailTriggered?: boolean;
  engine?: string;
}

interface PlantChatbotDrawerProps {
  isOpen: boolean;
  onClose: () => void;
}

export const PlantChatbotDrawer: React.FC<PlantChatbotDrawerProps> = ({
  isOpen,
  onClose,
}) => {
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: 'welcome',
      role: 'assistant',
      content:
        "👋 Welcome! I am **AgriHealth Botanical AI**, your dedicated plant science and agronomy advisor.\n\nAsk me anything regarding:\n- 🌱 Crop cultivation & greenhouse climate\n- 💧 Irrigation, soil chemistry & fertilization\n- 🐛 Organic pest control & companion planting\n- 🔬 Plant pathology & foliar health\n\n*Note: I am guardrailed strictly to plant science and agricultural topics!*",
      guardrailTriggered: false,
    },
  ]);
  const [input, setInput] = useState<string>('');
  const [loading, setLoading] = useState<boolean>(false);
  const [groqStatus, setGroqStatus] = useState<string>('Checking...');
  const [hasGroqKey, setHasGroqKey] = useState<boolean>(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (isOpen) {
      checkStatus();
    }
  }, [isOpen]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, loading]);

  const checkStatus = async () => {
    try {
      const res = await fetch('/api/chat/status');
      if (res.ok) {
        const data = await res.json();
        setHasGroqKey(data.has_groq_api_key);
        setGroqStatus(data.has_groq_api_key ? 'Groq Llama 3.3 Active' : 'Offline Knowledgebase');
      }
    } catch {
      setGroqStatus('Offline Knowledgebase');
    }
  };

  const handleSend = async (textToSend?: string) => {
    const text = (textToSend || input).trim();
    if (!text || loading) return;

    const userMsg: ChatMessage = {
      id: Date.now().toString(),
      role: 'user',
      content: text,
    };

    setMessages((prev) => [...prev, userMsg]);
    setInput('');
    setLoading(true);

    try {
      const historyPayload = messages
        .filter((m) => m.id !== 'welcome')
        .map((m) => ({ role: m.role, content: m.content }));

      const res = await fetch('/api/chat/general', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: text,
          history: historyPayload,
        }),
      });

      if (res.ok) {
        const data = await res.json();
        const botMsg: ChatMessage = {
          id: (Date.now() + 1).toString(),
          role: 'assistant',
          content: data.reply,
          guardrailTriggered: data.guardrail_triggered,
          engine: data.engine,
        };
        setMessages((prev) => [...prev, botMsg]);
      } else {
        const errData = await res.json();
        setMessages((prev) => [
          ...prev,
          {
            id: (Date.now() + 1).toString(),
            role: 'assistant',
            content: `⚠️ Error: ${errData.detail || 'Could not communicate with chat server.'}`,
          },
        ]);
      }
    } catch (err: any) {
      setMessages((prev) => [
        ...prev,
        {
          id: (Date.now() + 1).toString(),
          role: 'assistant',
          content: `⚠️ Network error: ${err.message || 'Failed to connect to backend.'}`,
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const suggestions = [
    'How do I prevent tomato blossom end rot?',
    'What are organic controls for aphids and whiteflies?',
    'Best drip irrigation schedule for flowering tomatoes',
    'How to spot nitrogen vs calcium deficiency',
  ];

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 overflow-hidden bg-black/40 backdrop-blur-xs flex justify-end">
      <div className="w-full max-w-lg bg-white h-full shadow-2xl flex flex-col border-l border-[#DCE6DE] animate-in slide-in-from-right duration-200">
        {/* Header */}
        <div className="p-4 bg-[#1B4332] text-white flex items-center justify-between shadow-xs">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-emerald-500/20 rounded-xl border border-emerald-400/30">
              <Sprout className="w-5 h-5 text-emerald-300" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h3 className="font-bold text-sm tracking-tight">AgriHealth Plant Advisor</h3>
                <span className="text-[10px] font-semibold bg-emerald-500/20 text-emerald-200 px-2 py-0.5 rounded-full border border-emerald-400/30 flex items-center gap-1">
                  <ShieldCheck className="w-3 h-3 text-emerald-300" /> Guardrailed
                </span>
              </div>
              <p className="text-[11px] text-emerald-200/80">
                General Botanical, Soil & Agronomic Assistant • {groqStatus}
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg text-white/70 hover:text-white hover:bg-white/10 transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Guardrail Disclaimer Banner */}
        <div className="px-4 py-2 bg-emerald-50 border-b border-emerald-100 flex items-center justify-between text-[11px] text-emerald-800">
          <span className="flex items-center gap-1.5">
            <Leaf className="w-3.5 h-3.5 text-emerald-600 shrink-0" />
            Strictly limited to plants, crops, soil, pests & agronomy.
          </span>
          {!hasGroqKey && (
            <span className="text-amber-700 font-semibold text-[10px] bg-amber-100/70 px-1.5 py-0.5 rounded">
              Offline Mode (.env key pending)
            </span>
          )}
        </div>

        {/* Chat Messages */}
        <div className="flex-1 overflow-y-auto p-4 space-y-3.5 bg-[#FAFBF9]">
          {messages.map((m) => {
            const isUser = m.role === 'user';
            return (
              <div
                key={m.id}
                className={`flex gap-2.5 ${isUser ? 'justify-end' : 'justify-start'}`}
              >
                {!isUser && (
                  <div className="w-7 h-7 rounded-lg bg-[#1B4332] text-white flex items-center justify-center shrink-0 mt-0.5 shadow-xs">
                    <Bot className="w-4 h-4 text-emerald-300" />
                  </div>
                )}
                <div
                  className={`max-w-[85%] rounded-2xl px-4 py-3 text-xs leading-relaxed shadow-xs ${
                    isUser
                      ? 'bg-[#1B4332] text-white rounded-br-xs'
                      : m.guardrailTriggered
                      ? 'bg-amber-50 text-amber-950 border border-amber-200 rounded-bl-xs'
                      : 'bg-white text-gray-800 border border-[#E3EBE4] rounded-bl-xs'
                  }`}
                >
                  <FormattedAiResponse content={m.content} isUser={isUser} />
                  {m.guardrailTriggered && (
                    <div className="mt-2 pt-1.5 border-t border-amber-200/60 text-[10px] text-amber-800 flex items-center gap-1">
                      <AlertCircle className="w-3 h-3 text-amber-600" /> Guardrail Policy Enforced
                    </div>
                  )}
                </div>
                {isUser && (
                  <div className="w-7 h-7 rounded-lg bg-[#2D6A4F] text-white flex items-center justify-center shrink-0 mt-0.5 shadow-xs">
                    <User className="w-4 h-4 text-emerald-200" />
                  </div>
                )}
              </div>
            );
          })}
          {loading && (
            <div className="flex gap-2.5 justify-start">
              <div className="w-7 h-7 rounded-lg bg-[#1B4332] text-white flex items-center justify-center shrink-0">
                <Bot className="w-4 h-4 text-emerald-300" />
              </div>
              <div className="bg-white border border-[#E3EBE4] rounded-2xl px-4 py-3 text-xs text-gray-500 flex items-center gap-2">
                <Loader2 className="w-3.5 h-3.5 animate-spin text-emerald-600" />
                <span>AgriHealth Advisor is researching agronomic guidance...</span>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Quick Suggestion Chips */}
        <div className="px-3 py-2 bg-white border-t border-[#E8EEE9] flex gap-1.5 overflow-x-auto text-[11px]">
          {suggestions.map((s, idx) => (
            <button
              key={idx}
              onClick={() => handleSend(s)}
              className="shrink-0 px-2.5 py-1 bg-[#F1F6F2] hover:bg-[#E2EDE4] text-[#1B4332] rounded-lg transition-colors border border-[#DEE9E0] font-medium"
            >
              {s}
            </button>
          ))}
        </div>

        {/* Input Bar */}
        <div className="p-3 bg-white border-t border-[#E3EBE4]">
          <form
            onSubmit={(e) => {
              e.preventDefault();
              handleSend();
            }}
            className="flex items-center gap-2"
          >
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask about crops, pests, soil, irrigation, fertilizer..."
              className="flex-1 px-3.5 py-2 text-xs border border-[#C8D9CB] rounded-xl focus:outline-none focus:ring-2 focus:ring-[#1B4332] focus:border-transparent bg-[#FAFCFA]"
              disabled={loading}
            />
            <button
              type="submit"
              disabled={loading || !input.trim()}
              className="p-2 bg-[#1B4332] hover:bg-[#133023] text-white rounded-xl disabled:opacity-40 transition-colors shadow-xs"
            >
              <Send className="w-4 h-4" />
            </button>
          </form>
        </div>
      </div>
    </div>
  );
};
