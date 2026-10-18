import React, { useState, useEffect, useRef } from 'react';
import {
  MessageSquare,
  Send,
  Sparkles,
  Bot,
  User,
  ShieldCheck,
  Loader2,
  Stethoscope,
  Info,
  ChevronRight,
} from 'lucide-react';
import { FormattedAiResponse } from './FormattedAiResponse';

interface ClinicalMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  engine?: string;
}

interface ClinicalDiseaseChatProps {
  diseaseName: string;
  cropType: string;
  confidence: number;
  severity: string;
  lesionPercentage?: number;
  affectedViews?: string[];
}

export const ClinicalDiseaseChat: React.FC<ClinicalDiseaseChatProps> = ({
  diseaseName,
  cropType,
  confidence,
  severity,
  lesionPercentage = 12.0,
  affectedViews = ['front', 'underside'],
}) => {
  const [messages, setMessages] = useState<ClinicalMessage[]>([]);
  const [input, setInput] = useState<string>('');
  const [loading, setLoading] = useState<boolean>(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Initialize or reset chat when disease changes
  useEffect(() => {
    setMessages([
      {
        id: 'initial_consult',
        role: 'assistant',
        content: `🔬 **Clinical Pathologist Case Opened: ${diseaseName}**\n\nI am your consultation specialist locked to this verified **${cropType}** case.\n\nAsk me about:\n- **Etiology & Causes**: Why did this pathogen infect your crop?\n- **Curative Prescriptions**: Organic bio-controls vs chemical spray programs\n- **Dosage & Timing**: Application rates, spray intervals, and FRAC resistance rotations\n- **Containment**: Pruning protocol, tool sterilization & preventing field spread.`,
      },
    ]);
  }, [diseaseName, cropType]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, loading]);

  const handleSend = async (customPrompt?: string) => {
    const text = (customPrompt || input).trim();
    if (!text || loading) return;

    const userMsg: ClinicalMessage = {
      id: Date.now().toString(),
      role: 'user',
      content: text,
    };

    setMessages((prev) => [...prev, userMsg]);
    setInput('');
    setLoading(true);

    try {
      const historyPayload = messages
        .filter((m) => m.id !== 'initial_consult')
        .map((m) => ({ role: m.role, content: m.content }));

      const res = await fetch('/api/chat/disease', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: text,
          context: {
            disease_name: diseaseName,
            crop_type: cropType,
            confidence: confidence,
            severity: severity,
            lesion_percentage: lesionPercentage,
            affected_views: affectedViews,
          },
          history: historyPayload,
        }),
      });

      if (res.ok) {
        const data = await res.json();
        const botMsg: ClinicalMessage = {
          id: (Date.now() + 1).toString(),
          role: 'assistant',
          content: data.reply,
          engine: data.engine,
        };
        setMessages((prev) => [...prev, botMsg]);
      } else {
        const err = await res.json();
        setMessages((prev) => [
          ...prev,
          {
            id: (Date.now() + 1).toString(),
            role: 'assistant',
            content: `⚠️ Consultation error: ${err.detail || 'Could not fetch guidance.'}`,
          },
        ]);
      }
    } catch (err: any) {
      setMessages((prev) => [
        ...prev,
        {
          id: (Date.now() + 1).toString(),
          role: 'assistant',
          content: `⚠️ Connection failure: ${err.message}`,
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const clinicalPrompts = [
    `What caused this ${diseaseName} to develop?`,
    'What is the immediate treatment spray program?',
    'Are organic bio-fungicides effective for this stage?',
    'How do I prevent this spreading to adjacent plants?',
  ];

  return (
    <div className="bg-white rounded-2xl border border-[#D5E2D7] shadow-sm overflow-hidden flex flex-col h-[520px]">
      {/* Header */}
      <div className="p-3.5 bg-[#1B4332] text-white flex items-center justify-between">
        <div className="flex items-center gap-2.5">
          <div className="p-1.5 bg-emerald-500/20 rounded-lg border border-emerald-400/30">
            <Stethoscope className="w-4 h-4 text-emerald-300" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h4 className="font-bold text-xs tracking-tight">Clinical Pathologist Consultation</h4>
              <span className="text-[10px] font-semibold bg-emerald-500/20 text-emerald-200 px-1.5 py-0.2 rounded border border-emerald-400/30">
                Case-Locked
              </span>
            </div>
            <p className="text-[10px] text-emerald-200/80">
              Direct agronomic consultation for <strong className="text-white">{diseaseName}</strong> ({severity})
            </p>
          </div>
        </div>
      </div>

      {/* Case Metrics Bar */}
      <div className="px-3.5 py-1.5 bg-[#F2F7F3] border-b border-[#E3EDE5] flex flex-wrap items-center justify-between gap-2 text-[11px] text-[#2D5A3C]">
        <span><strong>Crop:</strong> {cropType}</span>
        <span>•</span>
        <span><strong>Confidence:</strong> {(confidence * 100).toFixed(1)}%</span>
        <span>•</span>
        <span><strong>Severity:</strong> {severity}</span>
        <span>•</span>
        <span><strong>Lesion Area:</strong> ~{lesionPercentage.toFixed(1)}%</span>
      </div>

      {/* Messages Scroll Area */}
      <div className="flex-1 overflow-y-auto p-3.5 space-y-3 bg-[#FAFBF9]">
        {messages.map((m) => {
          const isUser = m.role === 'user';
          return (
            <div
              key={m.id}
              className={`flex gap-2 ${isUser ? 'justify-end' : 'justify-start'}`}
            >
              {!isUser && (
                <div className="w-6 h-6 rounded-md bg-[#1B4332] text-white flex items-center justify-center shrink-0 mt-0.5 shadow-xs">
                  <Bot className="w-3.5 h-3.5 text-emerald-300" />
                </div>
              )}
              <div
                className={`max-w-[88%] rounded-xl px-3.5 py-2.5 text-xs leading-relaxed shadow-xs ${
                  isUser
                    ? 'bg-[#1B4332] text-white rounded-br-xs'
                    : 'bg-white text-gray-800 border border-[#DEE7E0] rounded-bl-xs'
                }`}
              >
                <FormattedAiResponse content={m.content} isUser={isUser} />
              </div>
              {isUser && (
                <div className="w-6 h-6 rounded-md bg-[#2D6A4F] text-white flex items-center justify-center shrink-0 mt-0.5 shadow-xs">
                  <User className="w-3.5 h-3.5 text-emerald-200" />
                </div>
              )}
            </div>
          );
        })}
        {loading && (
          <div className="flex gap-2 justify-start">
            <div className="w-6 h-6 rounded-md bg-[#1B4332] text-white flex items-center justify-center shrink-0">
              <Bot className="w-3.5 h-3.5 text-emerald-300" />
            </div>
            <div className="bg-white border border-[#DEE7E0] rounded-xl px-3.5 py-2.5 text-xs text-gray-500 flex items-center gap-2">
              <Loader2 className="w-3 h-3 animate-spin text-emerald-600" />
              <span>Synthesizing clinical treatment advice...</span>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Suggested Quick Inquiries */}
      <div className="px-3 py-1.5 bg-white border-t border-[#E8EEE9] flex gap-1.5 overflow-x-auto text-[10px]">
        {clinicalPrompts.map((p, idx) => (
          <button
            key={idx}
            onClick={() => handleSend(p)}
            className="shrink-0 px-2 py-1 bg-[#F2F7F4] hover:bg-[#E5EFE7] text-[#1B4332] rounded-md transition-colors border border-[#DEE8E0] font-medium"
          >
            {p}
          </button>
        ))}
      </div>

      {/* Input */}
      <div className="p-2.5 bg-white border-t border-[#E3EBE4]">
        <form
          onSubmit={(e) => {
            e.preventDefault();
            handleSend();
          }}
          className="flex items-center gap-1.5"
        >
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder={`Ask pathologist about ${diseaseName} causes, treatments, dosages...`}
            className="flex-1 px-3 py-1.5 text-xs border border-[#CBDBCF] rounded-lg focus:outline-none focus:ring-1.5 focus:ring-[#1B4332] bg-[#FAFCFA]"
            disabled={loading}
          />
          <button
            type="submit"
            disabled={loading || !input.trim()}
            className="p-1.5 bg-[#1B4332] hover:bg-[#133023] text-white rounded-lg disabled:opacity-40 transition-colors shadow-xs"
          >
            <Send className="w-3.5 h-3.5" />
          </button>
        </form>
      </div>
    </div>
  );
};
