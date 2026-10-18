import React, { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import { Copy, Check, Sparkles, CheckSquare, Square, ChevronRight } from 'lucide-react';

interface FormattedAiResponseProps {
  content: string;
  isUser?: boolean;
}

export const FormattedAiResponse: React.FC<FormattedAiResponseProps> = ({ content, isUser = false }) => {
  const [copied, setCopied] = useState<boolean>(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  if (isUser) {
    return <div className="whitespace-pre-wrap font-sans text-xs leading-relaxed">{content}</div>;
  }

  return (
    <div className="relative group text-gray-800 text-xs leading-relaxed space-y-2">
      {/* Quick Copy Tool in corner */}
      <div className="absolute -top-1 -right-1 opacity-0 group-hover:opacity-100 transition-opacity">
        <button
          onClick={handleCopy}
          className="p-1 rounded-md bg-white border border-gray-200 text-gray-500 hover:text-[#1B4332] shadow-xs transition-colors flex items-center gap-1 text-[10px]"
          title="Copy formatted answer"
        >
          {copied ? (
            <>
              <Check className="w-3 h-3 text-emerald-600" />
              <span className="text-emerald-700 font-semibold">Copied</span>
            </>
          ) : (
            <>
              <Copy className="w-3 h-3" />
              <span>Copy</span>
            </>
          )}
        </button>
      </div>

      <ReactMarkdown
        components={{
          h1: ({ children }) => (
            <h1 className="text-sm font-black text-[#1B4332] tracking-tight pb-1.5 border-b border-[#D8E4DA] mt-2 mb-2 flex items-center gap-1.5">
              <Sparkles className="w-3.5 h-3.5 text-emerald-600 shrink-0" />
              {children}
            </h1>
          ),
          h2: ({ children }) => (
            <h2 className="text-xs font-extrabold text-[#1B4332] tracking-tight pb-1 border-b border-[#E2EBE4] mt-2.5 mb-1.5 flex items-center gap-1.5">
              <ChevronRight className="w-3 h-3 text-emerald-600 shrink-0" />
              {children}
            </h2>
          ),
          h3: ({ children }) => (
            <h3 className="text-xs font-bold text-[#245741] mt-2 mb-1 flex items-center gap-1">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-600 shrink-0" />
              {children}
            </h3>
          ),
          h4: ({ children }) => (
            <h4 className="text-[11px] font-bold uppercase tracking-wider text-emerald-800 mt-1.5 mb-0.5">
              {children}
            </h4>
          ),
          p: ({ children }) => <p className="mb-1.5 text-xs text-gray-800 leading-relaxed">{children}</p>,
          strong: ({ children }) => (
            <strong className="font-bold text-[#143527] bg-[#EAF2EC]/60 px-1 py-0.2 rounded">
              {children}
            </strong>
          ),
          ul: ({ children }) => <ul className="space-y-1 mb-2 pl-1">{children}</ul>,
          ol: ({ children }) => <ol className="list-decimal list-inside space-y-1 mb-2 text-xs">{children}</ol>,
          li: ({ children }) => {
            // Check if content starts with checkbox symbols
            const rawText = React.Children.toArray(children).map(c => typeof c === 'string' ? c : '').join('');
            const isCheckbox = rawText.includes('☐') || rawText.includes('[ ]') || rawText.includes('[x]');
            
            if (isCheckbox) {
              return (
                <li className="flex items-start gap-1.5 text-xs text-gray-700 bg-white/70 p-1.5 rounded-lg border border-gray-100 hover:border-emerald-200 transition-colors">
                  <Square className="w-3.5 h-3.5 text-emerald-700 mt-0.5 shrink-0" />
                  <span className="flex-1">{children}</span>
                </li>
              );
            }

            return (
              <li className="flex items-start gap-1.5 text-xs text-gray-700">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-700 mt-1.5 shrink-0" />
                <span className="flex-1">{children}</span>
              </li>
            );
          },
          blockquote: ({ children }) => (
            <blockquote className="p-2.5 my-2 bg-emerald-50/90 border-l-3 border-[#2D6A4F] rounded-r-lg text-emerald-950 text-xs italic space-y-1 shadow-xs">
              {children}
            </blockquote>
          ),
          hr: () => <hr className="my-2.5 border-t border-[#DDE7DF]" />,
          table: ({ children }) => (
            <div className="overflow-x-auto my-2 rounded-lg border border-gray-200">
              <table className="w-full text-left text-xs border-collapse">{children}</table>
            </div>
          ),
          thead: ({ children }) => <thead className="bg-[#EAF2EC] text-[#1B4332] font-bold">{children}</thead>,
          th: ({ children }) => <th className="p-2 border-b border-gray-200">{children}</th>,
          td: ({ children }) => <td className="p-2 border-b border-gray-100">{children}</td>,
          code: ({ children }) => (
            <code className="px-1.5 py-0.5 bg-gray-100 text-[#1B4332] rounded font-mono text-[11px] font-semibold border border-gray-200">
              {children}
            </code>
          ),
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
};
