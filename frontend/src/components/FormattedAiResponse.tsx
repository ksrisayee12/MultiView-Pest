import React, { useState, useMemo } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkBreaks from 'remark-breaks';
import { Copy, Check, Sparkles, Square, ChevronRight, Table } from 'lucide-react';

interface FormattedAiResponseProps {
  content: string;
  isUser?: boolean;
}

/**
 * Normalizes and formats AI responses:
 * 1. Splits concatenated markdown table rows (e.g., "| remarks | |---| | row |" onto newlines).
 * 2. Formats numbered sections (e.g., "1. Why No Immediate Spray Is Required") into markdown headers.
 * 3. Highlights bare key-value lines (e.g., "Etiology: ...", "Pathogen Presence: ...", "Host Crop: ...").
 */
function normalizeAiMarkdown(raw: string): string {
  if (!raw) return '';
  let text = raw;

  // Fix concatenated tables where rows are squashed onto a single line without newlines
  text = text.replace(/\|\s*\|(?=[ -])/g, '|\n|');
  text = text.replace(/\|\s*\|/g, '|\n|');

  // Ensure table header separator has proper linebreaks around it
  text = text.replace(/(\|[^\n]+\|)\s*(\|[-: ]+\|)/g, '$1\n$2');

  // Convert bare numbered section headers like "1. Why No Immediate Spray Is Required"
  text = text.replace(
    /(?:^|\n)(\d+\.\s+([A-Z][^\n]+))(?=\n|$)/g,
    '\n\n### $1\n'
  );

  // Convert bare diagnostic key-value lines at start of lines to bold bullet items
  text = text.replace(
    /^(Host Crop|Pathogen Status|Pathogen Presence|Etiology|Implication|Bottom line|Mode of action|Regulatory status|Symptoms|Location|Timing|Growing conditions|Watering|Soil\/Medium|Pests|Weather|Remarks|Dosage|Application interval|Active ingredient):\s*(.+)$/gim,
    '- **$1:** $2'
  );

  return text;
}

export const FormattedAiResponse: React.FC<FormattedAiResponseProps> = ({ content, isUser = false }) => {
  const [copied, setCopied] = useState<boolean>(false);

  const formattedContent = useMemo(() => {
    if (isUser) return content;
    return normalizeAiMarkdown(content);
  }, [content, isUser]);

  const handleCopy = () => {
    navigator.clipboard.writeText(content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  if (isUser) {
    return <div className="whitespace-pre-wrap font-sans text-xs leading-relaxed">{content}</div>;
  }

  return (
    <div className="relative group text-gray-800 text-xs leading-relaxed space-y-2.5">
      {/* Quick Copy Tool */}
      <div className="absolute -top-1 -right-1 opacity-0 group-hover:opacity-100 transition-opacity z-10">
        <button
          onClick={handleCopy}
          className="px-2 py-1 rounded-md bg-white border border-gray-200 text-gray-500 hover:text-[#1B4332] shadow-xs transition-colors flex items-center gap-1 text-[10px]"
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
        remarkPlugins={[remarkGfm, remarkBreaks]}
        components={{
          h1: ({ children }) => (
            <h1 className="text-sm font-black text-[#1B4332] tracking-tight pb-1.5 border-b border-[#D8E4DA] mt-3 mb-2 flex items-center gap-1.5">
              <Sparkles className="w-4 h-4 text-emerald-600 shrink-0" />
              <span>{children}</span>
            </h1>
          ),
          h2: ({ children }) => (
            <h2 className="text-xs font-black text-[#1B4332] tracking-tight pb-1 border-b border-[#E2EBE4] mt-3 mb-1.5 flex items-center gap-1.5">
              <ChevronRight className="w-3.5 h-3.5 text-emerald-600 shrink-0" />
              <span>{children}</span>
            </h2>
          ),
          h3: ({ children }) => (
            <h3 className="text-xs font-extrabold text-[#245741] mt-2.5 mb-1 flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full bg-emerald-600 shrink-0" />
              <span>{children}</span>
            </h3>
          ),
          h4: ({ children }) => (
            <h4 className="text-[11px] font-bold uppercase tracking-wider text-emerald-800 mt-2 mb-0.5">
              {children}
            </h4>
          ),
          p: ({ children }) => <p className="mb-2 text-xs text-gray-800 leading-relaxed">{children}</p>,
          strong: ({ children }) => (
            <strong className="font-bold text-[#143527] bg-[#EAF2EC] px-1.5 py-0.5 rounded text-[11px] border border-[#DEE8E0] inline-block mr-1">
              {children}
            </strong>
          ),
          ul: ({ children }) => <ul className="space-y-1.5 mb-2.5 pl-1">{children}</ul>,
          ol: ({ children }) => <ol className="list-decimal list-inside space-y-1 mb-2.5 text-xs">{children}</ol>,
          li: ({ children }) => {
            const rawText = React.Children.toArray(children).map(c => typeof c === 'string' ? c : '').join('');
            const isCheckbox = rawText.includes('☐') || rawText.includes('[ ]') || rawText.includes('[x]');

            if (isCheckbox) {
              return (
                <li className="flex items-start gap-2 text-xs text-gray-800 bg-white/80 p-2 rounded-lg border border-[#DCE8DE] shadow-2xs hover:border-emerald-300 transition-colors">
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
            <blockquote className="p-3 my-2.5 bg-emerald-50/90 border-l-3 border-[#2D6A4F] rounded-r-xl text-emerald-950 text-xs italic space-y-1 shadow-xs">
              {children}
            </blockquote>
          ),
          hr: () => <hr className="my-3 border-t border-[#DDE7DF]" />,
          // Responsive Markdown Table Rendering
          table: ({ children }) => (
            <div className="overflow-x-auto my-3 rounded-xl border border-[#D5E2D7] shadow-xs bg-white">
              <div className="px-3 py-1.5 bg-[#FAFBF9] border-b border-[#E3EDE5] flex items-center gap-1.5 text-[10px] font-bold text-[#1B4332] uppercase tracking-wider">
                <Table className="w-3 h-3 text-emerald-700" /> Clinical Schedule & Reference Matrix
              </div>
              <table className="w-full text-left text-xs border-collapse divide-y divide-gray-200">
                {children}
              </table>
            </div>
          ),
          thead: ({ children }) => (
            <thead className="bg-[#1B4332] text-white text-[11px] font-bold uppercase tracking-wider">
              {children}
            </thead>
          ),
          th: ({ children }) => (
            <th className="py-2.5 px-3 border-b border-gray-200 text-left font-bold text-white">
              {children}
            </th>
          ),
          tbody: ({ children }) => (
            <tbody className="divide-y divide-gray-100 bg-white text-gray-700 text-xs">
              {children}
            </tbody>
          ),
          tr: ({ children }) => (
            <tr className="hover:bg-[#F2F7F4] transition-colors">{children}</tr>
          ),
          td: ({ children }) => (
            <td className="py-2.5 px-3 border-b border-gray-100 text-xs align-top">
              {children}
            </td>
          ),
          code: ({ children }) => (
            <code className="px-1.5 py-0.5 bg-gray-100 text-[#1B4332] rounded font-mono text-[11px] font-semibold border border-gray-200">
              {children}
            </code>
          ),
        }}
      >
        {formattedContent}
      </ReactMarkdown>
    </div>
  );
};
