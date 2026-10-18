import React, { useState } from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import { ScanSummary, TileResult } from './types';

import { DashboardPage } from './pages/DashboardPage';
import { FieldScanPage } from './pages/FieldScanPage';
import { TileAnalysisPage } from './pages/TileAnalysisPage';
import { CloseUpAnalysisPage } from './pages/CloseUpAnalysisPage';
import { ResultsPage } from './pages/ResultsPage';
import { ScanHistoryPage } from './pages/ScanHistoryPage';
import { MultiViewPage } from './pages/MultiViewPage';
import { PlantChatbotDrawer } from './components/PlantChatbotDrawer';

import {
  LayoutDashboard,
  Scan,
  Grid,
  ZoomIn,
  FileText,
  History,
  Sprout,
  MessageSquare,
  Sparkles,
} from 'lucide-react';

const App = () => {
  const [activeTab, setActiveTab] = useState<string>('multiview');
  const [latestScan, setLatestScan] = useState<ScanSummary | null>(null);
  const [selectedTile, setSelectedTile] = useState<TileResult | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [isChatOpen, setIsChatOpen] = useState<boolean>(false);

  // Trigger 4x4 Field Analysis via Backend API
  const handleRunSimulation = async () => {
    setLoading(true);
    try {
      const response = await fetch('/api/analyze/field', {
        method: 'POST',
      });
      if (response.ok) {
        const scanData: ScanSummary = await response.json();
        setLatestScan(scanData);
        if (scanData.tile_results && scanData.tile_results.length > 0) {
          setSelectedTile(scanData.tile_results[0]);
        }
      } else {
        console.warn('API returned status:', response.status);
      }
    } catch (err) {
      console.error('Simulation error:', err);
    } finally {
      setLoading(false);
    }
  };

  const navItems = [
    { id: 'multiview', label: 'Multi-View Plant Clinic', icon: Sprout },
    { id: 'dashboard', label: 'Field Dashboard', icon: LayoutDashboard },
    { id: 'field-scan', label: 'Grid Scan', icon: Scan },
    { id: 'tile-analysis', label: 'Tile Diagnostics', icon: Grid },
    { id: 'closeup-analysis', label: 'Macro Analysis', icon: ZoomIn },
    { id: 'results', label: 'Audit Reports', icon: FileText },
    { id: 'scan-history', label: 'History Archive', icon: History },
  ];

  const renderContent = () => {
    switch (activeTab) {
      case 'dashboard':
        return (
          <DashboardPage
            onNavigate={setActiveTab}
            latestScan={latestScan}
            onRunSimulation={handleRunSimulation}
            loading={loading}
          />
        );
      case 'multiview':
        return <MultiViewPage />;

      case 'field-scan':
        return (
          <FieldScanPage
            scan={latestScan}
            onRunSimulation={handleRunSimulation}
            onSelectTile={setSelectedTile}
            onNavigate={setActiveTab}
            loading={loading}
          />
        );
      case 'tile-analysis':
        return (
          <TileAnalysisPage
            selectedTile={selectedTile}
            onNavigate={setActiveTab}
          />
        );
      case 'closeup-analysis':
        return (
          <CloseUpAnalysisPage
            scan={latestScan}
            onNavigate={setActiveTab}
          />
        );
      case 'results':
        return (
          <ResultsPage
            scan={latestScan}
            onNavigate={setActiveTab}
          />
        );
      case 'scan-history':
        return (
          <ScanHistoryPage
            onSelectScan={(scan) => {
              setLatestScan(scan);
              if (scan.tile_results && scan.tile_results.length > 0) {
                setSelectedTile(scan.tile_results[0]);
              }
            }}
            onNavigate={setActiveTab}
          />
        );
      default:
        return <MultiViewPage />;
    }
  };

  return (
    <div className="min-h-screen bg-[#F8FAF8] text-gray-900">
      {/* Header */}
      <header className="bg-[#1B4332] text-white shadow-sm border-b border-[#2D6A4F] sticky top-0 z-30">
        <div className="flex items-center justify-between px-6 py-3.5">
          <div className="flex items-center space-x-3 cursor-pointer" onClick={() => setActiveTab('multiview')}>
            <div className="bg-emerald-500/20 p-2 rounded-xl border border-emerald-400/30 backdrop-blur-xs">
              <Sprout className="w-6 h-6 text-emerald-300" />
            </div>
            <div>
              <h1 className="text-xl font-bold tracking-tight text-white flex items-center gap-2">
                AgriHealth Precision Clinic
                <span className="text-[11px] bg-emerald-500/20 text-emerald-200 px-2.5 py-0.5 rounded-full border border-emerald-400/30 font-semibold">
                  Multi-Model Active
                </span>
              </h1>
              <p className="text-xs text-emerald-200/80">
                Explainable Multi-View Crop Pathology, Pathogen Identification & Clinical Treatment
              </p>
            </div>
          </div>

          <div className="flex items-center gap-3">
            {/* Plant Assistant Button */}
            <button
              onClick={() => setIsChatOpen(true)}
              className="px-3.5 py-2 bg-emerald-500/20 hover:bg-emerald-500/30 text-emerald-100 border border-emerald-400/30 rounded-xl font-bold text-xs flex items-center gap-2 transition-all shadow-xs active:scale-[0.98]"
              title="Open General Plant & Agronomy Assistant (Guardrailed)"
            >
              <MessageSquare className="w-4 h-4 text-emerald-300" />
              <span>🌿 Plant Assistant</span>
            </button>

            <div className="hidden sm:block text-right border-l border-emerald-700/60 pl-3.5">
              <p className="text-[10px] uppercase font-bold text-emerald-300/80 tracking-wider">Engine Status</p>
              <p className="text-xs font-semibold text-emerald-100 flex items-center justify-end gap-1.5">
                <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
                3 AI Models & Groq Ready
              </p>
            </div>
          </div>
        </div>
      </header>

      <div className="flex min-h-[calc(100vh-65px)]">
        {/* Sidebar */}
        <aside className="w-64 bg-white border-r border-[#E3EDE5] p-5 shrink-0 flex flex-col justify-between">
          <div>
            <div className="mb-4">
              <p className="text-[11px] font-bold uppercase tracking-wider text-emerald-900/60">
                Diagnostic Console
              </p>
            </div>

            <nav className="space-y-1.5">
              {navItems.map((item) => {
                const Icon = item.icon;
                const isActive = activeTab === item.id;
                return (
                  <button
                    key={item.id}
                    onClick={() => setActiveTab(item.id)}
                    className={`w-full flex items-center gap-3 px-3.5 py-2.5 rounded-xl text-xs font-bold transition-all text-left ${
                      isActive
                        ? 'bg-[#EBF4ED] text-[#1B4332] shadow-xs border-l-4 border-[#1B4332]'
                        : 'text-gray-600 hover:bg-[#F4F8F5] hover:text-gray-900'
                    }`}
                  >
                    <Icon className={`w-4 h-4 ${isActive ? 'text-[#1B4332]' : 'text-gray-400'}`} />
                    <span>{item.label}</span>
                  </button>
                );
              })}
            </nav>
          </div>

          <div className="space-y-3">
            {/* Quick Chat Shortcut Card */}
            <div
              onClick={() => setIsChatOpen(true)}
              className="rounded-2xl bg-gradient-to-br from-[#F1F7F3] to-[#E5EFE8] border border-[#D5E5D8] p-3.5 cursor-pointer hover:shadow-xs transition-shadow"
            >
              <div className="flex items-center gap-2 text-xs font-bold text-[#1B4332]">
                <MessageSquare className="w-4 h-4 text-emerald-700" />
                <span>Botanical AI Advisor</span>
              </div>
              <p className="mt-1 text-[11px] text-gray-600 leading-relaxed">
                Need cultivation, soil, or pest advice? Consult the guardrailed assistant.
              </p>
              <span className="mt-2 inline-flex items-center text-[11px] font-bold text-emerald-800 hover:underline">
                Ask a question &rarr;
              </span>
            </div>

            <div className="rounded-xl bg-[#FAFBF9] border border-[#E3EDE5] p-3 text-[11px] text-gray-500">
              <span className="font-semibold text-gray-700 block">Ensemble Architecture:</span>
              YOLOv8n + LeafSeverityNet + PestVectorNet + DIP
            </div>
          </div>
        </aside>

        {/* Main Content */}
        <main className="flex-1 p-6 md:p-8 overflow-y-auto">
          {renderContent()}

          {/* Footer */}
          <footer className="mt-12 pt-6 border-t border-[#E3EDE5] text-center text-xs text-gray-500">
            AgriHealth Precision Crop Diagnostic Platform — Grounded Multi-Model AI & DIP Clinical Architecture
          </footer>
        </main>
      </div>

      {/* Global Plant Chatbot Slide-over Drawer */}
      <PlantChatbotDrawer isOpen={isChatOpen} onClose={() => setIsChatOpen(false)} />
    </div>
  );
};

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);