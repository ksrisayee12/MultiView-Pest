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


import {
  LayoutDashboard,
  Scan,
  Grid,
  ZoomIn,
  FileText,
  History,
  Sprout
} from 'lucide-react';

const App = () => {
  const [activeTab, setActiveTab] = useState<string>('dashboard');
  const [latestScan, setLatestScan] = useState<ScanSummary | null>(null);
  const [selectedTile, setSelectedTile] = useState<TileResult | null>(null);
  const [loading, setLoading] = useState<boolean>(false);

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
    { id: 'dashboard', label: 'Dashboard', icon: LayoutDashboard },
    { id: 'multiview', label: 'Multi-View Plant', icon: Sprout },
    { id: 'field-scan', label: 'Field Scan', icon: Scan },
    { id: 'tile-analysis', label: 'Tile Analysis', icon: Grid },
    { id: 'closeup-analysis', label: 'Close-up Analysis', icon: ZoomIn },
    { id: 'results', label: 'Results', icon: FileText },
    { id: 'scan-history', label: 'Scan History', icon: History },
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
        return (
          <DashboardPage
            onNavigate={setActiveTab}
            latestScan={latestScan}
            onRunSimulation={handleRunSimulation}
            loading={loading}
          />
        );
    }
  };

  return (
    <div className="min-h-screen bg-[#F4F8FC] text-gray-800">
      {/* Header */}
      <header className="bg-[#0D47A1] text-white shadow-md border-b border-[#1565C0] sticky top-0 z-30">
        <div className="flex items-center justify-between px-6 py-3.5">
          <div className="flex items-center space-x-3 cursor-pointer" onClick={() => setActiveTab('dashboard')}>
            <div className="bg-white/10 p-2 rounded-lg backdrop-blur-sm">
              <Sprout className="w-6 h-6 text-green-300" />
            </div>
            <div>
              <h1 className="text-xl font-bold tracking-tight text-white flex items-center gap-2">
                AgriHealth AI
                <span className="text-xs bg-blue-500/30 text-blue-100 px-2 py-0.5 rounded border border-blue-400/30 font-normal">
                  Academic Prototype
                </span>
              </h1>
              <p className="text-xs text-blue-200">
                Multi-View Agricultural Health Assessment & Dual-Evidence Fusion
              </p>
            </div>
          </div>

          <div className="flex items-center gap-4 text-right">
            <div>
              <p className="text-xs text-blue-200">System Status</p>
              <p className="text-xs font-semibold text-green-300 flex items-center justify-end gap-1">
                <span className="w-2 h-2 rounded-full bg-green-400 animate-pulse"></span>
                FastAPI Backend Online
              </p>
            </div>
          </div>
        </div>
      </header>

      <div className="flex min-h-[calc(100vh-65px)]">
        {/* Sidebar */}
        <aside className="w-64 bg-white border-r border-gray-200 p-5 shrink-0 flex flex-col justify-between">
          <div>
            <div className="mb-6">
              <p className="text-xs font-bold uppercase tracking-wider text-gray-400">
                Navigation
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
                    className={`w-full flex items-center gap-3 px-4 py-3 rounded-xl text-sm font-semibold transition-colors text-left ${
                      isActive
                        ? 'bg-[#E3F2FD] text-[#0D47A1] border-l-4 border-[#0D47A1] shadow-sm'
                        : 'text-gray-600 hover:bg-gray-100'
                    }`}
                  >
                    <Icon className={`w-4 h-4 ${isActive ? 'text-[#0D47A1]' : 'text-gray-400'}`} />
                    <span>{item.label}</span>
                  </button>
                );
              })}
            </nav>
          </div>

          <div className="mt-8 rounded-xl bg-[#F5F9FF] border border-blue-100 p-4">
            <p className="text-xs font-bold text-[#0D47A1] uppercase tracking-wide">
              Multi-View Workflow
            </p>
            <p className="mt-2 text-xs leading-relaxed text-gray-500">
              Field image $\rightarrow$ 4x4 Tiles $\rightarrow$ Close-up resolution $\rightarrow$ Dual-evidence assessment
            </p>
          </div>
        </aside>

        {/* Main Content */}
        <main className="flex-1 p-6 md:p-8 overflow-y-auto">
          {renderContent()}

          {/* Footer */}
          <footer className="mt-12 pt-6 border-t border-gray-200 text-center text-xs text-gray-400">
            Multi-View Drone-Based Agricultural Pest & Plant Disease Detection System — Dual-Evidence AI & DIP Architecture
          </footer>
        </main>
      </div>
    </div>
  );
};

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);