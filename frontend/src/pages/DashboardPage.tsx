import React from 'react';
import { Sprout, ShieldAlert, Activity, ArrowRight, CheckCircle2, AlertTriangle } from 'lucide-react';
import { ScanSummary } from '../types';

interface DashboardPageProps {
  onNavigate: (tab: string) => void;
  latestScan: ScanSummary | null;
  onRunSimulation: () => void;
  loading: boolean;
}

export const DashboardPage: React.FC<DashboardPageProps> = ({
  onNavigate,
  latestScan,
  onRunSimulation,
  loading
}) => {
  const healthyCount = latestScan ? latestScan.healthy_tiles : 12;
  const diseaseCount = latestScan ? latestScan.confirmed_disease_tiles : 2;
  const disputedCount = latestScan ? latestScan.disputed_tiles : 2;
  const healthIndex = latestScan ? latestScan.plant_health_index : 84.5;

  return (
    <div className="space-[#0D47A1] space-y-7">
      {/* Page Title & Quick Actions */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h2 className="text-2xl sm:text-3xl font-bold text-[#0D47A1]">
            Agricultural Health Dashboard
          </h2>
          <p className="mt-1 text-sm text-gray-500">
            Multi-view drone field monitoring and dual-evidence crop health assessment
          </p>
        </div>
        <div className="flex items-center gap-3">
          <button
            onClick={onRunSimulation}
            disabled={loading}
            className="bg-[#1565C0] hover:bg-[#0D47A1] text-white px-4 py-2.5 rounded-lg text-sm font-semibold transition-colors shadow-sm flex items-center gap-2 disabled:opacity-50"
          >
            {loading ? (
              <span className="flex items-center gap-2">
                <span className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></span>
                Processing Field...
              </span>
            ) : (
              <>
                <Activity className="w-4 h-4" />
                Run 4x4 Field Simulation
              </>
            )}
          </button>
          <button
            onClick={() => onNavigate('field-scan')}
            className="bg-[#E3F2FD] hover:bg-[#BBDEFB] text-[#0D47A1] px-4 py-2.5 rounded-lg text-sm font-semibold transition-colors flex items-center gap-2"
          >
            Field Analysis
            <ArrowRight className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Statistics Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-5">
        <div className="rounded-xl border border-gray-200 bg-white p-5 shadow-sm">
          <div className="flex items-center justify-between">
            <p className="text-sm font-medium text-gray-500">Plant Health Index</p>
            <Sprout className="w-5 h-5 text-green-600" />
          </div>
          <p className="mt-2 text-3xl font-bold text-[#0D47A1]">{healthIndex} / 100</p>
          <p className="mt-2 text-xs text-gray-400">
            {latestScan ? 'Calculated from 16 tiles' : 'Baseline evaluation'}
          </p>
        </div>

        <div className="rounded-xl border border-gray-200 bg-white p-5 shadow-sm">
          <div className="flex items-center justify-between">
            <p className="text-sm font-medium text-gray-500">Healthy Tiles</p>
            <CheckCircle2 className="w-5 h-5 text-green-600" />
          </div>
          <p className="mt-2 text-3xl font-bold text-green-600">{healthyCount}</p>
          <p className="mt-2 text-xs text-gray-400">YOLO Healthy & OpenCV Normal</p>
        </div>

        <div className="rounded-xl border border-gray-200 bg-white p-5 shadow-sm">
          <div className="flex items-center justify-between">
            <p className="text-sm font-medium text-gray-500">Confirmed Disease</p>
            <ShieldAlert className="w-5 h-5 text-red-600" />
          </div>
          <p className="mt-2 text-3xl font-bold text-red-600">{diseaseCount}</p>
          <p className="mt-2 text-xs text-gray-400">YOLO Disease & OpenCV Abnormal</p>
        </div>

        <div className="rounded-xl border border-gray-200 bg-white p-5 shadow-sm">
          <div className="flex items-center justify-between">
            <p className="text-sm font-medium text-gray-500">Disputed Tiles</p>
            <AlertTriangle className="w-5 h-5 text-amber-500" />
          </div>
          <p className="mt-2 text-3xl font-bold text-amber-500">{disputedCount}</p>
          <p className="mt-2 text-xs text-gray-400">Requires Close-up Verification</p>
        </div>
      </div>

      {/* Main Panels */}
      <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
        {/* Drone Field Monitoring */}
        <div className="rounded-xl border border-gray-200 bg-white p-6 shadow-sm xl:col-span-2">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h3 className="text-lg font-bold text-gray-800">Drone Field Monitoring</h3>
              <p className="text-sm text-gray-500">Simulated 4x4 Grid Composite Field Overview</p>
            </div>
            <span className="rounded-full bg-blue-50 border border-blue-200 px-3 py-1 text-xs font-semibold text-[#0D47A1]">
              SIMULATED 4x4 VIEW
            </span>
          </div>

          <div className="flex flex-col items-center justify-center min-h-[220px] rounded-lg border-2 border-dashed border-blue-200 bg-[#F5F9FF] p-6 text-center">
            <div className="w-12 h-12 rounded-full bg-blue-100 text-[#0D47A1] flex items-center justify-center mb-3">
              <Sprout className="w-6 h-6" />
            </div>
            <h4 className="font-semibold text-[#0D47A1] text-base">
              {latestScan ? `Scan ${latestScan.scan_id} Loaded` : 'Ready for Field Analysis'}
            </h4>
            <p className="mt-1 text-sm text-gray-500 max-w-md">
              {latestScan
                ? `16 leaf tiles processed via Dual-Evidence Fusion Engine. Health Score: ${latestScan.plant_health_index}/100.`
                : 'Click below to analyze the 4x4 field composite image using YOLO26 and OpenCV DIP.'}
            </p>
            <button
              onClick={() => onNavigate('field-scan')}
              className="mt-4 rounded-lg bg-[#0D47A1] px-5 py-2.5 text-sm font-semibold text-white hover:bg-[#083578] transition-colors"
            >
              Open 4x4 Field Scan Grid
            </button>
          </div>
        </div>

        {/* Health Summary */}
        <div className="rounded-xl border border-gray-200 bg-white p-6 shadow-sm">
          <h3 className="text-lg font-bold text-gray-800">Crop Health Breakdown</h3>
          <p className="mt-1 text-sm text-gray-500">Current Field Status Summary</p>

          <div className="mt-6 space-y-5">
            <div>
              <div className="flex justify-between text-sm mb-1.5">
                <span className="font-medium text-gray-700">Healthy</span>
                <span className="font-semibold text-green-600">{healthyCount} / 16 Tiles</span>
              </div>
              <div className="h-2.5 rounded-full bg-gray-100 overflow-hidden">
                <div
                  className="h-full bg-green-500 rounded-full transition-all duration-300"
                  style={{ width: `${(healthyCount / 16) * 100}%` }}
                ></div>
              </div>
            </div>

            <div>
              <div className="flex justify-between text-sm mb-1.5">
                <span className="font-medium text-gray-700">Confirmed Disease</span>
                <span className="font-semibold text-red-600">{diseaseCount} / 16 Tiles</span>
              </div>
              <div className="h-2.5 rounded-full bg-gray-100 overflow-hidden">
                <div
                  className="h-full bg-red-500 rounded-full transition-all duration-300"
                  style={{ width: `${(diseaseCount / 16) * 100}%` }}
                ></div>
              </div>
            </div>

            <div>
              <div className="flex justify-between text-sm mb-1.5">
                <span className="font-medium text-gray-700">Disputed (Needs Review)</span>
                <span className="font-semibold text-amber-500">{disputedCount} / 16 Tiles</span>
              </div>
              <div className="h-2.5 rounded-full bg-gray-100 overflow-hidden">
                <div
                  className="h-full bg-amber-500 rounded-full transition-all duration-300"
                  style={{ width: `${(disputedCount / 16) * 100}%` }}
                ></div>
              </div>
            </div>
          </div>

          <div className="mt-6 rounded-lg bg-blue-50 border border-blue-100 p-3.5 text-xs text-blue-900 leading-relaxed">
            <strong>Dual-Evidence Rule:</strong> Tiles are marked Disputed when YOLO DL prediction and OpenCV visual analysis disagree.
          </div>
        </div>
      </div>

      {/* Multi-View Workflow Cards */}
      <div className="rounded-xl border border-gray-200 bg-white p-6 shadow-sm">
        <h3 className="text-lg font-bold text-gray-800">Multi-View Analysis Workflow</h3>
        <p className="mt-1 text-sm text-gray-500">4-Stage Academic Evaluation Pipeline</p>

        <div className="mt-5 grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-4">
          <div
            onClick={() => onNavigate('field-scan')}
            className="rounded-lg bg-[#E3F2FD] border border-blue-200 p-4 cursor-pointer hover:bg-blue-100 transition-colors"
          >
            <p className="text-xs font-bold uppercase text-[#0D47A1]">Stage 01</p>
            <p className="mt-1 font-bold text-[#0D47A1]">Field View</p>
            <p className="mt-2 text-xs text-gray-600 leading-normal">
              Analyze the 4x4 field composite image captured by drone simulation.
            </p>
          </div>

          <div
            onClick={() => onNavigate('tile-analysis')}
            className="rounded-lg bg-[#E3F2FD] border border-blue-200 p-4 cursor-pointer hover:bg-blue-100 transition-colors"
          >
            <p className="text-xs font-bold uppercase text-[#0D47A1]">Stage 02</p>
            <p className="mt-1 font-bold text-[#0D47A1]">Tile View</p>
            <p className="mt-2 text-xs text-gray-600 leading-normal">
              Run YOLO26 DL classifier and OpenCV DIP engine side by side.
            </p>
          </div>

          <div
            onClick={() => onNavigate('closeup-analysis')}
            className="rounded-lg bg-[#E3F2FD] border border-blue-200 p-4 cursor-pointer hover:bg-blue-100 transition-colors"
          >
            <p className="text-xs font-bold uppercase text-[#0D47A1]">Stage 03</p>
            <p className="mt-1 font-bold text-[#0D47A1]">Close-up</p>
            <p className="mt-2 text-xs text-gray-600 leading-normal">
              Upload close-up leaf photos for disputed tile resolution.
            </p>
          </div>

          <div
            onClick={() => onNavigate('results')}
            className="rounded-lg bg-[#E3F2FD] border border-blue-200 p-4 cursor-pointer hover:bg-blue-100 transition-colors"
          >
            <p className="text-xs font-bold uppercase text-[#0D47A1]">Stage 04</p>
            <p className="mt-1 font-bold text-[#0D47A1]">Final Assessment</p>
            <p className="mt-2 text-xs text-gray-600 leading-normal">
              Generate Plant Health Index, priority list, and agricultural guidance.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};
