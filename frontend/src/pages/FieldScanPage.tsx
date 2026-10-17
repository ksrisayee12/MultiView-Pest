import React from 'react';
import { ScanSummary, TileResult } from '../types';
import { ShieldAlert, CheckCircle2, AlertTriangle, Activity, ArrowRight } from 'lucide-react';

interface FieldScanPageProps {
  scan: ScanSummary | null;
  onRunSimulation: () => void;
  onSelectTile: (tile: TileResult) => void;
  onNavigate: (tab: string) => void;
  loading: boolean;
}

export const FieldScanPage: React.FC<FieldScanPageProps> = ({
  scan,
  onRunSimulation,
  onSelectTile,
  onNavigate,
  loading
}) => {
  return (
    <div className="space-y-7">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h2 className="text-2xl sm:text-3xl font-bold text-[#0D47A1]">
            4x4 Field Scan & Heatmap
          </h2>
          <p className="mt-1 text-sm text-gray-500">
            Simulated Drone Field Image — 16 Tile Dual-Evidence Processing
          </p>
        </div>

        <button
          onClick={onRunSimulation}
          disabled={loading}
          className="bg-[#1565C0] hover:bg-[#0D47A1] text-white px-5 py-2.5 rounded-lg text-sm font-semibold transition-colors shadow-sm flex items-center gap-2 self-start disabled:opacity-50"
        >
          {loading ? (
            <span className="flex items-center gap-2">
              <span className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></span>
              Analyzing Field...
            </span>
          ) : (
            <>
              <Activity className="w-4 h-4" />
              Generate 4x4 Field Scan
            </>
          )}
        </button>
      </div>

      {/* Simulated Drone Notice */}
      <div className="rounded-lg bg-blue-50 border border-blue-200 p-4 text-xs text-blue-900 leading-relaxed flex items-start gap-3">
        <ShieldAlert className="w-5 h-5 text-[#1565C0] shrink-0 mt-0.5" />
        <div>
          <strong>Simulated Multi-View Field View:</strong> Drone view is simulated using a 4x4 composite of 16 leaf images because real drone field imagery is not available for this prototype.
        </div>
      </div>

      {/* Field Overview Summary Bar */}
      {scan && (
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
          <div className="bg-white rounded-xl p-4 border border-gray-200 shadow-sm">
            <p className="text-xs text-gray-500 font-medium">Scan ID</p>
            <p className="text-lg font-bold text-[#0D47A1] truncate mt-1">{scan.scan_id}</p>
          </div>

          <div className="bg-white rounded-xl p-4 border border-gray-200 shadow-sm">
            <p className="text-xs text-gray-500 font-medium">Plant Health Index</p>
            <p className="text-lg font-bold text-[#1565C0] mt-1">{scan.plant_health_index} / 100</p>
          </div>

          <div className="bg-white rounded-xl p-4 border border-gray-200 shadow-sm">
            <p className="text-xs text-gray-500 font-medium">YOLO Model Status</p>
            <span className={`inline-block mt-1 px-2.5 py-0.5 rounded text-xs font-semibold ${
              scan.yolo_model_loaded ? 'bg-green-100 text-green-800' : 'bg-amber-100 text-amber-800'
            }`}>
              {scan.yolo_model_loaded ? 'YOLO26 Active' : 'Model Pending Training'}
            </span>
          </div>

          <div className="bg-white rounded-xl p-4 border border-gray-200 shadow-sm">
            <p className="text-xs text-gray-500 font-medium">Tile Breakdown</p>
            <div className="flex items-center gap-2 mt-1 text-xs font-semibold">
              <span className="text-green-600">{scan.healthy_tiles} H</span> |
              <span className="text-red-600"> {scan.confirmed_disease_tiles} D</span> |
              <span className="text-amber-600"> {scan.disputed_tiles} Disp</span>
            </div>
          </div>
        </div>
      )}

      {/* 4x4 Interactive Tile Heatmap */}
      <div className="bg-white border border-gray-200 rounded-xl p-6 shadow-sm">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-bold text-gray-800">4x4 Field Tile Grid</h3>
          <div className="flex items-center gap-4 text-xs font-medium">
            <div className="flex items-center gap-1.5">
              <div className="w-3 h-3 rounded bg-green-500"></div>
              <span>Healthy</span>
            </div>
            <div className="flex items-center gap-1.5">
              <div className="w-3 h-3 rounded bg-red-500"></div>
              <span>Confirmed Disease</span>
            </div>
            <div className="flex items-center gap-1.5">
              <div className="w-3 h-3 rounded bg-amber-500"></div>
              <span>Disputed</span>
            </div>
          </div>
        </div>

        {scan && scan.tile_results.length > 0 ? (
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
            {scan.tile_results.map((tile) => {
              const isHealthy = tile.fusion_status === 'HEALTHY';
              const isDisease = tile.fusion_status === 'CONFIRMED DISEASE';
              const isDisputed = tile.fusion_status === 'DISPUTED';

              const bgColor = isHealthy
                ? 'bg-green-50 hover:bg-green-100 border-green-300'
                : isDisease
                ? 'bg-red-50 hover:bg-red-100 border-red-300'
                : 'bg-amber-50 hover:bg-amber-100 border-amber-300';

              const badgeColor = isHealthy
                ? 'bg-green-600 text-white'
                : isDisease
                ? 'bg-red-600 text-white'
                : 'bg-amber-500 text-white';

              return (
                <div
                  key={tile.tile_id}
                  onClick={() => {
                    onSelectTile(tile);
                    onNavigate('tile-analysis');
                  }}
                  className={`p-4 rounded-xl border-2 transition-all cursor-pointer shadow-sm relative group ${bgColor}`}
                >
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-xs font-bold text-gray-700">Tile #{tile.tile_id}</span>
                    <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${badgeColor}`}>
                      {tile.fusion_status}
                    </span>
                  </div>

                  <div className="space-y-1 text-xs text-gray-600">
                    <p>
                      <strong>YOLO:</strong> {tile.yolo_prediction.replace('Tomato___', '').replace('_', ' ')}
                    </p>
                    <p>
                      <strong>OpenCV:</strong> {tile.opencv_status.toUpperCase()}
                    </p>
                    <p>
                      <strong>Lesion %:</strong> {tile.lesion_percentage}%
                    </p>
                  </div>

                  <div className="mt-3 flex items-center justify-between text-[11px] font-semibold text-[#0D47A1] group-hover:underline">
                    <span>View Details</span>
                    <ArrowRight className="w-3.5 h-3.5" />
                  </div>
                </div>
              );
            })}
          </div>
        ) : (
          <div className="text-center py-12 text-gray-500">
            <Activity className="w-12 h-12 text-blue-300 mx-auto mb-3" />
            <p className="font-semibold text-gray-700">No Field Scan Data Active</p>
            <p className="text-sm mt-1">Click "Generate 4x4 Field Scan" above to run dual-evidence analysis.</p>
          </div>
        )}
      </div>
    </div>
  );
};
