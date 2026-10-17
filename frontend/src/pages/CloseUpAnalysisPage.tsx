import React, { useState } from 'react';
import { TileResult, ScanSummary } from '../types';
import { AlertTriangle, Upload, CheckCircle2, ShieldAlert, ArrowRight } from 'lucide-react';

interface CloseUpAnalysisPageProps {
  scan: ScanSummary | null;
  onNavigate: (tab: string) => void;
}

export const CloseUpAnalysisPage: React.FC<CloseUpAnalysisPageProps> = ({
  scan,
  onNavigate
}) => {
  const disputedTiles = scan ? scan.tile_results.filter((t) => t.fusion_status === 'DISPUTED') : [];

  const [selectedTileId, setSelectedTileId] = useState<number>(
    disputedTiles.length > 0 ? disputedTiles[0].tile_id : 1
  );
  const [analyzing, setAnalyzing] = useState(false);
  const [resolution, setResolution] = useState<any | null>(null);

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setAnalyzing(true);
    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('tile_id', selectedTileId.toString());
      if (scan) {
        formData.append('scan_id', scan.scan_id);
      }

      const response = await fetch('/api/analyze/closeup', {
        method: 'POST',
        body: formData,
      });

      if (response.ok) {
        const data = await response.json();
        setResolution(data);
      }
    } catch (err) {
      console.error('Close-up error:', err);
    } finally {
      setAnalyzing(false);
    }
  };

  return (
    <div className="space-y-7">
      {/* Page Title */}
      <div>
        <h2 className="text-2xl sm:text-3xl font-bold text-[#0D47A1]">
          Close-up Verification Analysis
        </h2>
        <p className="mt-1 text-sm text-gray-500">
          Multi-view verification tool for resolving disputed tiles requiring high-resolution leaf evaluation
        </p>
      </div>

      {/* Disputed Selector Card */}
      <div className="bg-white border border-gray-200 rounded-xl p-6 shadow-sm space-y-4">
        <h3 className="text-lg font-bold text-gray-800">Select Disputed Tile</h3>
        <p className="text-sm text-gray-500">
          {disputedTiles.length > 0
            ? `${disputedTiles.length} tile(s) currently marked DISPUTED due to evidence mismatch.`
            : 'No disputed tiles in current scan. Select any tile number below for close-up testing.'}
        </p>

        <div className="flex flex-wrap items-center gap-3">
          {(disputedTiles.length > 0 ? disputedTiles : [1, 2, 3, 4]).map((item: any) => {
            const tId = typeof item === 'number' ? item : item.tile_id;
            const isSelected = selectedTileId === tId;
            return (
              <button
                key={tId}
                onClick={() => {
                  setSelectedTileId(tId);
                  setResolution(null);
                }}
                className={`px-4 py-2 rounded-lg text-sm font-bold border transition-colors ${
                  isSelected
                    ? 'bg-[#0D47A1] text-white border-[#0D47A1]'
                    : 'bg-white text-gray-700 border-gray-300 hover:bg-gray-50'
                }`}
              >
                Tile #{tId}
              </button>
            );
          })}
        </div>
      </div>

      {/* Upload Close-up Leaf Box */}
      <div className="bg-white border border-gray-200 rounded-xl p-8 shadow-sm text-center">
        <div className="w-12 h-12 rounded-full bg-amber-100 text-amber-600 flex items-center justify-center mx-auto mb-3">
          <AlertTriangle className="w-6 h-6" />
        </div>

        <h3 className="font-bold text-gray-800 text-lg">
          Upload Close-up Image for Tile #{selectedTileId}
        </h3>
        <p className="text-sm text-gray-500 max-w-md mx-auto mt-1 mb-6">
          High-resolution leaf photographs allow the dual-evidence fusion engine to eliminate lighting ambiguity.
        </p>

        <label className="inline-flex items-center gap-2 bg-[#1565C0] hover:bg-[#0D47A1] text-white px-6 py-3 rounded-lg font-semibold text-sm cursor-pointer shadow-sm transition-colors">
          <Upload className="w-4 h-4" />
          <span>Upload Close-up Leaf Photo</span>
          <input
            type="file"
            accept="image/*"
            className="hidden"
            onChange={handleFileUpload}
            disabled={analyzing}
          />
        </label>

        {analyzing && (
          <p className="text-sm text-[#0D47A1] mt-4 flex items-center justify-center gap-2 font-medium">
            <span className="w-4 h-4 border-2 border-[#0D47A1] border-t-transparent rounded-full animate-spin"></span>
            Running close-up dual-evidence evaluation...
          </p>
        )}
      </div>

      {/* Resolution Output Card */}
      {resolution && (
        <div className="bg-white border border-gray-200 rounded-xl p-6 shadow-sm space-y-4">
          <div className="flex items-center justify-between border-b pb-3">
            <h3 className="font-bold text-gray-800 text-lg">
              Close-up Resolution Results for Tile #{resolution.tile_id}
            </h3>
            <span className={`px-3 py-1 rounded-full text-xs font-bold ${
              resolution.final_decision === 'HEALTHY'
                ? 'bg-green-100 text-green-800'
                : resolution.final_decision === 'CONFIRMED DISEASE'
                ? 'bg-red-100 text-red-800'
                : 'bg-amber-100 text-amber-800'
            }`}>
              {resolution.final_decision}
            </span>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
            <div className="p-4 bg-gray-50 rounded-lg border border-gray-200">
              <p className="text-xs text-gray-500 font-medium">Close-up YOLO DL</p>
              <p className="font-bold text-gray-800 mt-1 uppercase">
                {resolution.closeup_yolo_prediction}
              </p>
              <p className="text-xs text-gray-400 mt-0.5">
                {resolution.closeup_yolo_confidence > 0
                  ? `Confidence: ${(resolution.closeup_yolo_confidence * 100).toFixed(1)}%`
                  : 'Confidence: N/A (Model Not Trained)'}
              </p>
            </div>

            <div className="p-4 bg-gray-50 rounded-lg border border-gray-200">
              <p className="text-xs text-gray-500 font-medium">Close-up OpenCV DIP</p>
              <p className="font-bold text-gray-800 mt-1 uppercase">
                {resolution.closeup_opencv_status}
              </p>
              <p className="text-xs text-gray-400 mt-0.5">
                Lesion %: {resolution.closeup_lesion_percentage}%
              </p>
            </div>

            <div className="p-4 bg-gray-50 rounded-lg border border-gray-200">
              <p className="text-xs text-gray-500 font-medium">Final Decision</p>
              <p className="font-bold text-[#0D47A1] mt-1">{resolution.final_decision}</p>
              <p className="text-xs text-gray-400 mt-0.5">Severity: {resolution.severity}</p>
            </div>
          </div>

          {resolution.recommendation && (
            <div className="p-4 bg-blue-50 border border-blue-200 rounded-lg text-xs text-blue-950 space-y-1 mt-3">
              <p className="font-bold text-sm text-[#0D47A1]">
                {resolution.recommendation.title}
              </p>
              <p>{resolution.recommendation.treatment}</p>
            </div>
          )}
        </div>
      )}
    </div>
  );
};
