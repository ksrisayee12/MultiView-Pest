import React, { useState } from 'react';
import { TileResult } from '../types';
import { ShieldAlert, CheckCircle2, AlertTriangle, Upload, ArrowRight, FileText } from 'lucide-react';

interface TileAnalysisPageProps {
  selectedTile: TileResult | null;
  onNavigate: (tab: string) => void;
}

export const TileAnalysisPage: React.FC<TileAnalysisPageProps> = ({
  selectedTile,
  onNavigate
}) => {
  const [analyzing, setAnalyzing] = useState(false);
  const [customResult, setCustomResult] = useState<TileResult | null>(null);

  const activeTile = customResult || selectedTile;

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setAnalyzing(true);
    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch('/api/analyze/tile', {
        method: 'POST',
        body: formData,
      });

      if (response.ok) {
        const data = await response.json();
        setCustomResult({
          tile_id: 99,
          grid_row: 0,
          grid_col: 0,
          bbox: [0, 0, 256, 256],
          ...data
        });
      }
    } catch (err) {
      console.error('Tile upload error:', err);
    } finally {
      setAnalyzing(false);
    }
  };

  return (
    <div className="space-y-7">
      {/* Page Title */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h2 className="text-2xl sm:text-3xl font-bold text-[#0D47A1]">
            Tile Level Dual-Evidence Analysis
          </h2>
          <p className="mt-1 text-sm text-gray-500">
            Side-by-side evaluation of Method A (YOLO26 DL) and Method B (OpenCV DIP)
          </p>
        </div>

        {/* Upload Custom Leaf Button */}
        <label className="bg-[#1565C0] hover:bg-[#0D47A1] text-white px-4 py-2.5 rounded-lg text-sm font-semibold transition-colors cursor-pointer flex items-center gap-2 self-start shadow-sm">
          <Upload className="w-4 h-4" />
          <span>Upload Custom Leaf Image</span>
          <input
            type="file"
            accept="image/*"
            className="hidden"
            onChange={handleFileUpload}
            disabled={analyzing}
          />
        </label>
      </div>

      {analyzing && (
        <div className="p-4 rounded-lg bg-blue-50 border border-blue-200 text-blue-900 text-sm flex items-center gap-3">
          <span className="w-4 h-4 border-2 border-[#1565C0] border-t-transparent rounded-full animate-spin"></span>
          Processing leaf image through YOLO26 and OpenCV DIP pipeline...
        </div>
      )}

      {activeTile ? (
        <div className="space-y-6">
          {/* Header Card for Tile */}
          <div className="bg-white border border-gray-200 rounded-xl p-6 shadow-sm flex flex-col sm:flex-row sm:items-center justify-between gap-4">
            <div>
              <div className="flex items-center gap-3">
                <h3 className="text-xl font-bold text-gray-800">
                  Tile #{activeTile.tile_id} Assessment
                </h3>
                <span className={`px-3 py-1 rounded-full text-xs font-bold ${
                  activeTile.fusion_status === 'HEALTHY'
                    ? 'bg-green-100 text-green-800'
                    : activeTile.fusion_status === 'CONFIRMED DISEASE'
                    ? 'bg-red-100 text-red-800'
                    : 'bg-amber-100 text-amber-800'
                }`}>
                  {activeTile.fusion_status}
                </span>
              </div>
              <p className="text-xs text-gray-500 mt-1">
                Grid Coordinates: Row {activeTile.grid_row + 1}, Col {activeTile.grid_col + 1}
              </p>
            </div>

            {activeTile.fusion_status === 'DISPUTED' && (
              <button
                onClick={() => onNavigate('closeup-analysis')}
                className="bg-amber-500 hover:bg-amber-600 text-white px-4 py-2 rounded-lg text-xs font-bold transition-colors flex items-center gap-2"
              >
                <AlertTriangle className="w-4 h-4" />
                Resolve via Close-up Analysis
              </button>
            )}
          </div>

          {/* Side by Side Evidence Comparison */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Method A Card */}
            <div className="bg-white border border-gray-200 rounded-xl p-6 shadow-sm space-y-4">
              <div className="flex items-center justify-between border-b pb-3">
                <div>
                  <h4 className="font-bold text-[#0D47A1] text-base">Method A: YOLO26 DL</h4>
                  <p className="text-xs text-gray-500">Deep Learning Disease Classifier</p>
                </div>
                <span className="text-xs font-semibold px-2.5 py-1 bg-blue-50 text-[#0D47A1] rounded border border-blue-200">
                  AI Evidence
                </span>
              </div>

              <div className="space-y-3 text-sm">
                <div className="flex justify-between py-1 border-b border-gray-100">
                  <span className="text-gray-600">Predicted Class:</span>
                  <span className="font-bold text-gray-800 uppercase">
                    {activeTile.yolo_prediction.replace('Tomato___', '').replace('_', ' ')}
                  </span>
                </div>

                <div className="flex justify-between py-1 border-b border-gray-100">
                  <span className="text-gray-600">Confidence Score:</span>
                  <span className="font-bold text-gray-800">
                    {activeTile.yolo_confidence > 0
                      ? `${(activeTile.yolo_confidence * 100).toFixed(1)}%`
                      : 'N/A (Model Not Trained)'}
                  </span>
                </div>

                <div className="flex justify-between py-1 border-b border-gray-100">
                  <span className="text-gray-600">Classification Status:</span>
                  <span className={`font-semibold ${
                    activeTile.yolo_prediction === 'healthy'
                      ? 'text-green-600'
                      : activeTile.yolo_prediction === 'YOLO model not trained'
                      ? 'text-amber-600'
                      : 'text-red-600'
                  }`}>
                    {activeTile.yolo_prediction === 'healthy'
                      ? 'Healthy Foliage'
                      : activeTile.yolo_prediction === 'YOLO model not trained'
                      ? 'YOLO model not trained'
                      : 'Disease Detected'}
                  </span>
                </div>
              </div>
            </div>

            {/* Method B Card */}
            <div className="bg-white border border-gray-200 rounded-xl p-6 shadow-sm space-y-4">
              <div className="flex items-center justify-between border-b pb-3">
                <div>
                  <h4 className="font-bold text-[#0D47A1] text-base">Method B: OpenCV DIP</h4>
                  <p className="text-xs text-gray-500">Digital Image Processing Analysis</p>
                </div>
                <span className="text-xs font-semibold px-2.5 py-1 bg-blue-50 text-[#0D47A1] rounded border border-blue-200">
                  Visual Evidence
                </span>
              </div>

              <div className="space-y-3 text-sm">
                <div className="flex justify-between py-1 border-b border-gray-100">
                  <span className="text-gray-600">Visual Health Status:</span>
                  <span className={`font-bold uppercase ${
                    activeTile.opencv_status === 'normal' ? 'text-green-600' : 'text-red-600'
                  }`}>
                    {activeTile.opencv_status}
                  </span>
                </div>

                <div className="flex justify-between py-1 border-b border-gray-100">
                  <span className="text-gray-600">Greenness Index:</span>
                  <span className="font-bold text-gray-800">{activeTile.greenness_index}</span>
                </div>

                <div className="flex justify-between py-1 border-b border-gray-100">
                  <span className="text-gray-600">Estimated Lesion Area:</span>
                  <span className="font-bold text-red-600">{activeTile.lesion_percentage}%</span>
                </div>

                <div className="flex justify-between py-1 border-b border-gray-100">
                  <span className="text-gray-600">Green Pixel Ratio:</span>
                  <span className="font-bold text-gray-800">{activeTile.green_pixel_percentage}%</span>
                </div>
              </div>
            </div>
          </div>

          {/* Fusion Result & Treatment Card */}
          <div className="bg-white border border-gray-200 rounded-xl p-6 shadow-sm space-y-4">
            <h4 className="font-bold text-gray-800 text-base flex items-center gap-2">
              <FileText className="w-5 h-5 text-[#0D47A1]" />
              Management Guidance & Prevention
            </h4>

            {activeTile.recommendation && (
              <div className="space-y-3 text-sm text-gray-700">
                <p>
                  <strong>Diagnosis Title:</strong> {activeTile.recommendation.title}
                </p>
                <p>
                  <strong>Treatment Guidance:</strong> {activeTile.recommendation.treatment}
                </p>

                <div>
                  <strong className="block mb-1">Prevention Practices:</strong>
                  <ul className="list-disc pl-5 space-y-1 text-xs text-gray-600">
                    {activeTile.recommendation.prevention.map((tip, idx) => (
                      <li key={idx}>{tip}</li>
                    ))}
                  </ul>
                </div>

                <div className="p-3 bg-gray-50 border border-gray-200 rounded text-xs text-gray-500 italic mt-3">
                  {activeTile.recommendation.disclaimer}
                </div>
              </div>
            )}
          </div>
        </div>
      ) : (
        <div className="bg-white border border-gray-200 rounded-xl p-12 text-center text-gray-500 shadow-sm">
          <p className="font-semibold text-gray-700">No Tile Selected</p>
          <p className="text-sm mt-1">
            Please select a tile from the Field Scan page or upload a custom leaf image above.
          </p>
        </div>
      )}
    </div>
  );
};
