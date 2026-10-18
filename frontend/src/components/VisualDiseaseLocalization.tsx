import React, { useState } from 'react';
import {
  Crosshair,
  Flame,
  Eye,
  Columns,
  ZoomIn,
  AlertCircle,
  CheckCircle2,
  Sparkles,
  Upload,
  RefreshCw,
  MapPin,
  ShieldAlert,
  Info
} from 'lucide-react';

export interface DetectedRegion {
  region_id: number;
  label: string;
  condition: string;
  confidence: number;
  severity: 'CRITICAL' | 'MODERATE' | 'MILD';
  box_pixels: [number, number, number, number];
  norm_box: [number, number, number, number]; // [ymin, xmin, ymax, xmax]
  area_pct: number;
  location_description: string;
  source?: string;
}

export interface VisualLocalizationData {
  annotated_image: string;
  heatmap_image: string;
  original_image: string;
  detected_regions: DetectedRegion[];
  total_regions_count: number;
  condition_key: string;
  is_healthy: boolean;
}

interface VisualDiseaseLocalizationProps {
  visualData?: VisualLocalizationData | null;
  cropType?: string;
  diseaseName?: string;
  onUploadCustomLeaf?: (file: File) => Promise<void>;
  isLoading?: boolean;
}

export const VisualDiseaseLocalization: React.FC<VisualDiseaseLocalizationProps> = ({
  visualData,
  cropType = 'Tomato',
  diseaseName,
  onUploadCustomLeaf,
  isLoading = false,
}) => {
  const [viewMode, setViewMode] = useState<'annotated' | 'heatmap' | 'original' | 'split'>('annotated');
  const [selectedRegionId, setSelectedRegionId] = useState<number | null>(null);
  const [uploading, setUploading] = useState<boolean>(false);
  const [localData, setLocalData] = useState<VisualLocalizationData | null>(visualData || null);

  // Sync if prop updates
  React.useEffect(() => {
    if (visualData) {
      setLocalData(visualData);
    }
  }, [visualData]);

  const activeData = localData;

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    if (onUploadCustomLeaf) {
      await onUploadCustomLeaf(file);
      return;
    }

    setUploading(true);
    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('crop_type', cropType.toLowerCase());

      const res = await fetch('/api/ensemble/assess', {
        method: 'POST',
        body: formData,
      });

      if (res.ok) {
        const report = await res.json();
        if (report.visual_localization) {
          setLocalData(report.visual_localization);
        }
      }
    } catch (err) {
      console.error('Error localizing uploaded leaf:', err);
    } finally {
      setUploading(false);
    }
  };

  const isHealthy = activeData?.is_healthy ?? false;
  const regions = activeData?.detected_regions || [];

  return (
    <div className="bg-white rounded-2xl border border-[#D5E2D7] shadow-sm overflow-hidden space-y-0">
      {/* Visualizer Header */}
      <div className="bg-gradient-to-r from-[#1B4332] via-[#245741] to-[#2D6A4F] p-4 text-white flex flex-wrap items-center justify-between gap-3">
        <div className="space-y-0.5">
          <div className="flex items-center gap-2">
            <span className="px-2 py-0.5 rounded-full text-[10px] font-bold uppercase tracking-wider bg-emerald-400/20 text-emerald-200 border border-emerald-400/30 flex items-center gap-1">
              <Crosshair className="w-3 h-3 text-emerald-300" />
              Computer Vision Localization
            </span>
            <span className="text-white/60 text-xs">•</span>
            <span className="text-xs text-emerald-100/90 font-medium">
              {isHealthy ? 'Optimal Chlorophyll Balance' : `${regions.length} Diseased Zone(s) Detected`}
            </span>
          </div>
          <h3 className="text-base font-bold text-white tracking-tight flex items-center gap-2">
            Visual Disease Localization & Lesion Heatmap
          </h3>
        </div>

        {/* View Mode Pill Switcher */}
        <div className="flex items-center gap-1 bg-black/25 backdrop-blur-md p-1 rounded-xl border border-white/15">
          <button
            onClick={() => setViewMode('annotated')}
            className={`px-3 py-1.5 rounded-lg text-xs font-bold transition-all flex items-center gap-1.5 ${
              viewMode === 'annotated'
                ? 'bg-white text-[#1B4332] shadow-sm'
                : 'text-white/80 hover:text-white hover:bg-white/10'
            }`}
            title="Bounding boxes identifying detected lesion spots"
          >
            <Crosshair className="w-3.5 h-3.5 text-amber-500" />
            <span>Detected Lesions</span>
          </button>

          <button
            onClick={() => setViewMode('heatmap')}
            className={`px-3 py-1.5 rounded-lg text-xs font-bold transition-all flex items-center gap-1.5 ${
              viewMode === 'heatmap'
                ? 'bg-white text-[#1B4332] shadow-sm'
                : 'text-white/80 hover:text-white hover:bg-white/10'
            }`}
            title="Thermal infection heatmap showing necrotic foliar spread"
          >
            <Flame className="w-3.5 h-3.5 text-rose-500" />
            <span>Infection Heatmap</span>
          </button>

          <button
            onClick={() => setViewMode('original')}
            className={`px-3 py-1.5 rounded-lg text-xs font-bold transition-all flex items-center gap-1.5 ${
              viewMode === 'original'
                ? 'bg-white text-[#1B4332] shadow-sm'
                : 'text-white/80 hover:text-white hover:bg-white/10'
            }`}
            title="Clean, unannotated leaf image"
          >
            <Eye className="w-3.5 h-3.5 text-emerald-400" />
            <span>Original Leaf</span>
          </button>

          <button
            onClick={() => setViewMode('split')}
            className={`px-3 py-1.5 rounded-lg text-xs font-bold transition-all flex items-center gap-1.5 ${
              viewMode === 'split'
                ? 'bg-white text-[#1B4332] shadow-sm'
                : 'text-white/80 hover:text-white hover:bg-white/10'
            }`}
            title="Side-by-side comparison"
          >
            <Columns className="w-3.5 h-3.5 text-sky-400" />
            <span>Side-by-Side</span>
          </button>
        </div>
      </div>

      {/* Main Interactive Stage */}
      <div className="p-4 sm:p-5 space-y-4 bg-[#F8FAF9]">
        {isLoading || uploading ? (
          <div className="h-80 flex flex-col items-center justify-center bg-white rounded-xl border border-gray-200 text-gray-500 space-y-3">
            <RefreshCw className="w-8 h-8 text-[#1B4332] animate-spin" />
            <p className="text-sm font-semibold text-gray-700">
              Running multi-model visual localization on leaf foliage...
            </p>
            <p className="text-xs text-gray-400">
              YOLO bounding box extraction & optical necrotic contour mapping in progress
            </p>
          </div>
        ) : activeData ? (
          <div className="space-y-4">
            {/* Display Canvas Container */}
            <div className="relative bg-slate-900 rounded-xl overflow-hidden border border-gray-200 shadow-inner flex items-center justify-center min-h-[320px] max-h-[520px]">
              {viewMode === 'split' ? (
                <div className="grid grid-cols-1 sm:grid-cols-2 w-full h-full gap-1 p-1 bg-slate-950">
                  <div className="relative flex flex-col items-center justify-center bg-black/40 rounded-lg overflow-hidden p-2">
                    <span className="absolute top-2 left-2 z-10 px-2.5 py-1 rounded bg-black/70 backdrop-blur text-[11px] font-bold text-white flex items-center gap-1">
                      <Eye className="w-3 h-3 text-emerald-400" /> Original Foliage
                    </span>
                    <img
                      src={activeData.original_image}
                      alt="Original Leaf"
                      className="max-h-[380px] w-auto object-contain rounded"
                    />
                  </div>
                  <div className="relative flex flex-col items-center justify-center bg-black/40 rounded-lg overflow-hidden p-2">
                    <span className="absolute top-2 left-2 z-10 px-2.5 py-1 rounded bg-black/70 backdrop-blur text-[11px] font-bold text-white flex items-center gap-1">
                      <Crosshair className="w-3 h-3 text-amber-400" /> Localized Lesion Zones
                    </span>
                    <img
                      src={activeData.annotated_image}
                      alt="Annotated Disease Lesions"
                      className="max-h-[380px] w-auto object-contain rounded"
                    />
                  </div>
                </div>
              ) : (
                <div className="relative w-full flex items-center justify-center p-2">
                  <img
                    src={
                      viewMode === 'annotated'
                        ? activeData.annotated_image
                        : viewMode === 'heatmap'
                        ? activeData.heatmap_image
                        : activeData.original_image
                    }
                    alt="Foliar Disease Localization"
                    className="max-h-[460px] w-auto object-contain rounded-lg transition-all duration-300"
                  />

                  {/* Mode Watermark Overlay Badge */}
                  <div className="absolute top-4 left-4 pointer-events-none">
                    <div className="px-3 py-1.5 rounded-lg bg-black/75 backdrop-blur-md border border-white/20 text-white text-xs font-bold flex items-center gap-2 shadow-lg">
                      {viewMode === 'annotated' ? (
                        <>
                          <Crosshair className="w-4 h-4 text-amber-400" />
                          <span>Detected Disease Lesions Overlay</span>
                        </>
                      ) : viewMode === 'heatmap' ? (
                        <>
                          <Flame className="w-4 h-4 text-rose-400" />
                          <span>Infection Intensity Heatmap (TURBO)</span>
                        </>
                      ) : (
                        <>
                          <Eye className="w-4 h-4 text-emerald-400" />
                          <span>Pristine Foliar Lamina</span>
                        </>
                      )}
                    </div>
                  </div>

                  {/* Active Region Highlights Legend */}
                  {selectedRegionId !== null && (
                    <div className="absolute bottom-4 right-4 bg-black/85 backdrop-blur-md p-3 rounded-xl border border-amber-400/40 text-white text-xs space-y-1 shadow-2xl max-w-xs animate-fade-in">
                      {(() => {
                        const r = regions.find((x) => x.region_id === selectedRegionId);
                        if (!r) return null;
                        return (
                          <>
                            <div className="flex items-center justify-between gap-2">
                              <span className="font-bold text-amber-300 flex items-center gap-1">
                                <MapPin className="w-3.5 h-3.5 text-amber-400" />
                                {r.label}
                              </span>
                              <span className="text-[10px] font-mono bg-amber-400/20 text-amber-200 px-1.5 py-0.5 rounded">
                                {(r.confidence * 100).toFixed(1)}% Conf
                              </span>
                            </div>
                            <p className="text-[11px] text-gray-300 font-medium">
                              Location: <span className="text-white font-semibold">{r.location_description}</span>
                            </p>
                            <p className="text-[10px] text-gray-400">
                              Estimated surface involvement: ~{r.area_pct}% of total lamina
                            </p>
                          </>
                        );
                      })()}
                    </div>
                  )}
                </div>
              )}
            </div>

            {/* Localized Regions Directory */}
            {regions.length > 0 ? (
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <h4 className="text-xs font-bold uppercase tracking-wider text-[#1B4332] flex items-center gap-1.5">
                    <Crosshair className="w-3.5 h-3.5 text-emerald-700" />
                    Localized Lesion Pinpoints ({regions.length} Areas Detected)
                  </h4>
                  <span className="text-[11px] text-gray-500">
                    Hover or click a lesion to highlight its coordinates
                  </span>
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-2.5">
                  {regions.map((reg) => {
                    const isSelected = selectedRegionId === reg.region_id;
                    return (
                      <div
                        key={reg.region_id}
                        onMouseEnter={() => setSelectedRegionId(reg.region_id)}
                        onMouseLeave={() => setSelectedRegionId(null)}
                        onClick={() => setSelectedRegionId(isSelected ? null : reg.region_id)}
                        className={`p-3 rounded-xl border text-xs cursor-pointer transition-all duration-200 space-y-1.5 ${
                          isSelected
                            ? 'bg-amber-50/90 border-amber-400 shadow-md ring-1 ring-amber-300'
                            : 'bg-white border-gray-200 hover:border-emerald-300 hover:shadow-xs'
                        }`}
                      >
                        <div className="flex items-center justify-between">
                          <span className="font-bold text-gray-900 flex items-center gap-1.5">
                            <span className="w-5 h-5 rounded-full bg-amber-500/20 text-amber-800 text-[10px] font-black flex items-center justify-center">
                              #{reg.region_id}
                            </span>
                            <span className="truncate max-w-[130px]">{reg.label}</span>
                          </span>
                          <span
                            className={`px-1.5 py-0.5 rounded text-[10px] font-bold ${
                              reg.severity === 'CRITICAL'
                                ? 'bg-rose-100 text-rose-800'
                                : reg.severity === 'MODERATE'
                                ? 'bg-amber-100 text-amber-800'
                                : 'bg-emerald-100 text-emerald-800'
                            }`}
                          >
                            {reg.severity}
                          </span>
                        </div>

                        <div className="text-[11px] text-gray-600 flex items-center gap-1">
                          <MapPin className="w-3 h-3 text-emerald-700 shrink-0" />
                          <span className="font-medium text-gray-800">{reg.location_description}</span>
                        </div>

                        <div className="flex items-center justify-between text-[10px] text-gray-500 pt-1 border-t border-gray-100">
                          <span>Lamina Area: ~{reg.area_pct}%</span>
                          <span className="font-bold text-[#1B4332]">
                            {(reg.confidence * 100).toFixed(1)}% AI Conf
                          </span>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            ) : isHealthy ? (
              <div className="p-4 bg-emerald-50 border border-emerald-200 rounded-xl flex items-center gap-3">
                <CheckCircle2 className="w-5 h-5 text-emerald-600 shrink-0" />
                <div className="text-xs text-emerald-900">
                  <p className="font-bold">No Pathogenic Lesions Localized</p>
                  <p className="text-emerald-700 mt-0.5">
                    Foliar lamina exhibits continuous photosynthetic green tissue with no necrotic spots or chlorotic margins.
                  </p>
                </div>
              </div>
            ) : (
              <div className="p-4 bg-amber-50 border border-amber-200 rounded-xl flex items-center gap-3">
                <AlertCircle className="w-5 h-5 text-amber-600 shrink-0" />
                <div className="text-xs text-amber-900">
                  <p className="font-bold">Diffuse Micro-Symptom Pattern</p>
                  <p className="text-amber-700 mt-0.5">
                    Pathogen symptoms are diffusely distributed across foliar veins rather than concentrated in discrete spot lesions. Check the infection heatmap above for full intensity distribution.
                  </p>
                </div>
              </div>
            )}
          </div>
        ) : (
          <div className="h-48 flex flex-col items-center justify-center bg-white rounded-xl border border-dashed border-gray-300 text-gray-400 space-y-2">
            <Info className="w-8 h-8 text-gray-300" />
            <p className="text-xs font-semibold text-gray-600">No leaf image loaded for localization</p>
            <p className="text-[11px] text-gray-400">
              Run assessment on plant session or upload a leaf photo below
            </p>
          </div>
        )}

        {/* Upload / Test Any Leaf Photo Box */}
        <div className="pt-2 border-t border-gray-200 flex flex-wrap items-center justify-between gap-3 text-xs">
          <div className="text-gray-600">
            <span className="font-bold text-gray-800">Test another leaf photo:</span> Upload any leaf image to pinpoint infected zones in real time
          </div>
          <label className="inline-flex items-center gap-1.5 bg-[#1B4332] hover:bg-[#133023] text-white px-3 py-1.5 rounded-lg text-xs font-bold cursor-pointer transition-colors shadow-xs">
            <Upload className="w-3.5 h-3.5" />
            <span>{uploading ? 'Analyzing Leaf...' : 'Localize New Leaf Image'}</span>
            <input
              type="file"
              accept="image/*"
              className="hidden"
              onChange={handleFileChange}
              disabled={uploading}
            />
          </label>
        </div>
      </div>
    </div>
  );
};
