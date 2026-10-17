import React from 'react';
import { ScanSummary } from '../types';
import { ShieldAlert, FileText, CheckCircle2, AlertTriangle, Sprout } from 'lucide-react';

interface ResultsPageProps {
  scan: ScanSummary | null;
  onNavigate: (tab: string) => void;
}

export const ResultsPage: React.FC<ResultsPageProps> = ({ scan, onNavigate }) => {
  if (!scan) {
    return (
      <div className="bg-white border border-gray-200 rounded-xl p-12 text-center text-gray-500 shadow-sm">
        <p className="font-semibold text-gray-700">No Assessment Active</p>
        <p className="text-sm mt-1">Please run a 4x4 Field Scan to view assessment results.</p>
        <button
          onClick={() => onNavigate('field-scan')}
          className="mt-4 bg-[#0D47A1] text-white px-5 py-2.5 rounded-lg text-sm font-semibold hover:bg-[#083578] transition-colors"
        >
          Go to Field Scan
        </button>
      </div>
    );
  }

  return (
    <div className="space-y-7">
      {/* Title */}
      <div>
        <h2 className="text-2xl sm:text-3xl font-bold text-[#0D47A1]">
          Field Health Assessment Results
        </h2>
        <p className="mt-1 text-sm text-gray-500">
          Executive summary, priority inspection queue, and registered management recommendations
        </p>
      </div>

      {/* Score Overview Banner */}
      <div className="bg-white border border-gray-200 rounded-xl p-6 shadow-sm flex flex-col sm:flex-row items-center justify-between gap-6">
        <div className="flex items-center gap-4">
          <div className="w-16 h-16 rounded-full bg-blue-100 text-[#0D47A1] flex items-center justify-center font-bold text-2xl">
            {scan.plant_health_index}
          </div>
          <div>
            <h3 className="text-xl font-bold text-gray-800">
              Prototype Plant Health Index: {scan.plant_health_index} / 100
            </h3>
            <p className="text-xs text-gray-500 mt-1">
              Field Status: {scan.plant_health_index > 80 ? 'Good Condition' : 'Intervention Recommended'}
            </p>
          </div>
        </div>

        <div className="flex gap-3 text-xs font-semibold">
          <div className="px-3 py-2 bg-green-50 text-green-700 rounded-lg border border-green-200">
            {scan.healthy_tiles} Healthy Tiles
          </div>
          <div className="px-3 py-2 bg-red-50 text-red-700 rounded-lg border border-red-200">
            {scan.confirmed_disease_tiles} Confirmed Disease
          </div>
          <div className="px-3 py-2 bg-amber-50 text-amber-700 rounded-lg border border-amber-200">
            {scan.disputed_tiles} Disputed Tiles
          </div>
        </div>
      </div>

      {/* Priority Inspection Queue */}
      <div className="bg-white border border-gray-200 rounded-xl p-6 shadow-sm space-y-4">
        <div className="flex items-center justify-between">
          <h3 className="text-lg font-bold text-gray-800 flex items-center gap-2">
            <ShieldAlert className="w-5 h-5 text-red-600" />
            Priority Inspection Queue
          </h3>
          <span className="text-xs text-gray-500 font-medium">Sorted by Risk & Lesion %</span>
        </div>

        {scan.priority_list && scan.priority_list.length > 0 ? (
          <div className="divide-y divide-gray-100 border rounded-lg overflow-hidden">
            {scan.priority_list.map((item, idx) => (
              <div key={idx} className="p-4 flex flex-col sm:flex-row sm:items-center justify-between gap-3 hover:bg-gray-50">
                <div className="flex items-center gap-3">
                  <span className={`w-7 h-7 rounded-full flex items-center justify-center text-xs font-bold ${
                    item.priority_level === 1 ? 'bg-red-600 text-white' : 'bg-amber-500 text-white'
                  }`}>
                    P{item.priority_level}
                  </span>
                  <div>
                    <h4 className="font-bold text-gray-800 text-sm">
                      Tile #{item.tile_id} — {item.disease ? item.disease.replace('Tomato___', '') : 'Disputed Issue'}
                    </h4>
                    <p className="text-xs text-gray-500 mt-0.5">
                      Severity: <strong>{item.severity}</strong> | Estimated Damaged Area: {item.lesion_percentage}%
                    </p>
                  </div>
                </div>

                <button
                  onClick={() => onNavigate(item.fusion_status === 'DISPUTED' ? 'closeup-analysis' : 'tile-analysis')}
                  className="bg-[#E3F2FD] hover:bg-[#BBDEFB] text-[#0D47A1] px-3 py-1.5 rounded text-xs font-semibold transition-colors self-start sm:self-auto"
                >
                  {item.fusion_status === 'DISPUTED' ? 'Upload Close-up' : 'Inspect Tile'}
                </button>
              </div>
            ))}
          </div>
        ) : (
          <p className="text-sm text-gray-500 italic py-4">No priority items flagged. Field condition is optimal.</p>
        )}
      </div>

      {/* Recommended Agricultural Guidance */}
      <div className="bg-white border border-gray-200 rounded-xl p-6 shadow-sm space-y-4">
        <h3 className="text-lg font-bold text-gray-800 flex items-center gap-2">
          <FileText className="w-5 h-5 text-[#0D47A1]" />
          Controlled Treatment Guidance & Safety Requirements
        </h3>

        <div className="space-y-4">
          {scan.tile_results.slice(0, 3).map((t) => (
            <div key={t.tile_id} className="p-4 rounded-lg bg-gray-50 border border-gray-200 space-y-2 text-xs">
              <div className="flex items-center justify-between font-bold text-sm text-gray-800">
                <span>Tile #{t.tile_id}: {t.recommendation.title}</span>
                <span className="text-xs text-gray-500 font-normal">Status: {t.fusion_status}</span>
              </div>
              <p className="text-gray-700"><strong>Treatment:</strong> {t.recommendation.treatment}</p>
              <div>
                <strong>Prevention Steps:</strong>
                <ul className="list-disc pl-4 mt-1 space-y-0.5 text-gray-600">
                  {t.recommendation.prevention.map((prev, pIdx) => (
                    <li key={pIdx}>{prev}</li>
                  ))}
                </ul>
              </div>
            </div>
          ))}
        </div>

        <div className="p-3.5 bg-blue-50 border border-blue-200 rounded-lg text-xs text-blue-950 font-medium">
          NOTICE: All pesticide or control recommendations provided by this prototype are for educational guidance only. Always consult registered product labels and local agricultural extension officers before applying field controls.
        </div>
      </div>
    </div>
  );
};
