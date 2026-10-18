import React, { useState } from 'react';
import {
  ShieldAlert,
  CheckCircle2,
  AlertTriangle,
  Layers,
  Bug,
  Activity,
  FileCheck,
  ChevronDown,
  ChevronUp,
  Stethoscope,
  Sparkles,
  Info,
} from 'lucide-react';
import { VisualDiseaseLocalization, VisualLocalizationData } from './VisualDiseaseLocalization';

interface DiagnosticExplainerCardProps {
  diagnosis: string;
  conditionKey: string;
  isHealthy: boolean;
  confidence: number;
  severityGrade: string;
  lesionPercentage: number;
  affectedViews: string[];
  consensusScore: number;
  modelsCount?: number;
  causalPathogen?: string;
  causesText?: string;
  symptomsText?: string;
  treatmentCultural?: string[];
  treatmentOrganic?: string[];
  treatmentChemical?: string[];
  preventionText?: string;
  modelProvenance?: any;
  visualLocalization?: VisualLocalizationData | null;
  cropType?: string;
}

export const DiagnosticExplainerCard: React.FC<DiagnosticExplainerCardProps> = ({
  diagnosis,
  conditionKey,
  isHealthy,
  confidence,
  severityGrade,
  lesionPercentage,
  affectedViews,
  consensusScore,
  modelsCount = 3,
  causalPathogen,
  causesText,
  symptomsText,
  treatmentCultural = [],
  treatmentOrganic = [],
  treatmentChemical = [],
  preventionText,
  modelProvenance,
  visualLocalization,
  cropType = 'Tomato',
}) => {
  const [activeTab, setActiveTab] = useState<'cultural' | 'organic' | 'chemical' | 'prevention'>('cultural');
  const [showModelDetails, setShowModelDetails] = useState<boolean>(false);

  // Default fallback guidance if custom strings not passed
  const defaultCauses = isHealthy
    ? 'Foliage exhibits uniform chlorophyll distribution with no pathogenic lesions or chlorotic margins.'
    : 'Infection initiated via rain-splash or windborne spores landing on wet foliage under favorable microclimatic humidity (>80% RH).';

  const defaultSymptoms = isHealthy
    ? 'Turgid green lamina, intact leaf margins, active photosynthetic tissue.'
    : 'Visible necrotic foliar lesions with characteristic chlorotic halos identified across camera viewpoints.';

  return (
    <div className="bg-white rounded-2xl border border-[#D5E2D7] shadow-sm overflow-hidden space-y-0">
      {/* Top Banner: Diagnosis & Consensus */}
      <div
        className={`p-5 text-white flex flex-wrap items-center justify-between gap-4 ${
          isHealthy
            ? 'bg-gradient-to-r from-[#1B4332] to-[#2D6A4F]'
            : 'bg-gradient-to-r from-[#3A1414] via-[#5C1D1D] to-[#8B2626]'
        }`}
      >
        <div className="space-y-1">
          <div className="flex items-center gap-2">
            <span
              className={`px-2.5 py-0.5 rounded-full text-[11px] font-bold tracking-wide uppercase border ${
                isHealthy
                  ? 'bg-emerald-400/20 text-emerald-200 border-emerald-400/30'
                  : 'bg-amber-400/20 text-amber-200 border-amber-400/30'
              }`}
            >
              {severityGrade || (isHealthy ? 'Optimal Vitality' : 'Foliar Infection')}
            </span>
            <span className="text-white/70 text-xs">•</span>
            <span className="text-xs text-white/90 font-mono">
              ~{lesionPercentage.toFixed(1)}% Lesion Area
            </span>
          </div>
          <h2 className="text-xl font-bold tracking-tight text-white flex items-center gap-2">
            {isHealthy ? <CheckCircle2 className="w-6 h-6 text-emerald-400" /> : <ShieldAlert className="w-6 h-6 text-amber-400" />}
            {diagnosis}
          </h2>
          {causalPathogen && (
            <p className="text-xs text-white/80 italic">
              Causal Agent: {causalPathogen}
            </p>
          )}
        </div>

        {/* Multi-Model Consensus Badge */}
        <div className="bg-white/10 backdrop-blur-md px-4 py-2.5 rounded-xl border border-white/20 text-right">
          <p className="text-[10px] text-white/70 uppercase tracking-wider font-semibold">Multi-Model Ensemble</p>
          <p className="text-lg font-extrabold text-white flex items-center justify-end gap-1.5">
            <Sparkles className="w-4 h-4 text-amber-300" />
            {consensusScore.toFixed(1)}% Consensus
          </p>
          <p className="text-[10px] text-emerald-200/90 font-medium">
            {modelsCount} Independent Models Agreed
          </p>
        </div>
      </div>

      {/* Cause & Symptom Breakdown */}
      <div className="p-5 grid grid-cols-1 md:grid-cols-2 gap-4 border-b border-[#E3EDE5] bg-[#FAFBF9]">
        <div className="space-y-1.5">
          <h4 className="text-xs font-bold uppercase tracking-wider text-[#1B4332] flex items-center gap-1.5">
            <Info className="w-3.5 h-3.5 text-emerald-700" /> What Caused This?
          </h4>
          <p className="text-xs text-gray-700 leading-relaxed">
            {causesText || defaultCauses}
          </p>
        </div>
        <div className="space-y-1.5">
          <h4 className="text-xs font-bold uppercase tracking-wider text-[#1B4332] flex items-center gap-1.5">
            <Activity className="w-3.5 h-3.5 text-emerald-700" /> Diagnostic Signs & Symptoms
          </h4>
          <p className="text-xs text-gray-700 leading-relaxed">
            {symptomsText || defaultSymptoms}
          </p>
          {affectedViews && affectedViews.length > 0 && (
            <div className="flex items-center gap-1.5 pt-1">
              <span className="text-[11px] font-semibold text-gray-500">Confirmed on viewpoints:</span>
              {affectedViews.map((v) => (
                <span
                  key={v}
                  className="px-2 py-0.5 rounded text-[10px] font-bold bg-[#E8F1EA] text-[#1B4332] uppercase"
                >
                  {v}
                </span>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Pinpoint Visual Disease Localization & Lesion Heatmap */}
      <div className="p-5 border-b border-[#E3EDE5] bg-gray-50/50">
        <VisualDiseaseLocalization
          visualData={visualLocalization}
          cropType={cropType}
          diseaseName={diagnosis}
        />
      </div>

      {/* Actionable Treatment & Cure Plan */}
      <div className="p-5 space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <h3 className="text-sm font-bold text-gray-900 flex items-center gap-2">
            <Stethoscope className="w-4 h-4 text-[#1B4332]" />
            Actionable Treatment & Management Plan
          </h3>
          {/* Action Tabs */}
          <div className="flex gap-1 bg-[#F1F6F2] p-1 rounded-xl border border-[#DDE7DF]">
            <button
              onClick={() => setActiveTab('cultural')}
              className={`px-3 py-1 rounded-lg text-xs font-bold transition-all ${
                activeTab === 'cultural'
                  ? 'bg-[#1B4332] text-white shadow-xs'
                  : 'text-gray-600 hover:text-gray-900'
              }`}
            >
              1. Cultural Sanitation
            </button>
            <button
              onClick={() => setActiveTab('organic')}
              className={`px-3 py-1 rounded-lg text-xs font-bold transition-all ${
                activeTab === 'organic'
                  ? 'bg-[#1B4332] text-white shadow-xs'
                  : 'text-gray-600 hover:text-gray-900'
              }`}
            >
              2. Organic Bio-Controls
            </button>
            <button
              onClick={() => setActiveTab('chemical')}
              className={`px-3 py-1 rounded-lg text-xs font-bold transition-all ${
                activeTab === 'chemical'
                  ? 'bg-[#1B4332] text-white shadow-xs'
                  : 'text-gray-600 hover:text-gray-900'
              }`}
            >
              3. Targeted Chemical
            </button>
            <button
              onClick={() => setActiveTab('prevention')}
              className={`px-3 py-1 rounded-lg text-xs font-bold transition-all ${
                activeTab === 'prevention'
                  ? 'bg-[#1B4332] text-white shadow-xs'
                  : 'text-gray-600 hover:text-gray-900'
              }`}
            >
              4. Prevention
            </button>
          </div>
        </div>

        {/* Tab Content */}
        <div className="bg-[#FAFBF9] border border-[#E3EDE5] rounded-xl p-4 text-xs text-gray-800 leading-relaxed min-h-[110px]">
          {activeTab === 'cultural' && (
            <div className="space-y-2">
              <p className="font-semibold text-[#1B4332]">Immediate Sanitation & Physical Controls:</p>
              <ul className="space-y-1.5 list-disc list-inside">
                {treatmentCultural.length > 0 ? (
                  treatmentCultural.map((item, i) => <li key={i}>{item}</li>)
                ) : (
                  <>
                    <li>Immediately prune lower infected foliage showing necrotic spots and bag before disposal.</li>
                    <li>Switch strictly to drip or basal soaker hose irrigation to prevent splashing fungal conidia.</li>
                    <li>Sterilize pruning shears between cuts using a 70% isopropyl alcohol wipe.</li>
                  </>
                )}
              </ul>
            </div>
          )}

          {activeTab === 'organic' && (
            <div className="space-y-2">
              <p className="font-semibold text-emerald-800">Certified Organic & Biological Treatments:</p>
              <ul className="space-y-1.5 list-disc list-inside">
                {treatmentOrganic.length > 0 ? (
                  treatmentOrganic.map((item, i) => <li key={i}>{item}</li>)
                ) : (
                  <>
                    <li>Apply copper octanoate (copper soap fungicide) at 7-10 day intervals.</li>
                    <li>Foliar drench with Bacillus amyloliquefaciens or Bacillus subtilis bio-fungicide.</li>
                    <li>Apply cold-pressed neem oil (1% solution) to deter vector feeding on leaf undersides.</li>
                  </>
                )}
              </ul>
            </div>
          )}

          {activeTab === 'chemical' && (
            <div className="space-y-2">
              <p className="font-semibold text-blue-900">Registered Fungicides & Active Ingredients (FRAC Guidelines):</p>
              <ul className="space-y-1.5 list-disc list-inside">
                {treatmentChemical.length > 0 ? (
                  treatmentChemical.map((item, i) => <li key={i}>{item}</li>)
                ) : (
                  <>
                    <li>Chlorothalonil (FRAC M05) applied preventatively prior to anticipated rain events.</li>
                    <li>Mancozeb (FRAC M03) protectant spray with minimum 5-day pre-harvest interval (PHI).</li>
                    <li>Rotate FRAC groups to prevent pathogen resistance development.</li>
                  </>
                )}
              </ul>
            </div>
          )}

          {activeTab === 'prevention' && (
            <div className="space-y-2">
              <p className="font-semibold text-[#2D5A3C]">Long-Term Crop Protection & Resistance:</p>
              <p>
                {preventionText ||
                  'Enforce a 3-year crop rotation with non-solanaceous species (corn, brassicas). Maintain 2-3 inches of clean organic straw mulch under tomato rows to prevent soil-splash pathogens. Select disease-resistant hybrid seed cultivars.'}
              </p>
            </div>
          )}
        </div>

        {/* Model Provenance & Cross-Model Evidence Accordion */}
        <div className="pt-2 border-t border-[#EAEFEA]">
          <button
            onClick={() => setShowModelDetails(!showModelDetails)}
            className="w-full flex items-center justify-between text-xs font-semibold text-gray-600 hover:text-[#1B4332] py-1 transition-colors"
          >
            <span className="flex items-center gap-1.5">
              <Layers className="w-3.5 h-3.5 text-emerald-700" />
              Multi-Model Ensemble Technical Evidence & SHA-256 Provenance
            </span>
            {showModelDetails ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
          </button>

          {showModelDetails && (
            <div className="mt-3 grid grid-cols-1 sm:grid-cols-3 gap-3 p-3 bg-gray-50 rounded-xl border border-gray-200 text-[11px]">
              {/* Model 1 */}
              <div className="bg-white p-3 rounded-lg border border-gray-200 space-y-1">
                <span className="font-bold text-[#1B4332] block">1. YOLOv8n Vision</span>
                <span className="text-gray-500 block text-[10px]">Lesion & Pathogen Detector</span>
                <div className="font-mono text-[9px] text-gray-400 truncate">
                  SHA: {modelProvenance?.yolo_detector?.sha256 || 'ccc05ab88bb5...'}
                </div>
                <div className="text-gray-700 font-semibold pt-1">
                  Confidence: {modelProvenance?.yolo_detector?.confidence || (confidence * 100).toFixed(1)}%
                </div>
              </div>

              {/* Model 2 */}
              <div className="bg-white p-3 rounded-lg border border-gray-200 space-y-1">
                <span className="font-bold text-[#1B4332] block">2. LeafSeverityNet</span>
                <span className="text-gray-500 block text-[10px]">Deep Severity Neural Net</span>
                <div className="font-mono text-[9px] text-gray-400 truncate">
                  SHA: {modelProvenance?.leaf_severity_net?.sha256 || '38f9fd17ddf0...'}
                </div>
                <div className="text-gray-700 font-semibold pt-1">
                  Lesion Area: ~{lesionPercentage.toFixed(1)}%
                </div>
              </div>

              {/* Model 3 */}
              <div className="bg-white p-3 rounded-lg border border-gray-200 space-y-1">
                <span className="font-bold text-[#1B4332] block">3. PestVectorNet</span>
                <span className="text-gray-500 block text-[10px]">Insect Vector Surveillance</span>
                <div className="font-mono text-[9px] text-gray-400 truncate">
                  SHA: {modelProvenance?.pest_vector_net?.sha256 || '7529ed10d788...'}
                </div>
                <div className="text-gray-700 font-semibold pt-1">
                  Vector: {modelProvenance?.pest_vector_net?.predicted_vector || 'NONE DETECTED'}
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
