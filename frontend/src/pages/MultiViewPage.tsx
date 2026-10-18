import React, { useState, useEffect } from 'react';
import {
  Video,
  Upload,
  Play,
  CheckCircle,
  AlertTriangle,
  Info,
  ShieldAlert,
  Layers,
  Eye,
  FileDown,
  RefreshCw,
  PlusCircle,
  Sparkles,
  Camera,
  Activity,
  CheckCircle2,
  XCircle,
  Clock,
} from 'lucide-react';
import {
  PlantSession,
  ClipMeta,
  PlantAssessment,
  ViewpointLabel,
  ModelReadinessStatus,
  FrameObservation,
} from '../types';

export const MultiViewPage: React.FC = () => {
  // Session State
  const [sessions, setSessions] = useState<PlantSession[]>([]);
  const [currentSession, setCurrentSession] = useState<PlantSession | null>(null);
  const [newPlantId, setNewPlantId] = useState<string>('PLANT-001');
  const [cropType, setCropType] = useState<string>('tomato');
  const [sessionNotes, setSessionNotes] = useState<string>('');

  // Upload State
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [selectedView, setSelectedView] = useState<ViewpointLabel>('front');
  const [uploading, setUploading] = useState<boolean>(false);
  const [uploadError, setUploadError] = useState<string | null>(null);

  // Analysis / Job State
  const [analyzing, setAnalyzing] = useState<boolean>(false);
  const [analysisProgress, setAnalysisProgress] = useState<number>(0);
  const [analysisStage, setAnalysisStage] = useState<string>('');
  const [analysisMessage, setAnalysisMessage] = useState<string>('');

  // Observations State
  const [observations, setObservations] = useState<FrameObservation[]>([]);
  const [showObservations, setShowObservations] = useState<boolean>(false);

  // Model Readiness
  const [modelStatus, setModelStatus] = useState<ModelReadinessStatus | null>(null);

  // Fetch model readiness on mount
  useEffect(() => {
    fetchModelStatus();
    fetchSessions();
  }, []);

  const fetchModelStatus = async () => {
    try {
      const res = await fetch('/api/models/status');
      if (res.ok) {
        const data = await res.json();
        setModelStatus(data);
      }
    } catch (err) {
      console.warn('Failed to fetch model status:', err);
    }
  };

  const fetchSessions = async () => {
    try {
      const res = await fetch('/api/plant-sessions');
      if (res.ok) {
        const data: PlantSession[] = await res.json();
        setSessions(data);
        if (data.length > 0 && !currentSession) {
          setCurrentSession(data[0]);
        }
      }
    } catch (err) {
      console.warn('Failed to fetch sessions:', err);
    }
  };

  const handleCreateSession = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!newPlantId.trim()) return;

    try {
      const res = await fetch('/api/plant-sessions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          plant_id: newPlantId.trim(),
          crop_type: cropType.trim(),
          notes: sessionNotes.trim() || undefined,
        }),
      });

      if (res.ok) {
        const created: PlantSession = await res.json();
        setSessions((prev) => [created, ...prev]);
        setCurrentSession(created);
        setSessionNotes('');
      }
    } catch (err) {
      console.error('Failed to create session:', err);
    }
  };

  const handleUploadClip = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!currentSession || !selectedFile) return;

    setUploading(true);
    setUploadError(null);

    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('view_label', selectedView);

    try {
      const res = await fetch(`/api/plant-sessions/${currentSession.session_id}/videos`, {
        method: 'POST',
        body: formData,
      });

      if (res.ok) {
        const result = await res.json();
        // Refresh session
        const sessRes = await fetch(`/api/plant-sessions/${currentSession.session_id}`);
        if (sessRes.ok) {
          const updated: PlantSession = await sessRes.json();
          setCurrentSession(updated);
          setSessions((prev) =>
            prev.map((s) => (s.session_id === updated.session_id ? updated : s))
          );
        }
        setSelectedFile(null);
      } else {
        const errData = await res.json();
        setUploadError(errData.detail || 'Failed to upload video clip.');
      }
    } catch (err: any) {
      setUploadError(err.message || 'Network error during clip upload.');
    } finally {
      setUploading(false);
    }
  };

  const handleRunAnalysis = async () => {
    if (!currentSession || currentSession.clips.length === 0) return;

    setAnalyzing(true);
    setAnalysisProgress(10);
    setAnalysisStage('Starting video processing...');
    setAnalysisMessage('Dispatching frame decoders and singleton vision model...');

    try {
      const startRes = await fetch(`/api/plant-sessions/${currentSession.session_id}/analyze`, {
        method: 'POST',
      });

      if (!startRes.ok) {
        const err = await startRes.json();
        alert(err.detail || 'Failed to start analysis.');
        setAnalyzing(false);
        return;
      }

      // Poll status
      const pollInterval = setInterval(async () => {
        try {
          const stRes = await fetch(`/api/plant-sessions/${currentSession.session_id}/status`);
          if (stRes.ok) {
            const stData = await stRes.json();
            setAnalysisProgress(stData.progress_percent);
            setAnalysisStage(stData.stage);
            setAnalysisMessage(stData.message);

            if (stData.status === 'COMPLETED' || stData.has_assessment) {
              clearInterval(pollInterval);
              setAnalyzing(false);

              // Reload full session
              const sRes = await fetch(`/api/plant-sessions/${currentSession.session_id}`);
              if (sRes.ok) {
                const updatedSession = await sRes.json();
                setCurrentSession(updatedSession);
                setSessions((prev) =>
                  prev.map((s) => (s.session_id === updatedSession.session_id ? updatedSession : s))
                );
              }
            } else if (stData.status === 'FAILED') {
              clearInterval(pollInterval);
              setAnalyzing(false);
              alert(`Analysis failed: ${stData.message}`);
            }
          }
        } catch (pollErr) {
          console.warn('Status poll warning:', pollErr);
        }
      }, 1000);
    } catch (err) {
      console.error('Run analysis error:', err);
      setAnalyzing(false);
    }
  };

  const handleCreateSimulation = async () => {
    setAnalyzing(true);
    setAnalysisProgress(30);
    setAnalysisStage('Generating 4-view synthetic plant session...');
    setAnalysisMessage('Simulating front, back, left, top camera orbits...');

    try {
      const form = new FormData();
      form.append('plant_id', `PLANT-SIM-${Math.floor(100 + Math.random() * 900)}`);
      form.append('crop_type', 'tomato');
      form.append('include_symptoms', 'true');

      const res = await fetch('/api/plant-sessions/simulation', {
        method: 'POST',
        body: form,
      });

      if (res.ok) {
        const simSession: PlantSession = await res.json();
        setAnalysisProgress(100);
        setAnalysisStage('Completed');
        setAnalysisMessage('Deterministic synthetic multi-view session ready.');
        setSessions((prev) => [simSession, ...prev]);
        setCurrentSession(simSession);
      }
    } catch (err) {
      console.error('Simulation generation error:', err);
    } finally {
      setAnalyzing(false);
    }
  };

  const handleLoadObservations = async () => {
    if (!currentSession) return;
    try {
      const res = await fetch(`/api/plant-sessions/${currentSession.session_id}/observations`);
      if (res.ok) {
        const data = await res.json();
        setObservations(data.observations || []);
        setShowObservations(true);
      }
    } catch (err) {
      console.warn('Failed to load observations:', err);
    }
  };

  const handleExportAssessment = () => {
    if (!currentSession?.assessment) return;
    const blob = new Blob([JSON.stringify(currentSession.assessment, null, 2)], {
      type: 'application/json',
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${currentSession.plant_id}_assessment_${currentSession.session_id}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const assessment = currentSession?.assessment;

  const getStatusBadge = (status?: string) => {
    switch (status) {
      case 'HEALTHY_APPEARING':
        return (
          <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-bold bg-emerald-100 text-emerald-800 border border-emerald-300">
            <CheckCircle className="w-3.5 h-3.5" /> Healthy Appearing
          </span>
        );
      case 'SYMPTOMS_DETECTED':
        return (
          <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-bold bg-amber-100 text-amber-900 border border-amber-300">
            <AlertTriangle className="w-3.5 h-3.5" /> Symptoms Detected
          </span>
        );
      case 'UNCERTAIN':
        return (
          <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-bold bg-yellow-100 text-yellow-900 border border-yellow-300">
            <AlertTriangle className="w-3.5 h-3.5" /> Uncertain / Mixed Evidence
          </span>
        );
      case 'INSUFFICIENT_COVERAGE':
        return (
          <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-bold bg-blue-100 text-blue-800 border border-blue-300">
            <Info className="w-3.5 h-3.5" /> Insufficient Coverage
          </span>
        );
      case 'MODEL_UNAVAILABLE':
        return (
          <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-bold bg-gray-200 text-gray-800 border border-gray-400">
            <ShieldAlert className="w-3.5 h-3.5" /> Model Weights Missing
          </span>
        );
      default:
        return (
          <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-bold bg-gray-100 text-gray-700 border border-gray-300">
            {status || 'Pending Analysis'}
          </span>
        );
    }
  };

  const standardViews = ['front', 'back', 'left', 'right', 'top', 'underside'];

  return (
    <div className="space-y-7 pb-12">
      {/* Page Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2">
            <span className="px-2.5 py-0.5 rounded text-xs font-semibold bg-[#1565C0]/10 text-[#1565C0] border border-[#1565C0]/20">
              New Multi-View Architecture
            </span>
            <span className="text-xs text-gray-500">• Single-Plant Multi-Clip Fusion</span>
          </div>
          <h2 className="text-2xl sm:text-3xl font-bold text-[#0D47A1] mt-1">
            Multi-View Plant Health Analysis
          </h2>
          <p className="mt-1 text-sm text-gray-500">
            Ingest multiple video viewpoints of an individual plant, stream-sample frames,
            fuse dual-evidence observations across views, and inspect plant coverage.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <button
            onClick={handleCreateSimulation}
            disabled={analyzing}
            className="bg-indigo-600 hover:bg-indigo-700 text-white px-3.5 py-2 rounded-lg text-sm font-semibold transition-colors shadow-sm flex items-center gap-1.5 disabled:opacity-50"
            title="Generate deterministic 4-view synthetic session without uploading videos"
          >
            <Sparkles className="w-4 h-4" />
            Quick Demo (4-View Sim)
          </button>
        </div>
      </div>

      {/* Model Readiness Indicator */}
      {modelStatus && (
        <div
          className={`p-4 rounded-xl border flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 text-sm ${
            modelStatus.status === 'ready'
              ? 'bg-emerald-50 border-emerald-200 text-emerald-900'
              : 'bg-amber-50 border-amber-200 text-amber-900'
          }`}
        >
          <div className="flex items-center gap-2.5">
            <Activity className="w-5 h-5 flex-shrink-0" />
            <div>
              <span className="font-semibold">Model Status: </span>
              <span className="capitalize">{modelStatus.status.replace('_', ' ')}</span> —{' '}
              <span className="text-xs text-gray-600">{modelStatus.message}</span>
            </div>
          </div>
          <span className="text-xs font-mono bg-white/70 px-2.5 py-1 rounded border border-black/10">
            Task: YOLOv8-Cls (Tomato: 4 classes)
          </span>
        </div>
      )}

      {/* Top Grid: Sessions Bar & Create Session */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Active Session Picker */}
        <div className="lg:col-span-2 bg-white rounded-xl border border-gray-200 shadow-sm p-5 space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="font-bold text-gray-900 flex items-center gap-2">
              <Camera className="w-4 h-4 text-[#1565C0]" /> Active Plant Session
            </h3>
            <span className="text-xs text-gray-500 font-mono">
              {sessions.length} session(s) recorded
            </span>
          </div>

          <div className="flex flex-wrap gap-2">
            {sessions.map((s) => (
              <button
                key={s.session_id}
                onClick={() => setCurrentSession(s)}
                className={`px-3 py-2 rounded-lg text-xs font-medium border transition-all text-left flex items-center gap-2 ${
                  currentSession?.session_id === s.session_id
                    ? 'bg-[#1565C0] text-white border-[#1565C0] shadow-sm'
                    : 'bg-gray-50 text-gray-700 border-gray-200 hover:bg-gray-100'
                }`}
              >
                <span className="font-bold">{s.plant_id}</span>
                <span className="opacity-75">({s.clips?.length || 0} clips)</span>
              </button>
            ))}
            {sessions.length === 0 && (
              <p className="text-sm text-gray-400 italic">
                No sessions yet. Create one or click "Quick Demo" above.
              </p>
            )}
          </div>

          {currentSession && (
            <div className="p-3 bg-gray-50 rounded-lg border border-gray-200 flex flex-wrap items-center justify-between gap-3 text-xs text-gray-600">
              <div>
                <span className="font-semibold text-gray-800">Session ID:</span>{' '}
                <span className="font-mono">{currentSession.session_id}</span>
              </div>
              <div>
                <span className="font-semibold text-gray-800">Crop:</span>{' '}
                <span className="capitalize">{currentSession.crop_type}</span>
              </div>
              <div>
                <span className="font-semibold text-gray-800">Status:</span>{' '}
                {getStatusBadge(currentSession.assessment?.status || currentSession.status)}
              </div>
            </div>
          )}
        </div>

        {/* Create New Session Card */}
        <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-5">
          <h3 className="font-bold text-gray-900 flex items-center gap-2 mb-3">
            <PlusCircle className="w-4 h-4 text-[#1565C0]" /> New Plant Session
          </h3>
          <form onSubmit={handleCreateSession} className="space-y-3">
            <div>
              <label className="block text-xs font-medium text-gray-700 mb-1">
                Plant Identifier
              </label>
              <input
                type="text"
                value={newPlantId}
                onChange={(e) => setNewPlantId(e.target.value)}
                placeholder="e.g. PLANT-002"
                className="w-full text-sm px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-[#1565C0]"
                required
              />
            </div>
            <div>
              <label className="block text-xs font-medium text-gray-700 mb-1">Crop Type</label>
              <select
                value={cropType}
                onChange={(e) => setCropType(e.target.value)}
                className="w-full text-sm px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-[#1565C0]"
              >
                <option value="tomato">Tomato (Solanum lycopersicum)</option>
                <option value="bell_pepper">Bell Pepper</option>
                <option value="potato">Potato</option>
                <option value="custom">General / Custom Crop</option>
              </select>
            </div>
            <button
              type="submit"
              className="w-full bg-[#1565C0] hover:bg-[#0D47A1] text-white py-2 rounded-lg text-xs font-semibold transition-colors"
            >
              Create Session
            </button>
          </form>
        </div>
      </div>

      {/* Main Analysis Stage: Upload Video & Clip List */}
      {currentSession && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Upload Clip Panel */}
          <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-5 space-y-4">
            <h3 className="font-bold text-gray-900 flex items-center gap-2">
              <Upload className="w-4 h-4 text-[#1565C0]" /> Add Viewpoint Video Clip
            </h3>
            <p className="text-xs text-gray-500">
              Upload video clips filmed from distinct angles around {currentSession.plant_id}.
            </p>

            <form onSubmit={handleUploadClip} className="space-y-4">
              <div>
                <label className="block text-xs font-medium text-gray-700 mb-1">
                  Viewpoint Angle
                </label>
                <select
                  value={selectedView}
                  onChange={(e) => setSelectedView(e.target.value as ViewpointLabel)}
                  className="w-full text-sm px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-[#1565C0]"
                >
                  <option value="front">Front View (Face)</option>
                  <option value="back">Back View (Opposite)</option>
                  <option value="left">Left Flank View</option>
                  <option value="right">Right Flank View</option>
                  <option value="top">Top Canopy View</option>
                  <option value="underside">Underside / Low Angle</option>
                  <option value="arbitrary">Arbitrary / Orbit Angle</option>
                  <option value="unknown">Unknown</option>
                </select>
              </div>

              <div>
                <label className="block text-xs font-medium text-gray-700 mb-1">Video File</label>
                <input
                  type="file"
                  accept="video/mp4,video/avi,video/quicktime,video/webm,video/x-matroska"
                  onChange={(e) => setSelectedFile(e.target.files?.[0] || null)}
                  className="w-full text-xs text-gray-600 file:mr-3 file:py-2 file:px-3 file:rounded-lg file:border-0 file:text-xs file:font-semibold file:bg-[#1565C0]/10 file:text-[#1565C0] hover:file:bg-[#1565C0]/20"
                />
              </div>

              {uploadError && (
                <div className="p-2.5 rounded bg-red-50 border border-red-200 text-red-700 text-xs">
                  {uploadError}
                </div>
              )}

              <button
                type="submit"
                disabled={!selectedFile || uploading}
                className="w-full bg-[#1565C0] hover:bg-[#0D47A1] text-white py-2.5 rounded-lg text-xs font-semibold transition-colors flex items-center justify-center gap-2 disabled:opacity-50"
              >
                {uploading ? (
                  <>
                    <RefreshCw className="w-3.5 h-3.5 animate-spin" /> Uploading...
                  </>
                ) : (
                  <>
                    <Upload className="w-3.5 h-3.5" /> Upload Clip ({selectedView})
                  </>
                )}
              </button>
            </form>
          </div>

          {/* Clips List & Start Analysis Action */}
          <div className="lg:col-span-2 bg-white rounded-xl border border-gray-200 shadow-sm p-5 space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="font-bold text-gray-900 flex items-center gap-2">
                <Video className="w-4 h-4 text-[#1565C0]" /> Clips in Session ({currentSession.clips?.length || 0})
              </h3>
              <button
                onClick={handleRunAnalysis}
                disabled={analyzing || !currentSession.clips || currentSession.clips.length === 0}
                className="bg-emerald-600 hover:bg-emerald-700 text-white px-4 py-2 rounded-lg text-xs font-bold transition-colors shadow-sm flex items-center gap-2 disabled:opacity-50"
              >
                {analyzing ? (
                  <>
                    <RefreshCw className="w-3.5 h-3.5 animate-spin" /> Processing...
                  </>
                ) : (
                  <>
                    <Play className="w-3.5 h-3.5" /> Start Multi-View Analysis
                  </>
                )}
              </button>
            </div>

            {/* Clips Table */}
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs border-collapse">
                <thead>
                  <tr className="bg-gray-50 border-b border-gray-200 text-gray-600 font-semibold">
                    <th className="py-2.5 px-3">View Label</th>
                    <th className="py-2.5 px-3">Filename</th>
                    <th className="py-2.5 px-3">Status</th>
                    <th className="py-2.5 px-3 text-center">Frames Sampled</th>
                    <th className="py-2.5 px-3 text-center">Duplicates Skipped</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-100 text-gray-700">
                  {currentSession.clips?.map((c) => (
                    <tr key={c.clip_id} className="hover:bg-gray-50/70">
                      <td className="py-2.5 px-3">
                        <span className="font-bold capitalize bg-gray-100 px-2 py-0.5 rounded border border-gray-200 text-gray-800">
                          {c.view_label}
                        </span>
                      </td>
                      <td className="py-2.5 px-3 font-mono text-gray-600 max-w-xs truncate">
                        {c.original_filename}
                      </td>
                      <td className="py-2.5 px-3">
                        <span
                          className={`px-2 py-0.5 rounded text-[10px] font-semibold ${
                            c.processing_status === 'COMPLETED'
                              ? 'bg-emerald-100 text-emerald-800'
                              : c.processing_status === 'ERROR'
                              ? 'bg-red-100 text-red-800'
                              : 'bg-amber-100 text-amber-800'
                          }`}
                        >
                          {c.processing_status}
                        </span>
                      </td>
                      <td className="py-2.5 px-3 text-center font-semibold">
                        {c.frames_extracted || '—'}
                      </td>
                      <td className="py-2.5 px-3 text-center text-gray-500">
                        {c.frames_skipped_duplicate || 0}
                      </td>
                    </tr>
                  ))}
                  {(!currentSession.clips || currentSession.clips.length === 0) && (
                    <tr>
                      <td colSpan={5} className="py-6 text-center text-gray-400 italic">
                        No video clips uploaded yet. Upload a clip or run Quick Demo.
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>

            {/* Analysis Progress Bar */}
            {analyzing && (
              <div className="p-4 bg-blue-50 border border-blue-200 rounded-xl space-y-2">
                <div className="flex items-center justify-between text-xs font-semibold text-blue-900">
                  <span>{analysisStage}</span>
                  <span>{analysisProgress}%</span>
                </div>
                <div className="w-full bg-blue-200 h-2 rounded-full overflow-hidden">
                  <div
                    className="bg-[#1565C0] h-full transition-all duration-300"
                    style={{ width: `${analysisProgress}%` }}
                  />
                </div>
                <p className="text-[11px] text-blue-700">{analysisMessage}</p>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Assessment Results Card */}
      {assessment && (
        <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-6 space-y-6">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-gray-100 pb-4">
            <div>
              <div className="flex items-center gap-3">
                <h3 className="text-xl font-bold text-gray-900">
                  Assessment for {currentSession?.plant_id}
                </h3>
                {getStatusBadge(assessment.status)}
              </div>
              <p className="text-xs text-gray-500 mt-1">
                Fused multi-view dual-evidence assessment from {assessment.total_clips} video clip(s)
              </p>
            </div>

            <div className="flex items-center gap-2">
              <button
                onClick={handleLoadObservations}
                className="bg-gray-100 hover:bg-gray-200 text-gray-700 px-3 py-1.5 rounded-lg text-xs font-semibold transition-colors flex items-center gap-1.5"
              >
                <Eye className="w-3.5 h-3.5" />
                {showObservations ? 'Hide Frame Observations' : 'Inspect Frame Observations'}
              </button>
              <button
                onClick={handleExportAssessment}
                className="bg-[#1565C0] hover:bg-[#0D47A1] text-white px-3 py-1.5 rounded-lg text-xs font-semibold transition-colors flex items-center gap-1.5 shadow-sm"
              >
                <FileDown className="w-3.5 h-3.5" /> Export JSON
              </button>
            </div>
          </div>

          {/* Key Metrics Grid */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="p-4 bg-gray-50 rounded-xl border border-gray-200">
              <div className="text-xs text-gray-500 font-medium">Usable Frames Processed</div>
              <div className="text-2xl font-bold text-gray-900 mt-1">
                {assessment.total_frames_processed}
              </div>
              <div className="text-[11px] text-gray-500 mt-0.5">Stream sampled @ 2 fps</div>
            </div>

            <div className="p-4 bg-gray-50 rounded-xl border border-gray-200">
              <div className="text-xs text-gray-500 font-medium">Top Predicted Symptom</div>
              <div className="text-xl font-bold text-[#1565C0] mt-1 capitalize truncate">
                {assessment.top_disease_class ? assessment.top_disease_class.replace('_', ' ') : 'None'}
              </div>
              <div className="text-[11px] text-gray-500 mt-0.5">
                {assessment.disease_fraction !== undefined && assessment.disease_fraction !== null
                  ? `${Math.round(assessment.disease_fraction * 100)}% of frames`
                  : 'No symptoms'}
              </div>
            </div>

            <div className="p-4 bg-gray-50 rounded-xl border border-gray-200">
              <div className="text-xs text-gray-500 font-medium">Visible Symptom Area (Heuristic)</div>
              <div className="text-2xl font-bold text-gray-900 mt-1">
                {assessment.visible_symptom_area_percent !== null &&
                assessment.visible_symptom_area_percent !== undefined
                  ? `${assessment.visible_symptom_area_percent}%`
                  : 'N/A'}
              </div>
              <div className="text-[11px] text-gray-500 mt-0.5">OpenCV color threshold</div>
            </div>

            <div className="p-4 bg-gray-50 rounded-xl border border-gray-200">
              <div className="text-xs text-gray-500 font-medium">View Coverage</div>
              <div className="text-2xl font-bold text-gray-900 mt-1">
                {assessment.submitted_views?.length || 0} / 6
              </div>
              <div className="text-[11px] text-gray-500 mt-0.5">Standard viewpoints</div>
            </div>
          </div>

          {/* Viewpoint Coverage Matrix */}
          <div className="space-y-3">
            <h4 className="text-sm font-bold text-gray-900 flex items-center gap-2">
              <Layers className="w-4 h-4 text-[#1565C0]" /> Viewpoint Coverage Matrix
            </h4>
            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-6 gap-2.5">
              {standardViews.map((view) => {
                const isSubmitted = assessment.submitted_views?.includes(view);
                return (
                  <div
                    key={view}
                    className={`p-3 rounded-lg border text-center ${
                      isSubmitted
                        ? 'bg-emerald-50 border-emerald-300 text-emerald-900'
                        : 'bg-gray-50 border-gray-200 text-gray-400'
                    }`}
                  >
                    <div className="flex justify-center mb-1">
                      {isSubmitted ? (
                        <CheckCircle2 className="w-4 h-4 text-emerald-600" />
                      ) : (
                        <XCircle className="w-4 h-4 text-gray-300" />
                      )}
                    </div>
                    <div className="text-xs font-bold capitalize">{view}</div>
                    <div className="text-[10px] opacity-75">
                      {isSubmitted ? 'Submitted' : 'Unobserved'}
                    </div>
                  </div>
                );
              })}
            </div>
            <p className="text-[11px] text-gray-500 italic">
              {assessment.coverage_note}
            </p>
          </div>

          {/* Disease Evidence Table */}
          {assessment.disease_evidence && assessment.disease_evidence.length > 0 && (
            <div className="space-y-3">
              <h4 className="text-sm font-bold text-gray-900">
                Disease & Symptom Evidence (Model Predictions)
              </h4>
              <div className="overflow-x-auto border border-gray-200 rounded-lg">
                <table className="w-full text-left text-xs border-collapse">
                  <thead>
                    <tr className="bg-gray-50 border-b border-gray-200 text-gray-600 font-semibold">
                      <th className="py-2.5 px-3">Predicted Class</th>
                      <th className="py-2.5 px-3 text-center">Frames Observed</th>
                      <th className="py-2.5 px-3 text-center">Fraction of Frames</th>
                      <th className="py-2.5 px-3 text-center">Mean Model Confidence</th>
                      <th className="py-2.5 px-3">Interpretation Note</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-100 text-gray-700">
                    {assessment.disease_evidence.map((item, idx) => (
                      <tr key={idx} className="hover:bg-gray-50/70">
                        <td className="py-2.5 px-3 font-bold capitalize text-amber-900">
                          {item.predicted_class.replace('_', ' ')}
                        </td>
                        <td className="py-2.5 px-3 text-center font-semibold">{item.frame_count}</td>
                        <td className="py-2.5 px-3 text-center">
                          {Math.round(item.fraction_of_total * 100)}%
                        </td>
                        <td className="py-2.5 px-3 text-center font-mono font-semibold">
                          {(item.average_confidence * 100).toFixed(1)}%
                        </td>
                        <td className="py-2.5 px-3 text-gray-500 text-[11px]">{item.note}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* Clip Summaries */}
          {assessment.clips_summary && assessment.clips_summary.length > 0 && (
            <div className="space-y-3">
              <h4 className="text-sm font-bold text-gray-900">Clip-by-Clip Observations</h4>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-3">
                {assessment.clips_summary.map((clip) => (
                  <div key={clip.clip_id} className="p-3 bg-gray-50 rounded-lg border border-gray-200 text-xs space-y-1">
                    <div className="font-bold text-gray-800 capitalize flex items-center justify-between">
                      <span>{clip.view_label} View</span>
                      <span className="text-[10px] font-mono text-gray-400">{clip.clip_id}</span>
                    </div>
                    <div className="text-gray-600">
                      Total frames: <span className="font-semibold">{clip.total_frames}</span>
                    </div>
                    <div className="text-gray-600">
                      Disease frames: <span className="font-semibold text-amber-800">{clip.disease_frames}</span>
                    </div>
                    <div className="text-gray-600">
                      DIP abnormal: <span className="font-semibold">{clip.opencv_abnormal_frames}</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Limitations and Disclaimers */}
          <div className="p-4 bg-amber-50/80 border border-amber-200 rounded-xl space-y-2">
            <h4 className="text-xs font-bold text-amber-900 flex items-center gap-1.5">
              <ShieldAlert className="w-4 h-4 text-amber-700" /> Scientific & Biosecurity Limitations
            </h4>
            <ul className="text-[11px] text-amber-800 space-y-1 list-disc pl-4">
              {assessment.limitations?.map((lim, idx) => (
                <li key={idx}>{lim}</li>
              ))}
              {assessment.assessment_notes?.map((note, idx) => (
                <li key={`note-${idx}`}>{note}</li>
              ))}
            </ul>
          </div>
        </div>
      )}

      {/* Frame Observations Modal / Drawer */}
      {showObservations && (
        <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-6 space-y-4">
          <div className="flex items-center justify-between border-b pb-3">
            <h3 className="font-bold text-gray-900 flex items-center gap-2">
              <Eye className="w-4 h-4 text-[#1565C0]" /> Raw Frame Observations ({observations.length})
            </h3>
            <button
              onClick={() => setShowObservations(false)}
              className="text-xs text-gray-500 hover:text-gray-800 underline"
            >
              Close
            </button>
          </div>

          <div className="overflow-x-auto max-h-96">
            <table className="w-full text-left text-xs border-collapse">
              <thead>
                <tr className="bg-gray-50 border-b border-gray-200 text-gray-600 font-semibold sticky top-0">
                  <th className="py-2 px-2.5">Clip ID</th>
                  <th className="py-2 px-2.5">View</th>
                  <th className="py-2 px-2.5">Time (s)</th>
                  <th className="py-2 px-2.5">YOLO Prediction</th>
                  <th className="py-2 px-2.5">YOLO Conf</th>
                  <th className="py-2 px-2.5">OpenCV Status</th>
                  <th className="py-2 px-2.5">Lesion %</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-100 text-gray-700">
                {observations.map((obs, idx) => (
                  <tr key={idx} className="hover:bg-gray-50/70 font-mono text-[11px]">
                    <td className="py-1.5 px-2.5">{obs.clip_id}</td>
                    <td className="py-1.5 px-2.5 capitalize">{obs.view_label}</td>
                    <td className="py-1.5 px-2.5">{obs.timestamp_sec}s</td>
                    <td className="py-1.5 px-2.5 font-sans font-bold capitalize text-amber-900">
                      {obs.yolo_prediction}
                    </td>
                    <td className="py-1.5 px-2.5">
                      {obs.yolo_confidence !== undefined ? `${(obs.yolo_confidence * 100).toFixed(1)}%` : '—'}
                    </td>
                    <td className="py-1.5 px-2.5 font-sans">
                      <span
                        className={`px-1.5 py-0.5 rounded text-[10px] ${
                          obs.opencv_visual_status === 'Healthy' || obs.opencv_visual_status === 'normal'
                            ? 'bg-emerald-100 text-emerald-800'
                            : 'bg-amber-100 text-amber-800'
                        }`}
                      >
                        {obs.opencv_visual_status}
                      </span>
                    </td>
                    <td className="py-1.5 px-2.5">{obs.lesion_percentage}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
};
