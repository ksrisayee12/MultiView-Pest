import React, { useState, useEffect, useMemo } from 'react';
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
  Leaf,
  Trash2,
  Stethoscope,
} from 'lucide-react';
import {
  PlantSession,
  ClipMeta,
  PlantAssessment,
  ViewpointLabel,
  ModelReadinessStatus,
  FrameObservation,
} from '../types';
import { DiagnosticExplainerCard } from '../components/DiagnosticExplainerCard';
import { ClinicalDiseaseChat } from '../components/ClinicalDiseaseChat';

export const MultiViewPage: React.FC = () => {
  // Session State
  const [sessions, setSessions] = useState<PlantSession[]>([]);
  const [currentSession, setCurrentSession] = useState<PlantSession | null>(null);
  const [newPlantId, setNewPlantId] = useState<string>('PLANT-001');
  const [cropType, setCropType] = useState<string>('tomato');
  const [sessionNotes, setSessionNotes] = useState<string>('');
  const [showSessionManager, setShowSessionManager] = useState<boolean>(false);
  const [selectedSessionIds, setSelectedSessionIds] = useState<string[]>([]);

  // Duplicate sessions computation
  const duplicateInfo = useMemo(() => {
    const counts: Record<string, number> = {};
    sessions.forEach((s) => {
      const pid = (s.plant_id || '').trim().toUpperCase();
      counts[pid] = (counts[pid] || 0) + 1;
    });
    const dupPlantIds = Object.keys(counts).filter((id) => counts[id] > 1);
    let totalDuplicates = 0;
    dupPlantIds.forEach((id) => {
      totalDuplicates += counts[id] - 1;
    });
    return { dupPlantIds, totalDuplicates, counts };
  }, [sessions]);

  // Upload State
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [selectedView, setSelectedView] = useState<ViewpointLabel>('front');
  const [uploading, setUploading] = useState<boolean>(false);
  const [uploadError, setUploadError] = useState<string | null>(null);

  // Workflow Mode & Bulk Media Queue
  const [workflowMode, setWorkflowMode] = useState<'single' | 'batch'>('single');
  const [mediaQueue, setMediaQueue] = useState<Array<{
    id: string;
    file: File;
    viewLabel: ViewpointLabel;
    plantId: string;
    previewUrl?: string;
  }>>([]);
  const [batchNotice, setBatchNotice] = useState<string | null>(null);

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
        const diseaseModel = data.models?.disease_classifier;
        const rawStatus = diseaseModel?.status || data.overall_status || 'missing_weights';
        setModelStatus({
          status: String(rawStatus).toLowerCase() as any,
          message: diseaseModel?.meta?.reason || diseaseModel?.description || 'Model readiness evaluated',
          model_path: diseaseModel?.weights_path,
          classes: diseaseModel?.intended_classes,
        });
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

  const handleDeleteSession = async (sessionId: string) => {
    if (
      !window.confirm(
        `Are you sure you want to delete session '${sessionId}'? This will permanently remove its media files and assessment data.`
      )
    ) {
      return;
    }

    try {
      const res = await fetch(`/api/plant-sessions/${sessionId}`, {
        method: 'DELETE',
      });

      if (res.ok) {
        setSessions((prev) => {
          const next = prev.filter((s) => s.session_id !== sessionId);
          if (currentSession?.session_id === sessionId) {
            setCurrentSession(next.length > 0 ? next[0] : null);
          }
          return next;
        });
        setSelectedSessionIds((prev) => prev.filter((id) => id !== sessionId));
        setBatchNotice(`Session '${sessionId}' deleted successfully.`);
      } else {
        const err = await res.json();
        alert(err.detail || 'Failed to delete session.');
      }
    } catch (err: any) {
      alert(`Network error deleting session: ${err.message}`);
    }
  };

  const handleDeleteDuplicates = async () => {
    if (duplicateInfo.totalDuplicates === 0) {
      alert('No duplicate plant sessions detected.');
      return;
    }
    if (
      !window.confirm(
        `Found ${duplicateInfo.totalDuplicates} duplicate session(s) across plant IDs: ${duplicateInfo.dupPlantIds.join(
          ', '
        )}.\n\nKeep the newest / highest quality session for each plant ID and remove all redundant duplicates?`
      )
    ) {
      return;
    }

    try {
      const res = await fetch('/api/plant-sessions?duplicates_only=true', {
        method: 'DELETE',
      });
      if (res.ok) {
        const data = await res.json();
        setBatchNotice(data.message || 'Removed duplicate sessions.');
        await fetchSessions();
      } else {
        const err = await res.json();
        alert(err.detail || 'Failed to remove duplicates.');
      }
    } catch (err: any) {
      alert(`Error cleaning duplicates: ${err.message}`);
    }
  };

  const handleDeleteAllSessions = async () => {
    if (sessions.length === 0) return;
    if (
      !window.confirm(
        `Are you sure you want to delete ALL ${sessions.length} sessions? All uploaded media and assessments will be permanently removed.`
      )
    ) {
      return;
    }

    try {
      const res = await fetch('/api/plant-sessions?duplicates_only=false', {
        method: 'DELETE',
      });
      if (res.ok) {
        setSessions([]);
        setCurrentSession(null);
        setSelectedSessionIds([]);
        setBatchNotice('All sessions have been deleted successfully.');
      } else {
        const err = await res.json();
        alert(err.detail || 'Failed to delete all sessions.');
      }
    } catch (err: any) {
      alert(`Error during bulk deletion: ${err.message}`);
    }
  };

  const handleDeleteSelectedSessions = async () => {
    if (selectedSessionIds.length === 0) return;
    if (
      !window.confirm(
        `Are you sure you want to delete the ${selectedSessionIds.length} selected session(s)?`
      )
    ) {
      return;
    }

    try {
      for (const sid of selectedSessionIds) {
        await fetch(`/api/plant-sessions/${sid}`, { method: 'DELETE' });
      }
      setBatchNotice(`Deleted ${selectedSessionIds.length} selected session(s).`);
      setSelectedSessionIds([]);
      await fetchSessions();
    } catch (err: any) {
      alert(`Error deleting selected sessions: ${err.message}`);
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

  const loadSampleLeaf = async (filename: string, view: ViewpointLabel) => {
    try {
      setSelectedView(view);
      setUploadError(null);
      const res = await fetch(`/static/samples/${filename}`);
      if (!res.ok) throw new Error('Sample image not found on server');
      const blob = await res.blob();
      const file = new File([blob], filename, { type: 'image/jpeg' });
      setSelectedFile(file);
    } catch (err: any) {
      setUploadError(`Could not load sample leaf: ${err.message}`);
    }
  };

  const handleQueueFiles = (files: FileList | null) => {
    if (!files || files.length === 0) return;
    const viewsList: ViewpointLabel[] = ['front', 'back', 'left', 'right', 'top', 'underside'];
    const newItems = Array.from(files).map((file, idx) => ({
      id: `${Date.now()}_${idx}_${file.name}`,
      file,
      viewLabel: viewsList[idx % viewsList.length],
      plantId: currentSession?.plant_id || newPlantId || 'PLANT-001',
      previewUrl: file.type.startsWith('image/') ? URL.createObjectURL(file) : undefined,
    }));
    setMediaQueue((prev) => [...prev, ...newItems]);
    setUploadError(null);
    setBatchNotice(null);
  };

  const removeFromQueue = (id: string) => {
    setMediaQueue((prev) => prev.filter((item) => item.id !== id));
  };

  const updateQueueItem = (id: string, updates: Partial<{ viewLabel: ViewpointLabel; plantId: string }>) => {
    setMediaQueue((prev) =>
      prev.map((item) => (item.id === id ? { ...item, ...updates } : item))
    );
  };

  const handleUploadBatchQueue = async () => {
    if (mediaQueue.length === 0) return;

    setUploading(true);
    setUploadError(null);
    setBatchNotice(null);

    try {
      if (workflowMode === 'single') {
        if (!currentSession) {
          throw new Error('Please select or create a plant session first.');
        }
        const formData = new FormData();
        const metaMap: Record<string, string> = {};

        mediaQueue.forEach((item) => {
          formData.append('files', item.file, item.file.name);
          metaMap[item.file.name] = item.viewLabel;
        });
        formData.append('metadata', JSON.stringify(metaMap));

        const res = await fetch(`/api/plant-sessions/${currentSession.session_id}/media/batch`, {
          method: 'POST',
          body: formData,
        });

        if (res.ok) {
          const data = await res.json();
          setBatchNotice(data.message);
          setMediaQueue([]);

          const sessRes = await fetch(`/api/plant-sessions/${currentSession.session_id}`);
          if (sessRes.ok) {
            const updated = await sessRes.json();
            setCurrentSession(updated);
            setSessions((prev) => prev.map((s) => s.session_id === updated.session_id ? updated : s));
          }
        } else {
          const errData = await res.json();
          setUploadError(errData.detail || 'Batch media upload failed.');
        }
      } else {
        // Multi-plant batch mode
        const formData = new FormData();
        const manifestMap: Record<string, any> = {};

        mediaQueue.forEach((item) => {
          formData.append('files', item.file, item.file.name);
          manifestMap[item.file.name] = {
            plant_id: item.plantId.trim() || 'PLANT-001',
            view_label: item.viewLabel,
            crop_type: cropType,
          };
        });
        formData.append('batch_manifest', JSON.stringify(manifestMap));

        const res = await fetch('/api/plant-sessions/batch', {
          method: 'POST',
          body: formData,
        });

        if (res.ok) {
          const data = await res.json();
          setBatchNotice(data.message);
          setMediaQueue([]);
          fetchSessions();
        } else {
          const errData = await res.json();
          setUploadError(errData.detail || 'Multi-plant batch creation failed.');
        }
      }
    } catch (err: any) {
      setUploadError(err.message || 'Error executing batch submission.');
    } finally {
      setUploading(false);
    }
  };

  const handleDownloadReport = (format: 'markdown' | 'json') => {
    if (!currentSession) return;
    const url = `/api/plant-sessions/${currentSession.session_id}/export?format=${format}`;
    const link = document.createElement('a');
    link.href = url;
    link.download = `plant_report_${currentSession.plant_id}_${currentSession.session_id}.${format === 'markdown' ? 'md' : 'json'}`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
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
      form.append('crop_type', currentSession?.crop_type || 'tomato');
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
              <span className="capitalize">{typeof modelStatus.status === 'string' ? modelStatus.status.replace(/_/g, ' ') : 'Checked'}</span> —{' '}
              <span className="text-xs text-gray-600">{modelStatus.message || 'Ready'}</span>
            </div>
          </div>
          <span className="text-xs font-mono bg-white/70 px-2.5 py-1 rounded border border-black/10">
            Task: YOLO Disease Classifier
          </span>
        </div>
      )}

      {/* Workflow Mode Tabs */}
      <div className="flex items-center gap-3 border-b border-gray-200 pb-2">
        <button
          onClick={() => setWorkflowMode('single')}
          className={`px-4 py-2 rounded-xl text-xs font-bold transition-all flex items-center gap-2 ${
            workflowMode === 'single'
              ? 'bg-[#1B4332] text-white shadow-xs'
              : 'bg-white text-gray-600 border border-gray-200 hover:bg-gray-50'
          }`}
        >
          <Camera className="w-4 h-4" /> Single Plant Session
        </button>
        <button
          onClick={() => setWorkflowMode('batch')}
          className={`px-4 py-2 rounded-xl text-xs font-bold transition-all flex items-center gap-2 ${
            workflowMode === 'batch'
              ? 'bg-[#1B4332] text-white shadow-xs'
              : 'bg-white text-gray-600 border border-gray-200 hover:bg-gray-50'
          }`}
        >
          <Layers className="w-4 h-4" /> Multi-Plant Batch Mode (Strict Isolation)
        </button>
      </div>

      {batchNotice && (
        <div className="p-3 bg-emerald-50 border border-emerald-200 rounded-xl text-emerald-800 text-xs flex items-center justify-between">
          <span className="font-semibold">{batchNotice}</span>
          <button onClick={() => setBatchNotice(null)} className="text-emerald-600 hover:text-emerald-900 font-bold ml-2">✕</button>
        </div>
      )}

      {/* Duplicate Sessions Notice Banner */}
      {duplicateInfo.totalDuplicates > 0 && (
        <div className="p-3.5 bg-amber-50 border border-amber-300 rounded-xl text-amber-900 text-xs flex flex-wrap items-center justify-between gap-3 shadow-xs">
          <div className="flex items-center gap-2">
            <AlertTriangle className="w-4 h-4 text-amber-600 flex-shrink-0" />
            <span>
              <strong>{duplicateInfo.totalDuplicates} duplicate session(s) detected</strong> across plant IDs:{' '}
              <span className="font-mono font-bold">{duplicateInfo.dupPlantIds.join(', ')}</span>.
            </span>
          </div>
          <button
            onClick={handleDeleteDuplicates}
            className="px-3 py-1.5 bg-amber-600 hover:bg-amber-700 text-white font-bold rounded-lg text-xs transition-colors flex items-center gap-1.5 shadow-xs"
            title="Keep newest/highest quality session for each plant ID and remove redundant duplicates"
          >
            <Sparkles className="w-3.5 h-3.5" /> Clean Up {duplicateInfo.totalDuplicates} Duplicate(s)
          </button>
        </div>
      )}

      {/* Top Grid: Sessions Bar & Create Session */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Active Session Picker */}
        <div className="lg:col-span-2 bg-white rounded-2xl border border-[#DCE6DE] shadow-xs p-5 space-y-4">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <h3 className="font-bold text-gray-900 flex items-center gap-2">
              <Camera className="w-4 h-4 text-[#1B4332]" /> {workflowMode === 'single' ? 'Active Plant Session' : 'Registered Plant Sessions'}
            </h3>
            <div className="flex flex-wrap items-center gap-2">
              <span className="text-xs text-gray-500 font-mono">
                {sessions.length} session(s)
              </span>
              {duplicateInfo.totalDuplicates > 0 && (
                <button
                  onClick={handleDeleteDuplicates}
                  className="text-xs px-2.5 py-1 bg-amber-50 hover:bg-amber-100 text-amber-800 border border-amber-300 rounded-lg flex items-center gap-1 font-bold transition-colors"
                  title="Remove duplicate sessions, keeping the latest/most complete session per plant ID"
                >
                  <Sparkles className="w-3.5 h-3.5 text-amber-600" /> Clean Duplicates
                </button>
              )}
              <button
                onClick={() => setShowSessionManager((prev) => !prev)}
                className={`text-xs px-2.5 py-1 rounded-lg flex items-center gap-1 font-semibold transition-colors border ${
                  showSessionManager
                    ? 'bg-[#EBF4ED] text-[#1B4332] border-[#C4DBC7]'
                    : 'bg-gray-50 hover:bg-gray-100 text-gray-700 border-gray-300'
                }`}
                title="View session inventory table and bulk delete options"
              >
                <Layers className="w-3.5 h-3.5" /> {showSessionManager ? 'Hide Manager' : 'Manage All'}
              </button>
              {sessions.length > 0 && (
                <button
                  onClick={handleDeleteAllSessions}
                  className="text-xs px-2.5 py-1 bg-red-50 hover:bg-red-100 text-red-600 border border-red-200 rounded-lg flex items-center gap-1 font-semibold transition-colors"
                  title="Delete all recorded sessions permanently"
                >
                  <Trash2 className="w-3.5 h-3.5" /> Clear All
                </button>
              )}
            </div>
          </div>

          {/* Expandable Session Inventory Table */}
          {showSessionManager && sessions.length > 0 && (
            <div className="p-3.5 bg-gray-50 border border-gray-200 rounded-xl space-y-3">
              <div className="flex flex-wrap items-center justify-between gap-2 border-b border-gray-200 pb-2.5">
                <span className="font-bold text-gray-800 text-xs flex items-center gap-2">
                  Session Inventory ({sessions.length})
                  {duplicateInfo.totalDuplicates > 0 && (
                    <span className="px-2 py-0.5 rounded-full text-[10px] font-bold bg-amber-100 text-amber-800">
                      {duplicateInfo.totalDuplicates} Duplicates
                    </span>
                  )}
                </span>
                <div className="flex flex-wrap items-center gap-2">
                  <button
                    onClick={() => {
                      if (selectedSessionIds.length === sessions.length) {
                        setSelectedSessionIds([]);
                      } else {
                        setSelectedSessionIds(sessions.map((s) => s.session_id));
                      }
                    }}
                    className="px-2 py-1 bg-white border border-gray-300 rounded text-[11px] font-semibold text-gray-700 hover:bg-gray-100"
                  >
                    {selectedSessionIds.length === sessions.length ? 'Deselect All' : 'Select All'}
                  </button>
                  {selectedSessionIds.length > 0 && (
                    <button
                      onClick={handleDeleteSelectedSessions}
                      className="px-2.5 py-1 bg-red-600 hover:bg-red-700 text-white rounded text-[11px] font-bold flex items-center gap-1 shadow-xs"
                    >
                      <Trash2 className="w-3 h-3" /> Delete Selected ({selectedSessionIds.length})
                    </button>
                  )}
                </div>
              </div>

              <div className="max-h-56 overflow-y-auto divide-y divide-gray-200 bg-white rounded-lg border border-gray-200">
                {sessions.map((s) => {
                  const isSelected = selectedSessionIds.includes(s.session_id);
                  const isDup = (duplicateInfo.counts[(s.plant_id || '').trim().toUpperCase()] || 0) > 1;
                  const formattedDate = s.created_at ? new Date(s.created_at * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) : 'N/A';
                  return (
                    <div
                      key={s.session_id}
                      className={`flex items-center justify-between px-3 py-2 text-xs hover:bg-gray-50 transition-colors ${
                        isSelected ? 'bg-blue-50/50' : ''
                      }`}
                    >
                      <div className="flex items-center gap-3">
                        <input
                          type="checkbox"
                          checked={isSelected}
                          onChange={() => {
                            setSelectedSessionIds((prev) =>
                              isSelected ? prev.filter((id) => id !== s.session_id) : [...prev, s.session_id]
                            );
                          }}
                          className="rounded border-gray-300 text-[#1565C0] focus:ring-[#1565C0]"
                        />
                        <div>
                          <div className="flex items-center gap-2">
                            <span className="font-bold text-gray-900">{s.plant_id}</span>
                            {isDup && (
                              <span className="px-1.5 py-0.2 text-[9px] font-bold bg-amber-100 text-amber-800 rounded">
                                Duplicate
                              </span>
                            )}
                            <span className="font-mono text-[10px] text-gray-400">{s.session_id}</span>
                          </div>
                          <div className="text-[11px] text-gray-500 flex items-center gap-2 mt-0.5">
                            <span>Time: {formattedDate}</span>
                            <span>•</span>
                            <span>{s.clips?.length || 0} clip(s)</span>
                            <span>•</span>
                            <span>Status: {s.assessment?.status || s.status}</span>
                          </div>
                        </div>
                      </div>
                      <div className="flex items-center gap-2">
                        <button
                          onClick={() => {
                            setCurrentSession(s);
                            setShowSessionManager(false);
                          }}
                          className="px-2 py-1 text-[11px] text-[#1565C0] hover:bg-blue-50 rounded font-semibold"
                        >
                          Select
                        </button>
                        <button
                          onClick={() => handleDeleteSession(s.session_id)}
                          className="p-1 text-gray-400 hover:text-red-600 hover:bg-red-50 rounded transition-colors"
                          title={`Delete session ${s.session_id}`}
                        >
                          <Trash2 className="w-3.5 h-3.5" />
                        </button>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          <div className="flex flex-wrap gap-2">
            {sessions.map((s) => {
              const isDup = (duplicateInfo.counts[(s.plant_id || '').trim().toUpperCase()] || 0) > 1;
              return (
                <div
                  key={s.session_id}
                  className={`group flex items-center rounded-lg border transition-all ${
                    currentSession?.session_id === s.session_id
                      ? 'bg-[#1565C0] text-white border-[#1565C0] shadow-sm'
                      : 'bg-gray-50 text-gray-700 border-gray-200 hover:bg-gray-100'
                  }`}
                >
                  <button
                    onClick={() => setCurrentSession(s)}
                    className="px-3 py-2 text-xs font-medium text-left flex items-center gap-1.5"
                  >
                    <span className="font-bold">{s.plant_id}</span>
                    <span className="text-[10px] font-mono opacity-60">#{s.session_id.slice(-4)}</span>
                    <span className="opacity-75">({s.clips?.length || 0})</span>
                    {isDup && (
                      <span
                        className={`text-[9px] font-bold px-1 rounded ${
                          currentSession?.session_id === s.session_id
                            ? 'bg-amber-400 text-amber-950'
                            : 'bg-amber-100 text-amber-800'
                        }`}
                        title="Duplicate plant ID"
                      >
                        dup
                      </span>
                    )}
                  </button>
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      handleDeleteSession(s.session_id);
                    }}
                    title={`Delete session ${s.session_id}`}
                    className={`pr-2.5 pl-1 py-2 text-xs transition-colors ${
                      currentSession?.session_id === s.session_id
                        ? 'text-white/80 hover:text-white hover:bg-white/10'
                        : 'text-gray-400 hover:text-red-600 hover:bg-red-50'
                    }`}
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                  </button>
                </div>
              );
            })}
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
              <button
                onClick={() => handleDeleteSession(currentSession.session_id)}
                className="text-red-600 hover:text-white hover:bg-red-600 px-3 py-1.5 rounded-lg border border-red-300 font-bold transition-all flex items-center gap-1.5 text-xs ml-auto shadow-xs"
                title="Delete this session and its uploaded media"
              >
                <Trash2 className="w-3.5 h-3.5" /> Delete Active Session
              </button>
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
                <option value="bell_pepper">Bell Pepper / Capsicum</option>
                <option value="chilli">Chilli / Hot Pepper</option>
                <option value="potato">Potato (Solanum tuberosum)</option>
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
              <Upload className="w-4 h-4 text-[#1565C0]" /> {workflowMode === 'single' ? 'Add Viewpoint Media (Single or Multiple)' : 'Multi-Plant Batch Media Ingestion'}
            </h3>
            <p className="text-xs text-gray-500">
              {workflowMode === 'single'
                ? `Upload images or video clips for ${currentSession.plant_id}. Select one or multiple files.`
                : 'Select multiple images/videos and map them to Plant IDs with strict sample isolation.'}
            </p>

            {/* Drag & Drop / Multi-file Selector */}
            <div className="border-2 border-dashed border-gray-300 hover:border-[#1565C0] rounded-xl p-4 text-center transition-colors bg-gray-50/50">
              <input
                type="file"
                multiple
                id="bulk-media-input"
                accept="video/mp4,video/avi,video/quicktime,video/webm,image/jpeg,image/png,image/webp"
                onChange={(e) => handleQueueFiles(e.target.files)}
                className="hidden"
              />
              <label htmlFor="bulk-media-input" className="cursor-pointer block space-y-1.5">
                <div className="w-10 h-10 rounded-full bg-[#1565C0]/10 text-[#1565C0] flex items-center justify-center mx-auto">
                  <Upload className="w-5 h-5" />
                </div>
                <div className="text-xs font-semibold text-gray-800">
                  Click to select files <span className="text-[#1565C0]">or drag & drop</span>
                </div>
                <div className="text-[11px] text-gray-500">
                  Select 1 or many images (.jpg, .png) or videos (.mp4, .mov)
                </div>
              </label>
            </div>

            {/* Media Queue Table */}
            {mediaQueue.length > 0 && (
              <div className="space-y-3 pt-2">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-bold text-gray-800 flex items-center gap-1.5">
                    <Layers className="w-3.5 h-3.5 text-[#1565C0]" /> Queued Media ({mediaQueue.length})
                  </span>
                  <button
                    onClick={() => setMediaQueue([])}
                    className="text-[11px] text-red-600 hover:underline"
                  >
                    Clear All
                  </button>
                </div>

                <div className="max-h-56 overflow-y-auto divide-y divide-gray-100 border border-gray-200 rounded-lg bg-gray-50">
                  {mediaQueue.map((item) => (
                    <div key={item.id} className="p-2 flex items-center gap-2.5 text-xs">
                      {item.previewUrl ? (
                        <img src={item.previewUrl} alt="prev" className="w-8 h-8 rounded object-cover border border-gray-300 shrink-0" />
                      ) : (
                        <div className="w-8 h-8 rounded bg-blue-100 text-[#1565C0] flex items-center justify-center shrink-0">
                          <Video className="w-4 h-4" />
                        </div>
                      )}
                      <div className="min-w-0 flex-1">
                        <div className="font-semibold text-gray-800 truncate text-[11px]">{item.file.name}</div>
                        <div className="text-[10px] text-gray-500">{(item.file.size / 1024).toFixed(0)} KB</div>
                      </div>

                      {/* View Angle selector */}
                      <select
                        value={item.viewLabel}
                        onChange={(e) => updateQueueItem(item.id, { viewLabel: e.target.value as ViewpointLabel })}
                        className="text-[11px] px-2 py-1 border border-gray-300 rounded bg-white text-gray-700"
                      >
                        <option value="front">Front</option>
                        <option value="back">Back</option>
                        <option value="left">Left</option>
                        <option value="right">Right</option>
                        <option value="top">Top</option>
                        <option value="underside">Underside</option>
                        <option value="unknown">Unknown</option>
                      </select>

                      {/* Plant ID input for batch mode */}
                      {workflowMode === 'batch' && (
                        <input
                          type="text"
                          value={item.plantId}
                          onChange={(e) => updateQueueItem(item.id, { plantId: e.target.value })}
                          placeholder="Plant ID"
                          className="w-20 text-[11px] px-1.5 py-1 border border-gray-300 rounded bg-white font-mono"
                        />
                      )}

                      <button
                        onClick={() => removeFromQueue(item.id)}
                        className="text-gray-400 hover:text-red-600 p-1 font-bold text-xs"
                      >
                        ✕
                      </button>
                    </div>
                  ))}
                </div>

                <button
                  type="button"
                  onClick={handleUploadBatchQueue}
                  disabled={uploading}
                  className="w-full bg-[#1565C0] hover:bg-[#0D47A1] text-white py-2 rounded-lg text-xs font-bold transition-colors flex items-center justify-center gap-2 shadow-xs disabled:opacity-50"
                >
                  {uploading ? (
                    <>
                      <RefreshCw className="w-3.5 h-3.5 animate-spin" /> Ingesting Queue...
                    </>
                  ) : (
                    <>
                      <Upload className="w-3.5 h-3.5" /> {workflowMode === 'single' ? `Upload All (${mediaQueue.length}) to Session` : `Execute Multi-Plant Batch (${mediaQueue.length})`}
                    </>
                  )}
                </button>
              </div>
            )}

            {/* Quick Real Sample Leaf Pickers */}
            <div className="pt-2 border-t border-gray-100">
              <label className="block text-xs font-semibold text-gray-700 mb-1.5 flex items-center gap-1.5">
                <Leaf className="w-3.5 h-3.5 text-emerald-600" /> One-Click Real Leaf Samples:
              </label>
              <div className="grid grid-cols-2 gap-1.5 text-xs">
                <button
                  type="button"
                  onClick={() => loadSampleLeaf('tomato_healthy.jpg', 'front')}
                  className="p-1.5 bg-gray-50 hover:bg-emerald-50 hover:border-emerald-300 border border-gray-200 rounded text-left transition-colors text-[11px] font-medium text-gray-700"
                >
                  🌱 Healthy Tomato
                </button>
                <button
                  type="button"
                  onClick={() => loadSampleLeaf('tomato_early_blight.jpg', 'back')}
                  className="p-1.5 bg-gray-50 hover:bg-amber-50 hover:border-amber-300 border border-gray-200 rounded text-left transition-colors text-[11px] font-medium text-gray-700"
                >
                  🍂 Early Blight
                </button>
                <button
                  type="button"
                  onClick={() => loadSampleLeaf('tomato_late_blight.jpg', 'left')}
                  className="p-1.5 bg-gray-50 hover:bg-red-50 hover:border-red-300 border border-gray-200 rounded text-left transition-colors text-[11px] font-medium text-gray-700"
                >
                  🥀 Late Blight
                </button>
                <button
                  type="button"
                  onClick={() => loadSampleLeaf('tomato_septoria_leaf_spot.jpg', 'top')}
                  className="p-1.5 bg-gray-50 hover:bg-purple-50 hover:border-purple-300 border border-gray-200 rounded text-left transition-colors text-[11px] font-medium text-gray-700"
                >
                  🔍 Septoria Leaf Spot
                </button>
              </div>
            </div>

            {/* Single File Fallback Form */}
            {selectedFile && (
              <form onSubmit={handleUploadClip} className="pt-2 border-t border-gray-100 space-y-3">
                <div className="flex items-center gap-3 p-2 bg-emerald-50 rounded-lg border border-emerald-200">
                  {selectedFile.type.startsWith('image/') ? (
                    <img
                      src={URL.createObjectURL(selectedFile)}
                      alt="Selected leaf"
                      className="w-10 h-10 object-cover rounded border border-emerald-300"
                    />
                  ) : (
                    <Video className="w-6 h-6 text-emerald-700" />
                  )}
                  <div className="text-xs overflow-hidden flex-1">
                    <p className="font-semibold text-emerald-900 truncate">{selectedFile.name}</p>
                    <p className="text-emerald-700 text-[10px]">View: {selectedView}</p>
                  </div>
                </div>

                <button
                  type="submit"
                  disabled={uploading}
                  className="w-full bg-emerald-600 hover:bg-emerald-700 text-white py-2 rounded-lg text-xs font-semibold transition-colors flex items-center justify-center gap-1.5"
                >
                  {uploading ? <RefreshCw className="w-3.5 h-3.5 animate-spin" /> : <Upload className="w-3.5 h-3.5" />}
                  Upload Selected Sample
                </button>
              </form>
            )}

            {uploadError && (
              <div className="p-2.5 rounded bg-red-50 border border-red-200 text-red-700 text-xs">
                {uploadError}
              </div>
            )}
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
        <div className="space-y-6">
          {/* 1. Explainable Clinical Finding, Pathogen Etiology & Step-by-Step Cure */}
          <DiagnosticExplainerCard
            diagnosis={
              assessment.top_disease_class
                ? assessment.top_disease_class.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase())
                : (assessment.status === 'HEALTHY_APPEARING' ? `Healthy ${currentSession?.crop_type ? currentSession.crop_type.charAt(0).toUpperCase() + currentSession.crop_type.slice(1) : 'Plant'} Foliage` : 'Unspecified Condition')
            }
            conditionKey={assessment.top_disease_class || 'healthy'}
            isHealthy={assessment.status === 'HEALTHY_APPEARING' || (!assessment.top_disease_class || assessment.top_disease_class === 'healthy')}
            confidence={
              assessment.disease_evidence && assessment.disease_evidence.length > 0
                ? assessment.disease_evidence[0].average_confidence
                : 0.88
            }
            severityGrade={assessment.status}
            lesionPercentage={assessment.visible_symptom_area_percent || 0.0}
            affectedViews={assessment.submitted_views || []}
            consensusScore={
              assessment.disease_evidence && assessment.disease_evidence.length > 0
                ? assessment.disease_evidence[0].average_confidence * 100
                : 91.5
            }
            modelsCount={3}
            modelProvenance={assessment.model_provenance}
            visualLocalization={assessment.visual_localization}
            cropType={currentSession?.crop_type || 'Tomato'}
          />

          {/* 2. Interactive Clinical Pathologist Consultation (Chatbot 2) */}
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-bold text-gray-900 flex items-center gap-2">
                <Stethoscope className="w-4 h-4 text-emerald-800" />
                Case Consultation: Pathologist Interactive Q&A
              </h3>
              <span className="text-[11px] text-gray-500 font-medium">
                Ask about dilution rates, spread risk, organic treatments, or application intervals
              </span>
            </div>
            <ClinicalDiseaseChat
              diseaseName={
                assessment.top_disease_class
                  ? assessment.top_disease_class.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase())
                  : 'Healthy Foliage'
              }
              cropType={currentSession?.crop_type || 'Tomato'}
              confidence={
                assessment.disease_evidence && assessment.disease_evidence.length > 0
                  ? assessment.disease_evidence[0].average_confidence
                  : 0.88
              }
              severity={assessment.status}
              lesionPercentage={assessment.visible_symptom_area_percent || 0.0}
              affectedViews={assessment.submitted_views || []}
            />
          </div>

          <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-6 space-y-6">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-gray-100 pb-4">
              <div>
                <div className="flex items-center gap-3">
                  <h3 className="text-xl font-bold text-gray-900">
                    Audit Session Log for {currentSession?.plant_id}
                  </h3>
                  {getStatusBadge(assessment.status)}
                </div>
                <p className="text-xs text-gray-500 mt-1">
                  Multi-view dual-evidence assessment from {assessment.total_clips} video clip(s)
                </p>
              </div>

              <div className="flex items-center gap-2 flex-wrap">
                <button
                  onClick={handleLoadObservations}
                  className="bg-gray-100 hover:bg-gray-200 text-gray-700 px-3 py-1.5 rounded-lg text-xs font-semibold transition-colors flex items-center gap-1.5"
                >
                  <Eye className="w-3.5 h-3.5" />
                  {showObservations ? 'Hide Frames' : 'Inspect Frames'}
                </button>
                <button
                  onClick={() => handleDownloadReport('markdown')}
                  className="bg-[#1B4332] hover:bg-[#133023] text-white px-3 py-1.5 rounded-lg text-xs font-semibold transition-colors flex items-center gap-1.5 shadow-xs"
                >
                  <FileDown className="w-3.5 h-3.5" /> Report (.md)
                </button>
                <button
                  onClick={() => handleDownloadReport('json')}
                  className="bg-[#2D6A4F] hover:bg-[#1B4332] text-white px-3 py-1.5 rounded-lg text-xs font-semibold transition-colors flex items-center gap-1.5 shadow-xs"
                >
                  <FileDown className="w-3.5 h-3.5" /> JSON
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
                {assessment.top_disease_class
                  ? assessment.top_disease_class.replace(/_/g, ' ')
                  : assessment.status === 'HEALTHY_APPEARING'
                  ? 'Healthy Foliage'
                  : assessment.status === 'UNCERTAIN'
                  ? 'Symptoms Uncertain'
                  : 'Awaiting Evidence'}
              </div>
              <div className="text-[11px] text-gray-500 mt-0.5">
                {assessment.top_disease_class && assessment.disease_fraction !== undefined && assessment.disease_fraction !== null
                  ? `${Math.round(assessment.disease_fraction * 100)}% of observed frames`
                  : assessment.status === 'HEALTHY_APPEARING'
                  ? 'Optimal canopy vitality'
                  : 'Clear on observed angles'}
              </div>
            </div>

            <div className="p-4 bg-gray-50 rounded-xl border border-gray-200">
              <div className="text-xs text-gray-500 font-medium">Visible Symptom Area (Heuristic)</div>
              <div className="text-2xl font-bold text-gray-900 mt-1">
                {assessment.visible_symptom_area_percent !== null &&
                assessment.visible_symptom_area_percent !== undefined
                  ? `${assessment.visible_symptom_area_percent}%`
                  : '0.0%'}
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
