export interface Recommendation {
  problem: string;
  title: string;
  treatment: string;
  prevention: string[];
  disclaimer: string;
}

export interface TileResult {
  tile_id: number;
  grid_row: number;
  grid_col: number;
  bbox: [number, number, number, number];
  yolo_prediction: string;
  yolo_confidence: number;
  opencv_status: 'normal' | 'abnormal';
  greenness_index: number;
  green_pixel_percentage: number;
  lesion_percentage: number;
  fusion_status: 'HEALTHY' | 'CONFIRMED DISEASE' | 'DISPUTED';
  confirmed_disease?: string | null;
  severity: 'Low' | 'Moderate' | 'High' | 'Severe';
  requires_closeup: boolean;
  recommendation: Recommendation;
}

export interface PriorityItem {
  tile_id: number;
  fusion_status: 'CONFIRMED DISEASE' | 'DISPUTED';
  disease?: string;
  severity: 'Low' | 'Moderate' | 'High' | 'Severe';
  lesion_percentage: number;
  priority_level: number;
}

export interface ScanSummary {
  scan_id: string;
  created_at: string;
  crop: string;
  image_name: string;
  total_tiles: number;
  healthy_tiles: number;
  confirmed_disease_tiles: number;
  disputed_tiles: number;
  plant_health_index: number;
  yolo_model_loaded: boolean;
  tile_results: TileResult[];
  priority_list: PriorityItem[];
}

export interface AnalyticsData {
  total_scans: number;
  total_tiles: number;
  healthy_tiles: number;
  disease_tiles: number;
  disputed_tiles: number;
  average_health_index: number;
  disease_breakdown: Record<string, number>;
  accuracy_benchmarks?: {
    yolo_only_acc?: string;
    opencv_only_acc?: string;
    fusion_acc?: string;
  };
}

// ---------------------------------------------------------------------------
// Phase 3 Multi-View Plant Analysis Types
// ---------------------------------------------------------------------------

export type ViewpointLabel =
  | 'front'
  | 'back'
  | 'left'
  | 'right'
  | 'top'
  | 'underside'
  | 'unknown'
  | 'arbitrary';

export type AssessmentStatus =
  | 'HEALTHY_APPEARING'
  | 'SYMPTOMS_DETECTED'
  | 'UNCERTAIN'
  | 'INSUFFICIENT_COVERAGE'
  | 'PROCESSING_ERROR'
  | 'MODEL_UNAVAILABLE';

export interface ClipMeta {
  clip_id: string;
  session_id: string;
  original_filename: string;
  view_label: ViewpointLabel | string;
  upload_status: string;
  processing_status: string;
  duration_seconds?: number;
  fps?: number;
  total_video_frames?: number;
  frames_extracted: number;
  frames_skipped_duplicate: number;
  sample_fps?: number;
  error_message?: string;
  created_at: number;
  processed_at?: number;
}

export interface DiseaseEvidenceItem {
  predicted_class: string;
  frame_count: number;
  fraction_of_total: number;
  average_confidence: number;
  note: string;
}

export interface ClipSummaryItem {
  clip_id: string;
  view_label: string;
  total_frames: number;
  disease_frames: number;
  disease_fraction: number;
  opencv_abnormal_frames: number;
  avg_lesion_percent?: number | null;
}

export interface PlantAssessment {
  session_id: string;
  plant_id: string;
  crop_type: string;
  status: AssessmentStatus | string;
  top_disease_class?: string | null;
  disease_evidence: DiseaseEvidenceItem[];
  visible_symptom_area_percent?: number | null;
  opencv_abnormal_fraction?: number | null;
  disease_fraction?: number | null;
  total_clips: number;
  total_frames_processed: number;
  total_usable_frames?: number;
  submitted_views: string[];
  views_submitted?: string[];
  missing_views: string[];
  clips_summary: ClipSummaryItem[];
  model_available: boolean;
  model_provenance?: {
    model_name?: string;
    model_version?: string;
    sha256?: string;
    status?: string;
  };
  visual_localization?: any;
  coverage_note: string;
  limitations: string[];
  assessment_notes: string[];
  created_at: number;
}

export interface PlantSession {
  session_id: string;
  plant_id: string;
  crop_type: string;
  status: string;
  created_at: number;
  updated_at: number;
  notes?: string | null;
  clips: ClipMeta[];
  assessment?: PlantAssessment | null;
  observation_count: number;
}

export interface FrameObservation {
  clip_id: string;
  frame_index: number;
  timestamp_sec: number;
  view_label: string;
  yolo_prediction: string;
  yolo_confidence: number;
  opencv_visual_status: string;
  lesion_percentage: number;
  greenness_index?: number;
  green_pixel_percentage?: number;
  frame_shape?: number[];
  model_loaded?: boolean;
}

export interface ModelReadinessStatus {
  status: 'ready' | 'missing_weights' | 'load_error' | 'not_configured';
  message: string;
  model_path?: string;
  classes?: string[];
  version?: string;
}

