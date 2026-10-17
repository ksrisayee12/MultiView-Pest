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
  accuracy_benchmarks: {
    yolo_only_acc: string;
    opencv_only_acc: string;
    fusion_acc: string;
  };
}
