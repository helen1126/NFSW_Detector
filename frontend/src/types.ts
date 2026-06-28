export type AlertLevel = 'SAFE' | 'LOW' | 'MEDIUM' | 'HIGH';

export interface HealthResponse {
  status: string;
  model_loaded: boolean;
  device: string;
  version: string;
}

export interface HarmfulSegment {
  start_time: number;
  end_time: number;
  score: number;
  category: string;
  category_en: string;
}

export interface HarmfulContent {
  category_en: string;
  category_zh: string;
  confidence: number;
  time_segments: string;
  description: string;
  keyframe_url?: string | null;
}

export interface DetectionResult {
  video_id: string;
  duration: number;
  is_harmful: boolean;
  anomaly_score: number;
  predicted_categories: string[];
  category_scores: Record<string, number>;
  segment_scores?: number[] | number[][];
  harmful_segments: HarmfulSegment[];
  keyframe_urls: string[];
  calibrated_score?: number | null;
  ood_score?: number | null;
  is_ood?: boolean | null;
  extra_category_info?: Record<string, unknown> | null;
  processing_time?: number;
  detection_time?: string;
}

export interface AlertReport {
  report_id: string;
  video_id: string;
  video_path: string;
  scan_time: string;
  alert_level: AlertLevel;
  anomaly_score: number;
  harmful_contents: HarmfulContent[];
  summary: string;
  action_suggestion: string;
  processing_time: number;
}

export interface DetectResponse {
  detection: DetectionResult;
  report: AlertReport;
}
