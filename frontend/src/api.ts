import type { AlertReport, DetectResponse, HealthResponse } from './types';

export const API_BASE = (import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000').replace(/\/$/, '');

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, init);
  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`;
    try {
      const data = await response.json();
      message = data.detail || data.error || message;
    } catch {
      // Keep the HTTP message when the body is not JSON.
    }
    throw new Error(message);
  }
  return response.json() as Promise<T>;
}

export function keyframeUrl(path?: string | null): string {
  if (!path) return '';
  if (/^https?:\/\//.test(path)) return path;
  return `${API_BASE}${path.startsWith('/') ? path : `/${path}`}`;
}

export function getHealth() {
  return request<HealthResponse>('/api/v1/health');
}

export function getReport(reportId: string) {
  return request<AlertReport>(`/api/v1/reports/${encodeURIComponent(reportId)}`);
}

export function detectVideo(file: File, threshold?: number | null) {
  const form = new FormData();
  form.append('file', file);
  const query = new URLSearchParams();
  if (typeof threshold === 'number') {
    query.set('threshold', String(threshold));
  }
  const path = query.size ? `/api/v1/detect?${query.toString()}` : '/api/v1/detect';
  return request<DetectResponse>(path, {
    method: 'POST',
    body: form,
  });
}
