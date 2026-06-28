<script setup lang="ts">
import { computed } from 'vue';
import { keyframeUrl } from '../api';
import type { AlertReport, DetectionResult } from '../types';
import ScoreChart from './ScoreChart.vue';

const props = defineProps<{
  report: AlertReport;
  detection?: DetectionResult | null;
  threshold?: number | null;
}>();

const categoryRows = computed(() => {
  if (props.detection?.category_scores) {
    return Object.entries(props.detection.category_scores)
      .sort((a, b) => b[1] - a[1])
      .map(([label, score]) => ({ label, score }));
  }
  return props.report.harmful_contents.map((item) => ({
    label: `${item.category_zh} (${item.category_en})`,
    score: item.confidence,
  }));
});

const keyframes = computed(() => {
  const urls = props.detection?.keyframe_urls || [];
  if (urls.length) return urls.map((url, index) => ({ url, caption: `关键帧 ${index + 1}` }));
  return props.report.harmful_contents
    .filter((item) => item.keyframe_url)
    .map((item) => ({
      url: item.keyframe_url || '',
      caption: `${item.category_zh} ${item.time_segments}`,
    }));
});

function percent(value: number) {
  return `${Math.round(Math.max(0, Math.min(1, value)) * 100)}%`;
}

function downloadReport() {
  const blob = new Blob([JSON.stringify(props.report, null, 2)], { type: 'application/json;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = `${props.report.report_id}.json`;
  link.click();
  URL.revokeObjectURL(url);
}
</script>

<template>
  <section class="result-grid">
    <div class="panel summary-panel">
      <div class="result-head">
        <span class="level-badge" :class="report.alert_level.toLowerCase()">{{ report.alert_level }}</span>
        <div>
          <strong>{{ report.anomaly_score.toFixed(4) }}</strong>
          <small>异常分数</small>
        </div>
        <div>
          <strong>{{ report.processing_time.toFixed(3) }}s</strong>
          <small>报告耗时</small>
        </div>
      </div>
      <p class="summary">{{ report.summary }}</p>
      <p class="action">{{ report.action_suggestion || '暂无处置建议' }}</p>
      <div class="meta-row">
        <span>报告 ID</span>
        <code>{{ report.report_id }}</code>
      </div>
      <div class="meta-row">
        <span>扫描时间</span>
        <code>{{ report.scan_time }}</code>
      </div>
      <button class="ghost-button" type="button" @click="downloadReport">下载 JSON</button>
    </div>

    <div class="panel">
      <h2>类别分数</h2>
      <div v-if="categoryRows.length" class="score-list">
        <div v-for="row in categoryRows" :key="row.label" class="score-row">
          <div class="score-label">
            <span>{{ row.label }}</span>
            <strong>{{ row.score.toFixed(4) }}</strong>
          </div>
          <div class="bar-track">
            <div class="bar-fill" :class="{ danger: row.score >= 0.5 }" :style="{ width: percent(row.score) }"></div>
          </div>
        </div>
      </div>
      <p v-else class="empty">暂无类别命中</p>
    </div>

    <div v-if="detection" class="panel wide">
      <h2>异常时间曲线</h2>
      <ScoreChart :scores="detection.segment_scores" :threshold="threshold" />
    </div>

    <div class="panel wide">
      <h2>有害片段</h2>
      <div v-if="detection?.harmful_segments?.length" class="segment-list">
        <div v-for="segment in detection.harmful_segments" :key="`${segment.start_time}-${segment.end_time}`" class="segment-item">
          <strong>{{ segment.start_time.toFixed(1) }}s - {{ segment.end_time.toFixed(1) }}s</strong>
          <span>{{ segment.category }} / {{ segment.category_en }}</span>
          <code>{{ segment.score.toFixed(4) }}</code>
        </div>
      </div>
      <div v-else-if="report.harmful_contents.length" class="segment-list">
        <div v-for="item in report.harmful_contents" :key="item.category_en" class="segment-item">
          <strong>{{ item.time_segments || '未返回时间段' }}</strong>
          <span>{{ item.category_zh }} / {{ item.category_en }}</span>
          <code>{{ item.confidence.toFixed(4) }}</code>
        </div>
      </div>
      <p v-else class="empty">未检测到有害片段</p>
    </div>

    <div class="panel wide">
      <h2>关键帧</h2>
      <div v-if="keyframes.length" class="keyframe-grid">
        <figure v-for="frame in keyframes" :key="frame.url">
          <img :src="keyframeUrl(frame.url)" :alt="frame.caption" />
          <figcaption>{{ frame.caption }}</figcaption>
        </figure>
      </div>
      <p v-else class="empty">暂无关键帧</p>
    </div>
  </section>
</template>
