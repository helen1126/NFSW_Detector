<script setup lang="ts">
import { onMounted, ref } from 'vue';
import { RouterLink } from 'vue-router';
import { detectVideo, getHealth } from '../api';
import type { DetectResponse, HealthResponse } from '../types';
import ReportPanel from '../components/ReportPanel.vue';

const health = ref<HealthResponse | null>(null);
const file = ref<File | null>(null);
const previewUrl = ref('');
const useThreshold = ref(false);
const threshold = ref(0.5);
const loading = ref(false);
const error = ref('');
const result = ref<DetectResponse | null>(null);

async function refreshHealth() {
  try {
    health.value = await getHealth();
  } catch (exc) {
    error.value = exc instanceof Error ? exc.message : '无法连接后端服务';
  }
}

function onFileChange(event: Event) {
  const input = event.target as HTMLInputElement;
  const selected = input.files?.[0] || null;
  file.value = selected;
  result.value = null;
  error.value = '';
  if (previewUrl.value) URL.revokeObjectURL(previewUrl.value);
  previewUrl.value = selected ? URL.createObjectURL(selected) : '';
}

async function submit() {
  if (!file.value) {
    error.value = '请选择一个视频文件';
    return;
  }
  loading.value = true;
  error.value = '';
  try {
    result.value = await detectVideo(file.value, useThreshold.value ? threshold.value : null);
  } catch (exc) {
    error.value = exc instanceof Error ? exc.message : '检测失败';
  } finally {
    loading.value = false;
    await refreshHealth();
  }
}

onMounted(refreshHealth);
</script>

<template>
  <div class="workspace">
    <aside class="side-panel">
      <section class="panel">
        <div class="section-title">
          <h1>视频检测</h1>
          <button class="icon-button" type="button" title="刷新后端状态" @click="refreshHealth">↻</button>
        </div>
        <div class="health" :class="{ loaded: health?.model_loaded, offline: !health }">
          <span>{{ health?.model_loaded ? '模型已加载' : '模型未加载' }}</span>
          <small>{{ health ? `device: ${health.device} · API ${health.version}` : '等待后端服务' }}</small>
        </div>

        <label class="upload-box">
          <input type="file" accept=".mp4,.avi,.mov,.mkv,.wmv,.flv,.webm,video/*" @change="onFileChange" />
          <strong>{{ file?.name || '选择视频文件' }}</strong>
          <small>支持 mp4 / avi / mov / mkv / wmv / flv / webm</small>
        </label>

        <label class="toggle-row">
          <input v-model="useThreshold" type="checkbox" />
          <span>临时覆盖检测阈值</span>
        </label>
        <div class="threshold-row" :class="{ disabled: !useThreshold }">
          <input v-model.number="threshold" type="range" min="0" max="1" step="0.01" :disabled="!useThreshold" />
          <code>{{ threshold.toFixed(2) }}</code>
        </div>

        <button class="primary-button" type="button" :disabled="loading || !file" @click="submit">
          {{ loading ? '检测中...' : '开始检测' }}
        </button>
        <p v-if="error" class="error">{{ error }}</p>
      </section>

      <section v-if="previewUrl" class="panel">
        <h2>本地预览</h2>
        <video :src="previewUrl" controls />
      </section>
    </aside>

    <section class="content-area">
      <ReportPanel
        v-if="result"
        :report="result.report"
        :detection="result.detection"
        :threshold="useThreshold ? threshold : 0.5"
      />
      <div v-else class="empty-state">
        <h2>等待检测任务</h2>
        <p>上传视频后会在这里展示预警等级、类别分数、有害片段、关键帧和报告 ID。</p>
      </div>
      <RouterLink v-if="result" class="report-link" :to="`/reports/${result.report.report_id}`">
        打开报告详情
      </RouterLink>
    </section>
  </div>
</template>
