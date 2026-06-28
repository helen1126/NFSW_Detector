<script setup lang="ts">
import { onMounted, ref } from 'vue';
import { RouterLink, useRoute } from 'vue-router';
import { getReport } from '../api';
import type { AlertReport } from '../types';
import ReportPanel from '../components/ReportPanel.vue';

const route = useRoute();
const report = ref<AlertReport | null>(null);
const loading = ref(true);
const error = ref('');

async function loadReport() {
  loading.value = true;
  error.value = '';
  try {
    report.value = await getReport(String(route.params.reportId));
  } catch (exc) {
    error.value = exc instanceof Error ? exc.message : '报告加载失败';
  } finally {
    loading.value = false;
  }
}

onMounted(loadReport);
</script>

<template>
  <div class="report-page">
    <RouterLink class="back-link" to="/">返回检测页</RouterLink>
    <div v-if="loading" class="empty-state">正在加载报告...</div>
    <p v-else-if="error" class="error">{{ error }}</p>
    <ReportPanel v-else-if="report" :report="report" />
  </div>
</template>
