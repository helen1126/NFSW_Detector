<script setup lang="ts">
import { computed } from 'vue';

const props = defineProps<{
  scores?: number[] | number[][];
  threshold?: number | null;
}>();

const flatScores = computed(() => {
  const source = props.scores || [];
  return source.map((item) => Array.isArray(item) ? Math.max(...item) : item);
});

const points = computed(() => {
  const values = flatScores.value;
  if (!values.length) return '';
  const width = 640;
  const height = 180;
  const maxIndex = Math.max(values.length - 1, 1);
  return values.map((score, index) => {
    const x = (index / maxIndex) * width;
    const y = height - Math.max(0, Math.min(1, score)) * height;
    return `${x},${y}`;
  }).join(' ');
});

const thresholdY = computed(() => {
  const threshold = props.threshold ?? 0.5;
  return 180 - Math.max(0, Math.min(1, threshold)) * 180;
});
</script>

<template>
  <div class="chart">
    <svg viewBox="0 0 640 180" role="img" aria-label="异常分数曲线">
      <line x1="0" x2="640" :y1="thresholdY" :y2="thresholdY" class="threshold" />
      <polyline v-if="points" :points="points" class="score-line" />
      <text x="8" :y="Math.max(14, thresholdY - 6)" class="threshold-label">
        threshold {{ (threshold ?? 0.5).toFixed(2) }}
      </text>
    </svg>
    <p v-if="!flatScores.length" class="empty">暂无段级分数</p>
  </div>
</template>
