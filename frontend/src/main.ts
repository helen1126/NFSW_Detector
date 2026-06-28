import { createApp } from 'vue';
import { createRouter, createWebHistory } from 'vue-router';
import App from './App.vue';
import DetectView from './views/DetectView.vue';
import ReportView from './views/ReportView.vue';
import './styles.css';

const router = createRouter({
  history: createWebHistory(),
  routes: [
    { path: '/', component: DetectView },
    { path: '/reports/:reportId', component: ReportView },
  ],
});

createApp(App).use(router).mount('#app');
