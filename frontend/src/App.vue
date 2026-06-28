<template>
  <div class="shell">
    <header class="topbar">
      <RouterLink class="brand" to="/">
        <span class="brand-mark">ND</span>
        <span>
          <strong>NFSW Detector</strong>
          <small>视频内容审核工作台</small>
        </span>
      </RouterLink>
      <nav>
        <RouterLink to="/">检测</RouterLink>
      </nav>
    </header>
    <main>
      <RouterView />
    </main>
  </div>
</template>
