(function () {
  const video = document.getElementById('demoVideo');
  const canvas = document.getElementById('demoCanvas');
  const prompt = document.getElementById('demoPrompt');
  const startBtn = document.getElementById('startBtn');
  const camError = document.getElementById('camError');
  if (!video || !canvas || !startBtn) return;

  const ctx = canvas.getContext('2d');
  const elFps = document.getElementById('mFps');
  const elFaces = document.getElementById('mFaces');
  const elObj = document.getElementById('mObjects');
  const elLive = document.getElementById('mLive');
  const elClock = document.getElementById('hudClock');
  const elFeed = document.getElementById('detFeed');
  let running = false;
  let lastT = performance.now();
  let fps = 0;
  let prevFrame = null;
  let boxes = [];
  let frameN = 0;

  function feed(text, color) {
    const line = document.createElement('div');
    line.className = 'line';
    line.innerHTML = `<span class="t">${new Date().toLocaleTimeString('en-GB')}</span> · <span class="e" style="color:${color || 'var(--ok)'}">${text}</span>`;
    elFeed.prepend(line);
    while (elFeed.children.length > 12) elFeed.removeChild(elFeed.lastChild);
  }

  async function start() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 960, height: 540 }, audio: false });
      video.srcObject = stream;
      await video.play();
      canvas.width = video.videoWidth || 960;
      canvas.height = video.videoHeight || 540;
      prompt.classList.add('hidden');
      running = true;
      feed('SANDBOX INITIALIZED', 'var(--neon)');
      requestAnimationFrame(loop);
    } catch (err) {
      camError.textContent = `Camera unavailable: ${err.message || err.name}`;
    }
  }

  function detectMotion() {
    const w = 64;
    const h = 36;
    const tmp = document.createElement('canvas');
    tmp.width = w;
    tmp.height = h;
    const tctx = tmp.getContext('2d');
    tctx.drawImage(video, 0, 0, w, h);
    const cur = tctx.getImageData(0, 0, w, h).data;
    const out = [];
    if (prevFrame) {
      let minX = w, minY = h, maxX = 0, maxY = 0, count = 0;
      for (let y = 0; y < h; y += 1) {
        for (let x = 0; x < w; x += 1) {
          const i = (y * w + x) * 4;
          const d = Math.abs(cur[i] - prevFrame[i]) + Math.abs(cur[i + 1] - prevFrame[i + 1]) + Math.abs(cur[i + 2] - prevFrame[i + 2]);
          if (d > 70) {
            count += 1;
            minX = Math.min(minX, x); minY = Math.min(minY, y);
            maxX = Math.max(maxX, x); maxY = Math.max(maxY, y);
          }
        }
      }
      if (count > 36 && maxX > minX && maxY > minY) {
        out.push({
          x: (minX / w) * canvas.width,
          y: (minY / h) * canvas.height,
          w: ((maxX - minX) / w) * canvas.width,
          h: ((maxY - minY) / h) * canvas.height,
          conf: Math.min(0.98, 0.6 + count / 700),
        });
      }
    }
    prevFrame = cur;
    return out;
  }

  function drawBox(b, label) {
    ctx.strokeStyle = '#34e0a1';
    ctx.lineWidth = 2;
    ctx.shadowColor = '#34e0a1';
    ctx.shadowBlur = 8;
    ctx.strokeRect(b.x, b.y, b.w, b.h);
    ctx.shadowBlur = 0;
    ctx.save();
    ctx.scale(-1, 1);
    ctx.fillStyle = 'rgba(0,0,0,0.65)';
    ctx.fillRect(-(b.x + b.w), b.y - 18, Math.max(120, b.w), 16);
    ctx.fillStyle = '#34e0a1';
    ctx.font = '11px "Space Mono", monospace';
    ctx.fillText(label, -(b.x + b.w) + 4, b.y - 6);
    ctx.restore();
  }

  function loop() {
    if (!running) return;
    const now = performance.now();
    fps = fps * 0.9 + (1000 / Math.max(now - lastT, 1)) * 0.1;
    lastT = now;
    frameN += 1;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (frameN % 4 === 0) boxes = detectMotion();
    boxes.forEach((b, i) => drawBox(b, `TRACK_${i + 1} ${b.conf.toFixed(2)}`));
    elFps.textContent = fps.toFixed(1);
    elFaces.textContent = boxes.length ? 1 : 0;
    elObj.textContent = boxes.length;
    elLive.textContent = boxes.length ? 'CHECKING' : '-';
    elLive.style.color = boxes.length ? 'var(--warn)' : 'var(--text-dim)';
    if (frameN % 100 === 0 && boxes.length) feed('MOTION TRACKING');
    elClock.textContent = new Date().toLocaleTimeString('en-GB');
    requestAnimationFrame(loop);
  }

  startBtn.addEventListener('click', start);
  setInterval(() => { if (elClock) elClock.textContent = new Date().toLocaleTimeString('en-GB'); }, 1000);
})();