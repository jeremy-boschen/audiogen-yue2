/* Shared stage for the microscope explorers. Classic script: works from file://. */
(function () {
  'use strict';
  const T = window.THREE;
  const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const DPR = Math.min(window.devicePixelRatio || 1, 1.5);

  const PALETTE = {
    bg: 0x0a0f1c, navy: 0x111a2d, blue: 0x4f8cff, violet: 0x9b6bff, pink: 0xff5fb4, orange: 0xff9a4d, gold: 0xffd27a,
  };

  // GLSL: the blue -> violet -> pink -> orange -> gold ramp, used by every view.
  const GLSL_RAMP = `
    vec3 ramp(float x) {
      x = clamp(x, 0.0, 1.0);
      vec3 c0 = vec3(0.035, 0.055, 0.13);
      vec3 c1 = vec3(0.12, 0.30, 0.95);
      vec3 c2 = vec3(0.55, 0.36, 1.0);
      vec3 c3 = vec3(1.0, 0.33, 0.68);
      vec3 c4 = vec3(1.0, 0.58, 0.28);
      vec3 c5 = vec3(1.0, 0.80, 0.45);
      if (x < 0.2) return mix(c0, c1, x / 0.2);
      if (x < 0.45) return mix(c1, c2, (x - 0.2) / 0.25);
      if (x < 0.68) return mix(c2, c3, (x - 0.45) / 0.23);
      if (x < 0.88) return mix(c3, c4, (x - 0.68) / 0.2);
      return mix(c4, c5, (x - 0.88) / 0.12);
    }`;

  function rampJS(x) {
    const stops = [[0, [0.035, 0.055, 0.13]], [0.2, [0.12, 0.3, 0.95]], [0.45, [0.55, 0.36, 1]], [0.68, [1, 0.33, 0.68]],
                   [0.88, [1, 0.58, 0.28]], [1, [1, 0.8, 0.45]]];
    x = Math.max(0, Math.min(1, x));
    for (let i = 1; i < stops.length; i++) {
      if (x <= stops[i][0]) {
        const [a, ca] = stops[i - 1], [b, cb] = stops[i], f = (x - a) / (b - a);
        return ca.map((v, k) => v + (cb[k] - v) * f);
      }
    }
    return stops[stops.length - 1][1];
  }
  const css = (rgb, a = 1) => `rgba(${rgb.map(v => Math.round(v * 255)).join(',')},${a})`;

  function decode(b64, Type) {
    const bin = atob(b64);
    const bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    return new Type(bytes.buffer);
  }

  const fmt = (s) => {
    if (s == null || !isFinite(s)) return '–';
    const m = Math.floor(s / 60), r = s - m * 60;
    return `${m}:${r < 10 ? '0' : ''}${r.toFixed(1)}`;
  };
  const num = (v, d = 3) => (v == null || !isFinite(v)) ? '–' : Number(v).toFixed(d);

  /* Renderer + bloom + render loop that sleeps when the page or canvas is not visible. */
  function stage(host, { bloom = 0.9, radius = 0.55, threshold = 0.18, fov = 38 } = {}) {
    const canvas = document.createElement('canvas');
    canvas.className = 'gl';
    canvas.tabIndex = 0;
    host.prepend(canvas);
    const renderer = new T.WebGLRenderer({ canvas, antialias: true, powerPreference: 'high-performance' });
    renderer.setPixelRatio(DPR);
    renderer.setClearColor(PALETTE.bg, 1);
    renderer.toneMapping = T.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 0.92;
    const scene = new T.Scene();
    scene.fog = new T.FogExp2(PALETTE.bg, 0.0042);
    const camera = new T.PerspectiveCamera(fov, 1, 0.5, 3000);
    const controls = new T.OrbitControls(camera, canvas);
    controls.enableDamping = !reduced;
    controls.dampingFactor = 0.08;
    controls.screenSpacePanning = true;
    const composer = new T.EffectComposer(renderer);
    composer.addPass(new T.RenderPass(scene, camera));
    const bloomPass = new T.UnrealBloomPass(new T.Vector2(256, 256), bloom, radius, threshold);
    composer.addPass(bloomPass);
    composer.addPass(new T.OutputPass());

    function resize() {
      const w = host.clientWidth, h = host.clientHeight;
      renderer.setSize(w, h, false);
      composer.setPixelRatio(DPR);
      composer.setSize(w, h);
      camera.aspect = w / Math.max(1, h);
      camera.updateProjectionMatrix();
    }
    new ResizeObserver(resize).observe(host);
    resize();

    // ?timer drives frames from a timer and ignores visibility: for automated checks of a background tab only.
    const timer = new URLSearchParams(location.search).has('timer');
    const next = timer ? (f) => setTimeout(() => f(performance.now()), 33) : requestAnimationFrame;
    let onScreen = true, running = false, last = performance.now();
    const hooks = [];
    function draw(now) {
      const dt = Math.min(0.1, (now - last) / 1000);
      last = now;
      for (const h of hooks) h(dt, now / 1000);
      controls.update();
      composer.render();
    }
    function frame(now) {
      if (!running) return;
      draw(now);
      next(frame);
    }
    function sync() {
      const want = timer || (onScreen && !document.hidden);
      if (want && !running) { running = true; last = performance.now(); next(frame); }
      if (!want) running = false;
    }
    document.addEventListener('visibilitychange', sync);
    new IntersectionObserver((entries) => { onScreen = entries[0].isIntersecting; sync(); }).observe(canvas);
    sync();

    return { renderer, scene, camera, controls, composer, bloomPass, canvas, host,
             onFrame: (fn) => hooks.push(fn), renderOnce: () => draw(performance.now()),
             get running() { return running; } };
  }

  /* HTML labels pinned to 3D points; positions are refreshed each frame. */
  function labels(host, camera) {
    const layer = document.createElement('div');
    layer.className = 'labels';
    host.appendChild(layer);
    const items = [];
    const v = new T.Vector3();
    return {
      add(text, pos, cls = 'tick') {
        const el = document.createElement('div');
        el.className = cls;
        el.innerHTML = text;
        layer.appendChild(el);
        const item = { el, pos: pos.clone(), visible: true };
        items.push(item);
        return item;
      },
      clear() { items.splice(0).forEach(i => i.el.remove()); },
      update() {
        const w = host.clientWidth, h = host.clientHeight;
        for (const it of items) {
          if (!it.visible) { it.el.style.display = 'none'; continue; }
          v.copy(it.pos).project(camera);
          const behind = v.z > 1 || v.z < -1;
          it.el.style.display = behind ? 'none' : '';
          it.el.style.left = ((v.x + 1) / 2 * w).toFixed(1) + 'px';
          it.el.style.top = ((1 - v.y) / 2 * h).toFixed(1) + 'px';
        }
      },
    };
  }

  /* Background: a field of faint dust points, drifting unless motion is reduced. */
  function dust(scene, count = 1400, spread = 900) {
    const g = new T.BufferGeometry();
    const p = new Float32Array(count * 3), c = new Float32Array(count * 3);
    for (let i = 0; i < count; i++) {
      p[i * 3] = (Math.random() - 0.5) * spread;
      p[i * 3 + 1] = (Math.random() - 0.2) * spread * 0.5;
      p[i * 3 + 2] = (Math.random() - 0.5) * spread;
      const col = rampJS(0.25 + Math.random() * 0.7);
      c.set(col.map(x => x * 0.55), i * 3);
    }
    g.setAttribute('position', new T.BufferAttribute(p, 3));
    g.setAttribute('color', new T.BufferAttribute(c, 3));
    const m = new T.PointsMaterial({ size: 1.6, vertexColors: true, transparent: true, opacity: 0.55,
                                     depthWrite: false, blending: T.AdditiveBlending, sizeAttenuation: true });
    const pts = new T.Points(g, m);
    scene.add(pts);
    return pts;
  }

  /* A glowing floor grid that fades with distance. */
  function floor(scene, size = 600, y = -0.05, color = new T.Color(0x3a4f86)) {
    const m = new T.ShaderMaterial({
      transparent: true, depthWrite: false,
      uniforms: { uColor: { value: color }, uSize: { value: size } },
      vertexShader: `varying vec2 vP; void main(){ vP = position.xy; gl_Position = projectionMatrix * modelViewMatrix * vec4(position,1.0); }`,
      fragmentShader: `varying vec2 vP; uniform vec3 uColor; uniform float uSize;
        float line(float c, float w){ float d = abs(fract(c - 0.5) - 0.5) / fwidth(c); return 1.0 - min(d / w, 1.0); }
        void main(){
          float g = max(line(vP.x / 10.0, 1.0), line(vP.y / 10.0, 1.0)) * 0.35 + max(line(vP.x / 50.0, 1.2), line(vP.y / 50.0, 1.2)) * 0.5;
          float fade = 1.0 - smoothstep(uSize * 0.12, uSize * 0.5, length(vP));
          gl_FragColor = vec4(uColor, g * fade * 0.55);
        }`,
    });
    const mesh = new T.Mesh(new T.PlaneGeometry(size, size), m);
    mesh.rotation.x = -Math.PI / 2;
    mesh.position.y = y;
    scene.add(mesh);
    return mesh;
  }

  function nav(active) {
    const top = document.createElement('div');
    top.className = 'top';
    const tabs = [['focus.html', 'Coming into focus', 'Focus'], ['stack.html', 'From plan to sound', 'Plan to sound'], ['map.html', 'Take map', 'Map']];
    top.innerHTML = `<a class="brand" href="index.html"><span class="mark"></span><span><b>Microscope</b><small>YuE2 · explore</small></span></a>
      <nav class="tabs">${tabs.map(([h, t, s]) => `<a href="${h}" class="${h.startsWith(active) ? 'on' : ''}"><span class="full">${t}</span><span class="short">${s}</span></a>`).join('')}</nav>
      <div class="run" id="runlabel"></div>
      <a class="ghlink" href="https://github.com/jeremy-boschen/audiogen-yue2" target="_blank" rel="noopener" title="Source code on GitHub" aria-label="Source code on GitHub">
        <svg viewBox="0 0 16 16" width="16" height="16" aria-hidden="true"><path fill="currentColor" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"/></svg></a>`;
    document.body.prepend(top);
    return top;
  }

  function tip() {
    const el = document.createElement('div');
    el.className = 'tip';
    document.body.appendChild(el);
    return {
      show(html, x, y) {
        el.innerHTML = html; el.style.display = 'block';
        const r = el.getBoundingClientRect();
        el.style.left = Math.min(window.innerWidth - r.width - 12, x + 16) + 'px';
        el.style.top = Math.min(window.innerHeight - r.height - 12, y + 14) + 'px';
      },
      hide() { el.style.display = 'none'; },
    };
  }

  /* A tiny sparkline: one or two series over steps, with a marker at `at`. */
  function spark(canvas, series, at, colors) {
    const w = canvas.clientWidth, h = canvas.clientHeight;
    if (canvas.width !== Math.round(w * DPR)) { canvas.width = Math.round(w * DPR); canvas.height = Math.round(h * DPR); }
    const g = canvas.getContext('2d');
    g.setTransform(DPR, 0, 0, DPR, 0, 0);
    g.clearRect(0, 0, w, h);
    const n = series[0].length;
    g.strokeStyle = 'rgba(58,78,120,.35)'; g.lineWidth = 1;
    g.beginPath(); g.moveTo(0, h - 0.5); g.lineTo(w, h - 0.5); g.stroke();
    series.forEach((s, k) => {
      g.beginPath();
      s.forEach((v, i) => { const x = i / (n - 1) * w, y = h - 2 - Math.max(0, Math.min(1, v ?? 0)) * (h - 4); i ? g.lineTo(x, y) : g.moveTo(x, y); });
      g.strokeStyle = colors[k]; g.lineWidth = k === series.length - 1 ? 1.8 : 1.2;
      g.shadowColor = colors[k]; g.shadowBlur = k === series.length - 1 ? 6 : 0;
      g.stroke(); g.shadowBlur = 0;
    });
    const x = at / (n - 1) * w;
    g.fillStyle = '#fff'; g.fillRect(x - 0.75, 0, 1.5, h);
  }

  // "How to read this": plain words for people who have never seen a diffusion model. It opens by
  // itself the first time a page is visited (remembered per page) and from the ? in the top bar.
  // Two panels side by side: what you are looking at, then what to press.
  // Whose work this is. YuE2 and every tool below belong to their authors; this explorer only runs them on one take.
  const CREDITS = `<p class="credits"><a href="https://github.com/multimodal-art-projection/YuE" target="_blank" rel="noopener">YuE2</a> is made by the
    <a href="https://huggingface.co/m-a-p" target="_blank" rel="noopener">Multimodal Art Projection (m-a-p)</a> team:
    <a href="https://huggingface.co/m-a-p/YuE2-3B" target="_blank" rel="noopener">model</a>,
    <a href="https://github.com/multimodal-art-projection/YuE/blob/main/docs/technical_report.pdf" target="_blank" rel="noopener">technical report</a>.
    This explorer is an independent project by <a href="https://github.com/jeremy-boschen" target="_blank" rel="noopener">Jeremy Boschen</a>, built with Claude, and not theirs
    (<a href="https://github.com/jeremy-boschen/audiogen-yue2" target="_blank" rel="noopener">source</a>). It uses
    <a href="https://threejs.org" target="_blank" rel="noopener">three.js</a> for the 3D;
    <a href="https://github.com/facebookresearch/demucs" target="_blank" rel="noopener">Demucs</a> to estimate when parts come in;
    <a href="https://github.com/chrisdonahue/sheetsage" target="_blank" rel="noopener">SheetSage</a> for chords;
    and <a href="https://huggingface.co/Qwen/Qwen3-ASR-1.7B" target="_blank" rel="noopener">Qwen3-ASR</a> and the
    <a href="https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B" target="_blank" rel="noopener">Qwen3 forced aligner</a>, run through
    <a href="https://github.com/0xShug0/audio.cpp" target="_blank" rel="noopener">audio.cpp</a>, for the recognized words.</p>`;
  function help(key, about, controls) {
    const card = document.createElement('div');
    card.className = 'helpcard';
    card.setAttribute('role', 'dialog');
    card.setAttribute('aria-label', 'How to read this page');
    card.innerHTML = `<section class="helppanel">${about}${CREDITS}</section>
      <section class="helppanel"><button class="btn helpclose" aria-label="Close">✕</button>${controls}
        <div class="row" style="margin-top:16px"><button class="btn primary helpgo">Got it</button></div></section>`;
    document.body.appendChild(card);
    // The page's key list lives here, not over the landscape.
    const keys = document.querySelector('.keys');
    if (keys) card.querySelector('.helpgo').parentNode.before(keys);
    const seen = 'microscope-help-' + key;
    const show = (on) => { card.classList.toggle('on', on); if (!on) { try { localStorage.setItem(seen, '1'); } catch (e) { /* private mode */ } } };
    card.querySelector('.helpclose').onclick = card.querySelector('.helpgo').onclick = () => show(false);
    window.addEventListener('keydown', (e) => { if (e.key === 'Escape') show(false); if (e.key === '?') show(!card.classList.contains('on')); });
    const button = document.createElement('button');
    button.className = 'btn helpbtn'; button.textContent = '?'; button.title = 'How to read this page (?)';
    button.onclick = () => show(!card.classList.contains('on'));
    (document.querySelector('.top') || document.body).appendChild(button);
    let first = true;
    try { first = !localStorage.getItem(seen); } catch (e) { /* private mode: show it */ }
    if (first) show(true);
  }

  // A glossary for the jargon: <dfn data-g="key">word</dfn> anywhere on a page gets a small card on hover, with a
  // plain meaning first and one line for the technically minded under it. The text stays short; the depth is a hover away.
  const GLOSSARY = {
    step: ['A step', 'The model refines the whole song in 32 small passes, starting from random noise. Each pass is a step; the dial counts the starting noise as step 0.',
      'Flow matching: 32 midpoint ODE steps from t = 1 to t = 0, two network evaluations each (YuE2 technical report, appendix).'],
    ode: ['ODE', 'The recipe the model follows from noise to song: at every step it estimates which way the sound should move, and moves it a little.',
      'Ordinary differential equation. The network predicts a velocity v; this take uses the midpoint method, which checks v halfway through each step. Simplest form: x ← x − Δt·v.'],
    latent: ['Latent', 'The model\'s own compressed sketch of the sound. You can\'t listen to it directly; a decoder turns it into audio.',
      'Continuous VAE latent: 64 channels (not frequency bands) at 25 frames a second, decoded to 48 kHz audio.'],
    decoder: ['Decoder', 'Turns the model\'s sketch of the sound (the latent) into audio you can hear.', 'VAE decoder, latent → waveform.'],
    tokens: ['Semantic tokens', 'A rough draft of the sound as a string of codes, 25 a second, that steers the detailed sound made next. A storyboard before filming. The codes are labels, not amounts.',
      'MERT2 codes from a 32,768-entry codebook, sampled autoregressively; they condition the acoustic stage (YuE2 technical report).'],
    score: ['Score', 'The plan, written as sheet music in text: notes, chords and sections. The lyrics are given separately. The model normally writes it first; this take reused one written earlier.',
      'ABC notation, sampled with the model\'s ordinary text tokens and no grammar constraints; this run loaded it from a saved take.'],
    scoretime: ['Score time', 'Time as the sheet music counts it. The recording usually follows it closely, but it can drift.',
      'Seconds from the ABC score\'s tempo, not measured from the audio.'],
    spectrogram: ['Spectrogram', 'A picture of sound: time runs along it, low to high frequency runs across it, and more energy is taller or brighter.',
      'Summed STFT power in log-spaced bands, on a log scale.'],
    state: ['State', 'The song as it is at this step, leftover noise and all.', 'x_t, the saved latent at this step, decoded.'],
    predicted: ['Predicted final', 'A quick guess at the finished song, made by jumping from this step straight to the end. Later steps can end up somewhere else.',
      'x̂₀ = x_t − t·v, decoded. With z_t = (1−t)z₀ + tε and v = ε − z₀ (report Eq. 6) this is exact for the true velocity; the model\'s estimate makes it a guess.'],
    t: ['t', 'The solver\'s clock: 1 at the start, 0 at the end. It is not a measure of how much noise you can hear.', 'The flow-matching time, 1 → 0 across the 32 steps.'],
    settling: ['Settling wall', 'One bar per frequency band. A bar grows as that band\'s rise and fall across the song looks more like the final step\'s. It does not mean the band is finished.',
      'Whole-song correlation of each band\'s level envelope with step 32\'s. It compares patterns, not exact levels.'],
    nps: ['Notes per syllable', 'Written notes per estimated syllable, per phrase. About 1 is one note per syllable; higher means more notes than syllables. It says nothing about how long a syllable is held.',
      'Score note onsets ÷ a heuristic syllable count of the lyric line; pickup notes folded into a phrase are not counted.'],
  };
  const g = (key, text) => `<dfn data-g="${key}">${text}</dfn>`;
  const gcard = document.createElement('div');
  gcard.className = 'gcard';
  document.addEventListener('DOMContentLoaded', () => document.body.appendChild(gcard));
  document.addEventListener('pointerover', (e) => {
    const el = e.target.closest && e.target.closest('[data-g]');
    if (!el || !GLOSSARY[el.dataset.g]) { gcard.classList.remove('on'); return; }
    const [term, plain, tech] = GLOSSARY[el.dataset.g];
    gcard.innerHTML = `<b>${term}</b> ${plain}${tech ? `<small>${tech}</small>` : ''}`;
    if (!gcard.parentNode) document.body.appendChild(gcard);
    const r = el.getBoundingClientRect(), w = Math.min(300, innerWidth - 24);
    gcard.style.width = `${w}px`;
    gcard.style.left = `${Math.max(12, Math.min(innerWidth - w - 12, r.left + r.width / 2 - w / 2))}px`;
    const below = r.bottom + 10, h = gcard.offsetHeight || 90;
    gcard.style.top = `${below + h > innerHeight - 12 ? r.top - h - 10 : below}px`;
    gcard.classList.add('on');
  });

  // Web build: standard audio by default; "HD audio" in the top bar swaps every file for its HD copy (publish writes
  // the map), remembered across pages. The swap needs a reload, because the focus page downloads everything first.
  const HD_KEY = 'microscope-hd';
  const hdOn = (D) => { try { return !!D.hd && localStorage.getItem(HD_KEY) === '1'; } catch (e) { return false; } };
  const hdSwap = (D, v) => (!hdOn(D) ? v : typeof v === 'string' ? D.hd.map[v] || v
    : Array.isArray(v) ? v.map((x) => hdSwap(D, x)) : v && typeof v === 'object' ? Object.fromEntries(Object.entries(v).map(([k, x]) => [k, hdSwap(D, x)])) : v);
  function hdToggle(D) {
    if (!D.hd) return;
    const run = document.getElementById('runlabel'), on = hdOn(D);
    const b = document.createElement('button');
    b.className = `hdtoggle${on ? ' on' : ''}`; b.innerHTML = 'HD<span class="long"> audio</span>';
    b.title = on ? `Playing HD audio (${D.hd.rate}). Click for standard (${D.hd.standard_rate}), a smaller download. Reloads the page.`
      : `Switch to HD audio (${D.hd.rate}), a bigger download. Reloads the page.`;
    b.onclick = () => { try { localStorage.setItem(HD_KEY, on ? '0' : '1'); } catch (e) { /* private mode */ } location.reload(); };
    run.appendChild(b);
  }
  // Web build: the finished song at HD quality, as a file download (publish writes D.download).
  function download(D) {
    if (!D.download) return;
    const a = document.createElement('a');
    a.className = 'ghlink'; a.href = D.download.url; a.download = D.download.file;
    a.title = `Download the song (${D.download.file}, AAC ${D.download.rate}, ${(D.download.bytes / 1e6).toFixed(1)} MB)`;
    a.setAttribute('aria-label', a.title);
    a.innerHTML = '<svg viewBox="0 0 16 16" width="16" height="16" aria-hidden="true"><path fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" d="M8 2v8M4.5 6.8 8 10.3l3.5-3.5M2.5 13.5h11"/></svg>';
    document.querySelector('.top').appendChild(a);
  }
  window.EX = { download, hdOn, hdSwap, hdToggle, g, GLOSSARY, T, reduced, DPR, PALETTE, GLSL_RAMP, rampJS, css, decode, fmt, num, stage, labels, dust, floor, nav, tip, spark, help };
})();
