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
    const tabs = [['focus.html', 'Coming into focus', 'F'], ['stack.html', 'The stack', 'S'], ['map.html', 'Take map', 'M']];
    top.innerHTML = `<a class="brand" href="index.html"><span class="mark"></span><span><b>Microscope</b><small>YuE2 · explore</small></span></a>
      <nav class="tabs">${tabs.map(([h, t, k]) => `<a href="${h}" class="${h.startsWith(active) ? 'on' : ''}">${t}</a>`).join('')}</nav>
      <div class="run" id="runlabel"></div>`;
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
  function help(key, about, controls) {
    const card = document.createElement('div');
    card.className = 'helpcard';
    card.setAttribute('role', 'dialog');
    card.setAttribute('aria-label', 'How to read this page');
    card.innerHTML = `<section class="helppanel">${about}</section>
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
      'Flow matching: 32 midpoint steps along an ODE from t = 1 to t = 0, two network evaluations each.'],
    ode: ['ODE', 'The recipe the model follows from noise to song: at every step it estimates which way the sound should move, and moves it a little.',
      'Ordinary differential equation. The network predicts a velocity v; this take uses the midpoint method, which checks v halfway through each step. Simplest form: x ← x − Δt·v.'],
    latent: ['Latent', 'The model\'s own compressed sketch of the sound. You can\'t listen to it directly; a decoder turns it into audio.',
      'Continuous VAE latent: 64 channels (not frequency bands) at 25 frames a second, decoded to 48 kHz audio.'],
    decoder: ['Decoder', 'Turns the model\'s sketch of the sound (the latent) into audio you can hear.', 'VAE decoder, latent → waveform.'],
    tokens: ['Semantic tokens', 'A rough draft of the sound as a string of codes, 25 a second, that steers the detailed sound made next. A storyboard before filming. The codes are labels, not amounts.',
      'Discrete categorical codes sampled autoregressively by the language model; they condition the acoustic stage.'],
    score: ['Score', 'The plan, written as sheet music in text: notes, chords and sections. The lyrics are given separately. The model normally writes it first; this take reused one written earlier.',
      'ABC notation from the language model; this run loaded it from a saved take.'],
    scoretime: ['Score time', 'Time as the sheet music counts it. The recording usually follows it closely, but it can drift.',
      'Seconds from the ABC score\'s tempo, not measured from the audio.'],
    spectrogram: ['Spectrogram', 'A picture of sound: time runs along it, low to high frequency runs across it, and more energy is taller or brighter.',
      'Summed STFT power in log-spaced bands, on a log scale.'],
    state: ['State', 'The song as it is at this step, leftover noise and all.', 'x_t, the saved latent at this step, decoded.'],
    predicted: ['Predicted final', 'A quick guess at the finished song, made by jumping from this step straight to the end. Later steps can end up somewhere else.',
      'x̂₀ = x_t − t·v: the current velocity extrapolated to t = 0, decoded.'],
    t: ['t', 'The solver\'s clock: 1 at the start, 0 at the end. It is not a measure of how much noise you can hear.', 'The flow-matching time, 1 → 0 across the 32 steps.'],
    settling: ['Settling wall', 'One bar per frequency band. A bar grows as that band\'s level rises and falls over the song in the same pattern as the finished song.',
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

  window.EX = { g, GLOSSARY, T, reduced, DPR, PALETTE, GLSL_RAMP, rampJS, css, decode, fmt, num, stage, labels, dust, floor, nav, tip, spark, help };
})();
