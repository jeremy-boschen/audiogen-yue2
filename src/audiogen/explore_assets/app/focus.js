/* "Coming into focus": the ODE solve as a morphing spectrogram terrain. */
(function () {
  'use strict';
  const { T, reduced, decode, fmt, num, rampJS, css } = EX;
  const D = window.FOCUS;
  EX.nav('focus');
  EX.help('focus', `
    <h2>Watching a song come into focus</h2>
    <p>This is an exploration of how YuE2, an open AI music model, generates a song. Everything here was recorded from
      one real generation, without changing it.</p>
    <p>YuE2 doesn't record a song from start to finish. It first decides what the song is: the notes,
      the words, the arrangement. Then it makes the sound for the whole song at once, the way a photo develops: it starts
      from pure noise and cleans it up in 32 steps. This page lets you stop at any step, look at it, and listen to it.</p>
    <h3>The landscape</h3>
    <p>It is the sound, drawn as terrain. <b>Left to right</b> is time through the song. <b>Front to back</b> is pitch: bass at
      the front, the highest sounds at the back. <b>Height</b> is how loud that pitch is at that moment. At step 0 it is flat
      static; by step 32 it is the song.</p>
    `, `
    <h3 style="margin-top:0">What to press</h3>
    <dl>
      <dt>▶ and the step slider</dt><dd>Press play, then drag the slider (or use ← →) to move between steps while it plays.
        The sound switches in place.</dd>
      <dt>Resolve</dt><dd>Plays the whole process for you: step 0 to 32, with the sound following. By default it takes one
        pass of the song; drag <b>Resolve pace</b> to slow it down or speed it up while it plays.</dd>
      <dt>I hear…</dt><dd>While it resolves, tap a button over the landscape the moment you hear a voice, the words or the
        chords. Your marks stay in this browser.</dd>
      <dt>State · Predicted final · Finished</dt><dd><b>State</b> is the song exactly as it is at this step: still partly
        noise until the last few steps. <b>Predicted final</b> is where the model thinks it is heading from here, often
        recognizable surprisingly early. <b>Finished</b> is the end result.</dd>
      <dt>Ghost overlay</dt><dd>A see-through copy of the other view floating above: in State, where it is heading; in
        Predicted final, where it actually is. The gap is how far it still has to go.</dd>
      <dt>Marks on the dial</dt><dd>Where something first became hearable: the beat, a voice, the words. <b>Gold dots</b>
        are yours: press <kbd>M</kbd> while it plays and tap what you can hear. <b>Blue rings</b> are machines: a speech recognizer, a voice detector and a music
        transcriber, each comparing a step with its own reading of the finished song. Machines and ears often disagree;
        both are shown, side by side, in the list below the dial.</dd>
      <dt>Settling color</dt><dd>Color means that part of the sound already matches the finished song; gray and glittering means it
        is still forming. The low end and the beat tend to lock in first, fine detail last.</dd>
    </dl>
    <p style="color:var(--muted)">Drag to turn the view, scroll to zoom, click the land to jump to that moment. Press
      <kbd>?</kbd> to bring this back. The numbers in the side panel measure the signal; they are not verdicts on how it sounds.</p>`);
  document.getElementById('runlabel').innerHTML = `<b>${D.run}</b> · ${D.seconds.toFixed(0)} s · ${D.steps} steps`;

  const C = D.columns, B = D.bands, S = D.steps, NS = S + 1;
  const W = 150, DEPTH = 52, H = 17;
  const state = decode(D.state, Uint8Array), pred = decode(D.predicted, Uint8Array);
  const settleS = decode(D.settle_state, Uint8Array), settleP = decode(D.settle_predicted, Uint8Array);

  // Band depth positions from log-center frequency: low at the front.
  const lf = (f) => Math.log(f / 40) / Math.log(16000 / 40);
  const bandZ = D.band_lo_hz.map((lo, b) => (0.5 - lf(Math.sqrt(lo * D.band_hi_hz[b]))) * DEPTH);
  const zOfHz = (f) => (0.5 - lf(f)) * DEPTH;
  const xOfSec = (s) => (s / D.seconds - 0.5) * W;

  const st = EX.stage(document.getElementById('stage'), { bloom: 0.45, radius: 0.45, threshold: 0.6 });
  const { scene, camera, controls } = st;
  const home = { pos: new T.Vector3(-96, 72, 128), target: new T.Vector3(6, 2, -4) };
  camera.position.copy(home.pos); controls.target.copy(home.target);
  controls.minDistance = 30; controls.maxDistance = 420; controls.maxPolarAngle = Math.PI * 0.49;
  EX.dust(scene);
  EX.floor(scene, 700, -0.2);

  // --- textures ---------------------------------------------------------------
  function arrayTex(data) {
    const t = new T.DataArrayTexture(data, C, B, NS);
    t.format = T.RedFormat; t.type = T.UnsignedByteType;
    t.minFilter = T.LinearFilter; t.magFilter = T.LinearFilter; t.unpackAlignment = 1; t.needsUpdate = true;
    return t;
  }
  function settleTex(data) {
    const t = new T.DataTexture(data, B, NS, T.RedFormat, T.UnsignedByteType);
    t.minFilter = T.LinearFilter; t.magFilter = T.LinearFilter; t.unpackAlignment = 1; t.needsUpdate = true;
    return t;
  }
  const uniforms = {
    uState: { value: arrayTex(state) }, uPred: { value: arrayTex(pred) },
    uSetS: { value: settleTex(settleS) }, uSetP: { value: settleTex(settleP) },
    uStep: { value: 0 }, uSteps: { value: S }, uMix: { value: 0 }, uFin: { value: 0 },
    uH: { value: H }, uTime: { value: 0 }, uPlayX: { value: -W / 2 }, uSettleOn: { value: 1 },
    uTexel: { value: new T.Vector2(1 / C, 1 / B) }, uBg: { value: new T.Color(EX.PALETTE.bg) },
    uHoverZ: { value: -999 }, uLift: { value: 0 }, uGhostMix: { value: 1 }, uAlpha: { value: 1 },
  };

  const VERT_COMMON = `
    precision highp sampler2DArray;
    uniform sampler2DArray uState, uPred;
    uniform sampler2D uSetS, uSetP;
    uniform float uStep, uSteps, uMix, uFin, uH, uLift, uGhostMix;
    uniform vec2 uTexel;
    attribute vec2 aUV;
    varying float vH; varying float vSettle; varying vec3 vN; varying vec3 vW; varying float vDist;
    float lvl(sampler2DArray t, vec2 uv, float s0, float s1, float f) {
      return mix(texture(t, vec3(uv, s0)).r, texture(t, vec3(uv, s1)).r, f);
    }
    float hAt(vec2 uv, float mixv) {
      float s0 = floor(uStep), s1 = min(s0 + 1.0, uSteps), f = uStep - s0;
      float a = lvl(uState, uv, s0, s1, f);
      float b = lvl(uPred, uv, s0, s1, f);
      float fin = texture(uState, vec3(uv, uSteps)).r;
      return mix(mix(a, b, mixv), fin, uFin);
    }
    float lev(float h) { return clamp((h - 0.28) / 0.7, 0.0, 1.0); }
    float shape(float h) { return pow(lev(h), 1.5) * uH; }
    void terrain(float mixv) {
      float h = hAt(aUV, mixv);
      float hx = shape(hAt(aUV + vec2(uTexel.x, 0.0), mixv)) - shape(hAt(aUV - vec2(uTexel.x, 0.0), mixv));
      float hz = shape(hAt(aUV + vec2(0.0, uTexel.y), mixv)) - shape(hAt(aUV - vec2(0.0, uTexel.y), mixv));
      vN = normalize(vec3(-hx * 2.0, 1.6, hz * 0.9));
      vH = lev(h);
      float sv = (uStep + 0.5) / (uSteps + 1.0);
      vSettle = mix(mix(texture(uSetS, vec2(aUV.y, sv)).r, texture(uSetP, vec2(aUV.y, sv)).r, mixv), 1.0, uFin);
      vec3 p = position; p.y = shape(h) + uLift;
      vec4 w = modelMatrix * vec4(p, 1.0); vW = w.xyz;
      vec4 mv = viewMatrix * w; vDist = length(mv.xyz);
      gl_Position = projectionMatrix * mv;
    }`;
  const FRAG_COMMON = `
    uniform float uTime, uPlayX, uSettleOn, uHoverZ, uAlpha; uniform vec3 uBg;
    varying float vH; varying float vSettle; varying vec3 vN; varying vec3 vW; varying float vDist;
    ${EX.GLSL_RAMP}
    float hash(vec2 p) { return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453); }
    vec3 shade(bool lines) {
      vec3 base = ramp(0.05 + pow(vH, 1.15) * 0.95);
      float settled = mix(1.0, smoothstep(0.6, 0.98, vSettle), uSettleOn);   // most bands pass 0.6 by step 12: the ramp starts there so step 16-24 still shows what is forming
      float lum = dot(base, vec3(0.3, 0.5, 0.2));
      vec3 cool = vec3(lum) * vec3(0.55, 0.66, 1.05) * 0.85;
      float glit = step(0.985, hash(floor(vW.xz * 2.2) + floor(uTime * 9.0))) * (1.0 - settled) * 0.9;
      vec3 col = mix(cool, base, settled) + glit * vec3(0.7, 0.8, 1.0);
      if (!lines) {
        vec3 L = normalize(vec3(-0.35, 0.85, 0.45));
        float diff = max(dot(normalize(vN), L), 0.0);
        col *= 0.32 + diff * 0.62;
        float c = vH * 16.0; float iso = abs(fract(c) - 0.5) / max(fwidth(c), 1e-4);
        col += (1.0 - min(iso, 1.0)) * 0.12 * base * (0.4 + settled);
        col += base * pow(vH, 4.0) * 0.35 * settled;
      }
      float d = vW.x - uPlayX;
      col += vec3(1.0, 0.86, 1.0) * exp(-d * d * 1.6) * 1.35;
      col += base * exp(-abs(d) * 0.18) * 0.22 * step(d, 0.0);
      float hz = vW.z - uHoverZ; col += vec3(0.5, 0.7, 1.0) * exp(-hz * hz * 3.0) * 0.35;
      float fog = 1.0 - exp(-pow(vDist * 0.0036, 2.0));
      return mix(col, uBg, fog);
    }`;

  // --- terrain mesh --------------------------------------------------------------
  const positions = new Float32Array(C * B * 3), uvs = new Float32Array(C * B * 2);
  for (let b = 0; b < B; b++) for (let c = 0; c < C; c++) {
    const i = b * C + c;
    positions[i * 3] = (c / (C - 1) - 0.5) * W; positions[i * 3 + 2] = bandZ[b];
    uvs[i * 2] = (c + 0.5) / C; uvs[i * 2 + 1] = (b + 0.5) / B;
  }
  const idx = [];
  for (let b = 0; b < B - 1; b++) for (let c = 0; c < C - 1; c++) {
    const i = b * C + c;
    idx.push(i, i + C, i + 1, i + 1, i + C, i + C + 1);
  }
  const geo = new T.BufferGeometry();
  geo.setAttribute('position', new T.BufferAttribute(positions, 3));
  geo.setAttribute('aUV', new T.BufferAttribute(uvs, 2));
  geo.setIndex(idx);
  geo.boundingSphere = new T.Sphere(new T.Vector3(0, H / 2, 0), W);
  const terrainMat = new T.ShaderMaterial({
    uniforms, side: T.DoubleSide,
    vertexShader: VERT_COMMON + `void main(){ terrain(uMix); }`,
    fragmentShader: FRAG_COMMON + `void main(){ gl_FragColor = vec4(shade(false), 1.0); }`,
  });
  const terrain = new T.Mesh(geo, terrainMat);
  scene.add(terrain);

  // Ghost: the counterpart (predicted over state, state over predicted) as glowing contour lines per band.
  const lineIdx = [];
  for (let b = 0; b < B; b += 2) for (let c = 0; c < C - 1; c++) lineIdx.push(b * C + c, b * C + c + 1);
  const ghostGeo = new T.BufferGeometry();
  ghostGeo.setAttribute('position', geo.getAttribute('position'));
  ghostGeo.setAttribute('aUV', geo.getAttribute('aUV'));
  ghostGeo.setIndex(lineIdx);
  ghostGeo.boundingSphere = geo.boundingSphere;
  const ghostUniforms = Object.assign({}, uniforms, { uLift: { value: 7 }, uGhostMix: { value: 1 }, uAlpha: { value: 0.5 }, uFin: { value: 0 } });
  const ghost = new T.LineSegments(ghostGeo, new T.ShaderMaterial({
    uniforms: ghostUniforms, transparent: true, depthWrite: false, blending: T.AdditiveBlending,
    vertexShader: VERT_COMMON + `void main(){ terrain(uGhostMix); }`,
    fragmentShader: FRAG_COMMON + `void main(){ vec3 c = shade(true); gl_FragColor = vec4(c * 0.9 + vec3(0.08,0.05,0.14), uAlpha * (0.35 + vH * 0.9)); }`,
  }));
  scene.add(ghost);

  // Playhead: a glowing vertical plane of focus.
  const planeMat = new T.ShaderMaterial({
    transparent: true, depthWrite: false, blending: T.AdditiveBlending, side: T.DoubleSide,
    uniforms: { uTime: uniforms.uTime },
    vertexShader: `varying vec2 vUv; void main(){ vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position,1.0); }`,
    fragmentShader: `varying vec2 vUv; uniform float uTime;
      void main(){
        float edge = smoothstep(0.035, 0.0, vUv.x) + smoothstep(0.965, 1.0, vUv.x);
        float top = smoothstep(0.97, 1.0, vUv.y);
        float body = (1.0 - vUv.y) * 0.22 + 0.04;
        float scan = 0.5 + 0.5 * sin(vUv.y * 60.0 - uTime * 4.0);
        vec3 col = mix(vec3(0.62, 0.42, 1.0), vec3(1.0, 0.45, 0.75), vUv.x);
        gl_FragColor = vec4(col * (body + edge * 0.6 + top * 0.8 + scan * 0.03), 1.0);
      }`,
  });
  const plane = new T.Mesh(new T.PlaneGeometry(DEPTH + 8, H * 1.55), planeMat);
  plane.rotation.y = Math.PI / 2;
  plane.position.y = H * 1.55 / 2 - 0.3;
  scene.add(plane);
  const beamGeo = new T.BufferGeometry().setFromPoints([new T.Vector3(0, 0, -DEPTH / 2 - 4), new T.Vector3(0, 0, DEPTH / 2 + 4)]);
  const beam = new T.Line(beamGeo, new T.LineBasicMaterial({ color: 0xffe0f4, transparent: true, opacity: 0.9, blending: T.AdditiveBlending }));
  beam.position.y = H * 1.55 - 0.3;
  scene.add(beam);

  // Settling wall: one bar per band at the left edge, height = envelope correlation at this step.
  const wallGeo = new T.BoxGeometry(1, 1, 1); wallGeo.translate(0, 0.5, 0);
  const wall = new T.InstancedMesh(wallGeo, new T.MeshBasicMaterial({ toneMapped: false }), B);
  scene.add(wall);
  const wallX = -W / 2 - 6, m4 = new T.Matrix4(), col = new T.Color();
  const bandGap = DEPTH / B * 0.8;

  // Section bands on the floor, in score time, just in front of the terrain.
  const secGroup = new T.Group(); scene.add(secGroup);
  (D.sections || []).forEach((s, i) => {
    if (s.start >= D.seconds) return;
    const x0 = xOfSec(s.start), x1 = xOfSec(Math.min(s.end, D.seconds));
    const c = rampJS(0.3 + 0.6 * ((i * 0.37) % 1));
    const m = new T.Mesh(new T.PlaneGeometry(Math.max(0.1, x1 - x0 - 0.4), 2.2),
      new T.MeshBasicMaterial({ color: new T.Color(...c), transparent: true, opacity: 0.55, toneMapped: false }));
    m.rotation.x = -Math.PI / 2; m.position.set((x0 + x1) / 2, 0.02, DEPTH / 2 + 5);
    secGroup.add(m);
  });

  // --- labels ---------------------------------------------------------------------
  const L = EX.labels(st.host, camera);
  for (let s = 0; s <= D.seconds + 0.01; s += 20) L.add(`${s}s`, new T.Vector3(xOfSec(s), 0, DEPTH / 2 + 9));
  L.add('time →', new T.Vector3(W / 2 + 10, 0, DEPTH / 2 + 9), 'tick big');
  for (const [f, t] of [[100, '100 Hz'], [300, '300'], [1000, '1 kHz'], [3000, '3 kHz'], [10000, '10 kHz']]) {
    L.add(t, new T.Vector3(W / 2 + 7, 0, zOfHz(f)));
  }
  L.add('settling wall', new T.Vector3(wallX, 15, DEPTH / 2 + 2), 'tick big');
  (D.sections || []).forEach((s) => { if (s.start < D.seconds) L.add(s.label, new T.Vector3(xOfSec(s.start) + 1, 0, DEPTH / 2 + 5), 'tick'); });
  const secNote = L.add('sections: score time', new T.Vector3(-W / 2 - 14, 0, DEPTH / 2 + 5), 'tick');

  // --- state ------------------------------------------------------------------------
  // Resolve's pace, seconds per step while the song plays. By default the 32 steps take one pass of the song.
  const PACE_DEFAULT = Math.round(D.seconds / D.steps * 10) / 10;
  const ui = {
    pace: PACE_DEFAULT,
    target: 0, shown: 0, mode: 0, ghost: true, settle: true, playing: false,
    mixT: 0, finT: 0, animating: false,
  };

  // --- audio: Web Audio, every layer scheduled on one clock -----------------------------
  // The compressed files are fetched up front (the current step first) and decoded on demand,
  // a few at a time. Every layer is an AudioBufferSourceNode started at the same context time
  // and offset, so a step change is a crossfade between buffers that are already in step: no
  // media elements to keep in lockstep, and gain nodes work where element volume does not (iOS).
  const audio = window.focusAudio = (() => {
    const Ctx = window.AudioContext || window.webkitAudioContext;
    const ctx = new Ctx();
    const FADE = 0.06, KEEP = 8;
    const files = [D.audio.finished, ...D.audio.state.flatMap((s, i) => [s, D.audio.predicted[i]])];
    const unique = [...new Set(files)];
    const bytes = new Map(), buffers = new Map();
    let fetched = 0, failed = null;
    // The web build ships AAC (~2 MB a step), kept after download so any step decodes at once. The
    // local build points at the runs' WAV listening copies (~35 MB each): fetched when needed and
    // let go once decoded, never all held at once.
    const compressed = unique.every((u) => /\.m4a$/.test(u));

    function load(url) {
      if (!bytes.has(url)) {
        bytes.set(url, fetch(url).then((r) => { if (!r.ok) throw new Error(`${r.status} ${url}`); return r.arrayBuffer(); })
          .then((b) => { fetched += 1; status(); return b; }, (e) => { failed = e; status(); throw e; }));
      }
      return bytes.get(url);
    }
    // Decoding is queued here, two at a time, because Safari decodes one file at a time and
    // slowly: left to itself, a ramp through 32 steps queued ~100 decodes and the step it
    // stopped on waited behind all of them. A queued decode nobody still wants is dropped.
    const queue = [];
    let decoding = 0;
    function pump() {
      while (decoding < 2 && queue.length) {
        const job = queue.shift();
        decoding += 1;
        load(job.url).then((b) => ctx.decodeAudioData(b.slice(0)))    // slice: decoding detaches its input
          .then((buffer) => { if (!compressed) bytes.delete(job.url); return buffer; })
          .then(job.resolve, job.reject)
          .finally(() => { decoding -= 1; pump(); });
      }
    }
    // Most recently used last; beyond KEEP the oldest decoded buffers are let go (~36 MB each).
    function decoded(url) {
      let job = buffers.get(url);
      if (job) buffers.delete(url);
      else {
        job = { url };
        job.promise = new Promise((resolve, reject) => { job.resolve = resolve; job.reject = reject; });
        job.promise.catch(() => null);
        queue.push(job);
        pump();
      }
      buffers.set(url, job);
      for (const [key, old] of buffers) {
        if (buffers.size <= KEEP) break;
        if (!queue.includes(old)) buffers.delete(key);
      }
      return job.promise;
    }
    function prune(keep) {
      for (let i = queue.length - 1; i >= 0; i--) {
        if (keep.has(queue[i].url)) continue;
        const [job] = queue.splice(i, 1);
        buffers.delete(job.url);
        job.reject(new Error('superseded'));
      }
    }
    // Everything is fetched in the background, nearest the current step first, four at a time.
    function preload() {
      const s = Math.round(ui.target);
      const order = [...unique].sort((x, y) => distance(x, s) - distance(y, s));
      let next = 0;
      const worker = () => (next < order.length ? load(order[next++]).catch(() => null).then(worker) : null);
      for (let i = 0; i < 4; i++) worker();
    }
    function distance(url, s) {
      if (url === D.audio.finished) return -1;
      const i = D.audio.state.indexOf(url), j = D.audio.predicted.indexOf(url);
      return Math.abs((i >= 0 ? i : j) - s);
    }
    function status() {
      const el = document.getElementById('audiostatus');
      if (!el) return;
      el.textContent = failed ? `Audio failed to load (${failed.message})`
        : compressed && fetched < unique.length ? `Loading audio ${fetched} of ${unique.length}` : '';
    }

    // Playback: `origin` is the context time at which the song's 0 s would have played.
    let origin = 0, startAt = 0, playing = false, want = 0, layers = null, loadedStep = -1;
    const weights = () => [ui.mode === 0 ? 1 : 0, ui.mode === 1 ? 1 : 0, ui.mode === 2 ? 1 : 0];
    const time = () => (playing ? Math.min(D.seconds, ctx.currentTime - origin) : startAt);

    function voice(buffer, weight, when, offset) {
      const gain = ctx.createGain();
      gain.gain.setValueAtTime(0, when);
      gain.gain.linearRampToValueAtTime(weight, when + FADE);
      gain.connect(ctx.destination);
      const src = ctx.createBufferSource();
      src.buffer = buffer; src.connect(gain); src.start(when, Math.max(0, offset));
      return { src, gain };
    }
    function release(set, when) {
      (set || []).forEach(({ src, gain }) => {
        gain.gain.cancelScheduledValues(when);
        gain.gain.setValueAtTime(gain.gain.value, when);
        gain.gain.linearRampToValueAtTime(0, when + FADE);
        src.stop(when + FADE + 0.01);
      });
    }
    // Sound step `s` from the playhead, fading out whatever was sounding. Latest request wins.
    function sound(s) {
      want = s;
      const urls = [D.audio.state[s], D.audio.predicted[s], D.audio.finished];
      const near = [s - 1, s + 1].filter((n) => n >= 0 && n <= D.steps).flatMap((n) => [D.audio.state[n], D.audio.predicted[n]]);
      prune(new Set([...urls, ...near]));
      const ready = Promise.all(urls.map(decoded));
      // Then the steps either side, so a ramp finds its next step already decoded.
      near.forEach(decoded);
      return ready.then((bufs) => {
        if (!playing || want !== s) return;
        const when = ctx.currentTime + 0.03, w = weights();
        const next = bufs.map((b, i) => ({ ...voice(b, w[i], when, when - origin), url: urls[i], weight: w[i] }));
        release(layers, when);
        layers = next; loadedStep = s;
      }).catch(() => null);
    }
    function start() {
      if (location.protocol === 'file:') { failed = new Error('open this page through a web server'); status(); return; }
      ctx.resume();
      if (startAt >= D.seconds - 0.05) startAt = 0;
      origin = ctx.currentTime + 0.05 - startAt;
      playing = true; ui.playing = true;
      sound(Math.round(ui.target));
    }
    function pause() {
      startAt = time();
      release(layers, ctx.currentTime); layers = null; loadedStep = -1;
      playing = false; ui.playing = false;
    }
    function seek(t) {
      startAt = Math.max(0, Math.min(D.seconds - 0.05, t));
      if (playing) { release(layers, ctx.currentTime); layers = null; origin = ctx.currentTime + 0.05 - startAt; sound(Math.round(ui.target)); }
    }
    function stepChanged() { if (playing) sound(Math.round(ui.target)); }
    let lastMode = ui.mode;
    function tick() {
      if (layers && ui.mode !== lastMode) {
        const now = ctx.currentTime, w = weights();
        layers.forEach((layer, i) => { layer.weight = w[i]; });
        layers.forEach(({ gain }, i) => { gain.gain.cancelScheduledValues(now); gain.gain.setValueAtTime(gain.gain.value, now); gain.gain.linearRampToValueAtTime(w[i], now + FADE); });
      }
      lastMode = ui.mode;
      if (playing && time() >= D.seconds) { pause(); startAt = 0; setPlayIcon(); }
    }
    status();
    if (compressed) preload();
    return { start, pause, seek, stepChanged, tick, time, get loadedStep() { return loadedStep; },
             // What is audible now: the files with a non-zero gain (for tests and the curious).
             get sounding() { return (layers || []).filter((l) => l.weight > 0).map((l) => l.url); } };
  })();

  // --- panel ---------------------------------------------------------------------------
  const $ = (id) => document.getElementById(id);
  const MODES = ['ODE state', 'predicted final', 'finished take'];
  $('datanote').textContent = `${D.bands_note}. Audio: ${D.audio_note}.`;

  // ring
  const ring = $('ring');
  const A0 = -150, A1 = 150, R = 58, CX = 75, CY = 75;
  const ang = (s) => (A0 + (A1 - A0) * s / S) * Math.PI / 180;
  const pt = (a, r) => [CX + r * Math.sin(a), CY - r * Math.cos(a)];
  let ringHTML = `<defs><linearGradient id="rg" x1="0" x2="1"><stop offset="0" stop-color="#4f8cff"/><stop offset=".45" stop-color="#9b6bff"/><stop offset=".75" stop-color="#ff5fb4"/><stop offset="1" stop-color="#ff9a4d"/></linearGradient>
    <filter id="glow"><feGaussianBlur stdDeviation="2.4" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter></defs>`;
  for (let s = 0; s <= S; s++) {
    const [x0, y0] = pt(ang(s), R + 6), [x1, y1] = pt(ang(s), R + (s % 4 ? 10 : 14));
    ringHTML += `<line x1="${x0}" y1="${y0}" x2="${x1}" y2="${y1}" stroke="#3a4a70" stroke-width="${s % 4 ? 1 : 1.6}" data-t="${s}"/>`;
  }
  const arcPath = (s) => { const [x0, y0] = pt(ang(0), R), [x1, y1] = pt(ang(s), R); const large = (A1 - A0) * s / S > 180 ? 1 : 0; return `M${x0},${y0} A${R},${R} 0 ${large} 1 ${x1},${y1}`; };
  ringHTML += `<path d="${arcPath(S)}" stroke="#1a2640" stroke-width="7" fill="none" stroke-linecap="round"/>
    <path id="arc" d="" stroke="url(#rg)" stroke-width="7" fill="none" stroke-linecap="round" filter="url(#glow)"/>
    <circle id="knob" r="8" fill="#fff" filter="url(#glow)"/>
    <text x="75" y="72" text-anchor="middle" fill="#fff" font-size="30" font-weight="600" id="ringnum">0</text>
    <text x="75" y="92" text-anchor="middle" fill="#8d9ab3" font-size="10" letter-spacing="2">OF ${S}</text>`;
  ring.innerHTML = ringHTML;
  const ringTicks = [...ring.querySelectorAll('line')];
  function ringFrom(e) {
    const r = ring.getBoundingClientRect();
    const x = (e.clientX - r.left) / r.width * 150 - CX, y = (e.clientY - r.top) / r.height * 150 - CY;
    let a = Math.atan2(x, -y) * 180 / Math.PI;
    a = Math.max(A0, Math.min(A1, a));
    setStep(Math.round((a - A0) / (A1 - A0) * S));
  }
  ring.addEventListener('pointerdown', (e) => { ring.setPointerCapture(e.pointerId); ringFrom(e); });
  ring.addEventListener('pointermove', (e) => { if (e.buttons) ringFrom(e); });

  // --- listening marks: the first step at which a listener heard each thing ------------------
  // Only a person's ears fill these in (the run's analysis/annotations.json); nothing is measured.
  // Each gets a pulsing dot inside the dial and a row below it. Served locally, M marks a step.
  const HEARD = D.listening || { fields: [], state: {}, predicted: {} };
  // Served locally, marks go to the run on disk. Anywhere else, the published
  // demo included, a visitor's marks stay in their own browser and nowhere else.
  const LOCAL = Boolean(D.annotate) && location.protocol !== 'file:';
  const STORE = `microscope-marks-${D.run}`;
  if (!LOCAL) {
    let mine = {};
    try { mine = JSON.parse(localStorage.getItem(STORE) || '{}'); } catch (e) { mine = {}; }
    HEARD.state = mine.state || {}; HEARD.predicted = mine.predicted || {};
  }
  const LABEL = Object.fromEntries(HEARD.fields);
  const viewKey = () => (ui.mode === 1 ? 'predicted' : 'state');
  function firstHeard(view) {
    const first = {};
    Object.keys(HEARD[view] || {}).map(Number).sort((a, b) => a - b).forEach((s) => {
      Object.entries(HEARD[view][s]).forEach(([f, v]) => { if (v === true && first[f] === undefined) first[f] = s; });
    });
    const byStep = {};
    Object.entries(first).forEach(([f, s]) => (byStep[s] = byStep[s] || []).push(LABEL[f] || f));
    return byStep;
  }
  const markLayer = document.createElementNS('http://www.w3.org/2000/svg', 'g');
  ring.appendChild(markLayer);
  // Machine listeners' first steps (bin/machine_listen.py), shown apart from a listener's: hollow rings, "machine" rows.
  const MACHINE = D.machine;
  function machineFirst(view) {
    const byStep = {};
    Object.entries((MACHINE && MACHINE[view]) || {}).forEach(([f, s]) => (byStep[s] = byStep[s] || []).push(f));
    return byStep;
  }
  function drawMarks() {
    const view = viewKey(), byStep = firstHeard(view), byMachine = machineFirst(view), now = Math.round(ui.target);
    const reached = (s) => (+s <= now ? ' reached' : '');
    markLayer.innerHTML = Object.entries(byStep).map(([s, names]) => {
      const [x, y] = pt(ang(+s), R - 13);
      return `<g class="heardmark${reached(s)}"><circle cx="${x}" cy="${y}" r="3.2"/><title>Step ${s}: ${names.join(', ')}</title></g>`;
    }).join('') + Object.entries(byMachine).map(([s, fs]) => {
      const [x, y] = pt(ang(+s), R - 22);
      return `<g class="machinemark${reached(s)}"><circle cx="${x}" cy="${y}" r="3"/><title>Step ${s}, a machine: ${fs.map((f) => LABEL[f] || f).join(', ')}</title></g>`;
    }).join('');
    const mine = {}, theirs = (MACHINE && MACHINE[view]) || {};
    Object.entries(byStep).forEach(([st, names]) => names.forEach((n) => (mine[n] = +st)));
    const fields = HEARD.fields.filter(([f, label]) => mine[label] !== undefined || theirs[f] !== undefined);
    const cell = (who, st, tip) => st === undefined ? `<span class="who ${who} none">${who} –</span>`
      : `<span class="who ${who}${reached(st)}" title="${tip}">${who} ${st}</span>`;   // a label, not a control: a click used to jump the step
    const gap = (a, b) => a === undefined || b === undefined ? '' : a < b ? `${b - a} steps sooner` : a > b ? `${a - b} steps later` : 'same step';
    const empty = canMark ? `<div class="dim">${MACHINE ? 'Blue rings are where machines first picked something out. Can you hear it sooner? ' : ''}Press Resolve and tap the "I hear" buttons over the landscape, or mark more here.</div>` : '';
    $('heard').innerHTML = fields.length
      ? `<div class="dim">First heard</div>` + fields.sort(([fa, la], [fb, lb]) =>
          Math.min(mine[la] ?? 99, theirs[fa] ?? 99) - Math.min(mine[lb] ?? 99, theirs[fb] ?? 99)).map(([f, label]) =>
          `<div class="heardrow"><span>${label}</span>${cell('you', mine[label], 'Where you first heard it')}${cell('machine', theirs[f],
            MACHINE && MACHINE.says[f] ? MACHINE.says[f].replace(/"/g, '&quot;') : '')}<small>${gap(mine[label], theirs[f])}</small></div>`).join('')
        + (Object.keys(mine).length ? '' : empty)
      : empty;
  }
  const canMark = LOCAL || (() => { try { return !!window.localStorage; } catch (e) { return false; } })();
  let marking = false;
  function drawMarker() {
    const box = $('marker');
    box.hidden = !marking;
    if (!marking) return;
    if (ui.mode === 2) { box.innerHTML = '<div class="dim">Switch to State or Predicted final to mark a step.</div>'; return; }
    const s = Math.round(ui.target), view = viewKey(), now = (HEARD[view] || {})[s] || {};
    box.innerHTML = `<div class="dim">Step ${s}, ${view === 'state' ? 'State' : 'Predicted final'}: tap what you can hear.${LOCAL ? '' : ' Saved in this browser only.'}</div>`
      + (LOCAL || !(Object.keys(HEARD.state).length || Object.keys(HEARD.predicted).length) ? '' : '<button class="linkbtn" id="clearmarks">Clear my marks</button>')
      + HEARD.fields.map(([f, label]) => `<button class="markchip${now[f] === true ? ' on' : ''}" data-f="${f}">${label}</button>`).join('');
    const clear = box.querySelector('#clearmarks');
    if (clear) clear.onclick = () => { HEARD.state = {}; HEARD.predicted = {}; localStorage.removeItem(STORE); drawMarks(); drawMarker(); drawGame(); };
    box.querySelectorAll('.markchip').forEach((b) => (b.onclick = () => mark(view, s, b.dataset.f, now[b.dataset.f] === true ? null : true)));
  }
  function mark(view, step, field, value) {
    if (!LOCAL) {
      const steps = HEARD[view];
      steps[step] = { ...(steps[step] || {}), [field]: value };
      if (value === null) delete steps[step][field];
      if (!Object.keys(steps[step]).length) delete steps[step];
      localStorage.setItem(STORE, JSON.stringify({ state: HEARD.state, predicted: HEARD.predicted }));
      drawMarks(); drawMarker(); drawGame(); return;
    }
    fetch('/api/annotate', { method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ run: D.annotate, view, step, field, value }) })
      .then((r) => r.json().then((j) => { if (!r.ok) throw new Error(j.error || r.status); return j; }))
      .then((j) => { HEARD.state = j.state; HEARD.predicted = j.predicted; drawMarks(); drawMarker(); drawGame(); })
      .catch((e) => $('marker').insertAdjacentHTML('beforeend', `<div class="dim">Not saved: ${e.message}</div>`));
  }
  // The listening game: big "I hear..." buttons over the landscape for the things the machines mark, so a
  // visitor races them without finding a key. A tap marks the step on screen as the first time they heard it.
  const GAME = [['vocal_present', 'a voice'], ['words_partially_intelligible', 'some words'],
                ['words_intelligible', 'all the words'], ['harmony_recognizable', 'the chords']];
  function firstMine(view, field) {
    const steps = Object.keys(HEARD[view] || {}).map(Number).sort((a, b) => a - b);
    return steps.find((st) => HEARD[view][st][field] === true);
  }
  function claim(view, field, step) {                 // this step becomes the first; earlier and later taps of it go
    Object.keys(HEARD[view] || {}).map(Number).forEach((st) => {
      if (st !== step && HEARD[view][st][field] === true) mark(view, st, field, null);
    });
    mark(view, step, field, true);
  }
  function drawGame() {
    const box = $('game');
    box.hidden = !canMark || ui.mode === 2;
    if (box.hidden) return;
    const view = viewKey(), s = Math.round(ui.target), theirs = (MACHINE && MACHINE[view]) || {};
    const done = GAME.filter(([f]) => firstMine(view, f) !== undefined).length;
    $('gamecue').innerHTML = done === GAME.length
      ? 'All four marked. Compare with the machines in the panel, or switch view and try again.'
      : ui.playing || ui.animating
        ? 'Tap the moment you hear it'
        : `Press <b>Resolve</b> to hear the song come out of the noise, and tap the moment you hear each of these.${MACHINE ? ' Can you beat the machines?' : ''}`;
    $('gamepills').innerHTML = GAME.map(([f, what]) => {
      const mine = firstMine(view, f), them = theirs[f];
      if (mine === undefined) return `<button class="gamepill" data-f="${f}">I hear ${what}</button>`;
      const vs = them === undefined ? '' : mine < them ? ` · ${them - mine} before the machine` : mine > them ? ` · ${mine - them} after the machine` : ' · same as the machine';
      return `<button class="gamepill got" data-f="${f}" title="Tap to move it to step ${s}">${what} <b>step ${mine}</b>${vs}</button>`;
    }).join('');
    $('gamepills').querySelectorAll('.gamepill').forEach((b) => (b.onclick = () => {
      claim(view, b.dataset.f, Math.round(ui.target));
      b.classList.add('pop');
    }));
  }
  $('markbtn').hidden = !canMark;
  $('markbtn').onclick = () => { marking = !marking; $('markbtn').classList.toggle('on', marking); drawMarker(); };

  // metrics
  const series = (set, fn) => D.metrics.map((m) => fn(m[set] || {}));
  const meanSettle = (arr, s) => { let t = 0; for (let b = 0; b < B; b++) t += arr[s * B + b]; return t / B / 255; };
  const METRICS = [
    ['Waveform correlation with the take', 'sample by sample', (m) => m.correlation_final, (v) => v],
    ['Latent cosine with the final latent', '', (m) => m.cosine_final, (v) => v],
    ['Band envelope agreement', 'mean over bands (map below)', null, (v) => v],
    ['Spectral centroid', 'Hz', (m) => m.spectral_centroid_hz, (v) => v / 5000],
  ];
  const metricEls = METRICS.map(([name, sub]) => {
    const d = document.createElement('div'); d.className = 'metric';
    d.innerHTML = `<div class="name">${name} <small>${sub}</small></div><div class="val">–</div><canvas></canvas>`;
    $('metrics').appendChild(d);
    return { val: d.querySelector('.val'), cv: d.querySelector('canvas') };
  });
  const metricSeries = METRICS.map(([, , fn, norm], k) => {
    if (!fn) return [D.metrics.map((_, s) => meanSettle(settleS, s)), D.metrics.map((_, s) => meanSettle(settleP, s))];
    return [series('state', fn).map((v) => v == null ? null : norm(v)), series('predicted', fn).map((v) => v == null ? null : norm(v))];
  });
  const BANDNAMES = D.metrics[0].state.bands ? Object.keys(D.metrics[0].state.bands) : [];
  $('bands').innerHTML = BANDNAMES.map((n) => `<div class="b"><span>${n}</span><div class="track"><div class="fill"></div><div class="ghost"></div></div><span class="v">–</span></div>`).join('');
  const bandRows = [...$('bands').querySelectorAll('.b')];

  function renderPanel() {
    drawMarks(); drawMarker(); drawGame();
    const s = Math.round(ui.target);
    const m = D.metrics[s];
    const setName = ui.mode === 1 ? 'predicted' : 'state';
    const other = setName === 'state' ? 'predicted' : 'state';
    $('stepbig').innerHTML = `step ${s}`;
    $('ringnum').textContent = s;
    $('tlabel').textContent = m.t != null ? `t = ${m.t.toFixed(3)}` : '';
    const blurb = s === 0 ? 'the start: pure noise in the latent' : s === S ? 'the end: this state is the take' : `${Math.round(s / S * 100)}% of the way through the solve`;
    $('stepsub').innerHTML = `${blurb}<br>showing <b style="color:#ffc2e3">${MODES[ui.mode]}</b>`;
    $('arc').setAttribute('d', s ? arcPath(s) : '');
    const [kx, ky] = pt(ang(s), R); $('knob').setAttribute('cx', kx); $('knob').setAttribute('cy', ky);
    ringTicks.forEach((t, i) => t.setAttribute('stroke', i <= s ? rgbCss(i / S) : '#3a4a70'));
    $('slider').value = s;
    document.querySelectorAll('#modes button').forEach((b) => b.classList.toggle('on', +b.dataset.mode === ui.mode));
    $('ghost').classList.toggle('on', ui.ghost);
    $('settlebtn').classList.toggle('on', ui.settle);
    $('modenote').innerHTML = ui.mode === 0
      ? 'The solver\'s own state after this step, decoded. Early states are mostly noise with the take fading in.' + (ui.ghost ? ' Lines above: the predicted final.' : '')
      : ui.mode === 1 ? 'What the solver would finish at if it jumped straight to the end from here (x − t·v), decoded. This answers “what had it already decided?”' + (ui.ghost ? ' Lines above: the ODE state.' : '')
      : 'The finished take (step 32). The terrain ignores the step; the ghost still shows the predicted final at the step.';
    const sel = setName === 'predicted' ? 1 : 0;
    METRICS.forEach((row, k) => {
      const raw = row[2] ? row[2](m[setName] || {}) : meanSettle(sel ? settleP : settleS, s);
      metricEls[k].val.textContent = row[0].startsWith('Spectral') ? (raw == null ? '–' : Math.round(raw)) : num(raw);
      const [a, b] = metricSeries[k];
      EX.spark(metricEls[k].cv, sel ? [a, b] : [b, a], s, ['rgba(141,154,179,.45)', sel ? '#ff5fb4' : '#9b6bff']);
    });
    bandRows.forEach((row, i) => {
      const n = BANDNAMES[i];
      const v = (m[setName].bands || {})[n], o = (m[other].bands || {})[n];
      row.querySelector('.fill').style.width = `${Math.max(0, (v ?? 0)) * 100}%`;
      row.querySelector('.ghost').style.left = `${Math.max(0, (o ?? 0)) * 100}%`;
      row.querySelector('.v').textContent = num(v, 2);
    });
    drawSettleMap(s);
  }
  const rgbCss = (x) => css(rampJS(0.2 + 0.8 * x));

  const sm = $('settlemap');
  function drawSettleMap(s) {
    const w = sm.clientWidth, h = sm.clientHeight, dpr = EX.DPR;
    if (sm.width !== Math.round(w * dpr)) { sm.width = Math.round(w * dpr); sm.height = Math.round(h * dpr); }
    const g = sm.getContext('2d'); g.setTransform(dpr, 0, 0, dpr, 0, 0);
    const arr = ui.mode === 1 ? settleP : settleS, first = D.settled_from[ui.mode === 1 ? 'predicted' : 'state'];
    const cw = w / NS, rh = h / B;
    for (let k = 0; k < NS; k++) for (let b = 0; b < B; b++) {
      const v = arr[k * B + b] / 255;
      g.fillStyle = css(rampJS(Math.pow(Math.max(0, v), 2.2) * 0.98), 1);
      g.fillRect(k * cw, h - (b + 1) * rh, cw + 0.5, rh + 0.5);
    }
    g.fillStyle = '#fff';
    first.forEach((f, b) => { if (f != null) g.fillRect(f * cw, h - (b + 1) * rh + rh * 0.3, 1.5, rh * 0.4); });
    g.strokeStyle = '#fff'; g.lineWidth = 2; g.shadowColor = '#ff5fb4'; g.shadowBlur = 8;
    g.strokeRect(s * cw, 0.5, cw, h - 1); g.shadowBlur = 0;
  }
  sm.addEventListener('click', (e) => { const r = sm.getBoundingClientRect(); setStep(Math.floor((e.clientX - r.left) / r.width * NS)); });

  // scrub bar: loudness of the finished take + sections
  const sc = $('scrubcanvas');
  const loud = new Float32Array(C);
  for (let c = 0; c < C; c++) { let t = 0; for (let b = 0; b < B; b++) t += state[S * B * C + b * C + c]; loud[c] = t / B / 255; }
  function drawScrub(t) {
    const w = sc.clientWidth, h = sc.clientHeight, dpr = EX.DPR;
    if (sc.width !== Math.round(w * dpr)) { sc.width = Math.round(w * dpr); sc.height = Math.round(h * dpr); }
    const g = sc.getContext('2d'); g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, w, h);
    const grad = g.createLinearGradient(0, 0, w, 0);
    grad.addColorStop(0, '#4f8cff'); grad.addColorStop(0.45, '#9b6bff'); grad.addColorStop(0.75, '#ff5fb4'); grad.addColorStop(1, '#ff9a4d');
    const mid = h * 0.58;
    g.fillStyle = grad; g.globalAlpha = 0.85;
    g.beginPath(); g.moveTo(0, mid);
    for (let c = 0; c < C; c++) g.lineTo(c / (C - 1) * w, mid - Math.pow(loud[c], 1.6) * mid * 0.95);
    for (let c = C - 1; c >= 0; c--) g.lineTo(c / (C - 1) * w, mid + Math.pow(loud[c], 1.6) * (h - mid) * 0.6);
    g.closePath(); g.fill(); g.globalAlpha = 1;
    (D.sections || []).forEach((s) => { if (s.start < D.seconds) { const x = s.start / D.seconds * w; g.fillStyle = 'rgba(255,255,255,.35)'; g.fillRect(x, 0, 1, h); g.fillStyle = 'rgba(232,237,247,.7)'; g.font = '9px -apple-system'; g.fillText(s.label, x + 3, 9); } });
    g.fillStyle = 'rgba(10,15,28,.55)'; g.fillRect(t / D.seconds * w, 0, w, h);
    g.fillStyle = '#fff'; g.shadowColor = '#ff5fb4'; g.shadowBlur = 10; g.fillRect(t / D.seconds * w - 1, 0, 2, h); g.shadowBlur = 0;
  }
  const scrub = $('scrub');
  const scrubTo = (e) => { const r = scrub.getBoundingClientRect(); audio.seek((e.clientX - r.left) / r.width * D.seconds); };
  scrub.addEventListener('pointerdown', (e) => { scrub.setPointerCapture(e.pointerId); scrubTo(e); });
  scrub.addEventListener('pointermove', (e) => { if (e.buttons) scrubTo(e); });

  // --- actions ---------------------------------------------------------------------------
  function setStep(s, fromAnim) {
    s = Math.max(0, Math.min(S, s));
    if (!fromAnim) ui.animating = false;
    if (s === ui.target) return;
    ui.target = s;
    if (reduced) ui.shown = s;
    audio.stepChanged();
    renderPanel();
  }
  function setMode(m) { ui.mode = m; renderPanel(); }
  function setPlayIcon() { $('play').textContent = ui.playing ? '❚❚' : '▶'; if (typeof drawGame === 'function') drawGame(); }
  function togglePlay() { if (ui.playing) audio.pause(); else audio.start(); setPlayIcon(); }
  $('play').onclick = togglePlay;
  function setPace(v) {
    ui.pace = Math.min(6, Math.max(0.5, Math.round(v * 10) / 10));
    $('pacerange').value = -ui.pace;                  // the slider is speed: right is faster
    $('pacelabel').textContent = `1 step every ${ui.pace.toFixed(1)} s · the whole solve in ${fmtPace(ui.pace * D.steps)}`;
  }
  const fmtPace = (t) => (t >= 60 ? `${Math.floor(t / 60)}:${String(Math.round(t % 60)).padStart(2, '0')}` : `${Math.round(t)} s`);
  $('pacerange').oninput = (e) => setPace(-e.target.value);
  setPace(ui.pace);
  $('resolve').onclick = () => {
    ui.animating = !ui.animating; if (ui.animating && ui.target >= S) setStep(0, true); ui.animClock = 0;
    if (ui.animating && !ui.playing) { audio.start(); setPlayIcon(); }   // the process is something to hear, not only watch
    $('resolve').classList.toggle('on', ui.animating); drawGame();
  };
  $('slider').oninput = (e) => setStep(+e.target.value);
  document.querySelectorAll('#modes button').forEach((b) => (b.onclick = () => setMode(+b.dataset.mode)));
  $('ghost').onclick = () => { ui.ghost = !ui.ghost; renderPanel(); };
  $('settlebtn').onclick = () => { ui.settle = !ui.settle; renderPanel(); };
  window.addEventListener('keydown', (e) => {
    const k = e.key.toLowerCase();
    if (k === ' ') { e.preventDefault(); togglePlay(); }
    else if (k === 'arrowright') { e.preventDefault(); setStep(ui.target + 1); }
    else if (k === 'arrowleft') { e.preventDefault(); setStep(ui.target - 1); }
    else if (k === '1' || k === '2' || k === '3') setMode(+k - 1);
    else if (k === 'g') $('ghost').click();
    else if (k === 'm' && canMark) $('markbtn').click();
    else if (k === 'c') $('settlebtn').click();
    else if (k === 'a') $('resolve').click();
    else if (k === '[' || k === ']') setPace(ui.pace + (k === '[' ? 0.5 : -0.5));
    else if (k === ',') audio.seek(audio.time() - 5);
    else if (k === '.') audio.seek(audio.time() + 5);
    else if (k === 'r') { camera.position.copy(home.pos); controls.target.copy(home.target); }
    else if (k === 'home') setStep(0); else if (k === 'end') setStep(S);
  });

  // hover + click on the land
  const ray = new T.Raycaster(), mouse = new T.Vector2(), groundPlane = new T.Plane(new T.Vector3(0, 1, 0), -H * 0.25), hit = new T.Vector3();
  const tip = EX.tip();
  let downAt = null;
  function pick(e) {
    const r = st.canvas.getBoundingClientRect();
    mouse.set((e.clientX - r.left) / r.width * 2 - 1, -(e.clientY - r.top) / r.height * 2 + 1);
    ray.setFromCamera(mouse, camera);
    if (!ray.ray.intersectPlane(groundPlane, hit)) return null;
    if (Math.abs(hit.x) > W / 2 || Math.abs(hit.z) > DEPTH / 2 + 1) return null;
    const sec = (hit.x / W + 0.5) * D.seconds;
    let b = 0, best = 1e9; bandZ.forEach((z, i) => { const d = Math.abs(z - hit.z); if (d < best) { best = d; b = i; } });
    return { sec, b, c: Math.min(C - 1, Math.floor((hit.x / W + 0.5) * C)) };
  }
  st.canvas.addEventListener('pointermove', (e) => {
    const p = pick(e);
    if (!p || e.buttons) { tip.hide(); uniforms.uHoverZ.value = -999; return; }
    const s = Math.round(ui.target), at = (arr, k) => arr[k * B * C + p.b * C + p.c] / 255;
    const [lo, hi] = D.range_log10, db = (v) => ((lo + v * (hi - lo)) * 10 - hi * 10).toFixed(0);
    uniforms.uHoverZ.value = bandZ[p.b];
    tip.show(`<b>${fmt(p.sec)}</b> · ${Math.round(D.band_lo_hz[p.b])}–${Math.round(D.band_hi_hz[p.b])} Hz<br>
      <span class="m">level at step ${s} (dB below the loudest cell):</span><br>
      state ${db(at(state, s))} · predicted ${db(at(pred, s))} · finished ${db(at(state, S))}<br>
      <span class="m">band envelope agreement with the take:</span> ${num(settleS[s * B + p.b] / 255, 2)} / ${num(settleP[s * B + p.b] / 255, 2)}<br>
      <span class="m">click to play from here</span>`, e.clientX, e.clientY);
  });
  st.canvas.addEventListener('pointerleave', () => { tip.hide(); uniforms.uHoverZ.value = -999; });
  st.canvas.addEventListener('pointerdown', (e) => { downAt = [e.clientX, e.clientY]; });
  st.canvas.addEventListener('pointerup', (e) => {
    if (!downAt || Math.hypot(e.clientX - downAt[0], e.clientY - downAt[1]) > 4) return;
    const p = pick(e); if (!p) return;
    audio.seek(p.sec); if (!ui.playing) { audio.start(); setPlayIcon(); }
  });

  // --- frame loop ---------------------------------------------------------------------
  let intro = reduced ? 1 : 0;
  st.onFrame((dt, now) => {
    if (ui.animating) {
      ui.animClock = (ui.animClock || 0) + dt;
      const rate = ui.playing ? 1 / ui.pace : 2.6;     // steps per second: the listener's pace with sound, brisk without
      if (ui.animClock > 1 / rate) { ui.animClock = 0; if (ui.target >= S) { ui.animating = false; $('resolve').classList.remove('on'); drawGame(); } else setStep(ui.target + 1, true); }
    }
    const k = reduced ? 1 : 1 - Math.exp(-dt * 7);
    ui.shown += (ui.target - ui.shown) * k; if (Math.abs(ui.target - ui.shown) < 1e-3) ui.shown = ui.target;
    const mixGoal = ui.mode === 1 ? 1 : 0, finGoal = ui.mode === 2 ? 1 : 0;
    ui.mixT += (mixGoal - ui.mixT) * k; ui.finT += (finGoal - ui.finT) * k;
    uniforms.uStep.value = ui.shown; uniforms.uMix.value = ui.mixT; uniforms.uFin.value = ui.finT;
    uniforms.uSettleOn.value += ((ui.settle ? 1 : 0) - uniforms.uSettleOn.value) * k;
    uniforms.uTime.value = reduced ? 0 : now;
    ghostUniforms.uGhostMix.value = ui.mode === 1 ? 0 : 1;
    ghostUniforms.uAlpha.value += ((ui.ghost ? 0.3 : 0) - ghostUniforms.uAlpha.value) * k;
    ghost.visible = ghostUniforms.uAlpha.value > 0.01;

    audio.tick();
    const t = audio.time();
    const x = xOfSec(t);
    uniforms.uPlayX.value = x; plane.position.x = x; beam.position.x = x;
    $('time').textContent = `${fmt(t)} / ${fmt(D.seconds)}`;
    drawScrub(t);

    // settling wall
    const s0 = Math.floor(ui.shown), s1 = Math.min(S, s0 + 1), f = ui.shown - s0;
    for (let b = 0; b < B; b++) {
      const get = (arr) => (arr[s0 * B + b] * (1 - f) + arr[s1 * B + b] * f) / 255;
      const v = ui.finT > 0.99 ? 1 : get(ui.mixT > 0.5 ? settleP : settleS);
      m4.makeScale(2.2, 0.3 + Math.max(0, v) * 13, bandGap); m4.setPosition(wallX, 0, bandZ[b]);
      wall.setMatrixAt(b, m4);
      const c = rampJS(0.15 + 0.85 * Math.pow(Math.max(0, v), 1.5)); wall.setColorAt(b, col.setRGB(c[0] * 0.75, c[1] * 0.75, c[2] * 0.75));
    }
    wall.instanceMatrix.needsUpdate = true; wall.instanceColor.needsUpdate = true;

    if (intro < 1) {       // a slow settle-in of the camera, once
      intro = Math.min(1, intro + dt * 0.35);
      const e = 1 - Math.pow(1 - intro, 3);
      camera.position.lerpVectors(new T.Vector3(-190, 150, 240), home.pos, e);
    }
    L.update();
  });

  renderPanel();
  setPlayIcon();
  st.renderOnce();          // one frame even if the page opens in a background tab
})();
