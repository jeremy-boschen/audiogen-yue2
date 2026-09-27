/* "From plan to sound" (stack.html): score, semantic tokens, acoustic latent and audio on one time axis. */
(function () {
  'use strict';
  const { T, reduced, decode, fmt, num, rampJS, css } = EX;
  const D = window.STACK;
  EX.nav('stack');
  EX.help('stack', `
    <h2>From plan to sound</h2>
    <p>This is an exploration of how <a href="https://github.com/multimodal-art-projection/YuE" target="_blank" rel="noopener">YuE2</a>, an open-weight AI music model that anyone can download and run, generates a song. The rows are the saved results of one real
      generation. One network of about 3.6 billion parameters writes the score, the tokens and the sketch, reading the lyrics
      and style description all the way; a separately trained decoder makes the audio
      (<a href="https://github.com/multimodal-art-projection/YuE/blob/main/docs/technical_report.pdf" target="_blank" rel="noopener">technical
      report</a>, section 2).</p>
    <p>The replay (<kbd>P</kbd>) is a sped-up illustration, not a recording: token progress is estimated between recorded
      checkpoints, and the last stage is animated.</p>
    <p>Read it from top to bottom: each row is the same song in a different form, from the plan to the audio you hear, all
      on one timeline: <b>left to right is time</b>.</p>
    <dl>
      <dt>Score</dt><dd>The plan: melody, chords and sections, written as sheet music in text. The lyrics are given
        separately. The model normally writes this first; this take reused a score written earlier.</dd>
      <dt>Semantic tokens</dt><dd>A sketch of the sound: 25 codes a second that steer the detailed sound made next. The
        codes are labels, not amounts, so their heights and colors here are only a way to draw them.</dd>
      <dt>Latent</dt><dd>The detailed sound in compressed form: 64 numbers for every 25th of a second. This is the layer
        you watch come out of noise on the <a href="focus.html">Coming into focus</a> page.</dd>
      <dt>Audio</dt><dd>What you hear, made from the latent by a decoder.</dd>
    </dl>
    <p>Each phrase shows its line from the lyrics when the plan's phrases and the lyric lines match one for one, going by
      the score's timing, not by listening to the singing. Under it, <span class="heardtag">recognized</span> is what a
      speech recognizer picked out of the finished song there. It can differ from the lyrics, and it can mishear.</p>
    `, `
    <h3 style="margin-top:0">What to do</h3>
    <p>Hover anywhere to light the same moment through all four rows and read its lyric. Click to hear that phrase.
      Press <kbd>P</kbd> for a sped-up illustration of the stages: token progress estimated from recorded checkpoints, the last
      stage animated.</p>
    <p style="color:var(--muted)">Drag to turn the view, scroll to zoom. Press <kbd>?</kbd> to bring this back.</p>`);
  document.getElementById('runlabel').innerHTML = `<b>${D.run}</b> · audio ${D.seconds.toFixed(0)} s · score ${D.score.seconds.toFixed(0)} s (score time)`;
  D.audio = EX.hdSwap(D, D.audio);
  EX.hdToggle(D); EX.download(D);
  const $ = (id) => document.getElementById(id);

  const TMAX = Math.max(D.seconds, D.score.seconds);
  const K = 0.78, DEPTH = 34;
  const xOf = (t) => (t - TMAX / 2) * K;
  const tOf = (x) => x / K + TMAX / 2;
  const LAYERS = [
    { key: 'audio', y: 0, name: 'Audio', sub: `what you hear · ${EX.g('spectrogram', 'spectrogram')}` },
    { key: 'latent', y: 18, name: EX.g('latent', 'Latent'), sub: `the detailed sketch of the sound · ${D.latent.frames} × ${D.latent.channels}` },
    { key: 'semantic', y: 36, name: EX.g('tokens', 'Semantic tokens'), sub: `the rough draft · ${D.semantic.count} codes, 25 a second` },
    { key: 'score', y: 54, name: EX.g('score', 'Score'), sub: `the plan · ${D.score.key} · ${D.score.tempo}` },
  ];
  const Y = Object.fromEntries(LAYERS.map((l) => [l.key, l.y]));

  const st = EX.stage($('stage'), { bloom: 0.55, radius: 0.5, threshold: 0.45, fov: 36 });
  const { scene, camera, controls } = st;
  const home = { pos: new T.Vector3(-92, 104, 150), target: new T.Vector3(8, 24, 0) };
  const back = innerWidth < innerHeight ? 2.1 : innerHeight <= 500 ? 1.35 : 1;          // phones: step back so the labels fit
  home.pos.sub(home.target).multiplyScalar(back).add(home.target);
  camera.position.copy(home.pos); controls.target.copy(home.target);
  // Phones: the panel is a sheet along the bottom (upright) or a column down the right (sideways), so the view's
  // center moves into the space that is left. Desktop keeps the full-window view.
  function frameStack() {
    const w = st.host.clientWidth, h = st.host.clientHeight, r = document.getElementById('panel').getBoundingClientRect();
    if (w > 900 && h > 500) camera.clearViewOffset();
    else if (r.top > h / 3) camera.setViewOffset(w, h, -0.16 * w, (h - r.top) / 2, w, h);   // right a little: the layer names sit left
    else camera.setViewOffset(w, h, (w - r.left) / 2 - 0.13 * w, 0, w, h);
    camera.updateProjectionMatrix();
  }
  new ResizeObserver(frameStack).observe(st.host);
  frameStack();
  controls.minDistance = 30; controls.maxDistance = 600;
  scene.fog.density = 0.0024;
  EX.dust(scene, 1600, 1100);
  EX.floor(scene, 900, -3);
  scene.add(new T.HemisphereLight(0xbfd0ff, 0x1a1030, 1.4));
  const sun = new T.DirectionalLight(0xffffff, 1.6); sun.position.set(-60, 120, 80); scene.add(sun);

  // --- layer plates -----------------------------------------------------------------
  const plateMat = (tint) => new T.ShaderMaterial({
    transparent: true, depthWrite: false, side: T.DoubleSide,
    uniforms: { uTint: { value: new T.Color(tint) }, uBeam: { value: -9999 } },
    vertexShader: `varying vec2 vUv; varying vec3 vW; void main(){ vUv = uv; vec4 w = modelMatrix * vec4(position,1.0); vW = w.xyz; gl_Position = projectionMatrix * viewMatrix * w; }`,
    fragmentShader: `varying vec2 vUv; varying vec3 vW; uniform vec3 uTint; uniform float uBeam;
      void main(){
        float edge = max(smoothstep(0.012, 0.0, min(vUv.x, 1.0 - vUv.x)), smoothstep(0.03, 0.0, min(vUv.y, 1.0 - vUv.y)));
        float d = vW.x - uBeam;
        float a = 0.10 + edge * 0.55 + exp(-d * d * 0.8) * 0.5;
        gl_FragColor = vec4(uTint * (0.5 + edge + exp(-d * d * 0.8) * 1.4), a);
      }`,
  });
  const plates = [];
  LAYERS.forEach((l, i) => {
    const end = l.key === 'score' ? D.score.seconds : D.seconds;
    const w = end * K + 4;
    const tint = [0x4f8cff, 0x9b6bff, 0xff5fb4, 0xff9a4d][i];
    const m = new T.Mesh(new T.PlaneGeometry(w, DEPTH + 4), plateMat(tint));
    m.rotation.x = -Math.PI / 2; m.position.set(xOf(end / 2), l.y - 0.4, 0);
    m.userData.layer = l.key;
    scene.add(m); plates.push(m);
  });

  // --- audio: spectrogram heightfield + waveform ribbon ----------------------------------
  const SC = D.spectrogram.columns, SB = D.spectrogram.bands;
  const spec = decode(D.spectrogram.values, Uint8Array);            // [band][column]
  const lf = (f) => Math.log(f / 40) / Math.log(16000 / 40);
  const specZ = D.spectrogram.band_lo_hz.map((lo, b) => (0.5 - lf(Math.sqrt(lo * D.spectrogram.band_hi_hz[b]))) * DEPTH);
  const develop = { audio: 1, latent: 1 };
  {
    const pos = new Float32Array(SC * SB * 3), colr = new Float32Array(SC * SB * 3), idx = [];
    for (let b = 0; b < SB; b++) for (let c = 0; c < SC; c++) {
      const i = b * SC + c, v = Math.max(0, (spec[i] / 255 - 0.3) / 0.7);
      pos.set([xOf((c + 0.5) / SC * D.seconds), Y.audio + Math.pow(v, 1.5) * 9, specZ[b]], i * 3);
      colr.set(rampJS(0.05 + v * 0.95), i * 3);
      if (b < SB - 1 && c < SC - 1) idx.push(i, i + SC, i + 1, i + 1, i + SC, i + SC + 1);
    }
    const g = new T.BufferGeometry();
    g.setAttribute('position', new T.BufferAttribute(pos, 3)); g.setAttribute('color', new T.BufferAttribute(colr, 3));
    g.setIndex(idx); g.computeVertexNormals();
    const mat = new T.MeshStandardMaterial({ vertexColors: true, roughness: 0.55, metalness: 0.05, side: T.DoubleSide,
                                             emissive: 0x1a1030, transparent: true });
    const mesh = new T.Mesh(g, mat); scene.add(mesh);
    develop.audioMesh = mesh;
    // waveform ribbon at the front edge
    const peaks = decode(D.waveform, Int8Array), n = peaks.length / 2;
    const wp = new Float32Array(n * 2 * 3), wi = [];
    for (let i = 0; i < n; i++) {
      const x = xOf((i + 0.5) / n * D.seconds);
      wp.set([x, Y.audio + 4 + peaks[i * 2] / 127 * 4, DEPTH / 2 + 3, x, Y.audio + 4 + peaks[i * 2 + 1] / 127 * 4, DEPTH / 2 + 3], i * 6);
      if (i < n - 1) wi.push(i * 2, i * 2 + 1, i * 2 + 2, i * 2 + 1, i * 2 + 3, i * 2 + 2);
    }
    const wg = new T.BufferGeometry(); wg.setAttribute('position', new T.BufferAttribute(wp, 3)); wg.setIndex(wi);
    const wave = new T.Mesh(wg, new T.MeshBasicMaterial({ color: 0xc9b8ff, transparent: true, opacity: 0.85, side: T.DoubleSide, toneMapped: false }));
    scene.add(wave); develop.wave = wave;
  }

  // --- latent: displaced sheet, diverging color --------------------------------------------
  const LF = D.latent.frames, LC = D.latent.channels;
  const latTex = new T.DataTexture(decode(D.latent.values, Uint8Array), LF, LC, T.RedFormat, T.UnsignedByteType);
  latTex.minFilter = T.LinearFilter; latTex.magFilter = T.LinearFilter; latTex.unpackAlignment = 1; latTex.needsUpdate = true;
  const latUniforms = { uTex: { value: latTex }, uDevelop: { value: 1 }, uBeam: { value: -9999 }, uTime: { value: 0 } };
  {
    const cols = 1250, g = new T.PlaneGeometry(D.seconds * K, DEPTH, cols - 1, LC - 1);
    g.rotateX(-Math.PI / 2);
    g.translate(xOf(D.seconds / 2), Y.latent, 0);
    const mat = new T.ShaderMaterial({
      uniforms: latUniforms, side: T.DoubleSide, transparent: true,
      vertexShader: `uniform sampler2D uTex; uniform float uDevelop, uTime; varying float vV; varying vec3 vW; varying vec2 vUv;
        float hash(vec2 p){ return fract(sin(dot(p, vec2(41.3, 289.1))) * 43758.5); }
        void main(){
          vUv = vec2(uv.x, 1.0 - uv.y);
          float v = texture(uTex, vUv).r * 2.0 - 1.0;
          float n = hash(floor(vUv * vec2(1250.0, 64.0)) + floor(uTime * 6.0)) * 2.0 - 1.0;
          v = mix(n * 0.8, v, uDevelop);
          vV = v;
          vec3 p = position; p.y += v * 3.2;
          vec4 w = modelMatrix * vec4(p, 1.0); vW = w.xyz;
          gl_Position = projectionMatrix * viewMatrix * w;
        }`,
      fragmentShader: `uniform sampler2D uTex; uniform float uBeam, uDevelop; varying float vV; varying vec3 vW; varying vec2 vUv;
        void main(){
          float v = mix(vV, texture(uTex, vUv).r * 2.0 - 1.0, uDevelop);
          vec3 neg = vec3(0.18, 0.45, 1.0), pos = vec3(1.0, 0.55, 0.22), mid = vec3(0.22, 0.12, 0.38);
          vec3 c = v < 0.0 ? mix(mid, neg, pow(-v, 0.8)) : mix(mid, pos, pow(v, 0.8));
          float d = vW.x - uBeam; c += vec3(1.0, 0.9, 1.0) * exp(-d * d * 1.2) * 0.8;
          gl_FragColor = vec4(c * 1.05, 0.95);
        }`,
    });
    scene.add(new T.Mesh(g, mat));
  }

  // --- semantic tokens: id against time, one glowing cell per token -------------------------------
  const ids = decode(D.semantic.ids, Uint16Array);
  const box = new T.BoxGeometry(1, 1, 1);
  const semMesh = new T.InstancedMesh(box, new T.MeshBasicMaterial({ toneMapped: false }), ids.length);
  {
    const m = new T.Matrix4(), c = new T.Color();
    for (let i = 0; i < ids.length; i++) {
      const f = ids[i] / D.semantic.max, rep = i > 0 && ids[i] === ids[i - 1];
      m.makeScale(1 / D.semantic.rate * K * 0.95, rep ? 0.25 : 0.7 + f * 2.2, 0.55);
      m.setPosition(xOf((i + 0.5) / D.semantic.rate), Y.semantic + 0.4, (f - 0.5) * DEPTH);
      semMesh.setMatrixAt(i, m);
      const rgb = rampJS(0.25 + f * 0.75);
      semMesh.setColorAt(i, c.setRGB(rgb[0] * (rep ? 0.35 : 0.9), rgb[1] * (rep ? 0.35 : 0.9), rgb[2] * (rep ? 0.35 : 0.9)));
    }
    scene.add(semMesh);
  }

  // --- score: notes of both voices as bars, sections as colored floors -----------------------------
  const voiceNames = Object.keys(D.score.voices);
  const allPitch = voiceNames.flatMap((v) => D.score.voices[v].pitch);
  const pMin = Math.min(...allPitch), pMax = Math.max(...allPitch);
  const zOfPitch = (p) => (0.5 - (p - pMin) / Math.max(1, pMax - pMin)) * (DEPTH - 4);
  const noteMeshes = {};
  voiceNames.forEach((name) => {
    const v = D.score.voices[name], vocal = name.toLowerCase().startsWith('vocal');
    const mesh = new T.InstancedMesh(box, new T.MeshStandardMaterial({ roughness: 0.35, metalness: 0.1, emissive: vocal ? 0x401030 : 0x10183a }), v.t0.length);
    const m = new T.Matrix4(), c = new T.Color();
    v.t0.forEach((t0, i) => {
      const w = Math.max(0.15, v.dur[i] * K - 0.08);
      m.makeScale(w, vocal ? 2.4 : 0.8, vocal ? 1.5 : 0.8);
      m.setPosition(xOf(t0) + w / 2, Y.score + (vocal ? 1.6 : 0.4), zOfPitch(v.pitch[i]));
      mesh.setMatrixAt(i, m);
      const f = (v.pitch[i] - pMin) / Math.max(1, pMax - pMin);
      const rgb = vocal ? rampJS(0.62 + f * 0.38) : rampJS(0.22 + f * 0.3);
      mesh.setColorAt(i, c.setRGB(rgb[0] * 1.3, rgb[1] * 1.3, rgb[2] * 1.3));
    });
    scene.add(mesh); noteMeshes[name] = mesh;
  });
  const L = EX.labels(st.host, camera);
  D.score.sections.forEach((s, i) => {
    const rgb = rampJS(0.28 + ((i * 0.29) % 0.7));
    const w = (s.end - s.start) * K - 0.5;
    const m = new T.Mesh(new T.PlaneGeometry(w, 3), new T.MeshBasicMaterial({ color: new T.Color(...rgb), transparent: true, opacity: 0.7, toneMapped: false }));
    m.rotation.x = -Math.PI / 2; m.position.set(xOf(s.start) + w / 2 + 0.25, Y.score - 0.2, -DEPTH / 2 - 3.5);
    scene.add(m);
    L.add(s.label, new T.Vector3(xOf(s.start) + 2, Y.score + 1, -DEPTH / 2 - 3.5), 'tick sec');
  });

  // Where the render ends and the score keeps going.
  if (D.score.seconds > D.seconds + 0.5) {
    const h = Y.score + 8;
    const wallMat = new T.ShaderMaterial({
      transparent: true, depthWrite: false, side: T.DoubleSide, blending: T.AdditiveBlending,
      vertexShader: `varying vec2 vUv; void main(){ vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position,1.0); }`,
      fragmentShader: `varying vec2 vUv; void main(){ float s = step(0.5, fract((vUv.x + vUv.y * 0.4) * 22.0)); gl_FragColor = vec4(vec3(1.0, 0.6, 0.3) * (0.05 + s * 0.06), 1.0); }`,
    });
    const wall = new T.Mesh(new T.PlaneGeometry(DEPTH + 4, h + 4), wallMat);
    wall.rotation.y = Math.PI / 2; wall.position.set(xOf(D.seconds), h / 2 - 2, 0);
    scene.add(wall);
    L.add(`audio ends · ${D.seconds.toFixed(0)} s<br><span style="opacity:.7">score continues to ${D.score.seconds.toFixed(0)} s (score time)</span>`,
          new T.Vector3(xOf(D.seconds) + 2, Y.score + 10, DEPTH / 2), 'tick sec');
  }

  // Layer names and time ticks.
  LAYERS.forEach((l) => L.add(`${l.name}<br><span style="opacity:.6;letter-spacing:0;text-transform:none;font-weight:500">${l.sub}</span>`,
                              new T.Vector3(xOf(0) - 16, l.y, 0), 'tick big'));
  for (let s = 0; s <= TMAX; s += 20) L.add(`${s}s`, new T.Vector3(xOf(s), Y.audio - 1, DEPTH / 2 + 8));

  // --- the beam ---------------------------------------------------------------------------------
  const beamMat = new T.ShaderMaterial({
    transparent: true, depthWrite: false, blending: T.AdditiveBlending, side: T.DoubleSide,
    uniforms: { uTime: { value: 0 } },
    vertexShader: `varying vec2 vUv; void main(){ vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position,1.0); }`,
    fragmentShader: `varying vec2 vUv; uniform float uTime;
      void main(){ float c = 1.0 - abs(vUv.x - 0.5) * 2.0; float pulse = 0.85 + 0.15 * sin(vUv.y * 40.0 - uTime * 5.0);
        vec3 col = mix(vec3(0.45, 0.6, 1.0), vec3(1.0, 0.5, 0.8), vUv.y);
        gl_FragColor = vec4(col * pow(c, 3.0) * 1.3 * pulse, 1.0); }`,
  });
  const beamH = Y.score + 14;
  const beam = new T.Group();
  for (const r of [0, Math.PI / 2]) {
    const p = new T.Mesh(new T.PlaneGeometry(3, beamH), beamMat);
    p.rotation.y = r; p.position.y = beamH / 2 - 3; beam.add(p);
  }
  const ringGeo = new T.RingGeometry(1.6, 2.2, 40);
  LAYERS.forEach((l) => {
    const ring = new T.Mesh(ringGeo, new T.MeshBasicMaterial({ color: 0xffe6f5, transparent: true, opacity: 0.9, side: T.DoubleSide, toneMapped: false }));
    ring.rotation.x = -Math.PI / 2; ring.position.y = l.y + 0.2; beam.add(ring);
  });
  beam.visible = false;
  scene.add(beam);
  // phrase span highlight on the score layer
  const spanMesh = new T.Mesh(new T.PlaneGeometry(1, DEPTH), new T.MeshBasicMaterial({ color: 0xff5fb4, transparent: true, opacity: 0.18, depthWrite: false, toneMapped: false }));
  spanMesh.rotation.x = -Math.PI / 2; spanMesh.visible = false; scene.add(spanMesh);

  // --- phrases, moment card, audio --------------------------------------------------------------------
  const esc = (t) => String(t).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
  const words = (t) => String(t || '').toLowerCase().replace(/[^a-z' ]+/g, ' ').trim().split(/\s+/).join(' ');
  const asWritten = (p) => p.lyrics && words(p.lyrics) === words(p.heard);
  const phrases = D.phrases.map((p, i) => ({ ...p, i }));
  const secOf = (t) => D.score.sections.find((s) => t >= s.start && t < s.end);
  const phraseAt = (t) => phrases.find((p) => t >= p.start && t < p.start + (p.seconds || 0));
  $('phrasecount').innerHTML = `${phrases.length} · ${EX.g('scoretime', 'score time')} · ${EX.g('nps', 'notes per syllable')}`;
  $('phrases').innerHTML = phrases.map((p) => `<div class="item" data-i="${p.i}">
      <span class="dot" style="background:${css(rampJS(0.45 + (p.notes_per_syllable ? Math.min(1, (p.notes_per_syllable - 0.6) / 1.2) : 0) * 0.55))}"></span>
      <span class="t">${p.lyrics ? `<span class="lyric">${p.lyrics}</span>` : p.heard ? '' : `<span class="lyric none">${D.heard ? 'no words recognized here' : 'no lyric line paired'}</span>`}
        ${p.heard ? `<span class="heardwords ${p.lyrics ? '' : 'alone'}"><span class="heardtag">recognized</span> ${asWritten(p) ? 'as written' : esc(p.heard)}</span>` : ''}
        <small>${p.section} · phrase ${p.number} · ${fmt(p.start)}${p.start >= D.seconds ? ' · past the end of the audio' : ''}</small></span>
      <span class="n">${p.notes_per_syllable != null ? p.notes_per_syllable.toFixed(2) : '–'}</span></div>`).join('');
  const items = [...document.querySelectorAll('#phrases .item')];
  items.forEach((el) => (el.onclick = () => playPhrase(phrases[+el.dataset.i])));
  const notes = [];
  if (D.heard) notes.push(`Recognized words (may be wrong): ${esc(D.heard.source)}. Each phrase gets the words recognized from its start until the next phrase starts.`);
  if (D.score.pitch_note) notes.push(D.score.pitch_note);
  if (D.score.seconds > D.seconds) notes.push(`The score runs ${D.score.seconds.toFixed(1)} s in score time; the render is ${D.seconds.toFixed(0)} s, which alone doesn't show which parts of the plan were performed.`);
  notes.push(`Song-level notes/syllable ${num(D.song.notes_per_syllable, 3)} (${D.song.melody_notes} notes, ~${D.song.syllables} syllables).`);
  $('datanote').innerHTML = notes.join('<br>');

  // Played from a decoded buffer, not an <audio> element: seeking a media element needs the server to
  // honour byte ranges, and a plain static server that ignores them sends every phrase back to 0:00.
  const player = (() => {
    let ctx = null, buffer = null, loading = null, source = null, startedAt = 0, offset = 0;
    const load = () => loading || (loading = fetch(D.audio).then((r) => r.arrayBuffer())
      .then((bytes) => { ctx = ctx || new (window.AudioContext || window.webkitAudioContext)(); return ctx.decodeAudioData(bytes); })
      .then((b) => (buffer = b)));
    load().catch(() => null);
    const api = {
      get paused() { return !source; },
      get currentTime() { return source ? offset + ctx.currentTime - startedAt : offset; },
      pause() { if (source) { offset = api.currentTime; source.onended = null; source.stop(); source = null; } },
      async playAt(t) {
        api.pause(); offset = t;
        ctx = ctx || new (window.AudioContext || window.webkitAudioContext)();
        if (ctx.state === 'suspended') ctx.resume();          // resumed inside the click that asked for sound
        const want = ++api.ask; await load();
        if (want !== api.ask) return;                         // a later click won while this one loaded
        source = ctx.createBufferSource(); source.buffer = buffer; source.connect(ctx.destination);
        source.onended = () => { source = null; };
        startedAt = ctx.currentTime; source.start(0, Math.min(t, buffer.duration));
      },
      ask: 0,
    };
    return api;
  })();
  let stopAt = null, current = null, locked = null;
  function playPhrase(p) {
    current = p; locked = p.start + 0.01;
    items.forEach((el) => el.classList.toggle('on', +el.dataset.i === p.i));
    items[p.i].scrollIntoView({ block: 'nearest', behavior: reduced ? 'auto' : 'smooth' });
    showMoment(p.start + 0.01);
    if (p.start >= D.seconds) { player.pause(); $('moment').insertAdjacentHTML('beforeend', '<div class="chip warn" style="margin-top:6px">in the score, but after the audio ends — nothing to play</div>'); return; }
    stopAt = Math.min(D.seconds, p.start + (p.seconds || 4) + 0.4);
    player.playAt(p.start).catch(() => null);
  }
  function playFrom(t) {
    const p = phraseAt(t);
    if (p) return playPhrase(p);
    current = null; locked = t; showMoment(t);
    if (t >= D.seconds) return;
    stopAt = Math.min(D.seconds, t + 4); player.playAt(t).catch(() => null);
  }
  function showMoment(t) {
    const s = secOf(t), p = phraseAt(t), frame = Math.floor(t * D.semantic.rate);
    $('beamtime').textContent = `${fmt(t)} score time`;
    const tok = frame < ids.length ? `token <b>${ids[frame]}</b> at frame ${frame}${frame > 0 && ids[frame] === ids[frame - 1] ? ' (repeat of the previous)' : ''}` : 'past the last token';
    $('moment').innerHTML = `
      <div class="m-chips">${s ? `<span class="chip">${s.label}</span>` : ''} ${p && p.bin ? `<span class="chip">${p.notes_per_syllable.toFixed(2)} notes/syllable · ${p.bin}</span>` : ''}</div>
      <div class="m-lyric lyric ${p && p.lyrics ? '' : 'none'}">${p ? p.lyrics || 'no lyric line paired with this phrase' : 'no sung phrase here in the score'}</div>
      ${D.heard ? `<div class="m-line heardwords"><span class="heardtag">heard</span> ${p && p.heard ? esc(p.heard) : '<span style="color:var(--faint)">nothing</span>'}</div>` : ''}
      <div class="m-line note">${p ? `phrase ${p.number} of the ${p.section}: ${p.melody_notes} notes, ~${p.syllables} syllables, ${num(p.seconds, 1)} s` : ''}</div>
      <div class="m-line note" style="margin-top:6px">semantic: ${tok}</div>
      <div class="m-line note">latent: ${frame < LF ? `frame ${frame} of ${LF}` : 'past the end'} · audio: ${t < D.seconds ? fmt(t) : 'ended at ' + fmt(D.seconds)}</div>`;
    setBeam(t, p);
  }
  function setBeam(t, p) {
    beam.visible = true; beam.position.x = xOf(t);
    plates.forEach((m) => (m.material.uniforms.uBeam.value = xOf(t)));
    latUniforms.uBeam.value = xOf(t);
    if (p) { spanMesh.visible = true; spanMesh.scale.x = Math.max(0.3, p.seconds * K); spanMesh.position.set(xOf(p.start + p.seconds / 2), Y.score + 0.05, 0); }
    else spanMesh.visible = false;
  }

  // --- picking -------------------------------------------------------------------------------------
  const ray = new T.Raycaster(), mouse = new T.Vector2();
  const tip = EX.tip();
  let downAt = null;
  function pick(e) {
    const r = st.canvas.getBoundingClientRect();
    mouse.set((e.clientX - r.left) / r.width * 2 - 1, -(e.clientY - r.top) / r.height * 2 + 1);
    ray.setFromCamera(mouse, camera);
    const hit = ray.intersectObjects(plates, false)[0];
    if (!hit) return null;
    return { t: Math.max(0, Math.min(TMAX, tOf(hit.point.x))), layer: hit.object.userData.layer };
  }
  st.canvas.addEventListener('pointermove', (e) => {
    if (e.buttons) return;
    const p = pick(e);
    if (!p) { tip.hide(); if (locked != null) setBeam(locked, current); return; }
    const ph = phraseAt(p.t), s = secOf(p.t);
    showMoment(p.t);
    tip.show(`<b>${fmt(p.t)}</b> <span class="m">score time · on the ${p.layer} layer</span><br>
      ${s ? s.label : ''}${ph ? ` · phrase ${ph.number}` : ''}${ph && ph.lyrics ? `<div class="lyr">${ph.lyrics}</div>` : ''}${ph && ph.heard ? `<div class="m">heard: ${esc(ph.heard)}</div>` : ''}
      <span class="m">click to ${p.t < D.seconds ? 'play' : 'select'}${ph ? ' this phrase' : ' from here'}</span>`, e.clientX, e.clientY);
  });
  st.canvas.addEventListener('pointerleave', () => tip.hide());
  st.canvas.addEventListener('pointerdown', (e) => { downAt = [e.clientX, e.clientY]; });
  st.canvas.addEventListener('pointerup', (e) => {
    if (!downAt || Math.hypot(e.clientX - downAt[0], e.clientY - downAt[1]) > 4) return;
    const p = pick(e); if (p) playFrom(p.t);
  });

  // --- generation replay -------------------------------------------------------------------------
  // Sped up: the whole generation plays in ~22 s. Token progress follows the recorded checkpoints, straight
  // lines between them. A score loaded from an earlier take has no clock, so it is shown whole from the start.
  // The last stage is illustrated, not recorded: its real steps are on Coming into focus.
  const reused = !D.plan.length || D.plan.some((c) => c.generation_seconds == null);
  const plan = reused ? [] : D.plan, sem = D.semantic_checkpoints;
  const planEnd = plan.length ? plan[plan.length - 1].generation_seconds : 0;
  const semEnd = sem.length ? sem[sem.length - 1].generation_seconds : planEnd;
  const GEN_END = semEnd * 1.12;                       // an illustrative tail for the acoustic stage
  const lerpTable = (pts, x) => {                      // pts: [[x, y]...] sorted by x
    if (x <= pts[0][0]) return pts[0][1];
    for (let i = 1; i < pts.length; i++) if (x <= pts[i][0]) { const [a, ya] = pts[i - 1], [b, yb] = pts[i]; return ya + (yb - ya) * (x - a) / Math.max(1e-9, b - a); }
    return pts[pts.length - 1][1];
  };
  const planPts = (name) => [[0, 0], ...plan.map((c) => [c.generation_seconds, c.notes[name] || 0])];
  const planTokPts = [[0, 0], ...plan.map((c) => [c.generation_seconds, c.tokens])];
  const semPts = [[planEnd, 0], ...sem.map((c) => [c.generation_seconds, c.tokens])];
  let gen = GEN_END, replaying = false;
  const frontier = new T.Mesh(new T.PlaneGeometry(DEPTH + 4, 7), beamMat);
  frontier.rotation.y = Math.PI / 2; scene.add(frontier);
  function applyGen(g) {
    gen = g;
    const done = g >= GEN_END - 1e-6;
    voiceNames.forEach((n) => (noteMeshes[n].count = done || reused ? D.score.voices[n].t0.length : Math.floor(lerpTable(planPts(n), g))));
    const tokens = done ? ids.length : Math.floor(lerpTable(semPts, g));
    semMesh.count = g < planEnd ? 0 : tokens;
    const dev = done ? 1 : Math.max(0, Math.min(1, (g - semEnd) / (GEN_END - semEnd)));
    latUniforms.uDevelop.value = g < planEnd ? 0 : dev;
    develop.audioMesh.material.opacity = 0.08 + 0.92 * dev; develop.wave.material.opacity = 0.85 * dev;
    develop.audioMesh.visible = dev > 0 || g >= semEnd;
    let phase;
    if (done) { phase = 'The finished take.'; frontier.visible = false; }
    else if (g < planEnd) {
      const tok = Math.floor(lerpTable(planTokPts, g));
      const vocal = voiceNames.find((n) => n.toLowerCase().startsWith('vocal'));
      const k = noteMeshes[vocal] ? noteMeshes[vocal].count : 0, v = D.score.voices[vocal];
      const x = k > 0 ? v.t0[k - 1] + v.dur[k - 1] : 0;
      frontier.visible = true; frontier.position.set(xOf(x), Y.score + 3, 0);
      phase = `Writing the ${EX.g('score', 'score')}, the sheet music, left to right: symbol ~${tok}.`;
    } else if (g < semEnd) {
      frontier.visible = true; frontier.position.set(xOf(tokens / D.semantic.rate), Y.semantic + 3, 0);
      phase = `${reused ? `The ${EX.g('score', 'score')} was reused from an earlier take. ` : ''}Drafting the sound as ${EX.g('tokens', 'tokens')}: ${tokens} of ${ids.length}, ${(tokens / D.semantic.rate).toFixed(0)} s of music so far.`;
    } else { frontier.visible = false; phase = `Then the ${EX.g('latent', 'latent')} is refined from noise and decoded to audio. Illustrated here, not recorded; the real steps are on <a href="focus.html">Coming into focus</a>.`; }
    $('clock').textContent = done ? `tokens finished at ${fmt(semEnd)}${planEnd ? `, the score at ${fmt(planEnd)}` : ''}`
      : g > semEnd ? 'acoustic stage · timing not recorded here'
      : `≈ ${fmt(g)} into generation`;                   // estimated between recorded checkpoints
    $('phase').innerHTML = phase;
    $('gen').value = Math.round(g / GEN_END * 1000);
    $('replay').textContent = replaying ? '❚❚' : '▶';
  }
  $('gen').oninput = (e) => { replaying = false; applyGen(+e.target.value / 1000 * GEN_END); };
  function toggleReplay() { replaying = !replaying; if (replaying && gen >= GEN_END - 1e-6) gen = 0; applyGen(gen); }
  $('replay').onclick = toggleReplay;

  window.addEventListener('keydown', (e) => {
    const k = e.key.toLowerCase();
    if (k === ' ') { e.preventDefault(); if (!player.paused) player.pause(); else if (current) playPhrase(current); else playPhrase(phrases[0]); }
    else if (k === 'arrowright' || k === 'arrowleft') {
      e.preventDefault();
      const i = current ? current.i + (k === 'arrowright' ? 1 : -1) : 0;
      if (phrases[i]) playPhrase(phrases[i]);
    } else if (k === 'p') toggleReplay();
    else if (k === 'r') { camera.position.copy(home.pos); controls.target.copy(home.target); }
  });

  let intro = reduced ? 1 : 0;
  const introFrom = new T.Vector3(-260, 60, 300);
  st.onFrame((dt, now) => {
    beamMat.uniforms.uTime.value = reduced ? 0 : now;
    latUniforms.uTime.value = reduced ? 0 : now;
    if (replaying) {
      gen = Math.min(GEN_END, gen + dt * GEN_END / 22);  // the whole generation in ~22 s
      if (gen >= GEN_END) replaying = false;
      applyGen(gen);
    }
    if (stopAt != null && !player.paused && player.currentTime >= stopAt) { player.pause(); stopAt = null; }
    if (!player.paused) setBeam(player.currentTime, phraseAt(player.currentTime));
    if (intro < 1) { intro = Math.min(1, intro + dt * 0.3); camera.position.lerpVectors(introFrom, home.pos, 1 - Math.pow(1 - intro, 3)); }
    L.update();
  });
  applyGen(GEN_END);
  st.renderOnce();
})();
