/* "Coming into focus": the ODE solve as a morphing spectrogram terrain. */
(function () {
  'use strict';
  const { T, reduced, decode, fmt, num, rampJS, css } = EX;
  const D = window.FOCUS;
  EX.nav('focus');
  EX.help('focus', `
    <h2>Watching a song come into focus</h2>
    <p><a href="https://github.com/multimodal-art-projection/YuE" target="_blank" rel="noopener">YuE2</a> is an open-weight AI music model: anyone can download it and run it.
      It uses lyrics, a style description and a musical score to build a compressed representation of a song. In its last
      stage that representation starts as random values and is updated over 32 ${EX.g('step', 'steps')}. We decoded the
      saved state at every step, so you can hear it and see its frequencies.</p>
    <p>The landscape is the sound, a ${EX.g('spectrogram', 'spectrogram')} in 3D. <b>Left to right</b> is time, <b>front to
      back</b> is frequency (bass in front), <b>height</b> is level. With comparison coloring on, warmer
      color means a band's rise and fall across the song looks more like the final step's; frost teal means less alike.
      It does not mean a sound is finished.</p>
    <p style="color:var(--muted)">Dotted words explain themselves when you point at them. The model's own description is in the
      <a href="https://github.com/multimodal-art-projection/YuE/blob/main/docs/technical_report.pdf" target="_blank" rel="noopener">YuE2 technical report</a>.</p>
    `, `
    <h3 style="margin-top:0">Try it</h3>
    <p>Press <b>Resolve</b> and listen to the song come out of the noise. Tap <b>I hear…</b> the moment you hear a voice,
      the drums or the chords: a pin marks that moment in the song, and a dot on the dial marks the step. As the dots gather,
      you can see the steps where the song really comes through.</p>
    <p>The buttons show up around when a machine estimated each part comes in; your dots record where you tapped on this
      pass, not the earliest step a part could be heard.</p>
    <p><b>State</b> is the song at this step, <b>Predicted</b> a quick estimate of the end made from it, <b>Finished</b> the
      end result. Preview levels are adjusted for playback (each file to the same peak), so some steps may sound louder
      than others; web audio is AAC. The numbers come from the original audio and latents.</p>
    `);
  document.getElementById('runlabel').innerHTML = `<b>${D.run}</b> · ${D.seconds.toFixed(0)} s · ${D.steps} steps`;
  if (EX.hdOn(D)) { D.audio = EX.hdSwap(D, D.audio); D.audio_bytes = D.hd.bytes; }
  EX.hdToggle(D); EX.download(D);

  const C = D.columns, B = D.bands, S = D.steps, NS = S + 1;
  const W = 150, DEPTH = 78, H = 17;      // deep enough that the Hz bands spread out
  const state = decode(D.state, Uint8Array), pred = decode(D.predicted, Uint8Array);
  const settleS = decode(D.settle_state, Uint8Array), settleP = decode(D.settle_predicted, Uint8Array);

  // Band depth positions from log-center frequency: low at the front.
  const lf = (f) => Math.log(f / 40) / Math.log(16000 / 40);
  const bandZ = D.band_lo_hz.map((lo, b) => (0.5 - lf(Math.sqrt(lo * D.band_hi_hz[b]))) * DEPTH);
  const zOfHz = (f) => (0.5 - lf(f)) * DEPTH;
  const xOfSec = (s) => (s / D.seconds - 0.5) * W;
  // Five plain-named frequency bands: a ruler at the end of the land, and an optional tint across it.
  const PITCH = [
    ['bass', 40, 250, '#ff3d6e', 'Kick drum, bass guitar and the bottom of the piano: what you feel as much as hear.', 'Fundamentals of the low instruments.'],
    ['body', 250, 1000, '#ff9a1f', 'The warmth and weight of voices, guitars and keys. Too much of it sounds muddy.', 'Low-mid fundamentals and their first harmonics.'],
    ['voice & lead', 1000, 4000, '#22e6a0', 'Where the ear is most sensitive: much of the singing and the melody, and what makes words clear.', 'Vowel formants; the ear\'s most sensitive range.'],
    ['bite', 4000, 10000, '#2fa8ff', 'Consonants like s and t, the snap of a snare, the attack of cymbals.', 'Sibilance and transients.'],
    ['air', 10000, 16000, '#b45cff', 'Breath, shimmer and sparkle.', 'The top 0.7 octave of the view.'],
  ];
  const hzText = (f) => (f >= 1000 ? `${f / 1000}k` : `${f}`);
  PITCH.forEach(([name, lo, hi, , plain, tech], i) => { EX.GLOSSARY[`hz${i}`] = [`${hzText(lo)}–${hzText(hi)} Hz · ${name}`, plain, tech]; });
  const bandOf = (hz) => Math.max(0, PITCH.findIndex(([, , hi]) => hz < hi));
  let hoverBand = -1;


  const st = EX.stage(document.getElementById('stage'), { bloom: 0.45, radius: 0.45, threshold: 0.6 });
  const { scene, camera, controls } = st;
  const home = { pos: new T.Vector3(-96, 72, 128), target: new T.Vector3(0, 6, 0) };
  // Frame the land in the open space between the title, the panel and the dock rather than the middle
  // of the window: step the home view back until the whole scene fits there, then slide the projection
  // so its center is that space's center. Orbiting keeps the offset; R and resizes re-frame.
  const HOME_DIR = home.pos.clone().sub(home.target);
  const BOX = [];
  for (const x of [-W / 2 - 16, W / 2 + 16]) for (const y of [0, 30]) for (const z of [-DEPTH / 2 - 2, DEPTH / 2 + 12]) BOX.push(new T.Vector3(x, y, z));
  // Fog thins as the camera is framed further back (phones), so the land stays as bright as on a desktop.
  const FOG_REF = 286, fogK = { value: 1 };
  function frameView() {
    const w = st.host.clientWidth, h = st.host.clientHeight;
    const panel = document.getElementById('panel').getBoundingClientRect();
    const hero = document.querySelector('.hero').getBoundingClientRect();
    const dock = document.getElementById('dock').getBoundingClientRect();
    // The panel runs down the right (desktop, phones held sideways) or is a sheet along the bottom (phones upright).
    const side = panel.width > 0 && panel.left > w / 2, sheet = panel.width > 0 && panel.top > h / 3;
    const L0 = w < 760 ? 96 : 16, R0 = side ? panel.left - 16 : w - 16, T0 = hero.bottom + 12;
    const B0 = sheet ? panel.top - 64 : dock.height ? dock.top - 84 : h - 64;
    const keep = camera.position.clone(), p = new T.Vector3();
    camera.setViewOffset(w, h, 0, 0, w, h);
    let fit = 0.8, box;
    for (; fit < 9; fit += 0.05) {                     // narrow phones need the camera well back
      camera.position.copy(HOME_DIR).multiplyScalar(fit).add(home.target); camera.lookAt(home.target); camera.updateMatrixWorld();
      box = [Infinity, Infinity, -Infinity, -Infinity];
      for (const c of BOX) {
        p.copy(c).project(camera);
        const sx = (p.x + 1) / 2 * w, sy = (1 - p.y) / 2 * h;
        box = [Math.min(box[0], sx), Math.min(box[1], sy), Math.max(box[2], sx), Math.max(box[3], sy)];
      }
      if (box[2] - box[0] <= R0 - L0 && box[3] - box[1] <= B0 - T0) break;
    }
    home.pos.copy(camera.position);
    fogK.value = Math.min(1, FOG_REF / home.pos.distanceTo(home.target));
    if (scene.fog) scene.fog.density = 0.0042 * fogK.value;
    controls.maxDistance = Math.max(420, home.pos.distanceTo(home.target) * 1.3);
    camera.setViewOffset(w, h, (box[0] + box[2] - L0 - R0) / 2, (box[1] + box[3] - T0 - B0) / 2, w, h);
    camera.position.copy(keep); camera.lookAt(controls.target); camera.updateMatrixWorld();
  }
  new ResizeObserver(frameView).observe(st.host);
  frameView();
  camera.position.copy(home.pos); controls.target.copy(home.target);
  controls.minDistance = 30; controls.maxPolarAngle = Math.PI * 0.49;   // maxDistance: frameView
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
    uTexel: { value: new T.Vector2(1 / C, 1 / B) }, uBg: { value: new T.Color(EX.PALETTE.bg) }, uFog: fogK,
    uHoverZ: { value: -999 }, uLift: { value: 0 }, uGhostMix: { value: 1 }, uAlpha: { value: 1 },
  };

  const VERT_COMMON = `
    precision highp sampler2DArray;
    uniform sampler2DArray uState, uPred;
    uniform sampler2D uSetS, uSetP;
    uniform float uStep, uSteps, uMix, uFin, uH, uLift, uGhostMix;
    uniform vec2 uTexel;
    attribute vec2 aUV;
    varying float vH; varying float vFinH; varying float vSettle; varying vec3 vN; varying vec3 vW; varying float vDist;
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
      vFinH = lev(texture(uState, vec3(aUV, uSteps)).r);   // the height here in the finished song
      float sv = (uStep + 0.5) / (uSteps + 1.0);
      vSettle = mix(mix(texture(uSetS, vec2(aUV.y, sv)).r, texture(uSetP, vec2(aUV.y, sv)).r, mixv), 1.0, uFin);
      vec3 p = position; p.y = shape(h) + uLift;
      vec4 w = modelMatrix * vec4(p, 1.0); vW = w.xyz;
      vec4 mv = viewMatrix * w; vDist = length(mv.xyz);
      gl_Position = projectionMatrix * mv;
    }`;
  const FRAG_COMMON = `
    uniform float uTime, uPlayX, uSettleOn, uHoverZ, uAlpha, uFog; uniform vec3 uBg;
    varying float vH; varying float vFinH; varying float vSettle; varying vec3 vN; varying vec3 vW; varying float vDist;
    ${EX.GLSL_RAMP}
    float hash(vec2 p) { return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453); }
    const vec3 FROST = vec3(0.30, 0.80, 0.90);
    vec3 shade(bool lines) {
      vec3 base = ramp(0.05 + pow(vH, 1.15) * 0.95);
      // Settling color: what is still forming is frost, a cool teal set against the land's warm palette; each spot
      // takes on its color in the finished song, gently, as its band settles.
      float settled = mix(1.0, smoothstep(0.45, 0.99, vSettle), uSettleOn);
      vec3 toward = ramp(0.05 + pow(vFinH, 1.15) * 0.95);
      float lum = dot(base, vec3(0.3, 0.5, 0.2));
      vec3 cool = FROST * (0.3 + lum * 1.05);
      float glit = step(0.985, hash(floor(vW.xz * 2.2) + floor(uTime * 9.0))) * (1.0 - settled) * 0.9;
      vec3 col = mix(cool, mix(toward, base, settled * settled), pow(settled, 1.6)) + glit * mix(FROST, vec3(1.0), 0.5);
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
      float fog = 1.0 - exp(-pow(vDist * 0.0036 * uFog, 2.0));
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

  // Two bars in front of the land: the arrangement (verse, chorus: from the score, so in score time), and in
  // front of it the time line, where the pins go.
  const ARR_Z = DEPTH / 2 + 4.5, TIME_Z = DEPTH / 2 + 8;
  const SEC_RAMP = { verse: [0.02, 0.42], chorus: [0.55, 0.98], bridge: [0.4, 0.7], intro: [0.0, 0.25], outro: [0.0, 0.25] };   // stretches of the land's palette
  const secGroup = new T.Group(); scene.add(secGroup);
  (D.sections || []).forEach((s) => {
    if (s.start >= D.seconds) return;
    const x0 = xOfSec(s.start), x1 = xOfSec(Math.min(s.end, D.seconds));
    const [r0, r1] = SEC_RAMP[s.label] || [0.3, 0.6], len = Math.max(0.1, x1 - x0 - 0.5);
    const g = new T.PlaneGeometry(len, 1.5, 24, 1), gp = g.getAttribute('position'), gc = [];
    for (let i = 0; i < gp.count; i++) { const c = rampJS(r0 + (r1 - r0) * (gp.getX(i) / len + 0.5)); gc.push(c[0] * 1.2, c[1] * 1.2, c[2] * 1.2); }
    g.setAttribute('color', new T.Float32BufferAttribute(gc, 3));
    const m = new T.Mesh(g, new T.MeshBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.85, toneMapped: false }));
    m.rotation.x = -Math.PI / 2; m.position.set((x0 + x1) / 2, 0.02, ARR_Z);
    secGroup.add(m);
  });
  const timeBar = new T.Mesh(new T.PlaneGeometry(W, 1.6), new T.MeshBasicMaterial({ color: '#cdbfff', toneMapped: false }));
  timeBar.rotation.x = -Math.PI / 2; timeBar.position.set(0, 0.02, TIME_Z); scene.add(timeBar);
  for (let t = 0; t <= D.seconds + 0.01; t += 10) {
    const tick = new T.Mesh(new T.PlaneGeometry(0.45, t % 20 ? 2.2 : 3.4), new T.MeshBasicMaterial({ color: '#f4efff', toneMapped: false }));
    tick.rotation.x = -Math.PI / 2; tick.position.set(xOfSec(t), 0.03, TIME_Z + 0.9); scene.add(tick);
  }

  // --- labels ---------------------------------------------------------------------
  const L = EX.labels(st.host, camera);
  for (let s = 0; s <= D.seconds + 0.01; s += 20) L.add(`${s}s`, new T.Vector3(xOfSec(s), 0, TIME_Z + 4), 'tick tsec');
  L.add('time →', new T.Vector3(W / 2 + 10, 0, TIME_Z), 'tick big');
  // Hz bands: a see-through colored layer floating just above the noise, one strip per band across the whole land,
  // like the ghost. Their labels stand by the settling wall, whose bars take the band colors while it is on.
  const ROOF_Y = H * 1.15;
  const bandRoof = PITCH.map(([, lo, hi, color]) => {
    const z0 = zOfHz(lo), z1 = zOfHz(hi);
    const m = new T.Mesh(new T.PlaneGeometry(W, Math.abs(z0 - z1) - 0.35),
      new T.MeshBasicMaterial({ color, transparent: true, opacity: 0, depthWrite: false, blending: T.AdditiveBlending, side: T.DoubleSide, toneMapped: false }));
    m.rotation.x = -Math.PI / 2; m.position.set(0, ROOF_Y, (z0 + z1) / 2); m.visible = false; scene.add(m);
    return m;
  });
  const blockMat = new T.MeshBasicMaterial({ transparent: true, opacity: 0, depthWrite: false, blending: T.AdditiveBlending, side: T.DoubleSide, toneMapped: false });
  const block = new T.Mesh(new T.BoxGeometry(1, 1, 1), blockMat); block.visible = false; scene.add(block);
  const blockEdges = new T.LineSegments(new T.EdgesGeometry(new T.BoxGeometry(1, 1, 1)),
    new T.LineBasicMaterial({ transparent: true, opacity: 0, blending: T.AdditiveBlending, toneMapped: false }));
  block.add(blockEdges);
  let blockBand = -1;
  function placeBlock(i) {                    // the band's frequency range, across the whole song and the full height
    const [, lo, hi, color] = PITCH[i], z0 = zOfHz(lo), z1 = zOfHz(hi), top = H * 1.5;
    block.scale.set(W, top, Math.abs(z0 - z1)); block.position.set(0, top / 2, (z0 + z1) / 2);
    blockMat.color.set(color); blockEdges.material.color.set(color).multiplyScalar(1.4); blockBand = i;
  }
  const bandLabels = PITCH.map(([name, lo, hi, color], i) => {
    const el = L.add(`<dfn data-g="hz${i}">${hzText(lo)}–${hzText(hi)} Hz</dfn> <span>${name}</span>`,
      new T.Vector3(wallX - 3, 0, (zOfHz(lo) + zOfHz(hi)) / 2), 'tick hzband').el;
    el.style.setProperty('--c', color); el.dataset.g = `hz${i}`;      // the whole label opens its card
    el.addEventListener('pointerenter', () => { hoverBand = i; });
    el.addEventListener('pointerleave', () => { hoverBand = -1; });
    return el;
  });
  L.add(EX.g('settling', 'settling wall'), new T.Vector3(wallX, 16, -DEPTH / 2 - 3), 'tick big');   // at the back, clear of the band labels
  (D.sections || []).forEach((s) => {
    if (s.start < D.seconds) L.add(s.label, new T.Vector3((xOfSec(s.start) + xOfSec(Math.min(s.end, D.seconds))) / 2, 0.1, ARR_Z), 'tick arr');
  });
  L.add(`arrangement · ${EX.g('scoretime', 'score time')}`, new T.Vector3(-W / 2 - 3, 0, ARR_Z), 'tick side');

  // --- state ------------------------------------------------------------------------
  // Resolve's pace, seconds per step while the song plays from the top: the 32 steps finish a second before the
  // last chorus starts, so that chorus is heard fully resolved. Without sections, one pass of the song.
  const lastChorus = (D.sections || []).filter((x) => x.label === 'chorus' && x.start > 0).pop();
  const RESOLVE_BY = lastChorus ? lastChorus.start - 1 : D.seconds;
  const PACE_DEFAULT = RESOLVE_BY / D.steps;
  const ui = {
    pace: PACE_DEFAULT,
    target: 0, shown: 0, mode: 0, ghost: true, settle: true, bands: false, playing: false,
    mixT: 0, finT: 0, animating: false,
  };

  // --- audio: Web Audio, every layer scheduled on one clock -----------------------------
  // The web build downloads every file before anything plays, behind a progress bar, then decodes each when
  // its step is wanted (a decoded file is ~36 MB, so all 66 at once would hold ~2.3 GB). Every layer is an AudioBufferSourceNode started at the same context time
  // and offset, so a step change is a crossfade between buffers that are already in step: no
  // media elements to keep in lockstep, and gain nodes work where element volume does not (iOS).
  const audio = window.focusAudio = (() => {
    const Ctx = window.AudioContext || window.webkitAudioContext;
    const ctx = new Ctx();
    const FADE = 0.06, KEEP = 8, SWELL = 1.5;
    // Everything goes through one master gain, which swells in over SWELL seconds whenever playback starts: the
    // early steps are loud noise, and starting it at full volume made people jump.
    const master = ctx.createGain();
    master.connect(ctx.destination);
    const files = [D.audio.finished, ...D.audio.state.flatMap((s, i) => [s, D.audio.predicted[i]])];
    const unique = [...new Set(files)];
    const bytes = new Map(), buffers = new Map();
    let failed = null;
    // The web build ships AAC (~2 MB a step), all kept after download. The local build points at the runs'
    // WAV listening copies (~35 MB each): fetched when needed and let go once decoded, never all held at once.
    const compressed = unique.every((u) => /\.m4a$/.test(u));
    const got = new Map();                          // bytes received so far, per file, for the progress bar

    function load(url) {
      if (!bytes.has(url)) {
        bytes.set(url, fetch(url).then(async (r) => {
          if (!r.ok) throw new Error(`${r.status} ${url}`);
          if (!compressed || !r.body) return r.arrayBuffer();
          const reader = r.body.getReader(), parts = [];
          let n = 0;
          for (;;) {
            const { done, value } = await reader.read();
            if (done) break;
            parts.push(value); n += value.length; got.set(url, n); progress();
          }
          const out = new Uint8Array(n);
          parts.reduce((o, part) => { out.set(part, o); return o + part.length; }, 0);
          return out.buffer;
        }).catch((e) => { failed = e; status(); throw e; }));
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
    // Everything is downloaded before anything plays, six at a time, then step 0 and the finished take are
    // decoded, and only then do Play and Resolve come on. After that nothing waits on the network, though a step can
    // still take a moment to decode: the status under the buttons says so while it does.
    const TOTAL = D.audio_bytes || 0, MB = (n) => (n / 1e6).toFixed(0);
    let isReady = !compressed;
    function prepare() {
      document.body.classList.add('preparing');
      let next = 0;
      const worker = () => (next < unique.length ? load(unique[next++]).then(worker) : null);
      return Promise.all(Array.from({ length: 6 }, worker))
        .then(() => Promise.all([D.audio.state[0], D.audio.predicted[0], D.audio.finished].map(decoded)))
        .then(() => { isReady = true; document.body.classList.remove('preparing'); document.dispatchEvent(new Event('audioready')); })
        .catch(() => null);
    }
    function progress() {
      let n = 0; got.forEach((v) => { n += v; });
      const bar = document.getElementById('prepbar'), text = document.getElementById('preptext');
      if (!bar) return;
      const f = TOTAL ? Math.min(1, n / TOTAL) : got.size / unique.length;
      bar.style.width = `${(f * 100).toFixed(1)}%`;
      text.textContent = TOTAL ? `${MB(n)} of ${MB(TOTAL)} MB` : `${got.size} of ${unique.length} files`;
    }
    function status() {
      const el = document.getElementById('audiostatus');
      if (el) el.textContent = failed ? `Audio failed to load (${failed.message})` : '';
      const text = document.getElementById('preptext');
      if (failed && text) text.textContent = `Audio failed to load (${failed.message}). Reload to try again.`;
    }

    // Playback: `origin` is the context time at which the song's 0 s would have played.
    let origin = 0, startAt = 0, playing = false, want = 0, layers = null, loadedStep = -1;
    const weights = () => [ui.mode === 0 ? 1 : 0, ui.mode === 1 ? 1 : 0, ui.mode === 2 ? 1 : 0];
    const time = () => (playing ? Math.min(D.seconds, ctx.currentTime - origin) : startAt);

    function voice(buffer, weight, when, offset) {
      const gain = ctx.createGain();
      gain.gain.setValueAtTime(0, when);
      gain.gain.linearRampToValueAtTime(weight, when + FADE);
      gain.connect(master);
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
      if (!isReady) return;
      if (location.protocol === 'file:') { failed = new Error('open this page through a web server'); status(); return; }
      ctx.resume();
      if (startAt >= D.seconds - 0.05) startAt = 0;
      origin = ctx.currentTime + 0.05 - startAt;
      const now = ctx.currentTime;                          // an even rise in loudness: -60 dB to full
      master.gain.cancelScheduledValues(now);
      master.gain.setValueAtTime(0.001, now);
      master.gain.exponentialRampToValueAtTime(1, now + 0.05 + SWELL);
      playing = true; ui.playing = true; st.wake();      // it may start well after the click, once downloaded
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
    if (compressed) prepare();
    return { start, pause, seek, stepChanged, tick, time, get ready() { return isReady; }, get loadedStep() { return loadedStep; },
             // What is audible now: the files with a non-zero gain (for tests and the curious).
             get sounding() { return (layers || []).filter((l) => l.weight > 0).map((l) => l.url); } };
  })();

  // --- panel ---------------------------------------------------------------------------
  const $ = (id) => document.getElementById(id);
  const MODES = [EX.g('state', 'the state'), EX.g('predicted', 'the predicted final'), 'the finished song'];
  $('honest').title = `These are measurements of this take's saved latent states and decoded audio. ${D.bands_note}. Audio: ${D.audio_note}.`;

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
  function drawMarks() {
    const view = viewKey(), now = Math.round(ui.target), reached = (s) => (+s <= now ? ' reached' : '');
    const mine = heard[view] || {}, perStep = {};
    const own = LOCAL ? Object.entries(firstHeard(view)).map(([s, names]) => {   // the author's own marks, served locally
      const [x, y] = pt(ang(+s), R - 13);
      return `<g class="heardmark${reached(s)}"><circle cx="${x}" cy="${y}" r="3.2"/><title>Step ${s}: ${names.join(', ')}</title></g>`;
    }).join('') : '';
    markLayer.innerHTML = own + CUES.filter((c) => mine[c.id]).map((c) => {
      const s = mine[c.id].step, k = (perStep[s] = (perStep[s] || 0) + 1) - 1;
      const [x, y] = pt(ang(s), R - 13 - k * 7);
      return `<g class="cuemark${reached(s)}" style="--c:${c.color}"><circle cx="${x}" cy="${y}" r="3.2"/><title>Step ${s}: ${c.label}</title></g>`;
    }).join('');
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
    if (clear) clear.onclick = clearMine;
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
  // The listening game. The machines only decide what to listen for and when: each part of the finished song
  // has the time it first comes in (bin/find_entries.py), and its "I hear..." button appears once the song has
  // reached it, a few at a time. A tap flies to a pin at that moment on the time line, with a line down into the
  // open space under the land, and puts a dot on the dial at the step. Kept in this browser (and on disk locally).
  const PARTS = {
    drums: ['the drums', '#7fe0b4', 'beat_recognizable'], bass: ['the bass', '#8fb8ff', 'bass_recognizable'],
    voice: ['a voice', '#ff8fcf', 'vocal_present'], words: ['the words', '#ffb070', 'words_intelligible'],
    chords: ['the chords', '#c9a6ff', 'harmony_recognizable'], guitar: ['a guitar', '#ffe07a', 'instrument_identity_recognizable'],
    piano: ['a piano', '#9ff0ff', 'instrument_identity_recognizable'], other: ['the other instruments', '#d6dcef', 'instrument_identity_recognizable'],
  };
  const CUES = (D.entries || [{ id: 'voice', at: 0 }, { id: 'words', at: 0 }, { id: 'chords', at: 0 }])
    .filter((e) => PARTS[e.id]).map((e) => ({ id: e.id, at: e.at, label: e.label || PARTS[e.id][0], color: PARTS[e.id][1] }));
  const SHOW = 3;
  const CSTORE = `${STORE}-cues`;
  let heard = { state: {}, predicted: {} };
  try { heard = { ...heard, ...JSON.parse(localStorage.getItem(CSTORE) || '{}') }; } catch (e) { /* private mode */ }
  const saveHeard = () => { try { localStorage.setItem(CSTORE, JSON.stringify(heard)); } catch (e) { /* private mode */ } };
  // Pins: a dot per part at the moment on the time line; parts heard close together share one tag, a stack,
  // and the tags spread along the open space under the land on angled leader lines, the way the take map
  // clusters takes. Laid out again every frame, so they follow the camera.
  const pinBox = document.createElement('div'); pinBox.className = 'pins'; document.body.appendChild(pinBox);
  const pinLines = document.createElementNS('http://www.w3.org/2000/svg', 'svg'); pinBox.appendChild(pinLines);
  const anchors = {}, groups = new Map();
  function anchorFor(c) {
    if (!anchors[c.id]) {
      const el = document.createElement('i'); el.className = 'pinanchor'; el.style.setProperty('--c', c.color);
      pinBox.appendChild(el); anchors[c.id] = el;
    }
    return anchors[c.id];
  }
  const pv = new T.Vector3();
  function placePins() {
    const view = viewKey(), mine = heard[view] || {}, w = st.host.clientWidth, h = st.host.clientHeight;
    const off = ui.mode === 2;
    Object.entries(anchors).forEach(([id, el]) => { el.hidden = off || !mine[id]; });
    const pts = off ? [] : CUES.filter((c) => mine[c.id]).map((c) => {
      pv.set(xOfSec(mine[c.id].at), 0, TIME_Z).project(camera);
      const x = (pv.x + 1) / 2 * w, y = (1 - pv.y) / 2 * h;
      anchorFor(c).style.transform = `translate(${x.toFixed(1)}px, ${y.toFixed(1)}px)`;
      return { c, x, y };
    }).sort((p, q) => p.x - q.x);
    const clusters = [];
    pts.forEach((p) => {
      const last = clusters[clusters.length - 1], q = last && last[last.length - 1];
      if (q && Math.hypot(p.x - q.x, p.y - q.y) < 30) last.push(p); else clusters.push([p]);
    });
    const game = $('game').getBoundingClientRect(), floor = (game.height ? game.top : h - 120) - 10;
    const panel = $('panel').getBoundingClientRect(), L0 = 16, R0 = panel.left > w / 2 ? panel.left - 16 : w - 16;
    const seen = new Set();
    const laid = clusters.map((cl) => {
      const key = cl.map((p) => p.c.id).join();
      seen.add(key);
      let el = groups.get(key);
      if (!el) { el = document.createElement('div'); el.className = 'pintags'; pinBox.appendChild(el); groups.set(key, el); }
      const html = cl.map((p) => `<div style="--c:${p.c.color}"><i></i>${p.c.label.replace(/^(the|a) /, '')} <b>${mine[p.c.id].step}</b></div>`).join('');
      if (el.dataset.html !== html) { el.innerHTML = html; el.dataset.html = html; }
      el.hidden = false;
      const jx = cl.reduce((t, p) => t + p.x, 0) / cl.length, jy = Math.max(...cl.map((p) => p.y));
      return { cl, el, jx, jy, gw: el.offsetWidth, gh: el.offsetHeight };
    });
    groups.forEach((el, key) => { if (!seen.has(key)) el.hidden = true; });
    laid.forEach((g, i) => { g.x = Math.max(L0, g.jx - g.gw / 2, i ? laid[i - 1].x + laid[i - 1].gw + 14 : L0); });
    for (let i = laid.length - 1; i >= 0; i--) {         // pushed off the right edge: slide back left
      const limit = i < laid.length - 1 ? laid[i + 1].x - 14 : R0;
      laid[i].x = Math.max(L0, Math.min(laid[i].x, limit - laid[i].gw));
    }
    let svg = '';
    laid.forEach((g) => {
      const top = Math.max(floor - g.gh, g.jy + 22);
      g.el.style.transform = `translate(${g.x.toFixed(1)}px, ${top.toFixed(1)}px)`;
      const rows = [...g.el.children];
      g.cl.forEach((p, r) => {                      // each part's own line, ending at its own row's dot
        const row = rows[r], dot = row ? row.offsetTop + row.offsetHeight / 2 : 10;
        svg += `<line x1="${p.x}" y1="${p.y + 6}" x2="${g.x + 7}" y2="${top + dot}" stroke="${p.c.color}" stroke-opacity=".75"/>`;
      });
    });
    pinLines.innerHTML = svg;
  }
  function claim(c, chip) {
    const view = viewKey(), at = audio.time();
    const step = ui.playing && audio.loadedStep >= 0 ? audio.loadedStep : Math.round(ui.target);   // the step being heard
    heard[view] = { ...(heard[view] || {}), [c.id]: { step, at } }; saveHeard();
    if (LOCAL) mark(view, step, PARTS[c.id][2], true);
    placePins(); drawMarks();
    const from = chip.getBoundingClientRect(), pin = anchorFor(c);
    pv.set(xOfSec(at), 0, TIME_Z).project(camera);
    const tx = (pv.x + 1) / 2 * st.host.clientWidth, ty = (1 - pv.y) / 2 * st.host.clientHeight;
    const fx = from.left + from.width / 2, fy = from.top + from.height / 2;
    if (!reduced) {
      const fly = document.createElement('div'); fly.className = 'fly'; fly.style.setProperty('--c', c.color);
      document.body.appendChild(fly);
      pin.style.visibility = 'hidden';                 // the dot arrives, then the pin is there
      fly.animate([{ transform: `translate(${fx}px, ${fy}px) scale(1.4)` }, { transform: `translate(${tx}px, ${ty}px) scale(1)` }],
        { duration: 550, easing: 'cubic-bezier(.35, 0, .25, 1)', fill: 'forwards' }).onfinish = () => {
        fly.remove(); pin.style.visibility = ''; pin.classList.add('landing');
        setTimeout(() => pin.classList.remove('landing'), 650);
      };
    }
    drawGame();
  }
  function clearMine() {                               // a visitor's own marks only
    HEARD.state = {}; HEARD.predicted = {}; localStorage.removeItem(STORE);
    heard = { state: {}, predicted: {} }; localStorage.removeItem(CSTORE);
    drawMarks(); drawMarker(); drawGame(); placePins();
  }
  let shownKey = '';
  const switching = () => ui.playing && audio.loadedStep !== Math.round(ui.target);   // the chosen step is not sounding yet
  const gameKey = () => dueCues().map((c) => c.id).join() + (switching() ? '…' : '');
  function dueCues() {
    const mine = heard[viewKey()] || {}, t = audio.time();
    return CUES.filter((c) => !mine[c.id] && c.at <= t + 0.5).slice(0, SHOW);
  }
  function drawGame() {
    const box = $('game');
    box.hidden = !canMark || ui.mode === 2;
    if (box.hidden) return;
    const mine = heard[viewKey()] || {}, done = CUES.filter((c) => mine[c.id]).length, due = dueCues();
    shownKey = gameKey();
    const wait = switching() ? ' disabled' : '';
    $('gamecue').textContent = done === CUES.length ? 'Every part marked. The dots on the dial show the steps where the song came through.'
      : due.length ? (ui.playing || ui.animating ? 'Tap the moment you hear it' : '')
      : ui.playing || ui.animating ? 'Listen…' : 'Press Resolve, then tap the moment you hear each part come in.';
    const old = new Set([...$('gamepills').querySelectorAll('[data-id]')].map((b) => b.dataset.id));
    $('gamepills').innerHTML = due.map((c) => `<button class="gamepill${old.has(c.id) ? '' : ' enter'}" style="--c:${c.color}" data-id="${c.id}"${wait}>I hear ${c.label}</button>`).join('')
      + (!LOCAL && done ? '<button class="gamepill again" id="gameagain" title="Clear your marks and start from step 0" aria-label="Start over">↺</button>' : '');
    const again = $('gamepills').querySelector('#gameagain');
    if (again) again.onclick = () => { clearMine(); if (ui.animating) $('resolve').click(); setStep(0); audio.seek(0); };
    $('gamepills').querySelectorAll('[data-id]').forEach((b) => (b.onclick = () => claim(CUES.find((c) => c.id === b.dataset.id), b)));
  }
  st.onFrame(() => { placePins(); if (!$('game').hidden && gameKey() !== shownKey) drawGame(); });
  $('markbtn').hidden = !LOCAL;
  $('markbtn').onclick = () => { marking = !marking; $('markbtn').classList.toggle('on', marking); drawMarker(); };

  // Hover help: the panel's controls describe themselves in the space at its foot, not in tooltips.
  const HELP_IDLE = 'Point at anything in this panel and it is explained here.';
  const hh = $('hoverhelp');
  hh.innerHTML = HELP_IDLE;
  $('panel').addEventListener('pointerover', (e) => {
    if (e.target.closest('#hoverhelp')) return;
    const el = e.target.closest('[data-help]'); hh.innerHTML = el ? el.dataset.help : HELP_IDLE; hh.classList.toggle('on', !!el);
  });
  $('panel').addEventListener('pointerleave', () => { hh.innerHTML = HELP_IDLE; hh.classList.remove('on'); });

  // metrics
  const series = (set, fn) => D.metrics.map((m) => fn(m[set] || {}));
  const meanSettle = (arr, s) => { let t = 0; for (let b = 0; b < B; b++) t += arr[s * B + b]; return t / B / 255; };
  const METRICS = [
    ['Waveform match', '<b>Waveform match</b> compares the actual sound wave at this step with the finished song\'s, moment by moment. 1 means the shapes line up perfectly, which is not quite the same as identical; 0 means no straight-line relationship. It stays low until late, because tiny timing differences count against it even when the song already sounds right.<small class="tech">Pearson correlation of the samples with the final take\'s.</small>', (m) => m.correlation_final, (v) => v],
    ['Latent match', '<b>Latent match</b> compares the model\'s own internal sketch of the song (its ' + EX.g('latent', 'latent') + ') with its final sketch. Compare how this numerical similarity changes across the steps with how the waveform match changes.<small class="tech">Cosine similarity of the latent with the final latent.</small>', (m) => m.cosine_final, (v) => v],
    ['Band agreement', '<b>Band agreement</b> asks, for each frequency band, whether it gets louder and quieter at the same moments as the finished song, averaged over the bands. The map below shows it band by band.<small class="tech">Mean over bands of the envelope correlation with step 32\'s, negatives counted as 0.</small>', null, (v) => v],
    ['Brightness Hz', '<b>Brightness</b> is the average frequency of the sound, weighted by how strong each frequency is. Follow how it moves across the steps; it is not a direct measure of how much noise is left.<small class="tech">Spectral centroid of the magnitude spectrum, Hz.</small>', (m) => m.spectral_centroid_hz, (v) => v / 5000],
  ];
  const metricEls = METRICS.map(([name, sub]) => {
    const d = document.createElement('div'); d.className = 'metric';
    d.dataset.help = sub; d.innerHTML = `<div class="name">${name}</div><div class="val">–</div><canvas></canvas>`;
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
    const ms = ui.mode === 2 ? S : s;                    // Finished plays step 32, so its numbers are step 32's
    const m = D.metrics[ms];
    const setName = ui.mode === 1 ? 'predicted' : 'state';
    const other = setName === 'state' ? 'predicted' : 'state';
    $('stepbig').innerHTML = `step ${s}`;
    $('ringnum').textContent = s;
    const blurb = ui.mode === 2 || s === S ? 'the finished song, step 32'
      : s === 0 ? (ui.mode === 1 ? 'prediction from the initial state' : 'the starting noise') : `${Math.round(s / S * 100)}% of the steps done`;
    $('stepsub').innerHTML = `${blurb}<br>showing <b style="color:#ffc2e3">${MODES[ui.mode]}</b>${m.t != null ? ` · ${EX.g('t', 't')} ${m.t.toFixed(2)}` : ''}`;
    $('arc').setAttribute('d', s ? arcPath(s) : '');
    const [kx, ky] = pt(ang(s), R); $('knob').setAttribute('cx', kx); $('knob').setAttribute('cy', ky);
    ringTicks.forEach((t, i) => t.setAttribute('stroke', i <= s ? rgbCss(i / S) : '#3a4a70'));
    $('slider').value = s;
    document.querySelectorAll('#modes button').forEach((b) => b.classList.toggle('on', +b.dataset.mode === ui.mode));
    $('ghost').classList.toggle('on', ui.ghost);
    $('settlebtn').classList.toggle('on', ui.settle);
    $('bandsbtn').classList.toggle('on', ui.bands);
    const sel = setName === 'predicted' ? 1 : 0;
    METRICS.forEach((row, k) => {
      const raw = row[2] ? row[2](m[setName] || {}) : meanSettle(sel ? settleP : settleS, ms);
      metricEls[k].val.textContent = row[0].startsWith('Brightness') ? (raw == null ? '–' : Math.round(raw)) : num(raw);
      const [a, b] = metricSeries[k];
      EX.spark(metricEls[k].cv, sel ? [a, b] : [b, a], ms, ['rgba(141,154,179,.45)', sel ? '#ff5fb4' : '#9b6bff']);
    });
    bandRows.forEach((row, i) => {
      const n = BANDNAMES[i];
      const v = (m[setName].bands || {})[n], o = (m[other].bands || {})[n];
      row.querySelector('.fill').style.width = `${Math.max(0, (v ?? 0)) * 100}%`;
      row.querySelector('.ghost').style.left = `${Math.max(0, (o ?? 0)) * 100}%`;
      row.querySelector('.v').textContent = num(v, 2);
    });
    drawSettleMap(ms);
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
  function setPlayIcon() { $('play').textContent = ui.playing ? '❚❚ Pause' : '▶ Listen to this step'; if (typeof drawGame === 'function') drawGame(); }
  function setResolveLabel() { $('resolve').innerHTML = ui.animating ? '❚❚ Resolving' : '▶ Resolve <kbd>A</kbd>'; }
  function togglePlay() { if (ui.playing) audio.pause(); else audio.start(); setPlayIcon(); }
  $('play').onclick = togglePlay;
  // Resolve ties each step to a moment in the song (step × pace), so starting from any step plays from that
  // step's moment, and step 32 always lands a second before the last chorus.
  $('resolve').onclick = () => {
    if (!audio.ready) return;
    ui.animating = !ui.animating;
    if (ui.animating) {
      if (ui.target >= S) setStep(0, true);
      audio.seek(ui.target * ui.pace);
      if (!ui.playing) { audio.start(); setPlayIcon(); }   // the process is something to hear, not only watch
    }
    $('resolve').classList.toggle('on', ui.animating); setResolveLabel(); drawGame();
  };
  $('slider').oninput = (e) => setStep(+e.target.value);
  document.querySelectorAll('#modes button').forEach((b) => (b.onclick = () => setMode(+b.dataset.mode)));
  $('ghost').onclick = () => { ui.ghost = !ui.ghost; renderPanel(); };
  $('settlebtn').onclick = () => { ui.settle = !ui.settle; renderPanel(); };
  $('bandsbtn').onclick = () => { ui.bands = !ui.bands; renderPanel(); };
  window.addEventListener('keydown', (e) => {
    const k = e.key.toLowerCase();
    if (k === ' ') { e.preventDefault(); togglePlay(); }
    else if (k === 'arrowright') { e.preventDefault(); setStep(ui.target + 1); }
    else if (k === 'arrowleft') { e.preventDefault(); setStep(ui.target - 1); }
    else if (k === '1' || k === '2' || k === '3') setMode(+k - 1);
    else if (k === 'g') $('ghost').click();
    else if (k === 'm' && LOCAL) $('markbtn').click();
    else if (k === 'c') $('settlebtn').click();
    else if (k === 'b') $('bandsbtn').click();
    else if (k === 'a') $('resolve').click();
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
    if (!p || e.buttons) { tip.hide(); uniforms.uHoverZ.value = -999; hoverBand = -1; return; }
    const s = Math.round(ui.target), at = (arr, k) => arr[k * B * C + p.b * C + p.c] / 255;
    const view = ui.mode === 1 ? 'predicted' : 'state', ks = ui.mode === 2 ? S : s;
    const [lo, hi] = D.range_log10, db = (v) => ((lo + v * (hi - lo)) * 10 - hi * 10).toFixed(0);
    uniforms.uHoverZ.value = bandZ[p.b];
    const band = PITCH[bandOf(D.band_lo_hz[p.b])]; hoverBand = bandOf(D.band_lo_hz[p.b]);
    tip.show(`<b>${fmt(p.sec)}</b> into the song · <b>${band[0]}</b> <span class="m">(${Math.round(D.band_lo_hz[p.b])}–${Math.round(D.band_hi_hz[p.b])} Hz)</span><br>
      <span class="m">level here at step ${s}, in dB below the loudest spots (99.9th percentile):</span><br>
      now ${db(at(state, s))} · predicted ${db(at(pred, s))} · finished ${db(at(state, S))}<br>
      <span class="m">how closely this band's level rises and falls like the finished song's, ${view} view (1 = perfectly):</span> ${num((view === 'state' ? settleS : settleP)[ks * B + p.b] / 255, 2)}<br>
      <span class="m">click to play from here</span>`, e.clientX, e.clientY);
  });
  st.canvas.addEventListener('pointerleave', () => { tip.hide(); uniforms.uHoverZ.value = -999; hoverBand = -1; });
  st.canvas.addEventListener('pointerdown', (e) => { downAt = [e.clientX, e.clientY]; });
  st.canvas.addEventListener('pointerup', (e) => {
    if (!downAt || Math.hypot(e.clientX - downAt[0], e.clientY - downAt[1]) > 4) return;
    const p = pick(e); if (!p) return;
    audio.seek(p.sec); if (!ui.playing) { audio.start(); setPlayIcon(); }
  });

  // --- frame loop ---------------------------------------------------------------------
  let intro = reduced ? 1 : 0;
  st.onFrame((dt, now) => {
    // The dial moves at once, but a step's audio can take a moment to decode: say which step is still sounding.
    const want = Math.round(ui.target), heard = audio.loadedStep;
    const sw = !ui.playing || heard === want ? '' : heard < 0 ? `Loading step ${want}…` : `Loading step ${want} · still hearing step ${heard}`;
    if ($('switchstatus').textContent !== sw) $('switchstatus').textContent = sw;
    if (ui.animating && ui.playing) {                   // the step follows the song; paused, it holds
      const want = Math.min(S, Math.floor(audio.time() / ui.pace + 1e-6));
      if (want !== ui.target) setStep(want, true);
      if (want >= S) { ui.animating = false; $('resolve').classList.remove('on'); setResolveLabel(); drawGame(); }
    }
    const k = reduced ? 1 : 1 - Math.exp(-dt * 7);
    ui.shown += (ui.target - ui.shown) * k; if (Math.abs(ui.target - ui.shown) < 1e-3) ui.shown = ui.target;
    const mixGoal = ui.mode === 1 ? 1 : 0, finGoal = ui.mode === 2 ? 1 : 0;
    ui.mixT += (mixGoal - ui.mixT) * k; ui.finT += (finGoal - ui.finT) * k;
    uniforms.uStep.value = ui.shown; uniforms.uMix.value = ui.mixT; uniforms.uFin.value = ui.finT;
    uniforms.uSettleOn.value += ((ui.settle ? 1 : 0) - uniforms.uSettleOn.value) * k;
    bandRoof.forEach((m, i) => {
      const want = ui.bands ? (hoverBand === i ? 0.34 : hoverBand >= 0 ? 0.07 : 0.14) : 0;
      m.material.opacity += (want - m.material.opacity) * k; m.visible = m.material.opacity > 0.004;
    });
    bandLabels.forEach((el, i) => el.classList.toggle('hot', hoverBand === i));
    if (hoverBand >= 0 && hoverBand !== blockBand) placeBlock(hoverBand);
    blockMat.opacity += ((hoverBand >= 0 ? 0.16 : 0) - blockMat.opacity) * k;
    blockEdges.material.opacity = blockMat.opacity * 4; block.visible = blockMat.opacity > 0.004;
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
      const c = rampJS(0.1 + 0.9 * Math.max(0, v)), lift = 0.7 + 0.6 * Math.max(0, v); wall.setColorAt(b, col.setRGB(c[0] * lift, c[1] * lift, c[2] * lift));
      if (ui.bands) { const hb = bandOf(D.band_lo_hz[b]); wall.setColorAt(b, col.set(PITCH[hb][3]).multiplyScalar((0.35 + 0.65 * Math.max(0, v)) * (hoverBand < 0 || hoverBand === hb ? 1 : 0.4))); }
    }
    wall.instanceMatrix.needsUpdate = true; wall.instanceColor.needsUpdate = true;

    if (intro < 1) {       // a slow settle-in of the camera, once
      intro = Math.min(1, intro + dt * 0.35);
      const e = 1 - Math.pow(1 - intro, 3);
      camera.position.lerpVectors(new T.Vector3(-190, 150, 240), home.pos, e);
    }
    L.update();
    return ui.playing || ui.animating || intro < 1;    // busy; idle, the page stops drawing
  });

  renderPanel();
  setPlayIcon();
  st.renderOnce();          // one frame even if the page opens in a background tab
})();
