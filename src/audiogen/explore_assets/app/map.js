/* "Take map": every finished take as a glowing point, placed by envelope similarity. */
(function () {
  'use strict';
  const { T, reduced, fmt, num, css } = EX;
  const D = window.MAP;
  EX.nav('map');
  const $ = (id) => document.getElementById(id);
  const takes = D.takes.map((t, i) => ({ ...t, i }));
  document.getElementById('runlabel').innerHTML = `<b>${takes.length}</b> takes · ${D.runs.length} runs`;

  const GROUPS = {
    'baseline': { color: [1, 1, 1], label: 'the baseline take' },
    'seed study': { color: [1, 0.37, 0.71], label: 'seed study: new plan, tokens and noise' },
    'fixed ABC': { color: [1, 0.6, 0.3], label: 'fixed ABC: same plan, new tokens' },
    'fixed semantic': { color: [1, 0.82, 0.48], label: 'fixed semantic: same tokens' },
    'noise seed': { color: [0.61, 0.42, 1], label: 'same tokens, other acoustic noise' },
    'ODE steps': { color: [0.31, 0.55, 1], label: 'same tokens and noise, N-step solve' },
  };
  const groupOf = (t) => GROUPS[t.group] || { color: [0.7, 0.8, 1], label: t.group };
  const hidden = new Set();

  const st = EX.stage($('stage'), { bloom: 1.1, radius: 0.7, threshold: 0.1, fov: 40 });
  const { scene, camera, controls } = st;
  const SCALE = 60;
  const home = { pos: new T.Vector3(105, 62, 130), target: new T.Vector3(-8, -4, 0) };
  camera.position.copy(home.pos); controls.target.copy(home.target);
  controls.autoRotate = !reduced; controls.autoRotateSpeed = 0.35;
  controls.minDistance = 20; controls.maxDistance = 600;
  EX.dust(scene, 2200, 1000);
  const floorY = -SCALE * 1.15;
  EX.floor(scene, 800, floorY);
  const P = (t) => new T.Vector3(t.pos[0] * SCALE, t.pos[1] * SCALE, t.pos[2] * SCALE);

  // Points: glowing sprites (shader) + invisible spheres for picking.
  const n = takes.length;
  const pos = new Float32Array(n * 3), colr = new Float32Array(n * 3), size = new Float32Array(n), vis = new Float32Array(n).fill(1);
  takes.forEach((t, i) => { pos.set(P(t).toArray(), i * 3); colr.set(groupOf(t).color, i * 3); size[i] = t.group === 'baseline' ? 1.6 : t.group === 'seed study' || t.group === 'fixed ABC' ? 1.25 : 1; });
  const g = new T.BufferGeometry();
  g.setAttribute('position', new T.BufferAttribute(pos, 3));
  g.setAttribute('color', new T.BufferAttribute(colr, 3));
  g.setAttribute('aSize', new T.BufferAttribute(size, 1));
  g.setAttribute('aVis', new T.BufferAttribute(vis, 1));
  const pu = { uTime: { value: 0 }, uSel: { value: -1 }, uHover: { value: -1 }, uPlaying: { value: 0 }, uPx: { value: EX.DPR * window.innerHeight } };
  const pts = new T.Points(g, new T.ShaderMaterial({
    uniforms: pu, transparent: true, depthWrite: false, blending: T.AdditiveBlending, vertexColors: true,
    vertexShader: `attribute float aSize; attribute float aVis; uniform float uTime, uSel, uHover, uPlaying, uPx;
      varying vec3 vC; varying float vA; varying float vS;
      void main(){
        vC = color; float id = float(gl_VertexID);
        float sel = id == uSel ? 1.0 : 0.0, hov = id == uHover ? 1.0 : 0.0;
        vS = sel;
        float pulse = sel * uPlaying * (0.25 + 0.25 * sin(uTime * 5.0));
        vec4 mv = modelViewMatrix * vec4(position, 1.0);
        gl_PointSize = aVis * aSize * (8.0 + 3.0 * sel + 2.0 * hov + pulse * 5.0) * uPx / -mv.z;
        vA = aVis;
        gl_Position = projectionMatrix * mv;
      }`,
    fragmentShader: `varying vec3 vC; varying float vA; varying float vS;
      void main(){
        vec2 p = gl_PointCoord - 0.5; float r = length(p) * 2.0;
        if (r > 1.0) discard;
        float core = smoothstep(0.16, 0.0, r), halo = pow(1.0 - r, 3.0) * 0.75, ring = smoothstep(0.08, 0.0, abs(r - 0.8)) * vS;
        vec3 c = vC * (halo + core * 1.3) + vec3(1.0) * core * 0.9 + vec3(1.0, 0.85, 0.95) * ring;
        gl_FragColor = vec4(c * vA, 1.0);
      }`,
  }));
  scene.add(pts);
  const pickers = takes.map((t) => {
    const m = new T.Mesh(new T.SphereGeometry(2.4, 10, 8), new T.MeshBasicMaterial({ visible: false }));
    m.position.copy(P(t)); m.userData.i = t.i; scene.add(m); return m;
  });

  // Stems to the floor with a glowing foot, so depth reads.
  const stemPos = [], stemCol = [];
  takes.forEach((t) => { const p = P(t), c = groupOf(t).color; stemPos.push(p.x, p.y, p.z, p.x, floorY, p.z); stemCol.push(...c.map((v) => v * 0.5), ...c.map((v) => v * 0.05)); });
  const stems = new T.LineSegments(new T.BufferGeometry().setAttribute('position', new T.Float32BufferAttribute(stemPos, 3)).setAttribute('color', new T.Float32BufferAttribute(stemCol, 3)),
    new T.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.6, blending: T.AdditiveBlending, depthWrite: false }));
  scene.add(stems);
  const footGeo = new T.RingGeometry(0.6, 1.4, 24);
  takes.forEach((t) => { const p = P(t); const f = new T.Mesh(footGeo, new T.MeshBasicMaterial({ color: new T.Color(...groupOf(t).color), transparent: true, opacity: 0.45, side: T.DoubleSide, toneMapped: false })); f.rotation.x = -Math.PI / 2; f.position.set(p.x, floorY + 0.05, p.z); scene.add(f); t.foot = f; });

  // The N-step path (1 -> 2 -> ... -> the 32-step take -> 48 -> 64) and noise-seed spokes from their token siblings.
  const base = takes.find((t) => t.group === 'baseline');
  const ode = takes.filter((t) => t.group === 'ODE steps').sort((a, b) => a.ode_steps - b.ode_steps);
  const lines = new T.Group(); scene.add(lines);
  if (ode.length && base) {
    const seq = [...ode.filter((t) => t.ode_steps < 32), base, ...ode.filter((t) => t.ode_steps > 32)].filter((t) => t.run === base.run || t === base);
    if (seq.length > 1) {
      const curve = new T.CatmullRomCurve3(seq.map(P), false, 'centripetal');
      const tube = new T.Mesh(new T.TubeGeometry(curve, 200, 0.35, 8, false), new T.MeshBasicMaterial({ color: 0x4f8cff, transparent: true, opacity: 0.55, toneMapped: false }));
      lines.add(tube);
    }
  }
  const spokes = [];
  takes.filter((t) => t.group === 'noise seed').forEach((t) => {
    const sib = takes.find((b) => b.run === t.run && b.detail === 'the take itself') || base;
    if (sib) spokes.push(...P(sib).toArray(), ...P(t).toArray());
  });
  if (spokes.length) lines.add(new T.LineSegments(new T.BufferGeometry().setAttribute('position', new T.Float32BufferAttribute(spokes, 3)),
    new T.LineDashedMaterial({ color: 0x9b6bff, transparent: true, opacity: 0.7, dashSize: 1.2, gapSize: 0.8 })).computeLineDistances());

  const L = EX.labels(st.host, camera);
  takes.forEach((t) => { t.label3d = L.add(t.ode_steps != null ? `${t.ode_steps}` : t.label, P(t).add(new T.Vector3(0, 3.6, 0)), 'tick'); });

  // --- panel ----------------------------------------------------------------------------
  const counts = {};
  takes.forEach((t) => (counts[t.group] = (counts[t.group] || 0) + 1));
  $('groups').innerHTML = Object.keys(counts).map((gname) => `<div class="item on" data-g="${gname}">
      <span class="dot" style="background:${css(groupOf({ group: gname }).color)};box-shadow:0 0 10px ${css(groupOf({ group: gname }).color)}"></span>
      <span class="t">${gname}<small>${groupOf({ group: gname }).label}</small></span><span class="n">${counts[gname]}</span></div>`).join('');
  document.querySelectorAll('#groups .item').forEach((el) => (el.onclick = () => {
    const gname = el.dataset.g;
    hidden.has(gname) ? hidden.delete(gname) : hidden.add(gname);
    el.classList.toggle('on', !hidden.has(gname));
    takes.forEach((t) => { const v = hidden.has(t.group) ? 0 : 1; vis[t.i] = v; t.label3d.visible = !!v; t.foot.visible = !!v; pickers[t.i].visible = !!v; });
    g.getAttribute('aVis').needsUpdate = true;
  }));
  $('count').textContent = `${takes.length}`;
  $('takes').innerHTML = takes.map((t) => `<div class="item" data-i="${t.i}"><span class="dot" style="background:${css(groupOf(t).color)}"></span>
      <span class="t">${t.label}<small>${t.run}</small></span><span class="n">${num(t.similarity_to_first_baseline, 3)}</span></div>`).join('');
  const rows = [...document.querySelectorAll('#takes .item')];
  rows.forEach((el) => (el.onclick = () => select(+el.dataset.i, true)));
  $('method').innerHTML = `${D.method}.${D.skipped && D.skipped.length ? '<br>Skipped: ' + D.skipped.join('; ') : ''}<br>The right-hand number in “All takes” is envelope correlation with the baseline take.`;

  // --- audio: one element per take; switching can keep the moment ---------------------------------
  let el = null, sel = -1, keep = true;
  function select(i, play) {
    const t = takes[i];
    const at = el && keep ? el.currentTime : 0, wasPlaying = el && !el.paused;
    if (el) { el.pause(); el.removeAttribute('src'); el.load(); }
    el = new Audio(); el.preload = 'auto'; el.src = t.audio;
    if (at) el.addEventListener('loadedmetadata', () => { el.currentTime = Math.min(at, el.duration - 0.1); }, { once: true });
    sel = i; pu.uSel.value = i;
    if (play || wasPlaying) el.play().catch(() => null);
    el.onplay = el.onpause = () => { $('play').textContent = el.paused ? '▶' : '❚❚'; };
    rows.forEach((r) => r.classList.toggle('on', +r.dataset.i === i));
    rows[i].scrollIntoView({ block: 'nearest' });
    const a = t.aliases && t.aliases.length ? `<br><span class="m">identical PCM also in: ${t.aliases.join(', ')}</span>` : '';
    $('selhint').textContent = t.group;
    $('sel').innerHTML = `<div class="lyric">${t.label}</div>
      <div class="note">${t.detail} · ${t.run}${t.seed != null ? ` · seed ${t.seed}` : ''}${t.noise_seed != null ? ` · noise ${t.noise_seed}` : ''}${t.ode_steps != null ? ` · ${t.ode_steps} steps` : ''}<br>
      notes/syllable (score): <b>${num(t.notes_per_syllable, 3)}</b> · envelope r with baseline: <b>${num(t.similarity_to_first_baseline, 3)}</b>${a}</div>`;
    const near = D.similarity[i].map((r, j) => [r, j]).filter(([, j]) => j !== i).sort((a, b) => b[0] - a[0]).slice(0, 5);
    $('near').innerHTML = near.map(([r, j]) => `<div class="item" data-i="${j}"><span class="dot" style="background:${css(groupOf(takes[j]).color)}"></span>
      <span class="t">${takes[j].label}<small>${takes[j].group}</small></span><span class="n">${r.toFixed(3)}</span></div>`).join('');
    document.querySelectorAll('#near .item').forEach((r) => (r.onclick = () => select(+r.dataset.i, true)));
  }
  $('play').onclick = () => { if (!el) return select(base ? base.i : 0, true); el.paused ? el.play().catch(() => null) : el.pause(); };
  $('keep').onclick = () => { keep = !keep; $('keep').classList.toggle('on', keep); };
  $('keep').classList.toggle('on', keep);

  // --- picking ----------------------------------------------------------------------------
  const ray = new T.Raycaster(), mouse = new T.Vector2(), tip = EX.tip();
  let downAt = null;
  function pick(e) {
    const r = st.canvas.getBoundingClientRect();
    mouse.set((e.clientX - r.left) / r.width * 2 - 1, -(e.clientY - r.top) / r.height * 2 + 1);
    ray.setFromCamera(mouse, camera);
    const hit = ray.intersectObjects(pickers.filter((p) => p.visible), false)[0];
    return hit ? hit.object.userData.i : -1;
  }
  st.canvas.addEventListener('pointermove', (e) => {
    if (e.buttons) return;
    const i = pick(e); pu.uHover.value = i;
    st.canvas.style.cursor = i >= 0 ? 'pointer' : '';
    if (i < 0) return tip.hide();
    const t = takes[i];
    tip.show(`<b>${t.label}</b><br><span class="m">${t.group} · ${t.run}</span><br>notes/syllable ${num(t.notes_per_syllable, 2)} · r with baseline ${num(t.similarity_to_first_baseline, 3)}<br><span class="m">click to play</span>`, e.clientX, e.clientY);
  });
  st.canvas.addEventListener('pointerleave', () => { tip.hide(); pu.uHover.value = -1; });
  st.canvas.addEventListener('pointerdown', (e) => { downAt = [e.clientX, e.clientY]; controls.autoRotate = false; });
  st.canvas.addEventListener('pointerup', (e) => {
    if (!downAt || Math.hypot(e.clientX - downAt[0], e.clientY - downAt[1]) > 4) return;
    const i = pick(e); if (i >= 0) select(i, true);
  });
  window.addEventListener('keydown', (e) => {
    const k = e.key.toLowerCase();
    if (k === ' ') { e.preventDefault(); $('play').click(); }
    else if (k === 'arrowright' || k === 'arrowleft') { e.preventDefault(); select(((sel < 0 ? 0 : sel + (k === 'arrowright' ? 1 : -1)) + n) % n, true); }
    else if (k === 'k') $('keep').click();
    else if (k === 'o') controls.autoRotate = !controls.autoRotate && !reduced;
    else if (k === 'r') { camera.position.copy(home.pos); controls.target.copy(home.target); }
  });
  window.addEventListener('resize', () => (pu.uPx.value = EX.DPR * window.innerHeight));

  st.onFrame((dt, now) => {
    pu.uTime.value = reduced ? 0 : now;
    pu.uPlaying.value = el && !el.paused ? 1 : 0;
    $('ptime').textContent = el ? `${fmt(el.currentTime)} / ${fmt(el.duration)}` : '';
    L.update();
  });
  st.renderOnce();
})();
