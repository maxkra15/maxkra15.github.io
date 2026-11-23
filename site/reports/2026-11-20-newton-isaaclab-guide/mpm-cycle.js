(() => {
  const lesson = document.querySelector('#mpm-cycle');
  const canvas = lesson.querySelector('canvas');
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  const phases = [...lesson.querySelectorAll('.mpm-phase')];
  const selectors = [...lesson.querySelectorAll('[data-mpm-phase]')];
  const controls = lesson.querySelector('.mpm-controls');
  const [previous, next, play] = controls.querySelectorAll('button');
  const motion = window.matchMedia('(prefers-reduced-motion: reduce)');
  const colorScheme = window.matchMedia('(prefers-color-scheme: dark)');
  const labels = ['Particle state', 'Weighted transfer', 'Stress and contact', 'New particle state'];
  const width = 600;
  const height = 330;
  const floor = 270;
  const cell = 64;
  const originX = 100;
  const originY = 78;
  const gridRight = 484;
  const animationDuration = 650;
  const stageDuration = 2600;
  const sample = {x: 252, y: 222};
  const corners = [
    {x: 228, y: 206, labelX: -20, labelY: -15, free: [14, 22], solved: [9, 1]},
    {x: 292, y: 206, labelX: 18, labelY: -15, free: [23, 23], solved: [25, 5]},
    {x: 228, y: floor, labelX: -20, labelY: 23, free: [12, 27], solved: [8, 0]},
    {x: 292, y: floor, labelX: 18, labelY: 23, free: [20, 25], solved: [14, 0]},
  ];
  // Bilinear Q1 weights, in screen order from the top left.
  const u = (sample.x - corners[0].x) / cell;
  const v = (sample.y - corners[0].y) / cell;
  const weights = [(1 - u) * (1 - v), u * (1 - v), (1 - u) * v, u * v];
  const displacement = [0, 1].map(axis =>
    weights.reduce((sum, weight, i) => sum + weight * corners[i].solved[axis], 0));
  const particles = [];
  for (let row = 0; row < 6; row++) {
    for (let column = 0; column < 5; column++) {
      const x = 210 + 27 * column + row * 7 + Math.sin(row * 2 + column * 4) * 7;
      const y = 121 + row * 24 + Math.cos(row * 3 + column * 5) * 6;
      if (Math.hypot(x - sample.x, y - sample.y) >= 19) particles.push({x, y});
    }
  }

  let stage = 0;
  let raf = 0;
  let timer = 0;
  let playing = false;
  let elapsed = 0;
  let paused = false;
  let started = 0;
  let progress = 1;
  let visible = false;
  const interpolate = (from, to, t) => from + (to - from) * t;

  function draw() {
    const w = canvas.clientWidth;
    if (!w) return;
    const scale = Math.min(window.devicePixelRatio || 1, 2);
    const backing = Math.round(w * scale);
    if (canvas.width !== backing) {
      canvas.width = backing;
      canvas.height = Math.round(backing * height / width);
    }
    ctx.setTransform(canvas.width / width, 0, 0, canvas.height / height, 0, 0);
    const style = getComputedStyle(lesson);
    const colors = Object.fromEntries(['surface', 'ink', 'muted', 'line', 'grid', 'particle', 'node', 'strain', 'collider']
      .map(key => [key, style.getPropertyValue('--' + key).trim()]));
    const eased = 1 - (1 - progress) ** 3;

    function path(x1, y1, x2, y2, color, lineWidth = 1, dash = []) {
      ctx.strokeStyle = color;
      ctx.lineWidth = lineWidth;
      ctx.setLineDash(dash);
      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    function dot(x, y, size, color) {
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(x, y, size, 0, Math.PI * 2);
      ctx.fill();
    }

    function arrow(x, y, dx, dy, color, dashed = false) {
      const tipX = x + dx;
      const tipY = y + dy;
      const angle = Math.atan2(dy, dx);
      path(x, y, tipX, tipY, color, 2.2, dashed ? [4, 3] : []);
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.moveTo(tipX, tipY);
      ctx.lineTo(tipX - 8 * Math.cos(angle - 0.55), tipY - 8 * Math.sin(angle - 0.55));
      ctx.lineTo(tipX - 8 * Math.cos(angle + 0.55), tipY - 8 * Math.sin(angle + 0.55));
      ctx.closePath();
      ctx.fill();
    }

    function label(value, x, y, color = colors.muted, align = 'left') {
      ctx.textAlign = align;
      ctx.textBaseline = 'middle';
      ctx.font = '600 16px ' + style.getPropertyValue('--f-mono');
      ctx.fillStyle = colors.surface;
      const measured = ctx.measureText(value).width;
      const left = align === 'left' ? x : x - measured / 2;
      ctx.fillRect(left - 3, y - 11, measured + 6, 22);
      ctx.fillStyle = color;
      ctx.fillText(value, x, y);
    }

    ctx.fillStyle = colors.surface;
    ctx.fillRect(0, 0, width, height);
    if (stage === 1 || stage === 2) {
      for (let x = originX; x <= gridRight; x += cell) path(x, originY, x, floor, colors.line);
      for (let y = originY; y <= floor; y += cell) path(originX, y, gridRight, y, colors.line);
      ctx.globalAlpha = 0.14;
      ctx.fillStyle = stage === 2 ? colors.strain : colors.node;
      ctx.fillRect(corners[0].x, corners[0].y, cell, cell);
      ctx.globalAlpha = 1;
      ctx.strokeStyle = stage === 2 ? colors.strain : colors.node;
      ctx.lineWidth = 2;
      ctx.strokeRect(corners[0].x, corners[0].y, cell, cell);
    } else {
      for (let x = 34; x < width; x += 24) {
        for (let y = 54; y < floor; y += 24) dot(x, y, 1.1, colors.line);
      }
    }
    ctx.globalAlpha = 0.13;
    ctx.fillStyle = colors.collider;
    ctx.fillRect(34, floor, width - 68, 30);
    ctx.globalAlpha = 1;
    for (let x = 38; x <= width - 60; x += 12) {
      path(x, floor + 29, x + 18, floor + 2, colors.line);
    }
    path(34, floor, width - 34, floor, colors.collider, 2);
    label('Floor', width - 99, floor + 16, colors.collider);

    ctx.globalAlpha = stage === 1 || stage === 2 ? 0.34 : 0.72;
    for (const {x, y} of particles) {
      const offset = stage === 3 ? (10 + (y - 120) / 24) * eased : 0;
      dot(x + offset, y, 3.8, colors.particle);
    }
    ctx.globalAlpha = 1;

    if (stage === 1 || stage === 3) {
      corners.forEach((corner, index) => {
        const {x, y} = corner;
        const weight = weights[index];
        path(sample.x, sample.y, x, y, colors.node, 1 + 4 * weight, [4, 4]);
        dot(x, y, 4 + 8 * Math.sqrt(weight), colors.node);
        if (stage === 1) {
          label((100 * weight).toFixed(1) + '%', x + corner.labelX, y + corner.labelY, colors.node, 'center');
        }
        if (progress < 1) {
          const t = stage === 1 ? eased : 1 - eased;
          dot(interpolate(sample.x, x, t), interpolate(sample.y, y, t), 3.5, colors.particle);
        }
      });
    }

    if (stage === 2) {
      corners.forEach(({x, y, free, solved}) => {
        arrow(x, y, ...free, colors.muted, true);
        arrow(x, y, ...solved.map((value, axis) => interpolate(free[axis], value, eased)), colors.node);
        dot(x, y, 5, y === floor ? colors.collider : colors.node);
      });
      label('Floor blocks inward motion', 311, 238, colors.collider);
    }

    if (stage === 0) {
      path(sample.x - 105, sample.y, sample.x - 13, sample.y, colors.particle, 1.4);
      label('Sample point', 47, sample.y, colors.ink);
      arrow(sample.x, sample.y, 14, 26, colors.particle);
    }
    if (stage === 3) {
      arrow(sample.x, sample.y, ...displacement.map(value => value * 2.2), colors.particle);
      label('Updated motion', 325, 210, colors.ink);
    }
    const offset = stage === 3 ? eased : 0;
    const px = sample.x + offset * displacement[0];
    const py = sample.y + offset * displacement[1];
    dot(px, py, 11, colors.surface);
    dot(px, py, 7, colors.particle);
    label(labels[stage], 34, 29, colors.ink);
  }

  function updateControls() {
    previous.disabled = stage === 0 || playing;
    next.disabled = stage === phases.length - 1 || playing;
    play.hidden = motion.matches;
    play.textContent = playing ? 'Pause' : (paused ? 'Resume' : 'Play all stages');
  }

  function select(index, animate = true) {
    stage = index;
    paused = false;
    elapsed = 0;
    progress = animate && visible && !motion.matches && stage !== 0 ? 0 : 1;
    phases.forEach((phase, i) => { phase.hidden = i !== stage; });
    selectors.forEach((button, i) => button.setAttribute('aria-pressed', String(i === stage)));
    canvas.setAttribute('aria-label', phases[stage].dataset.diagram);
    updateControls();
    draw();
    runStage();
  }

  function runStage() {
    cancelAnimationFrame(raf);
    clearTimeout(timer);
    started = performance.now() - elapsed;
    if (progress < 1) raf = requestAnimationFrame(frame);
    if (playing) {
      timer = setTimeout(() => {
        if (stage < phases.length - 1) select(stage + 1);
        else stop();
      }, stageDuration - elapsed);
    }
  }

  function frame(now) {
    progress = Math.min(1, (now - started) / animationDuration);
    draw();
    if (progress < 1) raf = requestAnimationFrame(frame);
  }

  function stop() {
    playing = false;
    paused = false;
    progress = 1;
    cancelAnimationFrame(raf);
    clearTimeout(timer);
    updateControls();
    draw();
  }

  previous.addEventListener('click', () => select(stage - 1));
  next.addEventListener('click', () => select(stage + 1));
  selectors.forEach((button, i) => button.addEventListener('click', () => {
    playing = false;
    select(i);
  }));
  play.addEventListener('click', () => {
    if (playing) {
      playing = false;
      paused = true;
      elapsed = performance.now() - started;
      cancelAnimationFrame(raf);
      clearTimeout(timer);
      updateControls();
    } else if (!motion.matches) {
      playing = true;
      if (paused) {
        paused = false;
        updateControls();
        runStage();
      } else {
        select(0);
      }
    }
  });
  motion.addEventListener('change', stop);
  colorScheme.addEventListener('change', draw);
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) stop();
  });
  new IntersectionObserver(([entry]) => {
    visible = entry.isIntersecting;
    if (!visible) stop();
  }).observe(canvas);
  new ResizeObserver(draw).observe(canvas);
  new MutationObserver(draw).observe(document.documentElement, {
    attributes: true, attributeFilter: ['data-theme'],
  });
  controls.hidden = false;
  lesson.querySelector('.mpm-phase-select').hidden = false;
  canvas.hidden = false;
  select(0, false);
})();
