(() => {
  const lesson = document.querySelector('#mpm-cycle');
  const canvas = lesson.querySelector('canvas');
  const ctx = canvas.getContext('2d');
  if (!ctx) return;

  const handle = lesson.querySelector('#mpm-point');
  const diagram = lesson.querySelector('.mpm-figure');
  const phases = [...lesson.querySelectorAll('.mpm-phase')];
  const selectors = [...lesson.querySelectorAll('[data-mpm-phase]')];
  const floorInput = lesson.querySelector('#mpm-floor');
  const reset = lesson.querySelector('#mpm-reset');
  const scrubber = lesson.querySelector('#mpm-scrubber');
  const play = lesson.querySelector('#mpm-play');
  const announcement = lesson.querySelector('#mpm-announcement');
  const live = Object.fromEntries([...lesson.querySelectorAll('[data-mpm-value]')]
    .map(el => [el.dataset.mpmValue, el]));
  const bars = [...lesson.querySelectorAll('[data-mpm-weight]')];
  const motion = window.matchMedia('(prefers-reduced-motion: reduce)');
  const colorScheme = window.matchMedia('(prefers-color-scheme: dark)');

  // The cell is one metre wide. These three samples are fixed inputs to the step.
  const neighbors = [
    {x: 0.23, y: 0.73, mass: 1.1, velocity: [0.54, -0.44]},
    {x: 0.58, y: 0.59, mass: 1.1, velocity: [0.98, -0.86]},
    {x: 0.81, y: 0.19, mass: 0.9, velocity: [1.20, -1.25]},
  ];
  const initial = {x: 0.39, y: 0.22};
  const point = {...initial, mass: 1, velocity: [1.36, -1.0]};
  const corners = [[0, 1], [1, 1], [0, 0], [1, 0]];
  const phaseNames = ['Carry', 'Transfer to grid', 'Solve contact', 'Move point'];
  const dt = 0.10;
  const gravity = -9.81;

  let geometry;
  let pointer;
  let time = 0;
  let stage = -1;
  let data;
  let colors;
  let animationFrame = 0;
  let playing = false;
  let playbackOrigin = 0;
  let playbackOffset = 0;
  const holdDuration = 480;
  const phaseDuration = 1000;

  function weights(x, y) {
    return [(1 - x) * y, x * y, (1 - x) * (1 - y), x * (1 - y)];
  }

  function step() {
    const mass = [0, 0, 0, 0];
    const momentum = corners.map(() => [0, 0]);
    for (const particle of [...neighbors, point]) {
      weights(particle.x, particle.y).forEach((weight, i) => {
        const portion = weight * particle.mass;
        mass[i] += portion;
        momentum[i][0] += portion * particle.velocity[0];
        momentum[i][1] += portion * (particle.velocity[1] + dt * gravity);
      });
    }
    const free = momentum.map(([x, y], i) => [x / mass[i], y / mass[i]]);
    // Frictionless, stationary floor at y = 0. Only the lower two nodes touch it.
    const solved = free.map(([x, y], i) => [x, floorInput.checked && i >= 2 ? Math.max(0, y) : y]);
    const support = weights(point.x, point.y);
    const velocity = [0, 1].map(axis => solved.reduce((sum, v, i) => sum + support[i] * v[axis], 0));
    return {mass, free, solved, support, velocity};
  }

  function updateReadout() {
    live.position.textContent = `(${point.x.toFixed(2)}, ${point.y.toFixed(2)}) m`;
    live.start.textContent = formatVector(point.velocity) + ' m/s';
    data.support.forEach((weight, i) => {
      live['weight' + i].textContent = (100 * weight).toFixed(1) + '%';
      bars[i].style.setProperty('--share', (100 * weight) + '%');
    });
    live.left.textContent = formatSigned(data.free[2][1]) + ' → ' + formatSigned(data.solved[2][1]);
    live.right.textContent = formatSigned(data.free[3][1]) + ' → ' + formatSigned(data.solved[3][1]);
    live.velocity.textContent = formatVector(data.velocity) + ' m/s';
    live.displacement.textContent = formatVector(data.velocity.map(v => v * dt)) + ' m';
  }

  function formatSigned(value) {
    return (value < 0 ? '−' : '') + Math.abs(value).toFixed(2);
  }

  function formatVector([x, y]) {
    return '(' + formatSigned(x) + ', ' + formatSigned(y) + ')';
  }

  function readColors() {
    const style = getComputedStyle(lesson);
    colors = Object.fromEntries(['surface', 'ink', 'muted', 'line', 'particle', 'node', 'strain', 'collider', 'f-mono']
      .map(key => [key, style.getPropertyValue('--' + key).trim()]));
    draw();
  }

  function draw() {
    if (!data) return;
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;
    if (!width || !height) return;
    const ratio = Math.min(2, window.devicePixelRatio || 1);
    const backingWidth = Math.round(width * ratio);
    const backingHeight = Math.round(height * ratio);
    if (canvas.width !== backingWidth || canvas.height !== backingHeight) {
      canvas.width = backingWidth;
      canvas.height = backingHeight;
    }
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.clearRect(0, 0, width, height);
    const cell = Math.min(254, width - 96, height - 106);
    const left = (width - cell) / 2;
    const floor = height - 72;
    const toX = x => left + cell * x;
    const toY = y => floor - cell * y;
    const pointX = toX(point.x);
    const pointY = toY(point.y);
    const progress = time - Math.max(0, stage - 1);
    geometry = {left, floor, cell, pointX, pointY};
    handle.style.left = pointX + 'px';
    handle.style.top = pointY + 'px';

    function line(x1, y1, x2, y2, color, width = 1, dash = []) {
      ctx.strokeStyle = color;
      ctx.lineWidth = width;
      ctx.setLineDash(dash);
      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    function disk(x, y, radius, color, stroke = false) {
      ctx.beginPath();
      ctx.arc(x, y, radius, 0, 2 * Math.PI);
      if (stroke) {
        ctx.lineWidth = 1.6;
        ctx.strokeStyle = color;
        ctx.stroke();
      } else {
        ctx.fillStyle = color;
        ctx.fill();
      }
    }

    function arrow(x, y, dx, dy, color, dashed = false) {
      const length = Math.hypot(dx, dy);
      if (length < 2) return;
      const tipX = x + dx;
      const tipY = y + dy;
      const angle = Math.atan2(dy, dx);
      const head = Math.min(8, length * 0.45);
      line(x, y, tipX, tipY, color, 2, dashed ? [5, 4] : []);
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.moveTo(tipX, tipY);
      ctx.lineTo(tipX - head * Math.cos(angle - 0.52), tipY - head * Math.sin(angle - 0.52));
      ctx.lineTo(tipX - head * Math.cos(angle + 0.52), tipY - head * Math.sin(angle + 0.52));
      ctx.closePath();
      ctx.fill();
    }

    function label(text, x, y, color = colors.muted, align = 'center', weight = 500) {
      ctx.textAlign = align;
      ctx.textBaseline = 'middle';
      ctx.font = weight + ' 12px ' + colors['f-mono'];
      ctx.fillStyle = color;
      ctx.fillText(text, x, y);
    }

    // Keep the grid and floor fixed while the numerical state changes.
    const top = toY(1);
    ctx.globalAlpha = 0.27;
    ctx.fillStyle = colors.node;
    ctx.fillRect(left, top, cell, cell);
    ctx.globalAlpha = 1;
    ctx.fillStyle = colors.surface;
    ctx.fillRect(left + 1, top + 1, cell - 2, cell - 2);
    for (let i = 1; i < 4; i++) {
      const x = toX(i / 4);
      const y = toY(i / 4);
      line(x, top, x, floor, colors.line, 1, [2, 5]);
      line(left, y, left + cell, y, colors.line, 1, [2, 5]);
    }
    if (floorInput.checked) {
      ctx.globalAlpha = 0.09;
      ctx.fillStyle = colors.collider;
      ctx.fillRect(left - 25, floor, cell + 50, 34);
      ctx.globalAlpha = 0.42;
      for (let x = left - 25; x + 21 <= left + cell + 25; x += 9) {
        line(x, floor + 31, x + 21, floor + 1, colors.collider);
      }
      ctx.globalAlpha = 1;
    }
    line(left - 25, floor, left + cell + 25, floor,
      floorInput.checked ? colors.collider : colors.line, 1.8, floorInput.checked ? [] : [5, 4]);
    label(floorInput.checked ? 'Floor' : 'Floor off', left + cell + 24, floor + 53,
      floorInput.checked ? colors.collider : colors.muted, 'right');
    label('1 m', left + cell / 2, top - 16);

    for (const neighbor of neighbors) {
      ctx.globalAlpha = 0.48;
      disk(toX(neighbor.x), toY(neighbor.y), 4.1, colors.particle);
      ctx.globalAlpha = 1;
    }

    corners.forEach(([x, y], i) => {
      const cx = toX(x);
      const cy = toY(y);
      const weight = data.support[i];
      if (stage === 1) {
        ctx.globalAlpha = 0.23 + 0.57 * weight;
        line(pointX, pointY, cx, cy, colors.node, 1 + 4.5 * weight);
        ctx.globalAlpha = 1;
        const share = progress * weight;
        disk(cx, cy, 3 + 11 * Math.sqrt(share), colors.node);
        const labelX = cx + (x === 0 ? -12 : 12);
        const labelY = cy + (y === 0 ? 15 : -16);
        label((100 * weight).toFixed(1) + '%', labelX, labelY, colors.node,
          x === 0 ? 'right' : 'left', 600);
      } else if (stage === 2) {
        const [vx, vy] = data.free[i];
        const [sx, sy] = data.solved[i];
        const scale = dt * cell * 0.90;
        if (vy !== sy) arrow(cx, cy, vx * scale, -vy * scale, colors.muted, true);
        arrow(cx, cy, sx * scale, -(vy + (sy - vy) * progress) * scale, colors.node);
        disk(cx, cy, 3 + Math.sqrt(data.mass[i]) * 2, colors.node);
        if (vy !== sy) disk(cx, cy, 9, colors.collider, true);
      } else {
        disk(cx, cy, 3.5, colors.node);
      }
    });

    if (stage === 0) {
      arrow(pointX, pointY, point.velocity[0] * dt * cell, -point.velocity[1] * dt * cell, colors.particle);
    }
    if (stage === 3) {
      const [vx, vy] = data.velocity;
      const dx = vx * dt * cell;
      const dy = -vy * dt * cell;
      line(pointX, pointY, pointX + dx, pointY + dy, colors.particle, 1.5, [3, 4]);
      arrow(pointX, pointY, dx * progress, dy * progress, colors.particle);
      disk(pointX + dx * progress, pointY + dy * progress, 6, colors.particle);
    }
  }

  function setTime(value) {
    time = value;
    const nextStage = Math.ceil(value);
    if (nextStage !== stage) {
      stage = nextStage;
      phases.forEach((phase, index) => { phase.hidden = index !== stage; });
      selectors.forEach((button, index) => button.setAttribute('aria-pressed', String(index === stage)));
      diagram.dataset.stage = String(stage);
      scrubber.setAttribute('aria-valuetext', phaseNames[stage] + ', stage ' + (stage + 1) + ' of 4');
    }
    scrubber.value = String(time);
    draw();
  }

  function update() {
    data = step();
    updateReadout();
    draw();
  }

  function stopPlayback() {
    cancelAnimationFrame(animationFrame);
    playing = false;
    play.textContent = 'Play step';
  }

  function pausePlayback() {
    cancelAnimationFrame(animationFrame);
    playing = false;
    play.textContent = 'Resume step';
  }

  function frame(now) {
    playbackOffset = now - playbackOrigin;
    const progress = Math.max(0, (playbackOffset - holdDuration) / phaseDuration);
    setTime(Math.min(3, progress));
    if (time < 3) animationFrame = requestAnimationFrame(frame);
    else stopPlayback();
  }

  function chooseTime(value) {
    stopPlayback();
    setTime(value);
  }

  function updateMotion() {
    stopPlayback();
    scrubber.step = motion.matches ? '1' : '0.05';
    play.hidden = motion.matches;
    if (motion.matches) setTime(Math.ceil(time));
  }

  function setPoint(x, y) {
    stopPlayback();
    point.x = Math.max(0.03, Math.min(0.97, x));
    point.y = Math.max(0.04, Math.min(0.96, y));
    update();
  }

  function announcePoint() {
    announcement.textContent = 'Point at ' + live.position.textContent;
    if (stage === 1) {
      const percentages = data.support.map(w => (100 * w).toFixed(1) + ' percent');
      announcement.textContent += '. Top left ' + percentages[0] + ', top right ' + percentages[1]
        + ', bottom left ' + percentages[2] + ', bottom right ' + percentages[3] + '.';
    }
  }

  function localPoint(event) {
    const rect = canvas.getBoundingClientRect();
    return [event.clientX - rect.left, event.clientY - rect.top];
  }

  handle.addEventListener('pointerdown', event => {
    if (pointer || event.button !== 0) return;
    const [x, y] = localPoint(event);
    pointer = {id: event.pointerId, offsetX: x - geometry.pointX, offsetY: y - geometry.pointY};
    handle.setPointerCapture(pointer.id);
  });
  handle.addEventListener('pointermove', event => {
    if (pointer?.id !== event.pointerId) return;
    const [x, y] = localPoint(event);
    setPoint((x - pointer.offsetX - geometry.left) / geometry.cell,
      (geometry.floor - y + pointer.offsetY) / geometry.cell);
  });
  handle.addEventListener('lostpointercapture', () => {
    pointer = undefined;
    announcePoint();
  });
  handle.addEventListener('keydown', event => {
    const delta = {ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, 1], ArrowDown: [0, -1]}[event.key];
    if (!delta) return;
    event.preventDefault();
    const increment = event.shiftKey ? 0.10 : 0.025;
    setPoint(point.x + increment * delta[0], point.y + increment * delta[1]);
    announcePoint();
  });
  canvas.addEventListener('click', event => {
    const [x, y] = localPoint(event);
    const u = (x - geometry.left) / geometry.cell;
    const v = (geometry.floor - y) / geometry.cell;
    if (u < 0 || u > 1 || v < 0 || v > 1) return;
    setPoint(u, v);
    handle.focus({preventScroll: true});
    announcePoint();
  });

  selectors.forEach((button, index) => button.addEventListener('click', () => chooseTime(index)));
  scrubber.addEventListener('input', () => chooseTime(Number(scrubber.value)));
  floorInput.addEventListener('change', () => {
    stopPlayback();
    update();
  });
  reset.addEventListener('click', () => {
    setPoint(initial.x, initial.y);
    announcePoint();
  });
  play.addEventListener('click', () => {
    if (playing) {
      pausePlayback();
      return;
    }
    if (play.textContent !== 'Resume step') {
      playbackOffset = 0;
      setTime(0);
    }
    playing = true;
    play.textContent = 'Pause step';
    playbackOrigin = performance.now() - playbackOffset;
    animationFrame = requestAnimationFrame(frame);
  });
  const pauseWhenHidden = () => {
    if (playing) pausePlayback();
  };
  motion.addEventListener('change', updateMotion);
  colorScheme.addEventListener('change', readColors);
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) pauseWhenHidden();
  });
  new IntersectionObserver(([entry]) => {
    if (!entry.isIntersecting) pauseWhenHidden();
  }).observe(lesson);
  new ResizeObserver(draw).observe(canvas);
  new MutationObserver(readColors).observe(document.documentElement, {
    attributes: true, attributeFilter: ['data-theme'],
  });

  lesson.querySelectorAll('[data-mpm-enhanced]').forEach(el => { el.hidden = false; });
  readColors();
  updateMotion();
  setTime(0);
  update();
})();
