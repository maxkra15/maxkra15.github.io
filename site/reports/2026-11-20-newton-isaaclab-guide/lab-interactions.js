(() => {
  'use strict';

  const demo = document.querySelector('[data-clocks]');
  if (!demo) return;
  const motion = matchMedia('(prefers-reduced-motion: reduce)');
  const seek = demo.querySelector('[data-clock-seek]');
  const next = demo.querySelector('[data-clock-next]');
  const play = demo.querySelector('[data-clock-play]');
  const timeStep = demo.querySelector('#ltDt');
  const decimation = demo.querySelector('#ltDec');
  const substeps = demo.querySelector('#ltSub');
  const status = demo.querySelector('[data-clock-status]');
  const axisEnd = demo.querySelector('[data-clock-end]');
  const lanes = [...demo.querySelectorAll('[data-clock-lane]')];
  const rates = [...demo.querySelectorAll('[data-clock-rate]')];
  let active = 0;
  let timer = null;
  let tiles = [];
  const format = value => Number(value.toFixed(3)).toString();
  const count = () => +decimation.value * +substeps.value;

  function setStep(index) {
    active = index;
    const sub = +substeps.value;
    const dt = +timeStep.value;
    const physicsIndex = Math.floor(active / sub);
    const solverDt = dt / sub;
    const t0 = format(active * solverDt);
    const t1 = format((active + 1) * solverDt);

    [0, physicsIndex, active].forEach((selected, row) => {
      tiles[row].forEach((tile, i) => { tile.dataset.active = String(i === selected); });
    });
    seek.value = active + 1;
    seek.setAttribute('aria-valuetext', `Solver step ${active + 1} of ${count()}, physics step ${physicsIndex + 1}, ${t0} to ${t1} milliseconds`);
    status.textContent = `Physics ${physicsIndex + 1} of ${decimation.value} · solver ${active % sub + 1} of ${sub} in this step · ${t0}–${t1} ms`;
    next.textContent = active === count() - 1 ? 'Back to first step' : 'Next solver step';
  }

  function updateSettings() {
    pause();
    const dt = +timeStep.value;
    const dec = +decimation.value;
    const sub = +substeps.value;
    const groups = [1, dec, dec * sub];
    tiles = lanes.map((lane, row) => {
      const parts = Array.from({ length: groups[row] }, (_, i) => {
        const tile = document.createElement('span');
        tile.className = 'clock-tile';
        if (row === 2 && i % sub === 0) tile.classList.add('clock-boundary');
        if (row === 0) tile.textContent = 'Action held';
        return tile;
      });
      lane.style.setProperty('--steps', groups[row]);
      lane.replaceChildren(...parts);
      return parts;
    });
    [dt * dec, dt, dt / sub].forEach((period, i) => {
      rates[i].textContent = `${format(period)} ms · ${format(1000 / period)} Hz`;
    });
    demo.querySelector('#oLtDt').textContent = `${dt} ms`;
    demo.querySelector('#oLtDec').textContent = dec;
    demo.querySelector('#oLtSub').textContent = sub;
    axisEnd.textContent = `${format(dt * dec)} ms`;
    seek.max = count();
    seek.disabled = count() === 1;
    next.disabled = count() === 1;
    play.disabled = count() === 1;
    setStep(Math.min(active, count() - 1));
  }

  function pause() {
    clearInterval(timer);
    timer = null;
    play.textContent = 'Play one action';
  }

  next.addEventListener('click', () => {
    pause();
    setStep((active + 1) % count());
  });
  seek.addEventListener('input', () => {
    pause();
    setStep(+seek.value - 1);
  });
  for (const input of [timeStep, decimation, substeps]) input.addEventListener('input', updateSettings);
  play.addEventListener('click', () => {
    if (timer !== null) {
      pause();
      return;
    }
    if (motion.matches || count() === 1) return;
    setStep(active === count() - 1 ? 0 : active);
    play.textContent = 'Pause';
    timer = setInterval(() => {
      setStep(active + 1);
      if (active === count() - 1) pause();
    }, Math.max(65, 2800 / count()));
  });

  function syncMotion() {
    if (motion.matches) pause();
    play.hidden = motion.matches;
  }
  motion.addEventListener('change', syncMotion);
  document.addEventListener('visibilitychange', () => { if (document.hidden) pause(); });
  new IntersectionObserver(([entry]) => { if (!entry.isIntersecting) pause(); }).observe(demo);
  updateSettings();
  syncMotion();
  demo.hidden = false;
})();
