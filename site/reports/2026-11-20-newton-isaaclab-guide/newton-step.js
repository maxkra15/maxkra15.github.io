(() => {
  const widget = document.querySelector("[data-newton-step]");
  if (!widget) return;

  const gravity = 9.81;
  const dt = 0.08;
  const phaseCount = 4;
  const bufferNames = ["A", "B"];
  const motion = matchMedia("(prefers-reduced-motion: reduce)");
  const find = (selector) => widget.querySelector(selector);
  const phases = [...widget.querySelectorAll(".newton-step-rail li")];
  const buffers = [...widget.querySelectorAll("[data-step-buffer]")];
  const next = find("[data-step-next]");
  const previous = find("[data-step-previous]");
  const play = find("[data-step-play]");
  const rebound = find("[data-step-rebound]");
  const status = find("[data-step-status]");
  let steps = [];
  let index = 0;
  let timer = null;

  function advance(state, time) {
    return {
      height: state.height + state.velocity * time - gravity * time * time / 2,
      velocity: state.velocity - gravity * time,
    };
  }

  function calculateStep(current, bounciness) {
    const predicted = advance(current, dt);
    if (predicted.height > 0) return { current, predicted, solved: predicted, contact: false };

    // Solve z(t) = 0 for the downward crossing within this step.
    const impact = (current.velocity + Math.sqrt(current.velocity ** 2 + 2 * gravity * current.height)) / gravity;
    const reboundSpeed = -advance(current, impact).velocity * bounciness;
    // Treat a bounce shorter than a step's flight time as resting contact.
    const solved = reboundSpeed > gravity * dt / 2
      ? advance({ height: 0, velocity: reboundSpeed }, dt - impact)
      : { height: 0, velocity: 0 };
    return { current, predicted, solved, contact: true, impact };
  }

  function buildSteps() {
    let current = { height: 0.8, velocity: 0 };
    const bounciness = Number(rebound.value);
    steps = [];
    do {
      const step = calculateStep(current, bounciness);
      steps.push(step);
      current = step.solved;
    } while (current.height !== 0 || current.velocity !== 0);
  }

  function place(marker, state, visible) {
    const node = find(`[data-marker="${marker}"]`);
    node.style.setProperty("--ball-height", state.height);
    const key = find(`[data-marker-key="${marker}"]`);
    if (key) key.hidden = !visible;
    node.dataset.visible = String(visible);
  }

  function measure(value, unit) {
    const number = Math.abs(value) < 0.005 ? "0.00" : value.toFixed(2).replace("-", "−");
    return `${number} ${unit}`;
  }

  function render(announce = false) {
    const stepNumber = Math.floor(index / phaseCount);
    const phase = index % phaseCount;
    const step = steps[stepNumber];
    const inputBuffer = stepNumber % 2;
    const outputBuffer = 1 - inputBuffer;
    const swapped = phase === 3;
    const currentBuffer = swapped ? outputBuffer : inputBuffer;
    const current = swapped ? step.solved : step.current;
    const other = phase >= 2 ? step.solved : steps[stepNumber - 1]?.current;
    const values = inputBuffer === 0 ? [step.current, other] : [other, step.current];
    const finalPhase = steps.length * phaseCount - 1;

    place("current", current, true);
    place("predicted", step.predicted, phase === 1 || (phase === 2 && step.contact));
    place("output", step.solved, phase === 2);
    widget.dataset.contact = String(step.contact && phase >= 1);
    find("[data-step-time]").textContent = `t = ${measure((stepNumber + Number(swapped)) * dt, "s")}`;

    phases.forEach((node, number) => {
      if (number === phase) node.setAttribute("aria-current", "step");
      else node.removeAttribute("aria-current");
    });
    buffers.forEach((buffer, number) => {
      const value = values[number];
      const isCurrent = number === currentBuffer;
      buffer.dataset.current = String(isCurrent);
      buffer.querySelector("[data-buffer-role]").textContent = isCurrent ? "Read" : "Write";
      buffer.querySelector("[data-buffer-value]").textContent = value
        ? `${measure(value.height, "m")} · ${measure(value.velocity, "m/s")}`
        : "Unwritten";
    });

    const messages = [
      `Read state ${bufferNames[inputBuffer]}. The ball is ${measure(step.current.height, "m")} above the floor.`,
      step.contact
        ? `The predicted path crosses the floor. Impact occurs ${measure(step.impact, "s")} into this step.`
        : `The predicted height is ${measure(step.predicted.height, "m")}. The ball stays clear of the floor.`,
      step.contact
        ? `Solve the contact and write state ${bufferNames[outputBuffer]}. ${step.solved.velocity === 0 && step.solved.height === 0 ? "The ball comes to rest." : "The ball rebounds."}`
        : `Write state ${bufferNames[outputBuffer]} with the predicted height and velocity.`,
      `State ${bufferNames[outputBuffer]} is now the input. State ${bufferNames[inputBuffer]} can be overwritten on the next step.`,
    ];
    const message = messages[phase];
    find("[data-step-message]").textContent = message;
    if (announce) status.textContent = `${phases[phase].textContent}. ${message}`;
    previous.setAttribute("aria-disabled", String(index === 0));
    next.setAttribute("aria-disabled", String(index === finalPhase));
    play.textContent = timer !== null ? "Pause" : index === finalPhase ? "Replay" : "Play";
  }

  function stop() {
    if (timer === null) return;
    clearInterval(timer);
    timer = null;
    delete widget.dataset.playing;
    render();
  }

  function move(amount) {
    stop();
    const destination = index + amount;
    if (destination < 0 || destination >= steps.length * phaseCount) return;
    index = destination;
    render(true);
  }

  function syncMotion() {
    if (motion.matches) {
      stop();
      if (document.activeElement === play) next.focus();
    }
    play.hidden = motion.matches;
  }

  next.addEventListener("click", () => move(1));
  previous.addEventListener("click", () => move(-1));
  play.addEventListener("click", () => {
    if (timer !== null) {
      stop();
      return;
    }
    if (motion.matches) return;
    const end = steps.length * phaseCount - 1;
    if (index === end) index = 0;
    widget.dataset.playing = "";
    timer = setInterval(() => {
      index += 1;
      if (index === end) {
        stop();
        status.textContent = "Playback complete. The ball has come to rest.";
      } else {
        render();
      }
    }, 255);
    render();
  });
  find("[data-step-restart]").addEventListener("click", () => {
    stop();
    index = 0;
    render(true);
  });
  rebound.addEventListener("input", () => {
    stop();
    buildSteps();
    index = 0;
    find("[data-step-rebound-value]").value = Number(rebound.value).toFixed(2);
    render();
  });
  motion.addEventListener("change", syncMotion);
  document.addEventListener("visibilitychange", () => { if (document.hidden) stop(); });
  document.addEventListener("lessonchange", ({ detail }) => { if (detail.current !== "newton") stop(); });

  buildSteps();
  render();
  syncMotion();
  widget.hidden = false;
})();
