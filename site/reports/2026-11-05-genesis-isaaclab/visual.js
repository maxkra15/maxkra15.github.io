/* Site presentation: MIT license. No external dependencies. */
"use strict";

(() => {
  const source = document.getElementById("report-data");
  if (!source) return;
  let data;
  try {
    data = JSON.parse(source.textContent);
    if (!Array.isArray(data.groups)) throw new Error("Missing comparison groups");
  } catch (error) {
    console.error("The visual report data could not be loaded.", error);
    return;
  }
  document.documentElement.classList.add("has-js");

  const tasks = [
    { id: "anymal", name: "ANYmal-C", detail: "Velocity tracking", errorUnit: "m/s" },
    { id: "cartpole", name: "Cartpole", detail: "Balance", errorUnit: "rad" },
    { id: "franka", name: "Franka", detail: "Reaching", errorUnit: "m" },
  ];
  const backends = [
    { id: "genesis", name: "Genesis", short: "Genesis" },
    { id: "isaaclab", name: "Isaac Lab MJWarp", short: "MJWarp" },
  ];
  const metrics = {
    rollout: { mode: "rollout", field: "env_steps_per_s", title: "Rollout throughput", unit: "Environment transitions / s · higher is faster" },
    startup: { mode: "rollout", field: "startup_s", title: "Rollout startup", unit: "Seconds · lower is faster" },
    ppo: { mode: "learning", field: "training_s", title: "Fixed-budget PPO time", unit: "Seconds · lower is faster for this budget" },
  };
  const state = { metric: "rollout", vision: "none", batch: 4096 };
  const seeds = data.scope?.seeds || [0, 1, 2];
  const number = new Intl.NumberFormat("en", { maximumSignificantDigits: 3 });
  const whole = new Intl.NumberFormat("en", { maximumFractionDigits: 0 });
  const chart = document.getElementById("comparison-chart");
  const matrix = document.getElementById("learning-matrix");
  const batch = document.getElementById("batch");

  function node(tag, className, text) {
    const item = document.createElement(tag);
    if (className) item.className = className;
    if (text !== undefined) item.textContent = text;
    return item;
  }

  function format(value, compact = false) {
    if (!Number.isFinite(value)) return "—";
    if (compact && value >= 1e6) return `${number.format(value / 1e6)}M`;
    if (compact && value >= 1e3) return `${number.format(value / 1e3)}k`;
    return number.format(value);
  }

  function group(task, backend, mode, vision, count) {
    return data.groups.find((item) => item.task === task && item.backend === backend && item.mode === mode && item.vision === vision && item.num_envs === count);
  }

  function selection() {
    const metric = metrics[state.metric];
    const vision = state.vision;
    const count = state.metric === "ppo" ? 512 : vision === "rgb" ? 64 : state.batch;
    return { ...metric, vision, count };
  }

  function axisMaximum(maximum) {
    if (!(maximum > 0)) return 1;
    const rawStep = maximum / 4;
    const power = 10 ** Math.floor(Math.log10(rawStep));
    const step = [1, 2, 2.5, 5, 10].map((factor) => factor * power).find((value) => value >= rawStep);
    return Math.ceil(maximum / step) * step;
  }

  function configureBatch(current) {
    const values = state.metric === "ppo" ? [512] : current.vision === "rgb" ? [64] : [256, 4096, 8192];
    batch.replaceChildren(...values.map((value) => {
      const option = node("option", "", whole.format(value));
      option.value = value;
      return option;
    }));
    batch.value = current.count;
    batch.disabled = values.length === 1;
  }

  function makeBar(task, backend, current, maximum) {
    const row = node("div", `bar-row backend-${backend.id}`);
    row.append(node("span", "bar-label", backend.short));
    const track = node("div", "bar-track");
    const value = node("div", "chart-value");
    const measurement = group(task.id, backend.id, current.mode, current.vision, current.count);
    const metric = measurement?.metrics?.[current.field];
    if (!metric || !Number.isFinite(metric.median)) {
      value.append(node("span", "missing-value", "No data"));
      row.append(track, value);
      return row;
    }
    const percent = (amount) => `${100 * amount / maximum}%`;
    const bar = node("span", "bar-fill");
    bar.style.width = percent(metric.median);
    const whisker = node("span", "bar-whisker");
    whisker.style.left = percent(metric.min);
    whisker.style.width = percent(metric.max - metric.min);
    track.append(bar, whisker);
    for (const endpoint of [metric.min, metric.max]) {
      const cap = node("span", "bar-cap");
      cap.style.left = percent(endpoint);
      track.append(cap);
    }
    const values = [...metric.values].sort((a, b) => a.seed - b.seed);
    values.forEach((sample, index) => {
      const point = node("span", "bar-seed");
      point.style.left = percent(sample.value);
      point.style.top = `${8 + index * 8}px`;
      track.append(point);
    });
    const median = node("span", "bar-median");
    median.style.left = percent(metric.median);
    track.append(median);
    const description = `${backend.name}: median ${format(metric.median)}, range ${format(metric.min)} to ${format(metric.max)}. ${values.map((sample) => `Seed ${sample.seed}: ${format(sample.value)}`).join("; ")}.`;
    track.setAttribute("role", "img");
    track.setAttribute("aria-label", description);
    track.title = description;
    const medianValue = node("strong", "", format(metric.median, current.field.endsWith("per_s")));
    medianValue.title = String(metric.median);
    value.append(medianValue);
    if (state.metric === "ppo") {
      value.append(node("small", "", measurement.learning ? `${measurement.learning.reached}/${seeds.length} targets` : "Target data unavailable"));
    }
    row.append(track, value);
    return row;
  }

  function renderChart(current) {
    const selected = tasks.flatMap((task) => backends.map((backend) => group(task.id, backend.id, current.mode, current.vision, current.count)));
    const maximum = axisMaximum(Math.max(0, ...selected.map((item) => item?.metrics?.[current.field]?.max || 0)));
    const blocks = tasks.map((task) => {
      const block = node("div", "task-block");
      const heading = node("div", "task-title");
      heading.append(node("h4", "", task.name), node("span", "", task.detail));
      block.append(heading, ...backends.map((backend) => makeBar(task, backend, current, maximum)));
      return block;
    });
    const axis = node("div", "chart-axis");
    const ticks = node("div", "axis-ticks");
    for (let index = 0; index <= 4; index++) {
      const tick = node("span", index % 2 ? "minor-tick" : "", format(maximum * index / 4, current.field.endsWith("per_s")));
      tick.style.left = `${index * 25}%`;
      ticks.append(tick);
    }
    axis.append(node("span"), ticks, node("span"));
    chart.replaceChildren(...blocks, axis);
    document.getElementById("chart-title").textContent = current.title;
    document.getElementById("chart-unit").textContent = current.unit;
    const notes = {
      rollout: `Four physics steps per transition, plus task observations, rewards and resets${current.vision === "rgb" ? ", and native 64 × 64 RGB rendering" : ""}. Batch: ${whole.format(current.count)} environments.`,
      startup: `Selected ${current.vision === "rgb" ? "RGB" : "state"} rollout workload: runtime imports through the first usable observation, excluding Python and site initialization. Fresh workers; disk and JIT caches reused.`,
      ppo: "Full 500-iteration PPO budget: 6.144M training transitions, including evaluation and excluding startup. Target counts describe quality separately; the bars are not time-to-success.",
    };
    document.getElementById("chart-note").textContent = notes[state.metric];
    document.getElementById("chart-status").textContent = `${current.title}. ${current.unit}. ${whole.format(current.count)} environments. ${current.vision === "rgb" ? "State plus RGB" : "State"} scenes. Learning outcomes below use ${state.vision === "rgb" ? "state plus RGB" : "state"} observations.`;
  }

  function successCell(task, backend) {
    const cell = node("div", `success-cell backend-${backend.id}`);
    const learning = group(task.id, backend.id, "learning", state.vision, 512)?.learning;
    if (!learning) {
      cell.append(node("span", "missing-value", "No data"));
      return cell;
    }
    const top = node("div", "success-top");
    const dots = node("span", "seed-dots");
    for (const seed of [...learning.seeds].sort((a, b) => a.seed - b.seed)) {
      const dot = node("span", `seed-dot ${seed.reached ? "reached" : "censored"}`);
      const description = `Seed ${seed.seed}: ${seed.reached ? `target reached at ${format(seed.first_target_s)} training seconds` : "censored at the budget"}. Final error ${format(seed.final_error)} ${task.errorUnit}; survival ${format(seed.survival * 100)}%.`;
      dot.setAttribute("role", "img");
      dot.setAttribute("aria-label", description);
      dot.title = description;
      dots.append(dot);
    }
    top.append(dots, node("strong", "success-count", `${learning.reached}/${seeds.length}`));
    cell.append(top, node("p", "success-detail", `${learning.censored} censored`));
    if (learning.first_target_s) cell.append(node("p", "first-target", `First target ${format(learning.first_target_s.median)} s · reached seeds`));
    return cell;
  }

  function renderLearning() {
    const heading = node("div", "matrix-row matrix-header");
    heading.append(node("span", "", "Task"));
    for (const backend of backends) heading.append(node("span", `backend-${backend.id} ${backend.id}-text`, backend.name));
    const rows = tasks.map((task) => {
      const row = node("div", "matrix-row");
      row.append(node("span", "matrix-task", task.name), ...backends.map((backend) => successCell(task, backend)));
      return row;
    });
    matrix.replaceChildren(heading, ...rows);
    document.getElementById("learning-vision").textContent = state.vision === "rgb" ? "State + RGB observations" : "State observations";
  }

  function renderCurves() {
    const container = document.getElementById("learning-curves");
    const links = tasks.map((task) => {
      const figure = (data.learning_figures || []).find((item) => item.task === task.id && item.vision === state.vision);
      if (!figure) return null;
      const details = node("details", "task-curve");
      details.append(node("summary", "", task.name));
      const body = node("div", "curve-content");
      const link = node("a", "", "Open full-size SVG ↗");
      link.href = figure.path;
      link.target = "_blank";
      link.rel = "noopener";
      const image = node("img");
      image.src = figure.path;
      image.alt = `${task.name}, ${state.vision === "rgb" ? "state plus RGB" : "state"}: per-seed survival and task error against training time and environment transitions.`;
      image.loading = "lazy";
      body.append(link, image);
      details.append(body);
      return details;
    }).filter(Boolean);
    container.replaceChildren(...links);
  }

  function renderCameraGallery() {
    const gallery = document.getElementById("camera-gallery");
    const samples = tasks.flatMap((task) => backends.map((backend) => {
      const sample = (data.camera_samples || []).find((item) => item.task === task.id && item.backend === backend.id);
      if (!sample) return null;
      const figure = node("figure", `camera-sample backend-${backend.id}`);
      const link = node("a");
      link.href = sample.path;
      const image = node("img");
      image.src = sample.path;
      image.alt = `${task.name}: native ${backend.name} camera frame.`;
      image.width = 64;
      image.height = 64;
      image.loading = "lazy";
      link.append(image);
      const caption = node("figcaption", "", task.name);
      caption.append(node("span", "", backend.short));
      figure.append(link, caption);
      return figure;
    })).filter(Boolean);
    gallery.replaceChildren(...samples);
  }

  function renderDownloads() {
    const list = document.getElementById("figure-downloads");
    for (const task of tasks) {
      for (const vision of ["none", "rgb"]) {
        const figure = (data.learning_figures || []).find((item) => item.task === task.id && item.vision === vision);
        if (!figure) continue;
        const item = node("li");
        item.append(node("span", "", `${task.name} learning · ${vision === "rgb" ? "RGB" : "state"}`));
        for (const extension of ["svg", "png", "pdf"]) {
          const link = node("a", "", extension.toUpperCase());
          link.href = figure.path.replace(/\.svg$/i, `.${extension}`);
          item.append(link);
        }
        list.append(item);
      }
    }
    const pins = document.getElementById("source-pins");
    for (const [name, revision] of Object.entries(data.source_revisions || {})) {
      const value = typeof revision === "string" ? revision : revision.commit;
      if (!value) continue;
      const item = node("li", "", `${name}: `);
      const code = node("code", "", value.slice(0, 12));
      code.title = value;
      item.append(code);
      pins.append(item);
    }
  }

  function render() {
    const current = selection();
    configureBatch(current);
    renderChart(current);
    renderLearning();
    renderCurves();
  }

  for (const button of document.querySelectorAll("[data-metric]")) {
    button.addEventListener("click", () => {
      state.metric = button.dataset.metric;
      for (const item of document.querySelectorAll("[data-metric]")) item.setAttribute("aria-pressed", String(item === button));
      render();
    });
  }
  for (const radio of document.querySelectorAll('input[name="vision"]')) radio.addEventListener("change", () => { state.vision = radio.value; render(); });
  batch.addEventListener("change", () => { state.batch = Number(batch.value); render(); });
  if (Number.isFinite(data.scope?.runs)) document.getElementById("run-count").textContent = `${whole.format(data.scope.runs)} runs`;
  document.getElementById("seed-count").textContent = `${seeds.length} seeds`;
  const takeaway = data.takeaway;
  if (takeaway && Number.isFinite(takeaway.rollout_mjwarp_higher) && Number.isFinite(takeaway.rollout_settings) && Number.isFinite(takeaway.rollout_ratio_min) && Number.isFinite(takeaway.rollout_ratio_max)) {
    const ratio = new Intl.NumberFormat("en", { maximumFractionDigits: 1 });
    const range = `${ratio.format(takeaway.rollout_ratio_min)}–${ratio.format(takeaway.rollout_ratio_max)}×`;
    document.getElementById("takeaway").textContent = takeaway.rollout_mjwarp_higher === takeaway.rollout_settings
      ? `MJWarp rollout throughput: ${range} Genesis across all ${takeaway.rollout_settings} matched settings.`
      : `MJWarp rollout medians: ${range} Genesis; higher in ${takeaway.rollout_mjwarp_higher}/${takeaway.rollout_settings} matched settings.`;
  }
  renderCameraGallery();
  renderDownloads();
  render();
})();
