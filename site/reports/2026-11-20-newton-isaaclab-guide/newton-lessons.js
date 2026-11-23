const selection = document.querySelector("[data-newton-selection]");

if (selection) {
  const worlds = selection.querySelector("fieldset");
  const rows = Array.from(worlds.querySelectorAll(".newton-world"), (world) => ({
    checkbox: world.querySelector("input"),
    coordinates: world.querySelector("code"),
  }));
  const initialCoordinates = rows.map(({ coordinates }) => coordinates.textContent);
  const mask = selection.querySelector("[data-world-mask]");
  const status = selection.querySelector("[data-selection-status]");
  const write = selection.querySelector("[data-selection-write]");

  function renderMask() {
    const values = rows.map(({ checkbox }) => checkbox.checked ? "True" : "False");
    mask.textContent = `[${values.join(", ")}]`;
    write.disabled = values.every((value) => value === "False");
  }

  worlds.addEventListener("change", renderMask);
  write.addEventListener("click", () => {
    const updated = [];
    rows.forEach(({ checkbox, coordinates }, worldId) => {
      if (checkbox.checked) {
        coordinates.textContent = "0.00, 0.00";
        updated.push(worldId);
      }
    });
    const label = updated.length === 1 ? "world" : "worlds";
    status.textContent = `Set the joint coordinates to zero in ${label} ${updated.join(", ")}.`;
  });

  selection.querySelector("[data-selection-reset]").addEventListener("click", () => {
    rows.forEach(({ checkbox, coordinates }, worldId) => {
      checkbox.checked = checkbox.defaultChecked;
      coordinates.textContent = initialCoordinates[worldId];
    });
    status.textContent = "Restored the original joint coordinates and mask.";
    renderMask();
  });

  renderMask();
  worlds.disabled = false;
  selection.querySelector("[data-selection-controls]").hidden = false;
}
