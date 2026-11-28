<!--
SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
SPDX-License-Identifier: CC-BY-4.0
-->

# Cluster jobs of revision 3

NVIDIA OSMO jobs for the discretization study (`resolution/`) and for the sand and soil experiments
(`sand/`), one GPU each except the sand rod calibration, which spread the worlds of every batch over
four GPUs (`gpus` and `devices` in its job). Each folder has the job generator (`generate.py`, which snapshots the
`examples/mpm_soil` code of a newton-calibration checkout named by `NEWTON_CALIBRATION` and writes one
workflow per job), the task runner executed inside the job (`task.py`) and the analysis that turns the
downloaded results into the summaries exported to `../../data` (`analyze.py`). The bootstrap, source
pinning and requirements are those of `../osmo/`. `OSMO_PLATFORM` optionally selects a GPU platform
and `OSMO_POOL` the pool for `sand/submit_extra.py`. The runs of the report used NVIDIA L40S GPUs,
Newton 9deb629 and Warp 1.20.0.dev20261120.

`sand/generate.py` defines the first jobs. Reruns with corrected settings and later checks were
submitted with `sand/submit_extra.py sand/jobs_extra.json` (one entry per job, as submitted); where a
job exists in several versions (`-lit3`, `-cal3`, `-cal4`, `pullout-cal3`, `probe-*2`, `plate-*2`)
the report uses the latest. `rod-sand-cal3` exhausted its 6 h budget during the sensitivity analysis
and was rerun as `rod-sand-cal4` on four GPUs with a 16 h budget. That run searched in the rotated
combination coordinates of the sensitivity analysis, where 16 of its 30 search trials decoded outside the
parameter bounds; `rod-sand-cal5` searches the parameter box itself (`orthogonalize=False`, as the rake
case does) with up to 8 generations. `pullout-corner` checks the pull-out at the stiffest corner of its box.
