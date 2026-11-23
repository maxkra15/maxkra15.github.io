# Research reports

Short technical reports by Maximilian Krause.

- **Site:** https://maxkra15.github.io/
- **First report:** https://maxkra15.github.io/reports/2026-10-09-g1-learning/

## Structure

```text
site/
  index.html
  404.html
  reports/
    YYYY-MM-DD-topic/
      index.html
      report.pdf
      figures/
      data/
      videos/          # Optional MP4s, poster, and recording provenance
      SHA256SUMS
```

Only `site/` is uploaded to GitHub Pages. The homepage lists reports; each dated
directory preserves its paper, figures, measurements, source revisions, and
attribution. Stable URLs remain useful when the homepage changes.

The first report compares RSL PPO, Warp PPO, and FlashSAC for G1 locomotion.
Its JSON/CSV downloads are compact derived measurements; their provenance
records the hashes of the unchanged original sources. Native checkpoints and
full execution logs remain in the experiment archive.
The report also includes synchronized Newton RTX recordings of the three final
seed-0 policies, with individual MP4 downloads and separate recording provenance.

## Add a report

1. Create a feature branch.
2. Add the report to a new dated directory under `site/reports/`. Include a title,
   date, methods, figure captions, limitations, source revisions, and original
   algorithm attribution. Export reusable figures and a PDF when useful.
   For policy videos, document checkpoint selection, replay settings, and playback
   speed; use browser-compatible MP4s with a poster and native playback controls.
3. Use relative links within the report. Include only the files intended for
   publication, and keep the report's measurement version explicit.
4. Add a report card to `site/index.html`.
5. Preview locally, then merge the branch into `main`.

```bash
python -m http.server 8000 --directory site
# Open http://localhost:8000/
```

For corrections, retain the dated URL and document the change. Use a new dated
directory for a new experiment rather than replacing an earlier result.
The [G1 provenance](site/reports/2026-10-09-g1-learning/data/provenance.json)
illustrates how to distinguish published-file hashes from original-source hashes.

## Robotics Simulation Learning Hub

The hub keeps its original dated URL under [site/reports](site/reports/2026-11-20-newton-isaaclab-guide/). It uses static HTML and local CSS and JavaScript assets, so the existing GitHub Pages workflow serves it directly.

- Add a topic as a `main > .chapter[data-ch]` with a stable ID and a matching link in the topic navigation. Give it a short reading path in ordinary HTML. Use native `details` for derivations, source walkthroughs and configuration.
- Link directly to a section ID for deeper reading. `hub.js` selects the containing topic and opens disclosures around the target. Keep existing section IDs when revising lessons.
- Put new interactive examples in topic-specific assets. Keep the explanation available as HTML. Start playback on request, give readers a pause or step control, and use `prefers-reduced-motion` for a static or stepped equivalent. The MPM cycle is an example.
- Pin source links when implementation details depend on a revision. Mark experimental APIs and link to their current docs.
- Preview at desktop and narrow mobile widths. Check keyboard controls, expanded links from a fresh load, and both motion preferences. From the lesson directory, refresh the manifest with `sha256sum index.html *.css *.js > SHA256SUMS`.

The lesson structure follows [Diátaxis](https://diataxis.fr/). Progressive disclosure keeps the examples and the first reading path easy to find. The animation controls follow [W3C guidance on animation from interactions](https://www.w3.org/WAI/WCAG22/Understanding/animation-from-interactions.html).

## Deployment

Pushing changes to `site/` on `main` publishes automatically. The
**Publish reports** workflow can also be run manually from GitHub Actions.
It uses the official Pages actions, pinned to reviewed commit revisions.
The `github-pages` deployment environment accepts only `main`, and HTTPS is enabled.

This follows GitHub's supported
[prebuilt static-site workflow](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages).
The repository name provides a
[personal Pages site](https://docs.github.com/en/pages/getting-started-with-github-pages/what-is-github-pages)
at the root URL. There is no package installation or site build step.
If a future report needs Markdown notebooks or a generator, its exported HTML
can use the same publication directory and workflow.

## Credits and licenses

Site code is MIT licensed. The G1 report's HTML template is adapted from the
Isaac Lab benchmark report under BSD-3-Clause; its license is retained in
`licenses/IsaacLab-BSD-3-Clause.txt`. The paper preserves references and credits
to FlashSAC, PPO, RSL-RL, Warp-NN, Isaac Lab, Newton, and MuJoCo Warp.
Original algorithm implementations are not distributed by this site.
