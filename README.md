# mapgan

MapGAN is a deliberately tiny world-map approximator.

It does not try to reproduce coastlines directly. Instead, it treats the land mask as a hypothesis made from a small number of rotated, wrapped Gaussian blobs on an equirectangular grid. The solver sweeps from simpler to more complex hypotheses, scores each one against a downloaded target map, and reports the result as a measure rather than a pass/fail gate.

Search happens on a coarser grid. Verification happens on a higher-resolution grid. That gives the project an explicit overfit signal: if the coarse-grid score keeps improving while the verified score stops improving and begins to fall, the sweep can stop automatically.

## What it does

- Downloads a public world-country GeoJSON and rasterizes it into a land mask.
- Searches for a compact approximation using `N` blobs plus one bias term.
- Measures each hypothesis with IoU, F1, accuracy, precision, and recall.
- Verifies each complexity on a higher-resolution raster and can stop automatically on overfit.
- Saves preview images, diff images, and JSON model files into `out/`.

## Install

```bash
/home/t/PycharmProjects/mapgan/.venv/bin/python -m pip install -r requirements.txt
```

## Run

Fetch and rasterize a target map:

```bash
/home/t/PycharmProjects/mapgan/.venv/bin/python mapgan.py fetch-target --width 256 --height 128
```

Run the systematic hypothesis sweep until overfit is detected or the hard cap is reached:

```bash
/home/t/PycharmProjects/mapgan/.venv/bin/python mapgan.py solve --min-blobs 0 --max-blobs 48 --population 48 --steps 24
```

Verify a saved model explicitly:

```bash
/home/t/PycharmProjects/mapgan/.venv/bin/python mapgan.py verify --model out/best_overall.json --width 256 --height 128
```

## Notes

- The target data source defaults to `https://raw.githubusercontent.com/johan/world.geo.json/master/countries.geo.json`.
- “Shortest possible algorithm” is represented here by hypothesis complexity: fewer blobs means fewer parameters.
- There is no required threshold. The leaderboard is the point.
- Overfit detection compares the best verified IoU seen so far against later blob counts. By default, the sweep stops when the search-grid IoU keeps improving, the verified IoU has dropped by at least `0.003`, and that pattern has persisted for `4` consecutive blob counts after at least `6` blobs.
- If you want a pure fixed-range sweep, run `mapgan.py solve --no-stop-on-overfit`.