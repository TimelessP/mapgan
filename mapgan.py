from __future__ import annotations

import argparse
import json
import math
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "out"
TARGET_URL = "https://raw.githubusercontent.com/johan/world.geo.json/master/countries.geo.json"


@dataclass
class Candidate:
    bias: float
    blobs: np.ndarray

    def copy(self) -> "Candidate":
        return Candidate(float(self.bias), self.blobs.copy())

    def to_dict(self) -> dict:
        return {
            "bias": float(self.bias),
            "blobs": [
                {
                    "lon": float(blob[0]),
                    "lat": float(blob[1]),
                    "sx": float(blob[2]),
                    "sy": float(blob[3]),
                    "angle_deg": float(math.degrees(blob[4])),
                    "amp": float(blob[5]),
                }
                for blob in self.blobs
            ],
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "Candidate":
        blobs = []
        for blob in payload.get("blobs", []):
            angle = blob.get("angle")
            if angle is None:
                angle = math.radians(blob.get("angle_deg", 0.0))
            blobs.append(
                [
                    blob["lon"],
                    blob["lat"],
                    blob["sx"],
                    blob["sy"],
                    angle,
                    blob["amp"],
                ]
            )
        blob_array = np.asarray(blobs, dtype=np.float32).reshape((-1, 6)) if blobs else np.zeros((0, 6), dtype=np.float32)
        return cls(float(payload.get("bias", 0.0)), blob_array)


def count_params(candidate: Candidate) -> int:
    return 1 + (6 * candidate.blobs.shape[0])


def ensure_dirs() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    OUT_DIR.mkdir(exist_ok=True)


def download_target(url: str, refresh: bool = False) -> Path:
    ensure_dirs()
    target_path = DATA_DIR / "countries.geo.json"
    if refresh or not target_path.exists():
        with urllib.request.urlopen(url, timeout=60) as response, target_path.open("wb") as handle:
            handle.write(response.read())
    return target_path


def iter_polygons(geometry: dict):
    geometry_type = geometry.get("type")
    if geometry_type == "Polygon":
        yield geometry["coordinates"]
    elif geometry_type == "MultiPolygon":
        yield from geometry["coordinates"]


def polygon_crosses_dateline(ring: list, width: int) -> bool:
    xs = np.array([(((float(point[0]) + 180.0) % 360.0) / 360.0) * width for point in ring], dtype=np.float32)
    if xs.size < 2:
        return False
    return bool(xs.max() - xs.min() > (width / 2.0))


def project_ring(ring: list, width: int, height: int, crosses_dateline: bool):
    xs = np.array([(((float(point[0]) + 180.0) % 360.0) / 360.0) * width for point in ring], dtype=np.float32)
    ys = np.array([((90.0 - float(point[1])) / 180.0) * height for point in ring], dtype=np.float32)
    if crosses_dateline:
        xs = xs.copy()
        xs[xs < (width / 2.0)] += width
    return list(zip(xs.tolist(), ys.tolist()))


def rasterize_target(width: int, height: int, url: str = TARGET_URL, refresh: bool = False) -> np.ndarray:
    target_path = download_target(url, refresh=refresh)
    payload = json.loads(target_path.read_text(encoding="utf-8"))
    canvas = Image.new("L", (width * 3, height), 0)
    draw = ImageDraw.Draw(canvas)
    for feature in payload.get("features", []):
        geometry = feature.get("geometry")
        if not geometry:
            continue
        for polygon in iter_polygons(geometry):
            if not polygon:
                continue
            crosses_dateline = polygon_crosses_dateline(polygon[0], width)
            for ring_index, ring in enumerate(polygon):
                if len(ring) < 3:
                    continue
                points = project_ring(ring, width, height, crosses_dateline)
                fill = 255 if ring_index == 0 else 0
                for shift in (0.0, float(width), float(width * 2)):
                    shifted = [(x + shift, y) for x, y in points]
                    draw.polygon(shifted, fill=fill)
    cropped = canvas.crop((width, 0, width * 2, height))
    return np.asarray(cropped, dtype=np.uint8) > 0


def make_grid(width: int, height: int):
    lon = ((np.arange(width, dtype=np.float32) + 0.5) / width) * 360.0 - 180.0
    lat = 90.0 - ((np.arange(height, dtype=np.float32) + 0.5) / height) * 180.0
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    cos_lat = np.cos(np.radians(lat_grid)).astype(np.float32)
    return lon_grid, lat_grid, cos_lat


def wrap_longitude(value: float) -> float:
    return float(((value + 180.0) % 360.0) - 180.0)


def wrap_angle(value: float) -> float:
    return float(((value + math.pi) % (2.0 * math.pi)) - math.pi)


def wrap_delta(delta: np.ndarray) -> np.ndarray:
    return ((delta + 180.0) % 360.0) - 180.0


def clamp_amp(value: float) -> float:
    value = float(np.clip(value, -3.0, 3.0))
    if 0.0 <= value < 0.12:
        return 0.12
    if -0.12 < value < 0.0:
        return -0.12
    return value


def random_blob(rng: np.random.Generator) -> np.ndarray:
    amplitude = float(rng.choice((-1.0, 1.0)) * math.exp(rng.uniform(math.log(0.25), math.log(2.5))))
    return np.array(
        [
            rng.uniform(-180.0, 180.0),
            rng.uniform(-82.0, 84.0),
            rng.uniform(4.0, 75.0),
            rng.uniform(3.0, 44.0),
            rng.uniform(-math.pi, math.pi),
            clamp_amp(amplitude),
        ],
        dtype=np.float32,
    )


def random_candidate(blob_count: int, rng: np.random.Generator, target_land_fraction: float) -> Candidate:
    blobs = np.vstack([random_blob(rng) for _ in range(blob_count)]) if blob_count else np.zeros((0, 6), dtype=np.float32)
    bias = float(rng.normal(loc=target_land_fraction - 0.5, scale=0.75))
    return Candidate(bias=float(np.clip(bias, -3.0, 3.0)), blobs=blobs)


def grow_candidate(candidate: Candidate | None, blob_count: int, rng: np.random.Generator) -> Candidate | None:
    if candidate is None:
        return None
    grown = candidate.copy()
    if grown.blobs.shape[0] > blob_count:
        grown.blobs = grown.blobs[:blob_count].copy()
        return grown
    if grown.blobs.shape[0] == blob_count:
        return grown
    extras = np.vstack([random_blob(rng) for _ in range(blob_count - grown.blobs.shape[0])])
    grown.blobs = np.vstack([grown.blobs, extras]) if grown.blobs.size else extras
    return grown


def mutate_candidate(candidate: Candidate, rng: np.random.Generator, scale: float) -> Candidate:
    child = candidate.copy()
    child.bias = float(np.clip(child.bias + rng.normal(0.0, 0.45 * scale), -3.0, 3.0))
    for index in range(child.blobs.shape[0]):
        if rng.random() < (0.05 + 0.18 * scale):
            child.blobs[index] = random_blob(rng)
            continue
        blob = child.blobs[index]
        blob[0] = wrap_longitude(float(blob[0] + rng.normal(0.0, 42.0 * scale)))
        blob[1] = float(np.clip(blob[1] + rng.normal(0.0, 18.0 * scale), -88.0, 88.0))
        blob[2] = float(np.clip(blob[2] * math.exp(rng.normal(0.0, 0.35 * scale)), 2.5, 95.0))
        blob[3] = float(np.clip(blob[3] * math.exp(rng.normal(0.0, 0.35 * scale)), 2.0, 70.0))
        blob[4] = wrap_angle(float(blob[4] + rng.normal(0.0, 0.9 * scale)))
        amplitude = float(blob[5] * math.exp(rng.normal(0.0, 0.35 * scale)))
        if rng.random() < 0.08 * scale:
            amplitude *= -1.0
        blob[5] = clamp_amp(amplitude)
    return child


def perturb_candidate(candidate: Candidate, index: int, delta: float) -> Candidate:
    child = candidate.copy()
    if index == 0:
        child.bias = float(np.clip(child.bias + delta, -3.0, 3.0))
        return child
    blob_index, blob_param = divmod(index - 1, 6)
    blob = child.blobs[blob_index]
    if blob_param == 0:
        blob[0] = wrap_longitude(float(blob[0] + delta))
    elif blob_param == 1:
        blob[1] = float(np.clip(blob[1] + delta, -88.0, 88.0))
    elif blob_param == 2:
        blob[2] = float(np.clip(blob[2] * math.exp(delta), 2.5, 95.0))
    elif blob_param == 3:
        blob[3] = float(np.clip(blob[3] * math.exp(delta), 2.0, 70.0))
    elif blob_param == 4:
        blob[4] = wrap_angle(float(blob[4] + delta))
    else:
        blob[5] = clamp_amp(float(blob[5] + delta))
    return child


def render_mask(candidate: Candidate, grid) -> np.ndarray:
    lon_grid, lat_grid, cos_lat = grid
    field = np.full(lon_grid.shape, candidate.bias, dtype=np.float32)
    for lon0, lat0, sx, sy, angle, amp in candidate.blobs:
        dx = wrap_delta(lon_grid - lon0) * cos_lat
        dy = lat_grid - lat0
        cos_angle = math.cos(float(angle))
        sin_angle = math.sin(float(angle))
        xr = cos_angle * dx + sin_angle * dy
        yr = -sin_angle * dx + cos_angle * dy
        field += float(amp) * np.exp(-0.5 * ((xr / max(float(sx), 1e-3)) ** 2 + (yr / max(float(sy), 1e-3)) ** 2))
    return field > 0.0


def score_mask(mask: np.ndarray, target: np.ndarray) -> dict:
    tp = int(np.logical_and(mask, target).sum())
    fp = int(np.logical_and(mask, np.logical_not(target)).sum())
    fn = int(np.logical_and(np.logical_not(mask), target).sum())
    tn = int(mask.size - tp - fp - fn)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    iou = tp / max(tp + fp + fn, 1)
    f1 = (2 * tp) / max((2 * tp) + fp + fn, 1)
    accuracy = (tp + tn) / mask.size
    return {
        "iou": float(iou),
        "f1": float(f1),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "land_fraction": float(mask.mean()),
    }


def refine_candidate(candidate: Candidate, target: np.ndarray, grid, refine_passes: int) -> tuple[Candidate, dict]:
    best_candidate = candidate.copy()
    best_metrics = score_mask(render_mask(best_candidate, grid), target)
    if refine_passes <= 0:
        return best_candidate, best_metrics
    step_sizes = [
        0.28,
        14.0,
        7.0,
        0.22,
        0.22,
        0.32,
        0.18,
    ]
    parameter_count = 1 + (6 * best_candidate.blobs.shape[0])
    for refine_pass in range(refine_passes):
        scale = 0.55 ** refine_pass
        improved = False
        for parameter_index in range(parameter_count):
            base_step = step_sizes[0] if parameter_index == 0 else step_sizes[1 + ((parameter_index - 1) % 6)]
            delta = base_step * scale
            for direction in (-1.0, 1.0):
                trial = perturb_candidate(best_candidate, parameter_index, direction * delta)
                metrics = score_mask(render_mask(trial, grid), target)
                if metrics["iou"] > best_metrics["iou"]:
                    best_candidate = trial
                    best_metrics = metrics
                    improved = True
                    break
            if improved:
                continue
        if not improved:
            break
    return best_candidate, best_metrics


def compress_candidate(
    candidate: Candidate,
    search_target: np.ndarray,
    search_grid,
    verify_target: np.ndarray,
    verify_grid,
    min_blobs: int,
    refine_passes: int,
) -> list[dict]:
    current = candidate.copy()
    rows = []
    compression_passes = max(1, refine_passes // 2)
    while True:
        search_metrics = score_mask(render_mask(current, search_grid), search_target)
        verify_metrics = score_mask(render_mask(current, verify_grid), verify_target)
        rows.append(
            {
                "blobs": int(current.blobs.shape[0]),
                "params": count_params(current),
                "search": search_metrics,
                "verify": verify_metrics,
                "generalization_gap": float(search_metrics["iou"] - verify_metrics["iou"]),
                "status": "compressed",
                "candidate": current.copy(),
            }
        )
        if current.blobs.shape[0] <= min_blobs:
            break
        best_trial = None
        best_search_metrics = None
        best_verify_metrics = None
        for remove_index in range(current.blobs.shape[0]):
            trial = current.copy()
            trial.blobs = np.delete(trial.blobs, remove_index, axis=0)
            trial, trial_search_metrics = refine_candidate(trial, search_target, search_grid, compression_passes)
            trial_verify_metrics = score_mask(render_mask(trial, verify_grid), verify_target)
            if best_verify_metrics is None:
                best_trial = trial
                best_search_metrics = trial_search_metrics
                best_verify_metrics = trial_verify_metrics
                continue
            trial_score = (trial_verify_metrics["iou"], trial_verify_metrics["accuracy"], -count_params(trial))
            best_score = (best_verify_metrics["iou"], best_verify_metrics["accuracy"], -count_params(best_trial))
            if trial_score > best_score:
                best_trial = trial
                best_search_metrics = trial_search_metrics
                best_verify_metrics = trial_verify_metrics
        assert best_trial is not None
        current = best_trial
    return rows


def build_pareto_frontier(rows: list[dict]) -> list[dict]:
    frontier = []
    best_verify_iou = float("-inf")
    for row in sorted(rows, key=lambda item: (item["params"], item["blobs"], -item["verify"]["iou"])):
        if row["verify"]["iou"] > best_verify_iou + 1e-12:
            frontier.append(row)
            best_verify_iou = row["verify"]["iou"]
    return frontier


def strip_candidate(row: dict) -> dict:
    return {key: value for key, value in row.items() if key != "candidate"}


def optimize(
    target: np.ndarray,
    grid,
    blob_count: int,
    seed: int,
    population: int,
    steps: int,
    refine_passes: int,
    warm_start: Candidate | None,
) -> tuple[Candidate, dict]:
    rng = np.random.default_rng(seed)
    target_land_fraction = float(target.mean())
    current_population = []
    grown = grow_candidate(warm_start, blob_count, rng)
    if grown is not None:
        current_population.append(grown)
        for _ in range(max(3, population // 6)):
            current_population.append(mutate_candidate(grown, rng, 1.0))
    while len(current_population) < population:
        current_population.append(random_candidate(blob_count, rng, target_land_fraction))

    elite_count = max(4, population // 6)
    best_candidate = current_population[0].copy()
    best_metrics = score_mask(render_mask(best_candidate, grid), target)

    for step in range(steps):
        scored = []
        for candidate in current_population:
            metrics = score_mask(render_mask(candidate, grid), target)
            scored.append((metrics["iou"], metrics, candidate))
            if metrics["iou"] > best_metrics["iou"]:
                best_candidate = candidate.copy()
                best_metrics = metrics
        scored.sort(key=lambda item: item[0], reverse=True)
        if step == steps - 1:
            break
        elites = [item[2].copy() for item in scored[:elite_count]]
        next_population = elites[:]
        scale = max(0.06, (1.0 - (step / max(steps - 1, 1))) ** 2)
        random_injections = max(1, population // 10)
        while len(next_population) < population - random_injections:
            parent = elites[int(rng.integers(0, len(elites)))]
            next_population.append(mutate_candidate(parent, rng, scale))
        while len(next_population) < population:
            next_population.append(random_candidate(blob_count, rng, target_land_fraction))
        current_population = next_population

    best_candidate, best_metrics = refine_candidate(best_candidate, target, grid, refine_passes)
    return best_candidate, best_metrics


def save_mask_png(mask: np.ndarray, path: Path) -> None:
    Image.fromarray(np.where(mask, 255, 0).astype(np.uint8)).save(path)


def save_diff_png(mask: np.ndarray, target: np.ndarray, path: Path) -> None:
    diff = np.zeros((*mask.shape, 3), dtype=np.uint8)
    diff[:] = (11, 18, 28)
    diff[target] = (74, 142, 255)
    diff[mask] = (255, 111, 82)
    diff[np.logical_and(mask, target)] = (242, 242, 242)
    Image.fromarray(diff).save(path)


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def clean_solve_outputs() -> None:
    patterns = [
        "best_*.json",
        "best_*_*.png",
        "best_*_diff.png",
        "compressed_*.json",
        "compressed_*_*.png",
        "compressed_*_diff.png",
        "best_dense.json",
        "best_dense.png",
        "best_dense_diff.png",
        "best_overall.png",
        "best_overall_diff.png",
        "best_overall_verify.png",
        "best_overall_verify_diff.png",
        "leaderboard.json",
        "target_*.png",
    ]
    for pattern in patterns:
        for path in OUT_DIR.glob(pattern):
            path.unlink(missing_ok=True)


def print_leaderboard(rows: list[dict]) -> None:
    print(f"{'blobs':>5} {'params':>6} {'search':>8} {'verify':>8} {'gap':>8} {'status':>14}")
    for row in rows:
        print(
            f"{row['blobs']:5d} {row['params']:6d} {row['search']['iou']:8.4f} {row['verify']['iou']:8.4f} "
            f"{row['generalization_gap']:8.4f} {row['status']:>14}"
        )


def maybe_detect_overfit(
    blob_count: int,
    stagnation: int,
    best_verify_blob_count: int,
    best_verify_iou: float,
    best_verify_search_iou: float,
    current_verify_iou: float,
    max_search_iou_since_best: float,
    args: argparse.Namespace,
) -> dict | None:
    if not args.stop_on_overfit:
        return None
    if blob_count < args.overfit_min_blobs:
        return None
    if stagnation < args.overfit_patience:
        return None
    search_gain = max_search_iou_since_best - best_verify_search_iou
    verify_drop = best_verify_iou - current_verify_iou
    if search_gain < args.overfit_train_delta:
        return None
    if verify_drop < args.overfit_verify_delta:
        return None
    return {
        "detected": True,
        "blob_count": blob_count,
        "best_verify_blob_count": best_verify_blob_count,
        "stagnation": stagnation,
        "search_gain": float(search_gain),
        "verify_drop": float(verify_drop),
    }


def cmd_fetch_target(args: argparse.Namespace) -> None:
    target = rasterize_target(args.width, args.height, url=args.target_url, refresh=args.refresh)
    output_path = OUT_DIR / f"target_{args.width}x{args.height}.png"
    save_mask_png(target, output_path)
    print(json.dumps({"target": str(output_path), "land_fraction": float(target.mean())}, indent=2))


def cmd_solve(args: argparse.Namespace) -> None:
    clean_solve_outputs()
    search_target = rasterize_target(args.search_width, args.search_height, url=args.target_url, refresh=args.refresh)
    search_grid = make_grid(args.search_width, args.search_height)
    render_target = rasterize_target(args.render_width, args.render_height, url=args.target_url, refresh=False)
    render_grid = make_grid(args.render_width, args.render_height)

    save_mask_png(search_target, OUT_DIR / f"target_{args.search_width}x{args.search_height}.png")
    save_mask_png(render_target, OUT_DIR / f"target_{args.render_width}x{args.render_height}.png")

    rows = []
    warm_start = None
    best_overall = None
    best_verify_metrics = None
    best_verify_search_metrics = None
    best_verify_blob_count = None
    best_search_metrics = None
    best_search_blob_count = None
    overfit_stagnation = 0
    max_search_iou_since_best = float("-inf")
    stopping = {
        "reason": "max-blobs-reached",
        "blob_count": args.max_blobs,
        "best_verify_blob_count": None,
    }

    for blob_count in range(args.min_blobs, args.max_blobs + 1):
        candidate, search_metrics = optimize(
            target=search_target,
            grid=search_grid,
            blob_count=blob_count,
            seed=args.seed + (blob_count * 1009),
            population=args.population,
            steps=args.steps,
            refine_passes=args.refine_passes,
            warm_start=warm_start,
        )
        warm_start = candidate

        preview_mask = render_mask(candidate, search_grid)
        verify_mask = render_mask(candidate, render_grid)
        verify_metrics = score_mask(verify_mask, render_target)
        model_path = OUT_DIR / f"best_{blob_count:02d}.json"
        save_mask_png(preview_mask, OUT_DIR / f"best_{blob_count:02d}_{args.search_width}x{args.search_height}.png")
        save_diff_png(preview_mask, search_target, OUT_DIR / f"best_{blob_count:02d}_diff.png")
        write_json(model_path, candidate.to_dict())

        if best_search_metrics is None or search_metrics["iou"] > best_search_metrics["iou"]:
            best_search_metrics = search_metrics
            best_search_blob_count = blob_count

        is_new_best_verify = best_verify_metrics is None or verify_metrics["iou"] > (best_verify_metrics["iou"] + args.overfit_min_delta)
        if is_new_best_verify:
            best_overall = candidate.copy()
            best_verify_metrics = verify_metrics
            best_verify_search_metrics = search_metrics
            best_verify_blob_count = blob_count
            overfit_stagnation = 0
            max_search_iou_since_best = search_metrics["iou"]
            status = "best-verify"
        else:
            overfit_stagnation += 1
            max_search_iou_since_best = max(max_search_iou_since_best, search_metrics["iou"])
            status = "search-only"

        row = {
            "blobs": blob_count,
            "params": count_params(candidate),
            "search": search_metrics,
            "verify": verify_metrics,
            "generalization_gap": float(search_metrics["iou"] - verify_metrics["iou"]),
            "status": status,
            "model": model_path.name,
            "candidate": candidate.copy(),
        }
        rows.append(row)

        if not is_new_best_verify:
            assert best_verify_blob_count is not None
            assert best_verify_metrics is not None
            assert best_verify_search_metrics is not None
            overfit = maybe_detect_overfit(
                blob_count=blob_count,
                stagnation=overfit_stagnation,
                best_verify_blob_count=best_verify_blob_count,
                best_verify_iou=best_verify_metrics["iou"],
                best_verify_search_iou=best_verify_search_metrics["iou"],
                current_verify_iou=verify_metrics["iou"],
                max_search_iou_since_best=max_search_iou_since_best,
                args=args,
            )
            if overfit is not None:
                row["status"] = "overfit-stop"
                row["overfit"] = overfit
                stopping = {
                    "reason": "overfit-detected",
                    **overfit,
                }
                break

    assert best_overall is not None
    compression_rows = []
    if args.compress_best and best_overall.blobs.shape[0] > args.compress_min_blobs:
        compression_rows = compress_candidate(
            candidate=best_overall,
            search_target=search_target,
            search_grid=search_grid,
            verify_target=render_target,
            verify_grid=render_grid,
            min_blobs=max(args.min_blobs, args.compress_min_blobs),
            refine_passes=args.refine_passes,
        )
        for row in compression_rows:
            blob_count = row["blobs"]
            model_path = OUT_DIR / f"compressed_{blob_count:02d}.json"
            preview_mask = render_mask(row["candidate"], search_grid)
            save_mask_png(preview_mask, OUT_DIR / f"compressed_{blob_count:02d}_{args.search_width}x{args.search_height}.png")
            save_diff_png(preview_mask, search_target, OUT_DIR / f"compressed_{blob_count:02d}_diff.png")
            write_json(model_path, row["candidate"].to_dict())
            row["model"] = model_path.name
            if row["verify"]["iou"] > best_verify_metrics["iou"]:
                best_overall = row["candidate"].copy()
                best_verify_metrics = row["verify"]
                best_verify_search_metrics = row["search"]
                best_verify_blob_count = blob_count

    if stopping["reason"] == "max-blobs-reached":
        stopping["blob_count"] = rows[-1]["blobs"]
        stopping["best_verify_blob_count"] = best_verify_blob_count
    render_mask_best = render_mask(best_overall, render_grid)
    render_metrics = score_mask(render_mask_best, render_target)

    combined_rows = rows + compression_rows
    frontier = build_pareto_frontier(combined_rows)
    best_dense = next(
        row for row in frontier if row["verify"]["iou"] >= (render_metrics["iou"] * args.dense_fraction)
    )
    best_dense_mask = render_mask(best_dense["candidate"], render_grid)

    save_mask_png(render_mask_best, OUT_DIR / "best_overall.png")
    save_diff_png(render_mask_best, render_target, OUT_DIR / "best_overall_diff.png")
    write_json(OUT_DIR / "best_overall.json", best_overall.to_dict())
    save_mask_png(best_dense_mask, OUT_DIR / "best_dense.png")
    save_diff_png(best_dense_mask, render_target, OUT_DIR / "best_dense_diff.png")
    write_json(OUT_DIR / "best_dense.json", best_dense["candidate"].to_dict())

    report = {
        "search_resolution": [args.search_width, args.search_height],
        "render_resolution": [args.render_width, args.render_height],
        "population": args.population,
        "steps": args.steps,
        "refine_passes": args.refine_passes,
        "seed": args.seed,
        "target_url": args.target_url,
        "stop_on_overfit": args.stop_on_overfit,
        "overfit_min_delta": args.overfit_min_delta,
        "overfit_train_delta": args.overfit_train_delta,
        "overfit_verify_delta": args.overfit_verify_delta,
        "overfit_patience": args.overfit_patience,
        "overfit_min_blobs": args.overfit_min_blobs,
        "compress_best": args.compress_best,
        "compress_min_blobs": args.compress_min_blobs,
        "dense_fraction": args.dense_fraction,
        "results": [strip_candidate(row) for row in rows],
        "compression": [strip_candidate(row) for row in compression_rows],
        "pareto_frontier": [strip_candidate(row) for row in frontier],
        "best_search_metrics": best_search_metrics,
        "best_search_blob_count": best_search_blob_count,
        "best_verify_metrics": best_verify_metrics,
        "best_verify_blob_count": best_verify_blob_count,
        "best_verify_search_metrics": best_verify_search_metrics,
        "best_render_metrics": render_metrics,
        "best_dense": strip_candidate(best_dense),
        "stopping": stopping,
    }
    write_json(OUT_DIR / "leaderboard.json", report)
    print_leaderboard(rows)
    print(json.dumps({"best_render_metrics": render_metrics, "best_dense": strip_candidate(best_dense), "stopping": stopping, "outputs": str(OUT_DIR)}, indent=2))


def cmd_verify(args: argparse.Namespace) -> None:
    candidate = Candidate.from_dict(json.loads(Path(args.model).read_text(encoding="utf-8")))
    target = rasterize_target(args.width, args.height, url=args.target_url, refresh=args.refresh)
    mask = render_mask(candidate, make_grid(args.width, args.height))
    metrics = score_mask(mask, target)
    stem = Path(args.model).stem
    save_mask_png(mask, OUT_DIR / f"{stem}_verify.png")
    save_diff_png(mask, target, OUT_DIR / f"{stem}_verify_diff.png")
    print(json.dumps(metrics, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Approximate the world map with the smallest useful blob hypothesis.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    fetch_target = subparsers.add_parser("fetch-target", help="Download and rasterize the target land mask.")
    fetch_target.add_argument("--width", type=int, default=256)
    fetch_target.add_argument("--height", type=int, default=128)
    fetch_target.add_argument("--target-url", default=TARGET_URL)
    fetch_target.add_argument("--refresh", action="store_true")
    fetch_target.set_defaults(func=cmd_fetch_target)

    solve = subparsers.add_parser("solve", help="Sweep hypothesis complexity and save the leaderboard.")
    solve.add_argument("--min-blobs", type=int, default=0)
    solve.add_argument("--max-blobs", type=int, default=48)
    solve.add_argument("--population", type=int, default=48)
    solve.add_argument("--steps", type=int, default=24)
    solve.add_argument("--refine-passes", type=int, default=4)
    solve.add_argument("--seed", type=int, default=7)
    solve.add_argument("--search-width", type=int, default=128)
    solve.add_argument("--search-height", type=int, default=64)
    solve.add_argument("--render-width", type=int, default=256)
    solve.add_argument("--render-height", type=int, default=128)
    solve.add_argument("--stop-on-overfit", action=argparse.BooleanOptionalAction, default=True)
    solve.add_argument("--overfit-min-delta", type=float, default=0.001)
    solve.add_argument("--overfit-train-delta", type=float, default=0.003)
    solve.add_argument("--overfit-verify-delta", type=float, default=0.003)
    solve.add_argument("--overfit-patience", type=int, default=4)
    solve.add_argument("--overfit-min-blobs", type=int, default=6)
    solve.add_argument("--compress-best", action=argparse.BooleanOptionalAction, default=True)
    solve.add_argument("--compress-min-blobs", type=int, default=6)
    solve.add_argument("--dense-fraction", type=float, default=0.92)
    solve.add_argument("--target-url", default=TARGET_URL)
    solve.add_argument("--refresh", action="store_true")
    solve.set_defaults(func=cmd_solve)

    verify = subparsers.add_parser("verify", help="Score a saved model against the target land mask.")
    verify.add_argument("--model", required=True)
    verify.add_argument("--width", type=int, default=256)
    verify.add_argument("--height", type=int, default=128)
    verify.add_argument("--target-url", default=TARGET_URL)
    verify.add_argument("--refresh", action="store_true")
    verify.set_defaults(func=cmd_verify)

    return parser


def main() -> None:
    ensure_dirs()
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()