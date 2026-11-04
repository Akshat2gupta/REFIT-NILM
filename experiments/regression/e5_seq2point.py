"""E5: a small Seq2Point CNN on the 1-minute aggregate. Runs in the separate .venv-e5 environment.

Input is a window of aggregate minutes. Each window has its own minimum removed, so no scaling is learnt
from any home. Output is the washer's power at the target minute.

  past: the 121 minutes ending at t. Needs no future data.
  look: 60 minutes either side of t. Requires future data.

For a test home, the network trains on eight homes. Two other training homes choose the best epoch, and the
test home is used for nothing but the final prediction.
Usage: python -B e5_seq2point.py --track look --houses 16
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "10")

import numpy as np
import pandas as pd
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
CACHE, PRED = HERE / "cache", HERE / "outputs" / "predictions"
HOUSES = (2, 5, 6, 7, 10, 13, 15, 16, 17, 19, 20)
SEED, KEEP_EVERY, HARD_EXCESS_W, SCALE = 42, 8, 300, 1000.0
WINDOW = 121
OFFSETS = {"past": np.arange(-WINDOW + 1, 1), "look": np.arange(-(WINDOW // 2), WINDOW // 2 + 1)}
BATCH, MAX_EPOCHS, PATIENCE, LR = 512, 8, 2, 1e-3
VALID_EVERY = 5


class Seq2Point(nn.Module):
    """The layer pattern of Zhang et al. (2018), made smaller for a 121-minute window on a laptop CPU."""

    def __init__(self) -> None:
        super().__init__()
        layers, channels = [], 2
        for filters, kernel in ((30, 9), (30, 7), (40, 5), (50, 5), (50, 5)):
            layers += [nn.Conv1d(channels, filters, kernel, padding=kernel // 2), nn.ReLU()]
            channels = filters
        self.net = nn.Sequential(*layers, nn.Flatten(), nn.Linear(50 * WINDOW, 256), nn.ReLU(), nn.Linear(256, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


class Homes:
    """All homes joined into one array, with empty padding so that no window can cross between homes."""

    def __init__(self) -> None:
        parts, padding, position = [], np.full(WINDOW, np.nan, dtype="float32"), 0
        self.rows: dict[int, dict] = {}
        for house in HOUSES:
            series = pd.read_parquet(CACHE / f"house{house}_series.parquet")["aggregate"]
            base = pd.read_parquet(CACHE / f"house{house}_base.parquet", columns=["wm", "above_min_60", "eval_ok"])
            local = series.index.get_indexer(base.index)
            assert (local >= 0).all()
            self.rows[house] = {"pos": local + position, "y": base["wm"].to_numpy("float32"), "excess": base["above_min_60"].to_numpy("float32"), "ok": base["eval_ok"].to_numpy(), "index": base.index}
            parts += [series.to_numpy("float32"), padding]
            position += len(series) + WINDOW
        self.series = np.concatenate(parts)

    def windows(self, positions: np.ndarray, offsets: np.ndarray) -> torch.Tensor:
        raw = self.series[positions[:, None] + offsets]
        seen = np.isfinite(raw)
        low = np.min(np.where(seen, raw, np.inf), axis=1, keepdims=True)
        values = np.where(seen, (raw - low) / SCALE, 0.0).astype("float32")
        return torch.from_numpy(np.stack([values, seen.astype("float32")], axis=1))

    def training(self, houses: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The same rows and weights as reglib.training_rows."""
        pos, y, weight = [], [], []
        for house in houses:
            r = self.rows[house]
            hard = (r["y"] > 20) | (r["excess"] >= HARD_EXCESS_W)
            keep = hard | (np.random.default_rng(SEED + house).random(len(r["y"])) < 1 / KEEP_EVERY)
            pos.append(r["pos"][keep]); y.append(r["y"][keep]); weight.append(np.where(hard[keep], 1.0, float(KEEP_EVERY)).astype("float32"))
        return np.concatenate(pos), np.concatenate(y), np.concatenate(weight)


def predict(model: nn.Module, homes: Homes, positions: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    model.eval()
    out = []
    with torch.inference_mode():
        for start in range(0, len(positions), 4096):
            out.append(model(homes.windows(positions[start:start + 4096], offsets)).numpy())
    return np.clip(np.concatenate(out) * SCALE, 0, None)


def run_fold(homes: Homes, track: str, house: int) -> None:
    path = PRED / f"E5_{track}__h{house}.parquet"
    if path.exists():
        print(f"[{track}] house {house}: already done", flush=True)
        return
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    offsets = OFFSETS[track]
    i = HOUSES.index(house)
    held = [HOUSES[(i + 1) % len(HOUSES)], HOUSES[(i + 2) % len(HOUSES)]]
    train = [h for h in HOUSES if h != house and h not in held]
    pos, y, weight = homes.training(train)
    y_scaled = torch.from_numpy(y / SCALE)
    weight_t = torch.from_numpy(weight)
    valid = [(homes.rows[h]["pos"][homes.rows[h]["ok"]][::VALID_EVERY], homes.rows[h]["y"][homes.rows[h]["ok"]][::VALID_EVERY]) for h in held]
    model = Seq2Point()
    optimiser = torch.optim.Adam(model.parameters(), lr=LR)
    print(f"[{track}] house {house}: {len(pos):,} training windows from {train}, epochs chosen on {held}", flush=True)
    best, best_state, stale = np.inf, None, 0
    for epoch in range(1, MAX_EPOCHS + 1):
        started = time.time()
        model.train()
        order = rng.permutation(len(pos))
        total, weight_sum = 0.0, 0.0
        for start in range(0, len(order), BATCH):
            part = order[start:start + BATCH]
            loss_each = (model(homes.windows(pos[part], offsets)) - y_scaled[part]) ** 2
            loss = (loss_each * weight_t[part]).sum() / weight_t[part].sum()
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            total += float(loss.detach()) * float(weight_t[part].sum())
            weight_sum += float(weight_t[part].sum())
        trained = time.time() - started
        score = float(np.mean([np.sqrt(np.mean((predict(model, homes, p, offsets) - t) ** 2)) for p, t in valid]))
        print(f"[{track}] house {house} epoch {epoch}: train loss {total / weight_sum:.5f}, held-back RMSE {score:.2f}, {trained:.0f}s train, {time.time() - started:.0f}s total", flush=True)
        if score < best:
            best, best_state, stale = score, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    model.load_state_dict(best_state)
    r = homes.rows[house]
    pd.DataFrame({"pred": predict(model, homes, r["pos"], offsets).astype("float32")}, index=r["index"]).to_parquet(path)
    print(f"[{track}] house {house}: saved, held-back RMSE {best:.2f}", flush=True)


def check(homes: Homes) -> None:
    """A past window must not change when later readings change; a look window must."""
    position = homes.rows[16]["pos"][len(homes.rows[16]["pos"]) // 2:][:1]
    before = {t: homes.windows(position, o).numpy().copy() for t, o in OFFSETS.items()}
    saved = homes.series[position[0] + 1:position[0] + 61].copy()
    homes.series[position[0] + 1:position[0] + 61] = saved + 500
    after = {t: homes.windows(position, o).numpy() for t, o in OFFSETS.items()}
    homes.series[position[0] + 1:position[0] + 61] = saved
    assert np.array_equal(before["past"], after["past"], equal_nan=True), "past window read the future"
    assert not np.array_equal(before["look"], after["look"], equal_nan=True), "look window ignored the future"
    print("Window check passed: past reads no future minutes, look does", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--track", choices=tuple(OFFSETS), required=True)
    parser.add_argument("--houses", type=int, nargs="+", default=list(HOUSES))
    args = parser.parse_args()
    homes = Homes()
    check(homes)
    for house in args.houses:
        run_fold(homes, args.track, house)


if __name__ == "__main__":
    main()
