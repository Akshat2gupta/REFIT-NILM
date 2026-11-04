"""E3: same inputs, different loss or model structure. Each is judged against its own squared-error run."""
from __future__ import annotations

from reglib import report, run

TRACKS = {"past": (("jitter",), "E2_jitter"), "look": (("fut30", "fut60", "ctx"), "E1_L60_ctx")}
KINDS = ("poisson", "log", "hurdle", "mae", "states")


def main() -> None:
    for track, (groups, reference) in TRACKS.items():
        names = []
        for kind in KINDS:
            name = f"E3_{track}_{kind}"
            run(name, groups, kind=kind, note=f"{kind} on the {track} track's features")
            names.append(name)
        report(names, reference, f"E3 {track} track")


if __name__ == "__main__":
    main()
