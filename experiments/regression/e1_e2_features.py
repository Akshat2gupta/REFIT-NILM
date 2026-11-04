"""E1: look-ahead windows. E2: past-only feature groups, one at a time, then together. E2b: 8 s input.

All use P0's model and settings, so only the input changes. Each is judged against P0.
"""
from __future__ import annotations

from reglib import report, run

PAST = {
    "E2_long": ("long",), "E2_jitter": ("jitter",), "E2_blocks": ("blocks",), "E2_clock": ("clock",), "E2_history": ("history",),
    "E2_all": ("long", "jitter", "blocks", "clock", "history"),
}
LOOK = {
    "E1_L30": ("fut30",), "E1_L60": ("fut30", "fut60"), "E1_L120": ("fut30", "fut60", "fut120"),
    "E1_L60_ctx": ("fut30", "fut60", "ctx"),
    "E1_all": ("long", "jitter", "blocks", "clock", "history", "fut30", "fut60", "fut120", "ctx"),
}
FINE = {
    "E2b_fine_past": ("fine",), "E2b_fine_all_past": ("long", "jitter", "blocks", "clock", "history", "fine"),
    "E2b_fine_look": ("long", "jitter", "blocks", "clock", "history", "fut30", "fut60", "fut120", "ctx", "fine", "futfine"),
}


def main() -> None:
    for sets, title in ((LOOK, "E1 look-ahead"), (PAST, "E2 past-only"), (FINE, "E2b 8-second input")):
        for name, groups in sets.items():
            run(name, groups)
        report(list(sets), "P0", title)


if __name__ == "__main__":
    main()
