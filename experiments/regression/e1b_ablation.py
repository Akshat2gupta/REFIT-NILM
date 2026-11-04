"""Which parts of E1_L60_ctx earn their place? Remove one family at a time, then the plain future windows."""
from __future__ import annotations

from reglib import report, run

FULL = ("fut30", "fut60", "ctx")
FAMILIES = {
    "shape": ("ctx_in_block", "ctx_block_len", "ctx_block_height", "ctx_block_pos", "ctx_since_end"),
    "jitter": ("ctx_post_jit", "ctx_pre_jit", "ctx_next_post_jit", "ctx_next_pre_jit", "fut_jit_med_15", "fut_jit_med_30", "fut_jit_med_60"),
    "next": ("ctx_to_next", "ctx_next_len"),
    "count": ("ctx_blocks_within_60", "ctx_blocks_within_120", "ctx_blocks_within_180"),
    "stretch": ("ctx_since_quiet", "ctx_to_quiet", "ctx_stretch"),
}


def main() -> None:
    names = []
    for family, columns in FAMILIES.items():
        name = f"A_no_{family}"
        run(name, FULL, drop=columns, note=f"E1_L60_ctx without the {family} family")
        names.append(name)
    run("A_ctx_only", ("ctx",), note="Baseline features plus ctx, without the plain future windows")
    names.append("A_ctx_only")
    report(names, "E1_L60_ctx", "Ablation: difference from E1_L60_ctx (positive RMSE means the removed part helped)")


if __name__ == "__main__":
    main()
