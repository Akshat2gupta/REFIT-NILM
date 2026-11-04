"""Download and extract the REFIT archives into ``data/``.

Run from anywhere:  ``python scripts/download_data.py``

* Raw archive     -> only House 1 is extracted (the raw task uses House 1 only).
* Cleaned archive -> all houses are extracted to ``data/cleaned``.

The university's ``pureportal.strath.ac.uk/files/...`` links sit behind a browser
challenge, so we use ``pure.strath.ac.uk/ws/portalfiles/portal/...``: the same
repository's file endpoint and the same file IDs. Existing files are not downloaded
or extracted again, so the script is safe to re-run.
"""
from __future__ import annotations

import shutil
import urllib.request
from pathlib import Path

import py7zr

ROOT = Path(__file__).resolve().parents[1]
ARCHIVES = ROOT / "data" / "archives"
BASE_URL = "https://pure.strath.ac.uk/ws/portalfiles/portal"

FILES = {
    "raw": ("62090177/REFIT_RAW_081116.7z", 698_108_465),
    "cleaned": ("62090184/CLEAN_REFIT_081116.7z", 514_265_481),
}
README = "62090183/CLEAN_READ_ME_081116.txt"


def download(url: str, dest: Path, expected_size: int | None = None) -> None:
    """Download ``url`` to ``dest`` unless a complete copy already exists."""
    if dest.exists() and (expected_size is None or dest.stat().st_size == expected_size):
        print(f"[skip] {dest.name} already downloaded")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[get ] {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as response, open(tmp, "wb") as out:
        shutil.copyfileobj(response, out, length=1024 * 1024)
    if expected_size is not None and tmp.stat().st_size != expected_size:
        raise RuntimeError(f"{dest.name}: got {tmp.stat().st_size} bytes, expected {expected_size}")
    tmp.replace(dest)


def extract(archive: Path, out_dir: Path, house1_only: bool = False) -> None:
    """Extract CSVs not yet on disk. ``house1_only`` limits this to House 1 files."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with py7zr.SevenZipFile(archive) as z:
        names = [n for n in z.getnames() if n.lower().endswith(".csv")]
    if house1_only:
        names = [n for n in names if n.startswith("RAW_House1_")]
    missing = [n for n in names if not (out_dir / Path(n).name).exists()]
    if not missing:
        print(f"[skip] {archive.name}: {len(names)} files already extracted")
        return
    print(f"[7z  ] {archive.name}: extracting {len(missing)} files to {out_dir}")
    with py7zr.SevenZipFile(archive) as z:
        z.extract(path=out_dir, targets=missing)


def main() -> None:
    download(f"{BASE_URL}/{README}", ARCHIVES / "cleaned" / "CLEAN_READ_ME_081116.txt")
    for name, (path, size) in FILES.items():
        download(f"{BASE_URL}/{path}", ARCHIVES / name / Path(path).name, size)
    extract(ARCHIVES / "raw" / Path(FILES["raw"][0]).name, ROOT / "data" / "raw", house1_only=True)
    extract(ARCHIVES / "cleaned" / Path(FILES["cleaned"][0]).name, ROOT / "data" / "cleaned")
    print("Done.")


if __name__ == "__main__":
    main()
