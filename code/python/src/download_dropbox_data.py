"""Download the Allen Institute session files from the shared Dropbox folder.

Layout:
    data/raw/   <- .nwb recordings (large files, ~28 GB total)

Note: Reference notebooks are now in notebooks/ (version controlled).
      MATLAB code is in src/matlab/ (version controlled).

Files are streamed in fixed-size chunks straight to a ``.part`` file, so memory
use stays flat regardless of file size. Interrupted downloads resume via HTTP
Range, and a file is only renamed into place once its size matches the server.

Usage (from code/python/):
    python -m src.download_dropbox_data            # download .nwb files
    python -m src.download_dropbox_data --list     # show file sizes
"""

import argparse
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]  # code/python/src/ -> repo/


def load_folder_url() -> str:
    """Read DROPBOX_FOLDER_URL from the environment or the repo's .env file."""
    url = os.environ.get("DROPBOX_FOLDER_URL")
    env_file = ROOT / ".env"
    if not url and env_file.exists():
        for line in env_file.read_text().splitlines():
            key, sep, value = line.partition("=")
            if sep and key.strip() == "DROPBOX_FOLDER_URL":
                url = value.strip().strip("\"'")
    if not url:
        sys.exit(
            "DROPBOX_FOLDER_URL is not set. Copy .env.example to .env and paste the "
            "shared Dropbox folder link into it (see README)."
        )
    return url


FOLDER_URL = load_folder_url()

DATA_DIR = ROOT / "data"

# filename -> destination subdirectory under data/
# Note: Notebooks and MATLAB code are now version-controlled in the repo
FILES = {
    "688277-20230921-v001.nwb": "raw",
    "692644-20230907-v001.nwb": "raw",
}

CHUNK_SIZE = 8 * 1024 * 1024  # 8 MiB
MAX_RETRIES = 8


def file_url(name: str) -> str:
    parts = urllib.parse.urlsplit(FOLDER_URL)
    query = urllib.parse.parse_qs(parts.query)
    if "rlkey" not in query:
        sys.exit("DROPBOX_FOLDER_URL must be a full shared-folder link (it needs an rlkey=... part).")
    new_query = urllib.parse.urlencode({"rlkey": query["rlkey"][0], "dl": 1, "preview": name})
    return urllib.parse.urlunsplit(parts._replace(query=new_query))


def remote_size(name: str) -> int:
    req = urllib.request.Request(file_url(name), headers={"Range": "bytes=0-0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        content_range = resp.headers.get("Content-Range")  # "bytes 0-0/TOTAL"
        if content_range:
            return int(content_range.rsplit("/", 1)[1])
        return int(resp.headers["Content-Length"])


def fmt(n: float) -> str:
    for unit in ("B", "KB", "MB"):
        if n < 1024:
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} GB"


def download(name: str, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / name
    part = dest_dir / (name + ".part")

    total = remote_size(name)
    if dest.exists() and dest.stat().st_size == total:
        print(f"[skip] {dest} ({fmt(total)}) already complete")
        return dest

    for attempt in range(1, MAX_RETRIES + 1):
        have = part.stat().st_size if part.exists() else 0
        if have > total:
            part.unlink()
            have = 0
        if have == total:
            break

        req = urllib.request.Request(
            file_url(name), headers={"Range": f"bytes={have}-"}
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                if have and resp.status != 206:
                    part.unlink()
                    have = 0
                    continue
                print(f"[get ] {name}: {fmt(have)} / {fmt(total)} (attempt {attempt})")
                start, last_print = time.time(), time.time()
                got = 0
                with open(part, "ab") as f:
                    while chunk := resp.read(CHUNK_SIZE):
                        f.write(chunk)
                        got += len(chunk)
                        if time.time() - last_print >= 5:
                            done = have + got
                            rate = got / (time.time() - start)
                            print(
                                f"       {done / total:6.1%}  {fmt(done)} / {fmt(total)}"
                                f"  {fmt(rate)}/s",
                                end="\r",
                                flush=True,
                            )
                            last_print = time.time()
                print()
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            wait = min(2**attempt, 60)
            print(f"\n[warn] {name}: {e!r}; retrying in {wait}s", file=sys.stderr)
            time.sleep(wait)

    size = part.stat().st_size if part.exists() else 0
    if size != total:
        raise RuntimeError(f"{name}: got {size} of {total} bytes after {MAX_RETRIES} attempts")
    part.rename(dest)
    print(f"[done] {dest} ({fmt(total)})")
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Allen Institute data from Dropbox")
    parser.add_argument("--list", action="store_true", help="show files and sizes, then exit")
    args = parser.parse_args()

    if args.list:
        for name, sub in FILES.items():
            print(f"{fmt(remote_size(name)):>10}  data/{sub}/{name}")
        return

    for name, sub in FILES.items():
        download(name, DATA_DIR / sub)


if __name__ == "__main__":
    main()
