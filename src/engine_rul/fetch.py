"""rul-fetch: download the whole dataset once into data/raw/cmapss, so get_raw always finds the files locally.

Why not let get_raw download? Its fallback asks kagglehub for one file at a time, and for this dataset every
single-file download fails kagglehub's MD5 check (DataCorruptionError). Downloading the whole dataset works.
"""
import logging
import shutil
import sys
from pathlib import Path

from engine_rul.config import data_dir, load_config

logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config = load_config()
    dest = data_dir(config) / "raw" / "cmapss" / "CMaps"
    if (dest / "train_FD001.txt").exists():
        logger.info("data already in %s", dest)
        return
    try:
        import kagglehub
        src = Path(kagglehub.dataset_download(config["kaggle_handle"])) / "CMaps"
        shutil.copytree(src, dest, dirs_exist_ok=True)
        logger.info("copied %d files to %s", len(list(dest.iterdir())), dest)
    except Exception:
        logger.exception("Download failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
