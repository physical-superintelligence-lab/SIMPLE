"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.
"""

import hashlib
import json
import threading
from pathlib import Path
from typing import Dict, Optional
from platformdirs import user_cache_dir
import requests

class DownloadManager:
    """Manages large binary assets with on-demand downloading & caching."""

    _lock = threading.Lock()  # Prevent race conditions when downloading

    def __init__(self, package_name, assets: Dict[str, Dict[str, str]]):
        """
        Args:
            package_name: Unique name for your package (used for cache dir)
            assets: Mapping from asset name -> {url: str, sha256: str}
        """
        self.cache_dir = Path(user_cache_dir(package_name))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        # self.assets = assets

    def _sha256sum(self, path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    def _download(self, url: str, dest: Path):
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

    def get(self, name: str) -> Path:
        """Return the local path to the asset, downloading if necessary."""
        if name not in self.assets:
            raise ValueError(f"Unknown asset: {name}")

        asset_info = self.assets[name]
        path = self.cache_dir / name

        with self._lock:
            # If file exists & checksum matches, reuse
            if path.exists():
                if self._sha256sum(path) == asset_info["sha256"]:
                    return path
                else:
                    print(f"[AssetManager] Checksum mismatch, re-downloading {name}")
                    path.unlink()

            # Download fresh
            print(f"[AssetManager] Downloading {name}...")
            self._download(asset_info["url"], path)

            # Verify integrity
            if self._sha256sum(path) != asset_info["sha256"]:
                path.unlink()
                raise RuntimeError(f"Checksum failed for {name}")

        return path
