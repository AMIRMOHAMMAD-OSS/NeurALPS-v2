"""Versioned release assets, downloaded atomically and checked before use."""
import json
from pathlib import Path
import shutil
import stat
import tempfile
import urllib.error
import urllib.request
import zipfile
from .contracts import require, sha

NATURAL_RELEASE = "https://github.com/AMIRMOHAMMAD-OSS/NeurALPS-v2/releases/download/v0.2.0-20261010"
STAMP = "20261010"


def download(url, path, digest=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and digest and sha(path) == digest:
        return path
    temporary = path.with_name(path.name+".part")
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent":"NeurALPS"}), timeout=120) as response, temporary.open("wb") as out:
            shutil.copyfileobj(response, out, 1024*1024)
        require(digest is None or sha(temporary) == digest, "Downloaded asset checksum mismatch")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def extract(archive, destination, maximum_bytes=4*1024**3):
    destination = Path(destination)
    require(not destination.exists(), "Extraction destination already exists")
    with zipfile.ZipFile(archive) as z:
        names = set()
        require(sum(i.file_size for i in z.infolist()) <= maximum_bytes, "Asset exceeds extraction limit")
        for i in z.infolist():
            p = Path(i.filename)
            require(not p.is_absolute() and ".." not in p.parts and "\\" not in i.filename and
                    not stat.S_ISLNK(i.external_attr >> 16) and i.filename not in names, "Unsafe or duplicate ZIP member")
            names.add(i.filename)
        z.extractall(destination)


def natural_reference(base, release_url=NATURAL_RELEASE):
    from .reference import NaturalReference
    base = Path(base)
    destination = base/"natural_reference"
    if not destination.exists():
        asset = download(release_url+f"/NeurALPS_natural_reference_{STAMP}.zip", base/f"NeurALPS_natural_reference_{STAMP}.zip")
        extract(asset, destination)
    return NaturalReference(destination)


def natural_repertoire(base, release_url=NATURAL_RELEASE):
    from .repertoire import Repertoire
    base = Path(base)
    destination = base/"natural_repertoire"
    if not destination.exists():
        try:
            asset = download(release_url+f"/NeurALPS_repertoire_index_{STAMP}.zip", base/f"NeurALPS_repertoire_index_{STAMP}.zip")
        except urllib.error.HTTPError as e:
            if e.code == 404:
                raise FileNotFoundError("The natural repertoire is not published yet. Run the Jean Zay exporter and upload its ZIP assets as described in docs/EXPLORER_SETUP.md.") from e
            raise
        extract(asset, destination)
    def fetch(meta, root):
        asset = download(release_url+"/"+meta["archive"], base/meta["archive"], meta["archive_sha256"])
        # Shards contain one known filename and cannot overwrite other bank files.
        with zipfile.ZipFile(asset) as z:
            require(z.namelist() == [meta["path"]], "Unexpected vector shard member")
            require(z.getinfo(meta["path"]).file_size == meta["rows"]*1152*4, "Vector shard size differs")
            target = root/meta["path"]
            require(target.resolve().is_relative_to(root.resolve()), "Unsafe vector shard path")
            temporary = target.with_suffix(".part")
            with z.open(meta["path"]) as stream, temporary.open("wb") as out:
                shutil.copyfileobj(stream, out, 1024*1024)
            require(sha(temporary) == meta["sha256"], "Extracted vector shard checksum mismatch")
            temporary.replace(target)
        asset.unlink()
    return Repertoire(destination, fetch_shard=fetch)
