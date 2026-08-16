"""Retrieval of the balanced datasets used in the CutClean experiments.

The two datasets that are redistributed as archives (Corrupted-CIFAR10 and
Waterbirds) are downloaded from the Hugging Face Hub, verified against their
SHA-256 checksum and extracted once. CelebA is not redistributed: only the split
manifests are published, and the images must be obtained from the official
CelebA release and placed under ``<root>/celeba`` as described in the README.

The download location follows, in order of precedence:

1. the ``--datapath`` / ``--corruptedcifarunbiased_root`` / ``--waterbirds_root``
   arguments, when the corresponding directory already exists;
2. the ``CUTCLEAN_DATA`` environment variable;
3. ``./data`` relative to the repository root.

Run as a script to fetch everything up front::

    python src/data_setup.py                 # download and extract into CUTCLEAN_DATA
    python src/data_setup.py --check         # verify an existing copy, download nothing
    python src/data_setup.py --root /scratch/data
"""

import argparse
import hashlib
import os
import shutil
import sys
import tarfile

HF_REPO_ID = os.environ.get("CUTCLEAN_HF_REPO", "imDalton/cutclean-datasets")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# name -> (archive in the Hub repository, directory it expands to, sha256 of the archive)
ARCHIVES = {
    "corruptedcifarunbiased": (
        "corrupted_cifar10_unbiased.tar.gz",
        "corrupted_cifar_unbiased",
        "91c8e0e29e38f6e98d9a3429872050a06d54a2acacefe7300f8bb7d80303e9ce",
    ),
    "unbiasedWaterbirds": (
        "waterbirds_unbiased.tar.gz",
        "waterbirds_unbiased",
        "53f930e31466ca1141818208595ed758f94f54c9653cee832a87cb74c557c691",
    ),
}

CELEBA_MANIFESTS = [
    "celeba_manifests/celeba-Blond_Hair-Male.csv",
    "celeba_manifests/celeba-Heavy_Makeup-Male.csv",
]


def default_root():
    """Directory holding the datasets when no explicit path is given."""
    return os.environ.get("CUTCLEAN_DATA", os.path.join(REPO_ROOT, "data"))


def sha256(path, chunk=1 << 20):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_expected_checksums(root):
    """Read SHA256SUMS shipped next to the archives, if it was downloaded."""
    path = os.path.join(root, "SHA256SUMS")
    if not os.path.isfile(path):
        return {}
    sums = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            parts = line.split()
            if len(parts) == 2:
                sums[os.path.basename(parts[1].lstrip("*"))] = parts[0]
    return sums


def ensure_dataset(name, root=None, force=False):
    """Return the local directory of `name`, downloading and extracting if needed."""
    if name not in ARCHIVES:
        raise KeyError(f"Unknown dataset '{name}'; expected one of {sorted(ARCHIVES)}")

    root = root or default_root()
    archive_name, directory, expected = ARCHIVES[name]
    target = os.path.join(root, directory)

    if os.path.isdir(target) and not force:
        return target

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise RuntimeError(
            "huggingface_hub is required to download the datasets; "
            "install it with `pip install huggingface_hub`."
        ) from exc

    os.makedirs(root, exist_ok=True)
    print(f"[data] downloading {archive_name} from {HF_REPO_ID}", flush=True)
    archive = hf_hub_download(repo_id=HF_REPO_ID, filename=archive_name,
                              repo_type="dataset", local_dir=root)

    expected = expected or _load_expected_checksums(root).get(archive_name)
    if expected:
        digest = sha256(archive)
        if digest != expected:
            raise RuntimeError(
                f"Checksum mismatch for {archive_name}: expected {expected}, got {digest}"
            )
        print(f"[data] checksum verified for {archive_name}", flush=True)

    print(f"[data] extracting into {target}", flush=True)
    staging = target + ".partial"
    shutil.rmtree(staging, ignore_errors=True)
    os.makedirs(staging, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(staging)
    # the archive already contains the top-level directory
    inner = os.path.join(staging, directory)
    shutil.move(inner if os.path.isdir(inner) else staging, target)
    shutil.rmtree(staging, ignore_errors=True)
    return target


def fetch_celeba_manifests(root=None):
    """Download the CelebA split manifests; the images are not redistributed."""
    from huggingface_hub import hf_hub_download

    root = root or default_root()
    os.makedirs(root, exist_ok=True)
    paths = []
    for name in CELEBA_MANIFESTS:
        paths.append(hf_hub_download(repo_id=HF_REPO_ID, filename=name,
                                     repo_type="dataset", local_dir=root))
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default=None,
                        help="destination directory (default: $CUTCLEAN_DATA or ./data)")
    parser.add_argument("--check", action="store_true",
                        help="only report what is present, download nothing")
    args = parser.parse_args()

    root = args.root or default_root()
    print(f"[data] root: {root}")

    if args.check:
        missing = 0
        for name, (_archive, directory, _sha) in ARCHIVES.items():
            path = os.path.join(root, directory)
            ok = os.path.isdir(path)
            missing += 0 if ok else 1
            print(f"  {'present' if ok else 'MISSING'}  {name:26s} {path}")
        celeba = os.path.join(root, "celeba")
        print(f"  {'present' if os.path.isdir(celeba) else 'MISSING'}  {'celeba':26s} {celeba}"
              "   (obtain the official release manually; see the README)")
        return 1 if missing else 0

    for name in ARCHIVES:
        path = ensure_dataset(name, root=root)
        print(f"[data] {name}: {path}")
    for path in fetch_celeba_manifests(root=root):
        print(f"[data] manifest: {path}")
    print("[data] CelebA images are not redistributed: obtain the official release "
          f"manually and place it under {os.path.join(root, 'celeba')} (see the README).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
