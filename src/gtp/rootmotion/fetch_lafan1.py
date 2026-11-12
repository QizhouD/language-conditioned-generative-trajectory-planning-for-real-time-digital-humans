"""Download and unpack the LAFAN1 motion-capture archive.

LAFAN1 is published by Ubisoft La Forge under CC BY-NC-ND 4.0. Do not commit the
archive, the BVH files, or tracks extracted from them to this repository.
"""

from __future__ import annotations

import argparse
import hashlib
import urllib.request
import zipfile
from pathlib import Path

URL = (
    "https://media.githubusercontent.com/media/ubisoft/"
    "ubisoft-laforge-animation-dataset/master/lafan1/lafan1.zip"
)
SHA256 = "ea918082b500a5d158e9d3aa39039df04cd42e25f5c02fe8f7e88e8e9365a977"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/lafan1"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    archive = args.output / "lafan1.zip"
    if not archive.exists():
        print(f"downloading {URL}")
        urllib.request.urlretrieve(URL, archive)  # noqa: S310 - fixed https URL
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest != SHA256:
        raise SystemExit(f"checksum mismatch for {archive}: {digest}")
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.namelist():
            # Extract flat and refuse anything that is not a plain BVH name.
            name = Path(member).name
            if name.endswith(".bvh") and name == member.split("/")[-1]:
                (args.output / name).write_bytes(bundle.read(member))
    print(f"{len(list(args.output.glob('*.bvh')))} BVH files in {args.output}")


if __name__ == "__main__":
    main()
