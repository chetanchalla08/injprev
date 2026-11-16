"""Download the MediaPipe Pose Landmarker weights from Google's public model bucket."""
import sys
import urllib.request
from pathlib import Path

BASE = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_{m}/float16/latest/pose_landmarker_{m}.task"


def main() -> None:
    out = Path(__file__).resolve().parent.parent / "models"
    out.mkdir(exist_ok=True)
    for m in sys.argv[1:] or ["heavy", "full"]:
        dst = out / f"pose_landmarker_{m}.task"
        if dst.exists() and dst.stat().st_size > 1_000_000:
            print(f"{dst.name}: already present")
            continue
        print(f"downloading {dst.name} ...")
        urllib.request.urlretrieve(BASE.format(m=m), dst)
        print(f"  {dst.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
