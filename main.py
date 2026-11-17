from pathlib import Path

from mypylibs.optical_flow import compute_optical_flow


if __name__ == '__main__':
    root = Path(__file__).resolve().parent
    for video in sorted((root / 'in').glob('*.mp4')):
        compute_optical_flow(video, root / 'out')
