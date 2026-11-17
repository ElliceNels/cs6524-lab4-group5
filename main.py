from pathlib import Path

from mypylibs.optical_flow import compute_optical_flow
from mypylibs.tracking import track_object


if __name__ == '__main__':
    root = Path(__file__).resolve().parent
    out = root / 'out'
    out.mkdir(exist_ok=True)
    for video in sorted((root / 'in').glob('*.mp4')):
        measurements = compute_optical_flow(video, out)
        tracker = track_object(video, measurements, out)
        for event in tracker.reacquisitions:
            print(f'{video.name}: reacquired at frame {event.frame} after {event.gap} frames, '
                  f'prediction error {event.error:.1f} px')
