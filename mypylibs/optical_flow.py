"""Sparse Lucas-Kanade optical flow and per-frame object measurements."""
import csv
from dataclasses import dataclass

import cv2
import numpy as np

LK_PARAMS = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
)
FEATURE_PARAMS = dict(maxCorners=300, qualityLevel=0.01, minDistance=8, blockSize=7)
MOTION_FEATURE_PARAMS = dict(maxCorners=200, qualityLevel=0.001, minDistance=6, blockSize=7)
MOTION_THRESHOLD = 15

REDETECT_EVERY = 5
FB_MAX_ERROR = 1.0
MIN_SPEED = 1.0
MIN_MOVING_POINTS = 5
FLOW_TOLERANCE = 0.3
STATIC_PATIENCE = 50
TRAIL_LENGTH = 15


@dataclass
class Measurement:
    frame: int
    position: np.ndarray
    velocity: np.ndarray
    position_std: np.ndarray
    velocity_std: np.ndarray
    n_points: int


def _inliers(values, k=3.0):
    median = np.median(values, axis=0)
    mad = np.median(np.abs(values - median), axis=0) * 1.4826 + 1e-6
    return np.all(np.abs(values - median) <= k * mad, axis=1)


def _dominant_motion(flow):
    tolerance = np.maximum(FLOW_TOLERANCE * np.linalg.norm(flow, axis=1), 1.0)
    agree = np.linalg.norm(flow[:, None] - flow[None], axis=2) <= tolerance[:, None]
    return agree[np.argmax(agree.sum(axis=1))]


def _summarise(frame_idx, points, flow):
    keep = _inliers(points)
    if keep.sum() < MIN_MOVING_POINTS:
        return None
    points, flow = points[keep], flow[keep]
    return Measurement(frame_idx, points.mean(axis=0), flow.mean(axis=0),
                       points.std(axis=0), flow.std(axis=0), len(points))


def _measure(frame_idx, points, flow):
    """Split moving points into groups of coherent motion, largest first."""
    # Separate movers (object, hands, occluder edges) each get their own measurement
    measurements = []
    while len(points) >= MIN_MOVING_POINTS:
        group = _dominant_motion(flow)
        measurement = _summarise(frame_idx, points[group], flow[group])
        if measurement is not None:
            measurements.append(measurement)
        points, flow = points[~group], flow[~group]
    return measurements


def _track(prev_gray, gray, points):
    nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, points, None, **LK_PARAMS)
    back, st_back, _ = cv2.calcOpticalFlowPyrLK(gray, prev_gray, nxt, None, **LK_PARAMS)
    # Forward-backward consistency rejects points lost at occluder edges
    fb_error = np.linalg.norm(points - back, axis=2).ravel()
    return nxt, (st.ravel() == 1) & (st_back.ravel() == 1) & (fb_error < FB_MAX_ERROR)


def _detect(gray, existing, params, mask=None):
    if mask is None:
        mask = np.full(gray.shape, 255, np.uint8)
    for x, y in existing.reshape(-1, 2).astype(int):
        cv2.circle(mask, (x, y), params['minDistance'], 0, -1)
    found = cv2.goodFeaturesToTrack(gray, mask=mask, **params)
    return np.empty((0, 1, 2), np.float32) if found is None else found.astype(np.float32)


def _motion_mask(prev_gray, gray):
    diff = cv2.absdiff(cv2.GaussianBlur(prev_gray, (5, 5), 0), cv2.GaussianBlur(gray, (5, 5), 0))
    mask = np.uint8(diff > MOTION_THRESHOLD) * 255
    return cv2.dilate(mask, np.ones((15, 15), np.uint8))


def _direction_colour(flow):
    angle = (np.degrees(np.arctan2(flow[1], flow[0])) % 360) / 2
    hsv = np.uint8([[[angle, 255, 255]]])
    return tuple(int(c) for c in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0])


def _draw(frame, trails, moving, flow, measurements):
    overlay = frame.copy()
    for trail, is_moving, f in zip(trails, moving, flow):
        if not is_moving:
            cv2.circle(overlay, tuple(map(int, trail[-1])), 2, (160, 160, 160), -1)
            continue
        colour = _direction_colour(f)
        cv2.polylines(overlay, [np.int32(trail)], False, colour, 1, cv2.LINE_AA)
        cv2.circle(overlay, tuple(map(int, trail[-1])), 3, colour, -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

    for other in measurements[1:]:
        cv2.drawMarker(frame, tuple(map(int, other.position)), (200, 200, 200), cv2.MARKER_CROSS, 12, 1)
    if measurements:
        measurement = measurements[0]
        centre = measurement.position
        radius = np.maximum(measurement.position_std, 1).astype(int)
        cv2.ellipse(frame, tuple(map(int, centre)), tuple(map(int, radius)), 0, 0, 360,
                    (255, 255, 255), 1, cv2.LINE_AA)
        # Velocity is scaled so per-frame displacements are visible
        tip = centre + 10 * measurement.velocity
        cv2.arrowedLine(frame, tuple(map(int, centre)), tuple(map(int, tip)),
                        (0, 255, 255), 2, cv2.LINE_AA, tipLength=0.2)
        cv2.drawMarker(frame, tuple(map(int, centre)), (0, 0, 255), cv2.MARKER_CROSS, 16, 2)
        text = (f'pos ({centre[0]:.0f}, {centre[1]:.0f}) px  '
                f'vel ({measurement.velocity[0]:+.1f}, {measurement.velocity[1]:+.1f}) px/frame  '
                f'n={measurement.n_points}  groups={len(measurements)}')
    else:
        text = 'no measurement'
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(frame, (5, 5), (25 + w, 25 + h), (0, 0, 0), -1)
    cv2.putText(frame, text, (15, 15 + h), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)


def _write_csv(path, measurements, n_frames):
    by_frame = {}
    for m in measurements:
        by_frame.setdefault(m.frame, []).append(m)
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['frame', 'group', 'x', 'y', 'vx', 'vy', 'sx', 'sy', 'svx', 'svy', 'n'])
        for i in range(n_frames):
            if i not in by_frame:
                writer.writerow([i] + [''] * 10)
            for group, m in enumerate(by_frame.get(i, [])):
                writer.writerow([i, group, *np.round(m.position, 2), *np.round(m.velocity, 3),
                                 *np.round(m.position_std, 2), *np.round(m.velocity_std, 3),
                                 m.n_points])


def compute_optical_flow(video_path, output_dir):
    """Track LK keypoints, write `XX_of.mp4` and `XX_of.csv`, return every motion group of every frame."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    writer = cv2.VideoWriter(str(output_dir / f'{video_path.stem}_of.mp4'),
                             cv2.VideoWriter_fourcc(*'mp4v'), fps, size)

    ok, frame = cap.read()
    prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    points = _detect(prev_gray, np.empty((0, 1, 2), np.float32), FEATURE_PARAMS)
    trails = [[p] for p in points.reshape(-1, 2)]
    still = np.zeros(len(points), int)
    measurements = []
    writer.write(frame)

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        flow = np.empty((0, 2), np.float32)
        moving = np.zeros(0, bool)
        if len(points):
            nxt, good = _track(prev_gray, gray, points)
            flow = (nxt - points).reshape(-1, 2)
            moving = np.linalg.norm(flow, axis=1) >= MIN_SPEED
            still = np.where(moving, 0, still + 1)
            # Background points are kept only long enough to pick up objects entering them
            good &= still < STATIC_PATIENCE
            points, flow, moving, still = nxt[good], flow[good], moving[good], still[good]
            trails = [(t + [p])[-TRAIL_LENGTH:]
                      for t, p, g in zip(trails, nxt.reshape(-1, 2), good) if g]

        frame_measurements = []
        if moving.sum() >= MIN_MOVING_POINTS:
            frame_measurements = _measure(frame_idx, points.reshape(-1, 2)[moving], flow[moving])
        measurements += frame_measurements

        _draw(frame, trails, moving, flow, frame_measurements)
        writer.write(frame)

        # Moving regions are seeded every frame: smooth objects rarely win the global corner budget
        new = _detect(gray, points, MOTION_FEATURE_PARAMS, _motion_mask(prev_gray, gray))
        if frame_idx % REDETECT_EVERY == 0:
            new = np.concatenate([new, _detect(gray, np.concatenate([points, new]), FEATURE_PARAMS)])
        if len(new):
            points = np.concatenate([points, new])
            trails += [[p] for p in new.reshape(-1, 2)]
            still = np.concatenate([still, np.zeros(len(new), int)])
        prev_gray = gray

    cap.release()
    writer.release()
    _write_csv(output_dir / f'{video_path.stem}_of.csv', measurements, frame_idx + 1)
    return measurements
