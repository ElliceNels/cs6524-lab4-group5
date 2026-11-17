"""Kalman tracking of the optical-flow measurements through occlusions."""
import csv
from dataclasses import dataclass, field

import cv2
import numpy as np

from mypylibs.kalman import KalmanFilter, process_noise

SIGMA_A = 2.0
POS_STD_FLOOR = 2.0
VEL_STD_FLOOR = 0.3
# Chi-square 99.9% and 99.99% quantiles for 4 degrees of freedom
GATE = 18.47
REACQUIRE_GATE = 23.51
CONFIRM_HITS = 3
MAX_CANDIDATE_GAP = 2
MIN_OCCLUSION = 5
EXTENT_HISTORY = 25
VELOCITY_EVERY = 5
VELOCITY_SCALE = 4

MEASURED = (255, 255, 255)
REJECTED = (140, 140, 140)
CORRECTED = (0, 210, 0)
PREDICTED = (0, 140, 255)
GHOST = (255, 0, 255)
REACQUIRED = (0, 0, 255)
VELOCITY = (0, 255, 255)


@dataclass(eq=False)
class Observation:
    measurement: object
    accepted: bool = False
    d2: float = None


@dataclass
class Record:
    frame: int
    track_id: int
    status: str
    x: np.ndarray
    P: np.ndarray
    K: np.ndarray
    observation: Observation = None


@dataclass
class Ghost:
    start: int
    kf: KalmanFilter
    points: list
    alive: bool = True


@dataclass(eq=False)
class Candidate:
    """Tentative track built from measurements the main track did not take."""
    kf: KalmanFilter
    start: int
    first: object
    # Main filter's prior at `start`, coasted alongside so it can become the ghost
    fork: KalmanFilter = None
    fork_d2: float = None
    fork_points: list = field(default_factory=list)
    records: list = field(default_factory=list)
    spreads: list = field(default_factory=list)
    hits: int = 1
    misses: int = 0


@dataclass
class Reacquisition:
    frame: int
    gap: int
    predicted: np.ndarray
    measured: np.ndarray
    d2: float

    @property
    def error(self):
        return float(np.linalg.norm(self.measured - self.predicted))


def _extent(spreads):
    return np.median(spreads[-EXTENT_HISTORY:], axis=0) if spreads else None


def _noise(measurement, extent=None):
    position_std = np.maximum(measurement.position_std, POS_STD_FLOOR)
    if extent is not None:
        # A partially visible object has a tight point spread but a centroid biased towards the visible part
        position_std = np.maximum(position_std, extent)
    std = np.concatenate([position_std, np.maximum(measurement.velocity_std, VEL_STD_FLOOR)])
    return np.diag(std ** 2)


def _observation(measurement):
    return np.concatenate([measurement.position, measurement.velocity])


def _record(frame_idx, track_id, status, kf, observation=None):
    return Record(frame_idx, track_id, status, kf.x.copy(), kf.P.copy(), kf.K.copy(), observation)


class Tracker:
    def __init__(self, frame_size, sigma_a=SIGMA_A):
        self.frame_size = frame_size
        self.Q = process_noise(sigma_a)
        self.kf = None
        self.candidates = []
        self.track_id = -1
        self.missed = 0
        self.last_corrected = -1
        self.spreads = []
        self.records = []
        self.ghosts = []
        self.reacquisitions = []
        self.observations = []

    def _inside(self, position):
        w, h = self.frame_size
        return 0 <= position[0] < w and 0 <= position[1] < h

    def _advance_ghosts(self, frame_idx):
        for ghost in self.ghosts:
            if ghost.alive:
                ghost.kf.predict()
                ghost.alive = self._inside(ghost.kf.x[:2])
                if ghost.alive:
                    ghost.points.append((frame_idx, ghost.kf.x.copy()))

    def _spawn(self, frame_idx, observation):
        m = observation.measurement
        # Initial uncertainty is the spread of the optical-flow points behind the first measurement
        c = Candidate(KalmanFilter(_observation(m), _noise(m), self.Q), frame_idx, m)
        if self.kf is not None:
            c.fork = self.kf.copy()
            c.fork_d2 = c.fork.mahalanobis2(_observation(m), _noise(m, _extent(self.spreads)))
            c.fork_points.append((frame_idx, c.fork.x.copy()))
        c.records.append(_record(frame_idx, -1, 'corrected', c.kf, observation))
        c.spreads.append(m.position_std)
        self.candidates.append(c)

    def _update_candidates(self, frame_idx, free):
        for c in self.candidates:
            c.kf.predict()
            if c.fork is not None:
                c.fork.predict()
                c.fork_points.append((frame_idx, c.fork.x.copy()))
        pairs = []
        for i, c in enumerate(self.candidates):
            for j, o in enumerate(free):
                z, R = _observation(o.measurement), _noise(o.measurement, _extent(c.spreads))
                d2 = c.kf.mahalanobis2(z, R)
                if d2 < GATE:
                    pairs.append((d2, i, j))
        taken_c, taken_o = set(), set()
        for _, i, j in sorted(pairs):
            if i in taken_c or j in taken_o:
                continue
            taken_c.add(i)
            taken_o.add(j)
            c, o = self.candidates[i], free[j]
            c.kf.update(_observation(o.measurement), _noise(o.measurement, _extent(c.spreads)))
            c.records.append(_record(frame_idx, -1, 'corrected', c.kf, o))
            c.spreads.append(o.measurement.position_std)
            c.hits += 1
            c.misses = 0
        for i, c in enumerate(self.candidates):
            if i not in taken_c:
                c.misses += 1
                c.records.append(_record(frame_idx, -1, 'predicted', c.kf))
        self.candidates = [c for c in self.candidates if c.misses <= MAX_CANDIDATE_GAP]
        for j, o in enumerate(free):
            if j not in taken_o:
                self._spawn(frame_idx, o)

    def _promote(self, frame_idx):
        confirmed = [c for c in self.candidates if c.hits >= CONFIRM_HITS]
        if not confirmed:
            return
        if self.kf is None:
            c = max(confirmed, key=lambda c: np.mean([r.observation.measurement.n_points
                                                      for r in c.records if r.observation]))
            self.track_id += 1
            self.spreads = []
        elif self.missed > 0:
            plausible = [c for c in confirmed if c.fork is not None and c.fork_d2 < REACQUIRE_GATE]
            if not plausible:
                return
            c = min(plausible, key=lambda c: c.fork_d2)
            gap = c.start - self.last_corrected - 1
            if gap >= MIN_OCCLUSION:
                self.reacquisitions.append(Reacquisition(
                    c.start, gap, c.fork_points[0][1][:2], c.first.position.copy(), c.fork_d2))
                self.ghosts.append(Ghost(c.start, c.fork, c.fork_points))
            self.records = [r for r in self.records if r.frame < c.start]
        else:
            return
        self.records += [Record(r.frame, self.track_id, r.status, r.x, r.P, r.K, r.observation)
                         for r in c.records]
        for r in c.records:
            if r.observation is not None:
                r.observation.accepted = True
        self.spreads += c.spreads
        self.kf = c.kf
        self.missed = 0
        self.last_corrected = frame_idx
        self.candidates.remove(c)

    def step(self, frame_idx, measurements):
        self._advance_ghosts(frame_idx)
        observations = [Observation(m) for m in measurements]
        self.observations += observations
        free = list(observations)

        status, used = None, None
        if self.kf is not None:
            self.kf.predict()
            extent = _extent(self.spreads)
            for o in observations:
                o.d2 = self.kf.mahalanobis2(_observation(o.measurement), _noise(o.measurement, extent))
            best = min(observations, key=lambda o: o.d2, default=None)
            # After an occlusion the gate is wide enough to admit clutter, so reacquisition needs confirmation
            if best is not None and self.missed < MIN_OCCLUSION and best.d2 < GATE:
                self.kf.update(_observation(best.measurement), _noise(best.measurement, extent))
                self.spreads.append(best.measurement.position_std)
                best.accepted = True
                free.remove(best)
                self.missed = 0
                self.last_corrected = frame_idx
                status, used = 'corrected', best
            else:
                self.missed += 1
                status = 'predicted'
            self.records.append(_record(frame_idx, self.track_id, status, self.kf, used))

        self._update_candidates(frame_idx, free)
        self._promote(frame_idx)
        if self.kf is not None and not self._inside(self.kf.x[:2]):
            self.kf = None
        current = self.records[-1] if self.records and self.records[-1].frame == frame_idx else None
        return current.status if current is not None else None


def _pt(p):
    return tuple(int(round(v)) for v in p[:2])


def _draw_path(frame, records, thickness):
    for a, b in zip(records, records[1:]):
        if a.track_id == b.track_id and b.frame - a.frame == 1:
            colour = CORRECTED if b.status == 'corrected' else PREDICTED
            cv2.line(frame, _pt(a.x), _pt(b.x), colour, thickness, cv2.LINE_AA)
    for r in records:
        if r.frame % VELOCITY_EVERY == 0:
            cv2.arrowedLine(frame, _pt(r.x), _pt(r.x[:2] + VELOCITY_SCALE * r.x[2:]),
                            VELOCITY, max(1, thickness - 1), cv2.LINE_AA, tipLength=0.25)


def _draw_ellipse(frame, x, P, colour, thickness):
    # 2-sigma contour of the position covariance
    values, vectors = np.linalg.eigh(P[:2, :2])
    axes = tuple(int(round(2 * np.sqrt(max(v, 0)))) for v in values)
    angle = np.degrees(np.arctan2(vectors[1, 0], vectors[0, 0]))
    cv2.ellipse(frame, _pt(x), axes, angle, 0, 360, colour, thickness, cv2.LINE_AA)


def _text_box(frame, lines, origin, scale, anchor_bottom=False):
    font, thickness = cv2.FONT_HERSHEY_SIMPLEX, max(1, int(round(2 * scale)))
    sizes = [cv2.getTextSize(text, font, scale, thickness)[0] for text, _ in lines]
    line_h = int(max(h for _, h in sizes) * 1.8)
    width = max(w for w, _ in sizes) + 20
    height = line_h * len(lines) + 10
    x0, y0 = origin
    if anchor_bottom:
        y0 -= height
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + width, y0 + height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    for i, (text, colour) in enumerate(lines):
        cv2.putText(frame, text, (x0 + 10, y0 + (i + 1) * line_h), font, scale, colour,
                    thickness, cv2.LINE_AA)


def _draw(frame, tracker, frame_idx, status, thickness, scale):
    for observation in tracker.observations:
        p = _pt(observation.measurement.position)
        if observation.accepted:
            cv2.circle(frame, p, thickness + 1, MEASURED, -1, cv2.LINE_AA)
        elif observation.measurement.frame == frame_idx:
            cv2.drawMarker(frame, p, REJECTED, cv2.MARKER_TILTED_CROSS, 6 * thickness, thickness)

    for ghost in tracker.ghosts:
        path = np.int32([g[1][:2] for g in ghost.points])
        cv2.polylines(frame, [path], False, GHOST, thickness, cv2.LINE_AA)
        if ghost.alive:
            cv2.circle(frame, _pt(ghost.points[-1][1]), 4 * thickness, GHOST, -1, cv2.LINE_AA)

    _draw_path(frame, tracker.records, thickness)

    for event in tracker.reacquisitions:
        cv2.line(frame, _pt(event.predicted), _pt(event.measured), REACQUIRED, thickness, cv2.LINE_AA)
        cv2.drawMarker(frame, _pt(event.predicted), REACQUIRED, cv2.MARKER_CROSS, 8 * thickness, thickness)

    lines = [(f'frame {frame_idx}', MEASURED)]
    current = tracker.records[-1] if tracker.records and tracker.records[-1].frame == frame_idx else None
    if current is not None:
        colour = CORRECTED if status == 'corrected' else PREDICTED
        _draw_ellipse(frame, current.x, current.P, colour, thickness)
        cv2.circle(frame, _pt(current.x), 4 * thickness, colour, -1, cv2.LINE_AA)
        sx, sy = np.sqrt(np.diag(current.P)[:2])
        lines += [
            (f'{status}: pos ({current.x[0]:.0f}, {current.x[1]:.0f})  '
             f'vel ({current.x[2]:+.1f}, {current.x[3]:+.1f}) px/frame', colour),
            (f'sigma pos ({sx:.1f}, {sy:.1f}) px   K diag {np.round(np.diag(current.K), 2)}', MEASURED),
        ]
    else:
        lines.append(('no track', REJECTED))
    for event in tracker.reacquisitions:
        lines.append((f'reacquired @{event.frame} after {event.gap} frames: '
                      f'error {event.error:.0f} px (d2 {event.d2:.1f})', REACQUIRED))
    for ghost in tracker.ghosts:
        if ghost.alive and current is not None:
            gap = np.linalg.norm(ghost.points[-1][1][:2] - current.x[:2])
            lines.append((f'ghost from @{ghost.start} is {gap:.0f} px from track', GHOST))
    _text_box(frame, lines, (10, 10), scale)

    legend = [('measured (OF)', MEASURED), ('rejected by gate', REJECTED),
              ('corrected', CORRECTED), ('predicted only', PREDICTED),
              ('ghost: if never reacquired', GHOST), ('prediction vs reacquisition', REACQUIRED),
              (f'velocity x{VELOCITY_SCALE}', VELOCITY)]
    _text_box(frame, legend, (10, frame.shape[0] - 10), scale, anchor_bottom=True)


def _write_csv(path, tracker):
    ghosts = {}
    for ghost in tracker.ghosts:
        for frame_idx, x in ghost.points:
            ghosts.setdefault(frame_idx, x)
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['frame', 'track', 'status', 'x', 'y', 'vx', 'vy', 'sx', 'sy',
                         'mx', 'my', 'mvx', 'mvy', 'd2', 'ghost_x', 'ghost_y'])
        for r in tracker.records:
            o, g = r.observation, ghosts.get(r.frame)
            measured = ['', '', '', '', '']
            if o is not None:
                measured = [*np.round(o.measurement.position, 2), *np.round(o.measurement.velocity, 3),
                            '' if o.d2 is None else round(o.d2, 2)]
            writer.writerow([r.frame, r.track_id, r.status, *np.round(r.x, 2),
                             *np.round(np.sqrt(np.diag(r.P)[:2]), 2), *measured,
                             *(np.round(g[:2], 2) if g is not None else ['', ''])])


def track_object(video_path, measurements, output_dir, sigma_a=SIGMA_A):
    """Kalman-track the measurements, write `XX_tr.mp4` and `XX_tr.csv`, return the tracker."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    writer = cv2.VideoWriter(str(output_dir / f'{video_path.stem}_tr.mp4'),
                             cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    thickness = max(1, round(min(size) / 360))
    scale = min(size) / 1200

    by_frame = {}
    for m in measurements:
        by_frame.setdefault(m.frame, []).append(m)
    tracker = Tracker(size, sigma_a)
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        status = tracker.step(frame_idx, by_frame.get(frame_idx, []))
        _draw(frame, tracker, frame_idx, status, thickness, scale)
        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    _write_csv(output_dir / f'{video_path.stem}_tr.csv', tracker)
    return tracker
