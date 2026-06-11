"""Track geometry: sampled centerlines with curvature, track-frame errors.

A track is a polyline centerline sampled every DS meters, with heading psi,
curvature kappa and arc length s at each sample. Two generators:

  Track.circle(radius)        closed circular track (constant curvature)
  Track.random_track(rng)     open track whose curvature is a piecewise-linear
                              random profile bounded by KAPPA_LO <= |k| <= KAPPA_HI

Conventions: e_y > 0 means the car is LEFT of the centerline (w.r.t. track
direction); positive curvature turns left.
"""

import numpy as np

DS = 0.5            # centerline sampling step [m]
HALF_WIDTH = 4.0    # track half-width [m]

# random-track curvature bounds: corner radii between 22 m and 60 m,
# tight enough to demand drifting at speed, wide enough to be drivable
KAPPA_LO = 1.0 / 60.0
KAPPA_HI = 1.0 / 22.0
SEG_LEN = (30.0, 80.0)   # arc-length range between curvature knots [m]


class Track:
    def __init__(self, xy, psi, kappa, closed):
        self.xy = xy                      # (N, 2) centerline points
        self.psi = psi                    # (N,) tangent heading
        self.kappa = kappa                # (N,) curvature
        self.closed = closed
        self.n = len(xy)
        self.s = np.arange(self.n) * DS
        self.length = self.n * DS
        self.half_width = HALF_WIDTH
        normal = np.stack([-np.sin(psi), np.cos(psi)], axis=1)  # left normal
        self.left = xy + HALF_WIDTH * normal    # boundary polylines (rendering)
        self.right = xy - HALF_WIDTH * normal
        self._normal = normal

    # ------------------------------------------------------------- generators
    @classmethod
    def circle(cls, radius=30.0):
        n = int(round(2 * np.pi * radius / DS))
        th = np.arange(n) * 2 * np.pi / n
        xy = radius * np.stack([np.cos(th), np.sin(th)], axis=1)
        psi = th + np.pi / 2.0            # CCW travel
        kappa = np.full(n, 1.0 / radius)
        return cls(xy, psi, kappa, closed=True)

    @classmethod
    def random_track(cls, rng, length=500.0):
        # curvature knots: random spacing, random sign, |kappa| in [LO, HI];
        # linear interpolation between knots gives a smooth bounded profile
        knot_s, knot_k = [0.0], [0.0]      # start straight
        while knot_s[-1] < length:
            knot_s.append(knot_s[-1] + rng.uniform(*SEG_LEN))
            sign = rng.choice([-1.0, 1.0])
            knot_k.append(sign * rng.uniform(KAPPA_LO, KAPPA_HI))
        s = np.arange(int(length / DS)) * DS
        kappa = np.interp(s, knot_s, knot_k)
        psi = np.concatenate([[0.0], np.cumsum(kappa[:-1]) * DS])
        x = np.concatenate([[0.0], np.cumsum(np.cos(psi[:-1])) * DS])
        y = np.concatenate([[0.0], np.cumsum(np.sin(psi[:-1])) * DS])
        return cls(np.stack([x, y], axis=1), psi, kappa, closed=False)

    # ---------------------------------------------------------------- queries
    def nearest(self, x, y, hint):
        """Index of the nearest centerline sample, searched near `hint`."""
        w = 80  # search window: +-40 m, far more than one step of car motion
        if self.closed:
            idx = (np.arange(hint - w, hint + w)) % self.n
        else:
            idx = np.arange(max(hint - w, 0), min(hint + w, self.n))
        d2 = np.sum((self.xy[idx] - [x, y]) ** 2, axis=1)
        return int(idx[np.argmin(d2)])

    def frame(self, x, y, psi, hint):
        """Track-frame errors and curvature preview at the car position.

        Returns (e_y, e_psi, kappa_preview[3], idx). Preview samples the
        curvature 0, 10 and 25 m ahead along the centerline.
        """
        i = self.nearest(x, y, hint)
        e_y = float(np.dot([x, y] - self.xy[i], self._normal[i]))
        e_psi = (psi - self.psi[i] + np.pi) % (2 * np.pi) - np.pi
        prev = []
        for d in (0.0, 10.0, 25.0):
            j = i + int(d / DS)
            j = j % self.n if self.closed else min(j, self.n - 1)
            prev.append(self.kappa[j])
        return e_y, float(e_psi), np.array(prev), i

    def at_end(self, idx):
        return (not self.closed) and idx >= self.n - 4
