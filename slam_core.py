"""
slam_core.py
============
Lightweight SLAM (self-localization) core.

Algorithm:
  1. Prediction
     All the robot knows is the 'commanded odometry'. We integrate it into the
     estimated pose. Because the true motion carries slip error, drift
     accumulates if left uncorrected.

  2. Correction (Correlative Scan Matching on a Likelihood Field)
     From the known wall shape we precompute a 'likelihood field' (a map of the
     distance from each cell to the nearest wall). We transform the LiDAR point
     cloud into world coordinates for a candidate pose and score how well the
     points match the walls. We do a local search over (dx, dy, dtheta) and take
     the correction with the highest score.

  This is a lightweight approximation of ICP / correlative scan matching. It
  needs no matrix library and only scans an integer grid, so CPU load stays low.
"""

import math
from collections import deque

from room_simulator import WALL, FREE


class SlamCore:
    def __init__(self, known_grid, init_pose):
        """
        known_grid : the known wall map (a 2D list with only outer/fixed walls
                     marked WALL). Movable objects such as furniture are not
                     included = the prior map.
        init_pose  : initial estimated pose (x, y, theta). Assumed known at
                     start-up.
        """
        self.size = len(known_grid)
        self.known_grid = known_grid
        self.est_pose = tuple(init_pose)
        self.likelihood = self._build_likelihood_field(known_grid)
        self.trajectory = [self.est_pose]
        # estimation-error log
        self.last_correction = (0.0, 0.0, 0.0)

    # ------------------------------------------------------------------
    # Precompute the likelihood field (distance transform to walls) with BFS
    # ------------------------------------------------------------------
    def _build_likelihood_field(self, grid):
        n = self.size
        INF = 10 ** 9
        dist = [[INF] * n for _ in range(n)]
        q = deque()
        for y in range(n):
            for x in range(n):
                if grid[y][x] == WALL:
                    dist[y][x] = 0
                    q.append((x, y))
        # Distance transform via 4-neighbor BFS (integers only)
        while q:
            x, y = q.popleft()
            d = dist[y][x]
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = x + dx, y + dy
                if 0 <= nx < n and 0 <= ny < n and dist[ny][nx] > d + 1:
                    dist[ny][nx] = d + 1
                    q.append((nx, ny))
        return dist

    def _wall_distance(self, ix, iy):
        if 0 <= ix < self.size and 0 <= iy < self.size:
            return self.likelihood[iy][ix]
        return 0  # out of bounds is treated as a wall

    def _raycast_known(self, px, py, ang, max_range=12.0, step=0.1):
        """Cast one ray on the known wall map and return the expected range to a wall."""
        dx, dy = math.cos(ang), math.sin(ang)
        d = 0.0
        n = self.size
        while d < max_range:
            d += step
            ix = int(math.floor(px + dx * d))
            iy = int(math.floor(py + dy * d))
            if ix < 0 or iy < 0 or ix >= n or iy >= n:
                return d
            if self.known_grid[iy][ix] == WALL:
                return d
        return max_range

    # ------------------------------------------------------------------
    # Scan match score (higher is better = smaller residual)
    # ------------------------------------------------------------------
    def _scan_score(self, pose, scan, beam_stride=4):
        """
        Per bearing, take the residual between the 'expected range obtained by
        raycasting from the candidate pose against the known wall map' and the
        'LiDAR measured range', and score it robustly.
        - A measurement much shorter than expected = a beam that hit
          furniture / a dynamic obstacle -> excluded.
        - The residual is evaluated with a capped square to stay robust to
          outliers.
        Because it is evaluated per bearing, the 'corner-snapping' bias does
        not arise.
        """
        px, py, pth = pose
        score = 0.0
        cap = 1.0  # residual cap (cells)
        for i in range(0, len(scan), beam_stride):
            beam = scan[i]
            if not beam['hit']:
                continue
            ang = pth + beam['angle']
            expected = self._raycast_known(px, py, ang)
            measured = beam['dist']
            # A beam that hit nearer than the known wall (= furniture / dynamic
            # obstacle) is not used for alignment
            if measured < expected - 1.0:
                continue
            r = measured - expected
            if r > cap:
                r = cap
            elif r < -cap:
                r = -cap
            score += (cap * cap - r * r)   # max at residual 0, zero for outliers
        return score

    # ------------------------------------------------------------------
    # One-step update: prediction + scan matching correction
    # ------------------------------------------------------------------
    def update(self, odom_u, scan):
        """
        odom_u : (forward, dtheta) — the motion the robot commanded (= thinks it
                 measured).
        scan   : the LiDAR point cloud obtained from the current true pose.
        Returns: the corrected estimated pose (x, y, theta).
        """
        forward, dtheta = odom_u
        x, y, th = self.est_pose

        # --- 1) prediction step ---
        th_pred = th + dtheta
        x_pred = x + math.cos(th_pred) * forward
        y_pred = y + math.sin(th_pred) * forward
        predicted = (x_pred, y_pred, th_pred)

        # --- 2) correct with correlative scan matching (coarse-to-fine search) ---
        # The coarse search grabs the global correction direction; the fine
        # search improves accuracy. Keeping the candidate count low minimizes
        # CPU load.
        best_pose = (x_pred, y_pred, th_pred)
        best_score = self._scan_score(best_pose, scan)

        # Stage 1: coarse search (+/-0.6 cell, +/-0.1 rad)
        for dxx in (-0.6, 0.0, 0.6):
            for dyy in (-0.6, 0.0, 0.6):
                for dth in (-0.10, 0.0, 0.10):
                    cand = (x_pred + dxx, y_pred + dyy, th_pred + dth)
                    s = self._scan_score(cand, scan)
                    if s > best_score:
                        best_score, best_pose = s, cand

        # Stage 2: fine search (+/-0.2 cell, +/-0.05 rad around the coarse best)
        cx, cy, cth = best_pose
        for dxx in (-0.2, 0.0, 0.2):
            for dyy in (-0.2, 0.0, 0.2):
                for dth in (-0.05, 0.0, 0.05):
                    cand = (cx + dxx, cy + dyy, cth + dth)
                    s = self._scan_score(cand, scan)
                    if s > best_score:
                        best_score, best_pose = s, cand

        self.last_correction = (
            best_pose[0] - x_pred,
            best_pose[1] - y_pred,
            best_pose[2] - th_pred,
        )
        self.est_pose = best_pose
        self.trajectory.append(best_pose)
        return best_pose

    def pose_error(self, true_pose):
        """Euclidean distance (cells) between estimated and true pose. For evaluation."""
        return math.hypot(self.est_pose[0] - true_pose[0],
                          self.est_pose[1] - true_pose[1])
