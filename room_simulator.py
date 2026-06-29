"""
room_simulator.py
==================
Provides the "virtual indoor environment" and the "simple LiDAR" for a robot
moving through a home.

Design intent (embedded-oriented):
  * Pure Python, with no dependency on external frameworks (ROS / numpy, etc.).
  * 20x20 grid. One cell = 0.25 m (= 25 cm), representing a ~5 m x 5 m room.
  * Three obstacle types:
      WALL            : outer / fixed walls (the known shape SLAM relies on)
      HUMAN_FURNITURE : chairs, tables, etc. Used by people, so we want to
                        raise the surrounding cost.
      PET_ZONE        : pet bowls / toys. Strictly off-limits (a 50 cm margin
                        around them is forbidden too).

Coordinate system:
  grid[y][x]. x points east (right), y points south (down).
  Robot pose = (x, y, theta) is continuous [in cell units].
  theta is in radians. 0 = +x (east); it increases counter-clockwise (math
  convention).
"""

import math
import random

# ----- Cell types (semantic tags) -----
FREE = 0
WALL = 1
HUMAN_FURNITURE = 2   # chairs / tables
PET_ZONE = 3          # pet bowls / toys (hazard / off-limits)

GRID_SIZE = 20
CELL_METERS = 0.25    # physical size of one cell (m)


class RoomSimulator:
    """The 'ground truth' environment holding the true map and true robot pose."""

    def __init__(self, size=GRID_SIZE, seed=7):
        self.size = size
        self.rng = random.Random(seed)
        # grid[y][x]
        self.grid = [[FREE for _ in range(size)] for _ in range(size)]
        self._build_walls()
        self._place_furniture()
        # The robot's 'true' pose (SLAM does not know this directly)
        self.true_pose = (2.0, 2.0, 0.0)

    # ------------------------------------------------------------------
    # Map construction
    # ------------------------------------------------------------------
    def _build_walls(self):
        n = self.size
        for x in range(n):
            self.grid[0][x] = WALL
            self.grid[n - 1][x] = WALL
        for y in range(n):
            self.grid[y][0] = WALL
            self.grid[y][n - 1] = WALL
        # Interior partition (L-shaped) — gives scan matching some features
        for x in range(6, 13):
            self.grid[10][x] = WALL
        for y in range(10, 15):
            self.grid[y][12] = WALL

    def _place_block(self, x0, y0, w, h, kind):
        for yy in range(y0, y0 + h):
            for xx in range(x0, x0 + w):
                if 0 <= xx < self.size and 0 <= yy < self.size:
                    if self.grid[yy][xx] == FREE:
                        self.grid[yy][xx] = kind

    def _place_furniture(self):
        # Human furniture: a table (large) and chairs (small)
        self._place_block(4, 4, 2, 2, HUMAN_FURNITURE)    # table
        self._place_block(15, 3, 1, 1, HUMAN_FURNITURE)   # chair
        self._place_block(3, 15, 1, 1, HUMAN_FURNITURE)   # chair
        # Pet bowl (a small single-cell hazard) and a toy.
        # The bowl sits on the start->goal diagonal so the detour is clearly
        # observable.
        self._place_block(14, 14, 1, 1, PET_ZONE)         # pet bowl
        self._place_block(8, 6, 1, 1, PET_ZONE)           # toy

    # ------------------------------------------------------------------
    # Obstacle test
    # ------------------------------------------------------------------
    def is_obstacle(self, x, y):
        """Whether the continuous coordinate (x, y) is an obstacle cell."""
        ix, iy = int(math.floor(x)), int(math.floor(y))
        if ix < 0 or iy < 0 or ix >= self.size or iy >= self.size:
            return True
        return self.grid[iy][ix] != FREE

    def cell(self, ix, iy):
        if 0 <= ix < self.size and 0 <= iy < self.size:
            return self.grid[iy][ix]
        return WALL

    # ------------------------------------------------------------------
    # Simple LiDAR (360-degree point cloud)
    # ------------------------------------------------------------------
    def lidar_scan(self, pose=None, num_beams=72, max_range=12.0, step=0.05,
                   noise_std=0.01):
        """
        Cast 360 degrees of rays from the given pose and return the point cloud.

        Returns: a list of per-beam dicts
            { 'angle': beam relative angle (rad), 'dist': range (cells), 'hit': bool }
        num_beams=72 -> 5-degree steps. A light resolution even for embedded use.
        """
        if pose is None:
            pose = self.true_pose
        px, py, ptheta = pose
        scan = []
        for i in range(num_beams):
            rel = (2.0 * math.pi) * i / num_beams
            ang = ptheta + rel
            dx, dy = math.cos(ang), math.sin(ang)
            dist = 0.0
            hit = False
            while dist < max_range:
                dist += step
                sx = px + dx * dist
                sy = py + dy * dist
                if self.is_obstacle(sx, sy):
                    hit = True
                    break
            # Sensor noise
            meas = dist + self.rng.gauss(0.0, noise_std)
            scan.append({'angle': rel, 'dist': max(0.0, meas), 'hit': hit})
        return scan

    # ------------------------------------------------------------------
    # The robot's true motion (odometry + slip error)
    # ------------------------------------------------------------------
    def apply_true_motion(self, u, slip_std=0.04, rot_std=0.03):
        """
        Apply control input u=(forward, dtheta) to the true pose.
        Deliberately adds random slip (translation) and rotation error to
        reproduce odometry drift.
        Returns: the ideal odometry u the robot 'thinks it measured' (the
        commanded value, free of error).
        """
        forward, dtheta = u
        x, y, th = self.true_pose
        # --- the true motion carries error ---
        true_forward = forward + self.rng.gauss(0.0, slip_std)
        true_dtheta = dtheta + self.rng.gauss(0.0, rot_std)
        th2 = th + true_dtheta
        nx = x + math.cos(th2) * true_forward
        ny = y + math.sin(th2) * true_forward
        # If it would dig into a wall, cancel the translation (collision)
        if not self.is_obstacle(nx, ny):
            self.true_pose = (nx, ny, th2)
        else:
            self.true_pose = (x, y, th2)
        # All the robot knows is the 'commanded value (ideal)'
        return (forward, dtheta)

    # ------------------------------------------------------------------
    # Injecting a dynamic obstacle (e.g. a chair suddenly moves)
    # ------------------------------------------------------------------
    def set_cell(self, ix, iy, kind):
        if 0 <= ix < self.size and 0 <= iy < self.size:
            self.grid[iy][ix] = kind
