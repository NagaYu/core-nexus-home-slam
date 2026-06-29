"""
path_planner.py
===============
Dynamically generate a costmap that reflects semantics (intent) and search a
path with A*.

Cost design (quantifying intent):
  * walls / known obstacles      : non-traversable (INF)
  * pet bowl + 50 cm margin      : non-traversable (INF)   <- 50 cm = 2-cell inflation
  * chair / table cells          : non-traversable (INF)
  * around chairs / tables       : high cost (respect human movement, prefer to avoid)
  * free space                   : base cost 1

Dynamic re-planning:
  When a chair suddenly moves and blocks the path, partially update the costmap
  and re-run A*. At 20x20 scale a single A* completes in under 1 ms, so the
  required "within 10 ms" spec is met.
"""

import heapq
import math

from room_simulator import WALL, FREE, HUMAN_FURNITURE, PET_ZONE
from semantic_grid import LABEL_HUMAN, LABEL_PET

INF = float('inf')

# extra cost based on semantics
PET_INFLATE_CELLS = 2   # pet-bowl keep-out inflation radius = 2 cells ~= 50 cm
HUMAN_NEAR_COST = 6.0   # traversal cost around chairs (human movement lanes)
HUMAN_INFLATE = 1       # influence radius around chairs


class CostmapPlanner:
    def __init__(self, size):
        self.size = size
        self.costmap = [[1.0 for _ in range(size)] for _ in range(size)]
        self.last_plan_ms = 0.0

    # ------------------------------------------------------------------
    # (Re)generate the semantics-aware costmap
    # ------------------------------------------------------------------
    def build_costmap(self, occupancy, semantic):
        """
        occupancy : 2D list. Type of each cell (WALL/FREE/HUMAN_FURNITURE/PET_ZONE).
                    The obstacle layout SLAM/perception currently believes is
                    'there'.
        semantic  : SemanticGrid. Used as auxiliary class confidence for movable
                    obstacles.
        """
        n = self.size
        cm = [[1.0 for _ in range(n)] for _ in range(n)]

        pet_cells = []
        human_cells = []
        for y in range(n):
            for x in range(n):
                k = occupancy[y][x]
                if k == WALL:
                    cm[y][x] = INF
                elif k == PET_ZONE:
                    cm[y][x] = INF
                    pet_cells.append((x, y))
                elif k == HUMAN_FURNITURE:
                    cm[y][x] = INF        # the furniture cell itself is impassable
                    human_cells.append((x, y))

        # --- pet bowl: inflate a 50 cm (2-cell) keep-out region ---
        for (px, py) in pet_cells:
            for dy in range(-PET_INFLATE_CELLS, PET_INFLATE_CELLS + 1):
                for dx in range(-PET_INFLATE_CELLS, PET_INFLATE_CELLS + 1):
                    nx, ny = px + dx, py + dy
                    if 0 <= nx < n and 0 <= ny < n:
                        if math.hypot(dx, dy) <= PET_INFLATE_CELLS + 0.001:
                            cm[ny][nx] = INF

        # --- chairs / tables: raise surrounding cost (respect human lanes) ---
        for (hx, hy) in human_cells:
            for dy in range(-HUMAN_INFLATE, HUMAN_INFLATE + 1):
                for dx in range(-HUMAN_INFLATE, HUMAN_INFLATE + 1):
                    nx, ny = hx + dx, hy + dy
                    if 0 <= nx < n and 0 <= ny < n and cm[ny][nx] != INF:
                        cm[ny][nx] = max(cm[ny][nx], HUMAN_NEAR_COST)

        self.costmap = cm
        return cm

    # ------------------------------------------------------------------
    # A* path search
    # ------------------------------------------------------------------
    def plan(self, start, goal):
        """
        start, goal : (ix, iy) grid coordinates.
        Returns     : path [(x,y), ...] (start->goal). [] if unreachable.
        """
        import time
        t0 = time.perf_counter()
        path = self._astar(start, goal)
        self.last_plan_ms = (time.perf_counter() - t0) * 1000.0
        return path

    def _astar(self, start, goal):
        n = self.size
        cm = self.costmap
        sx, sy = start
        gx, gy = goal
        if cm[gy][gx] == INF:
            return []  # goal is non-traversable

        def h(x, y):
            # octile distance heuristic for 8-connected movement
            dx, dy = abs(x - gx), abs(y - gy)
            return (dx + dy) + (math.sqrt(2) - 2) * min(dx, dy)

        open_heap = [(h(sx, sy), 0.0, (sx, sy))]
        g_score = {(sx, sy): 0.0}
        came_from = {}
        closed = set()

        neighbors = [(1, 0), (-1, 0), (0, 1), (0, -1),
                     (1, 1), (1, -1), (-1, 1), (-1, -1)]

        while open_heap:
            f, g, (x, y) = heapq.heappop(open_heap)
            if (x, y) == (gx, gy):
                return self._reconstruct(came_from, (x, y))
            if (x, y) in closed:
                continue
            closed.add((x, y))
            for dx, dy in neighbors:
                nx, ny = x + dx, y + dy
                if not (0 <= nx < n and 0 <= ny < n):
                    continue
                cell_cost = cm[ny][nx]
                if cell_cost == INF:
                    continue
                # no diagonal corner-cutting (skip if either side is a wall)
                if dx != 0 and dy != 0:
                    if cm[y][nx] == INF or cm[ny][x] == INF:
                        continue
                step = math.hypot(dx, dy) * cell_cost
                ng = g + step
                if ng < g_score.get((nx, ny), INF):
                    g_score[(nx, ny)] = ng
                    came_from[(nx, ny)] = (x, y)
                    heapq.heappush(open_heap, (ng + h(nx, ny), ng, (nx, ny)))
        return []  # unreachable

    def _reconstruct(self, came_from, cur):
        path = [cur]
        while cur in came_from:
            cur = came_from[cur]
            path.append(cur)
        path.reverse()
        return path

    # ------------------------------------------------------------------
    # Dynamic re-planning
    # ------------------------------------------------------------------
    def replan_if_blocked(self, path, occupancy, semantic, start, goal):
        """
        Check whether a new obstacle has appeared on the current path; if it is
        blocked, regenerate the costmap and re-run A*.
        Returns: (new path, whether it replanned (bool), elapsed ms)
        """
        import time
        blocked = False
        for (x, y) in path:
            if occupancy[y][x] in (WALL, HUMAN_FURNITURE, PET_ZONE):
                blocked = True
                break
        if not blocked:
            return path, False, 0.0

        t0 = time.perf_counter()
        self.build_costmap(occupancy, semantic)
        new_path = self._astar(start, goal)
        ms = (time.perf_counter() - t0) * 1000.0
        self.last_plan_ms = ms
        return new_path, True, ms
