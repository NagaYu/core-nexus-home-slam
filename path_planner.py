"""
path_planner.py
===============
セマンティクス(意図)を反映したコストマップを動的生成し、A*で経路探索する。

コスト設計 (意図の数値化):
  * 壁 / 既知障害物          : 進入不可 (INF)
  * ペットの皿 + 周囲50cm    : 進入不可 (INF)   ← 50cm = 2セル膨張
  * 椅子・テーブルのセル自体 : 進入不可 (INF)
  * 椅子・テーブルの周囲     : 高コスト(人間の動線を尊重し避けたい)
  * 自由空間                 : 基本コスト 1

動的リプランニング:
  椅子が突然動いて経路を塞いだ場合、コストマップを部分更新して A* を
  再実行する。20x20 規模では 1回の A* が 1ms 未満で完了するため、
  要求仕様「10ms 以内」を満たす。
"""

import heapq
import math

from room_simulator import WALL, FREE, HUMAN_FURNITURE, PET_ZONE
from semantic_grid import LABEL_HUMAN, LABEL_PET

INF = float('inf')

# 意味論に基づく追加コスト
PET_INFLATE_CELLS = 2   # ペット皿の進入禁止膨張半径 = 2セル ≈ 50cm
HUMAN_NEAR_COST = 6.0   # 椅子周辺(人間の動線)の通過コスト
HUMAN_INFLATE = 1       # 椅子周辺の影響半径


class CostmapPlanner:
    def __init__(self, size):
        self.size = size
        self.costmap = [[1.0 for _ in range(size)] for _ in range(size)]
        self.last_plan_ms = 0.0

    # ------------------------------------------------------------------
    # セマンティクス反映コストマップの(再)生成
    # ------------------------------------------------------------------
    def build_costmap(self, occupancy, semantic):
        """
        occupancy : 2D リスト。各セルの種別 (WALL/FREE/HUMAN_FURNITURE/PET_ZONE)。
                    SLAM/知覚が現時点で『そこにある』と信じる障害物配置。
        semantic  : SemanticGrid。可動障害物のクラス確信度を補助的に使用。
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
                    cm[y][x] = INF        # 家具セル自体は通れない
                    human_cells.append((x, y))

        # --- ペット皿: 周囲50cm(2セル)を進入禁止に膨張 ---
        for (px, py) in pet_cells:
            for dy in range(-PET_INFLATE_CELLS, PET_INFLATE_CELLS + 1):
                for dx in range(-PET_INFLATE_CELLS, PET_INFLATE_CELLS + 1):
                    nx, ny = px + dx, py + dy
                    if 0 <= nx < n and 0 <= ny < n:
                        if math.hypot(dx, dy) <= PET_INFLATE_CELLS + 0.001:
                            cm[ny][nx] = INF

        # --- 椅子・テーブル: 周囲を高コスト化(人間の動線を尊重) ---
        for (hx, hy) in human_cells:
            for dy in range(-HUMAN_INFLATE, HUMAN_INFLATE + 1):
                for dx in range(-HUMAN_INFLATE, HUMAN_INFLATE + 1):
                    nx, ny = hx + dx, hy + dy
                    if 0 <= nx < n and 0 <= ny < n and cm[ny][nx] != INF:
                        cm[ny][nx] = max(cm[ny][nx], HUMAN_NEAR_COST)

        self.costmap = cm
        return cm

    # ------------------------------------------------------------------
    # A* 経路探索
    # ------------------------------------------------------------------
    def plan(self, start, goal):
        """
        start, goal : (ix, iy) グリッド座標。
        戻り値      : 経路 [(x,y), ...] (start→goal)。到達不可なら []。
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
            return []  # 目的地が侵入不可

        def h(x, y):
            # 8近傍移動のためオクタイル距離をヒューリスティックに
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
                # 斜め移動は角抜け禁止 (両隣が壁ならスキップ)
                if dx != 0 and dy != 0:
                    if cm[y][nx] == INF or cm[ny][x] == INF:
                        continue
                step = math.hypot(dx, dy) * cell_cost
                ng = g + step
                if ng < g_score.get((nx, ny), INF):
                    g_score[(nx, ny)] = ng
                    came_from[(nx, ny)] = (x, y)
                    heapq.heappush(open_heap, (ng + h(nx, ny), ng, (nx, ny)))
        return []  # 到達不可

    def _reconstruct(self, came_from, cur):
        path = [cur]
        while cur in came_from:
            cur = came_from[cur]
            path.append(cur)
        path.reverse()
        return path

    # ------------------------------------------------------------------
    # 動的リプランニング
    # ------------------------------------------------------------------
    def replan_if_blocked(self, path, occupancy, semantic, start, goal):
        """
        現在の経路 path 上に新たな障害物が出現していないか確認し、
        塞がれていればコストマップを再生成して A* を再実行する。
        戻り値: (新しい経路, 再計画したか bool, 所要ms)
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
