"""
slam_core.py
============
軽量SLAM(自己位置推定)コア。

アルゴリズム:
  1. 予測 (Prediction)
     ロボットが知るのは『指令オドメトリ』のみ。これを推定姿勢に積分する。
     真の運動には滑り誤差が乗っているため、放置するとドリフトが蓄積する。

  2. 補正 (Correlative Scan Matching on a Likelihood Field)
     既知の壁形状から『尤度場(各セルから最寄りの壁までの距離マップ)』を
     前計算しておく。LiDAR点群を候補姿勢で世界座標に変換し、点が壁に
     どれだけ一致するかをスコア化。(dx, dy, dtheta) を局所探索して
     スコア最大の補正量を採用する。

  これは ICP / 相関型スキャンマッチングの軽量近似であり、
  行列演算ライブラリ不要・整数グリッド走査のみで CPU 負荷を小さく抑えられる。
"""

import math
from collections import deque

from room_simulator import WALL, FREE


class SlamCore:
    def __init__(self, known_grid, init_pose):
        """
        known_grid : 既知の壁マップ (外壁・固定壁のみを WALL とした 2D リスト)。
                     家具など可動物は含めない = 事前地図。
        init_pose  : 初期推定姿勢 (x, y, theta)。起動時は既知とする。
        """
        self.size = len(known_grid)
        self.known_grid = known_grid
        self.est_pose = tuple(init_pose)
        self.likelihood = self._build_likelihood_field(known_grid)
        self.trajectory = [self.est_pose]
        # 推定誤差ログ
        self.last_correction = (0.0, 0.0, 0.0)

    # ------------------------------------------------------------------
    # 尤度場 (壁までの距離変換) を BFS で前計算
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
        # 4近傍 BFS による距離変換 (整数のみ)
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
        return 0  # 範囲外は壁扱い

    def _raycast_known(self, px, py, ang, max_range=12.0, step=0.1):
        """既知壁マップ上で1本のレイを飛ばし、壁までの期待距離を返す。"""
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
    # スキャン一致スコア (大きいほど良い = 残差が小さい)
    # ------------------------------------------------------------------
    def _scan_score(self, pose, scan, beam_stride=4):
        """
        方位ごとに『候補姿勢から既知壁マップへ raycast した期待距離』と
        『LiDAR実測距離』の残差を取り、ロバストにスコア化する。
        - 期待距離より大幅に短い実測 = 家具/動的障害物に当たったビーム → 除外。
        - 残差はキャップ付き二乗で評価し、外れ値に頑健にする。
        方位対応で評価するため『隅への吸着』バイアスが生じない。
        """
        px, py, pth = pose
        score = 0.0
        cap = 1.0  # 残差キャップ(セル)
        for i in range(0, len(scan), beam_stride):
            beam = scan[i]
            if not beam['hit']:
                continue
            ang = pth + beam['angle']
            expected = self._raycast_known(px, py, ang)
            measured = beam['dist']
            # 既知壁より手前で当たった(=家具/動的障害物) ビームは整合に使わない
            if measured < expected - 1.0:
                continue
            r = measured - expected
            if r > cap:
                r = cap
            elif r < -cap:
                r = -cap
            score += (cap * cap - r * r)   # 残差0で最大、外れで0
        return score

    # ------------------------------------------------------------------
    # 1ステップ更新: 予測 + スキャンマッチング補正
    # ------------------------------------------------------------------
    def update(self, odom_u, scan):
        """
        odom_u : (forward, dtheta) — ロボットが指令した(=計測したつもりの)運動。
        scan   : 現在の真の姿勢から得た LiDAR 点群。
        戻り値 : 補正後の推定姿勢 (x, y, theta)。
        """
        forward, dtheta = odom_u
        x, y, th = self.est_pose

        # --- 1) 予測ステップ ---
        th_pred = th + dtheta
        x_pred = x + math.cos(th_pred) * forward
        y_pred = y + math.sin(th_pred) * forward
        predicted = (x_pred, y_pred, th_pred)

        # --- 2) 相関型スキャンマッチングで補正 (粗→密の2段階探索) ---
        # 粗探索で大域的な補正方向を掴み、密探索で精度を上げる。
        # 候補数を抑えることで CPU 負荷を最小化する。
        best_pose = (x_pred, y_pred, th_pred)
        best_score = self._scan_score(best_pose, scan)

        # 段1: 粗探索 (±0.6セル, ±0.1rad)
        for dxx in (-0.6, 0.0, 0.6):
            for dyy in (-0.6, 0.0, 0.6):
                for dth in (-0.10, 0.0, 0.10):
                    cand = (x_pred + dxx, y_pred + dyy, th_pred + dth)
                    s = self._scan_score(cand, scan)
                    if s > best_score:
                        best_score, best_pose = s, cand

        # 段2: 密探索 (粗探索の最良点まわり ±0.2セル, ±0.05rad)
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
        """推定姿勢と真の姿勢のユークリッド距離(セル)。評価用。"""
        return math.hypot(self.est_pose[0] - true_pose[0],
                          self.est_pose[1] - true_pose[1])
