"""
room_simulator.py
==================
家庭内を移動するロボットの「仮想室内環境」と「簡易LiDAR」を提供する。

設計方針 (組込み前提):
  * 外部フレームワーク(ROS/numpy 等)に一切依存しない純Python実装。
  * グリッドは 20x20。1セル = 0.25m(=25cm) とみなし、約 5m x 5m の部屋を表現する。
  * 障害物は3種類:
      WALL            : 外壁・固定壁 (SLAMの基準となる既知形状)
      HUMAN_FURNITURE : 椅子・テーブル等。人間が使うため周囲コストを上げたい。
      PET_ZONE        : ペットの皿・おもちゃ。絶対進入禁止 (周囲50cmも禁止)。

座標系:
  grid[y][x]。x は東(右)方向、y は南(下)方向。
  ロボット姿勢 pose = (x, y, theta) は連続値[セル単位]。
  theta はラジアン。0 = +x(東)方向、反時計回り(数学的)に増加。
"""

import math
import random

# ----- セルの種別 (意味論的タグ) -----
FREE = 0
WALL = 1
HUMAN_FURNITURE = 2   # 椅子・テーブル
PET_ZONE = 3          # ペットの皿・おもちゃ (危険物 / 進入禁止)

GRID_SIZE = 20
CELL_METERS = 0.25    # 1セルあたりの実寸 (m)


class RoomSimulator:
    """真の室内マップと真のロボット姿勢を保持する『地上真実(ground truth)』環境。"""

    def __init__(self, size=GRID_SIZE, seed=7):
        self.size = size
        self.rng = random.Random(seed)
        # grid[y][x]
        self.grid = [[FREE for _ in range(size)] for _ in range(size)]
        self._build_walls()
        self._place_furniture()
        # ロボットの『真の』姿勢 (SLAMはこれを直接知らない)
        self.true_pose = (2.0, 2.0, 0.0)

    # ------------------------------------------------------------------
    # マップ構築
    # ------------------------------------------------------------------
    def _build_walls(self):
        n = self.size
        for x in range(n):
            self.grid[0][x] = WALL
            self.grid[n - 1][x] = WALL
        for y in range(n):
            self.grid[y][0] = WALL
            self.grid[y][n - 1] = WALL
        # 室内の仕切り壁 (L字) — スキャンマッチングに特徴を与える
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
        # 人間用家具: テーブル(大) と 椅子(小)
        self._place_block(4, 4, 2, 2, HUMAN_FURNITURE)    # テーブル
        self._place_block(15, 3, 1, 1, HUMAN_FURNITURE)   # 椅子
        self._place_block(3, 15, 1, 1, HUMAN_FURNITURE)   # 椅子
        # ペットの皿 (1セルの小さな危険物) と おもちゃ
        # 皿は出発→目的地の対角線上に置き、迂回行動を明確に観測できるようにする
        self._place_block(14, 14, 1, 1, PET_ZONE)         # ペットの皿
        self._place_block(8, 6, 1, 1, PET_ZONE)           # おもちゃ

    # ------------------------------------------------------------------
    # 障害物判定
    # ------------------------------------------------------------------
    def is_obstacle(self, x, y):
        """連続座標 (x, y) が障害物セルかどうか。"""
        ix, iy = int(math.floor(x)), int(math.floor(y))
        if ix < 0 or iy < 0 or ix >= self.size or iy >= self.size:
            return True
        return self.grid[iy][ix] != FREE

    def cell(self, ix, iy):
        if 0 <= ix < self.size and 0 <= iy < self.size:
            return self.grid[iy][ix]
        return WALL

    # ------------------------------------------------------------------
    # 簡易LiDAR (360度点群)
    # ------------------------------------------------------------------
    def lidar_scan(self, pose=None, num_beams=72, max_range=12.0, step=0.05,
                   noise_std=0.01):
        """
        与えられた姿勢から360度のレイキャストを行い、点群を返す。

        戻り値: ビームごとの dict のリスト
            { 'angle': ビーム相対角(rad), 'dist': 距離(セル), 'hit': bool }
        num_beams=72 → 5度刻み。組込みでも軽い解像度。
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
            # センサ雑音
            meas = dist + self.rng.gauss(0.0, noise_std)
            scan.append({'angle': rel, 'dist': max(0.0, meas), 'hit': hit})
        return scan

    # ------------------------------------------------------------------
    # ロボットの真の運動 (オドメトリ + 滑り誤差)
    # ------------------------------------------------------------------
    def apply_true_motion(self, u, slip_std=0.04, rot_std=0.03):
        """
        制御入力 u=(forward, dtheta) を真の姿勢に適用する。
        わざとランダムな滑り(並進)・回転誤差を加え、オドメトリのドリフトを再現。
        戻り値: ロボットが『計測したつもり』の理想オドメトリ u (誤差なしの指令値)。
        """
        forward, dtheta = u
        x, y, th = self.true_pose
        # --- 真の運動には誤差が乗る ---
        true_forward = forward + self.rng.gauss(0.0, slip_std)
        true_dtheta = dtheta + self.rng.gauss(0.0, rot_std)
        th2 = th + true_dtheta
        nx = x + math.cos(th2) * true_forward
        ny = y + math.sin(th2) * true_forward
        # 壁にめり込む場合は移動をキャンセル(衝突)
        if not self.is_obstacle(nx, ny):
            self.true_pose = (nx, ny, th2)
        else:
            self.true_pose = (x, y, th2)
        # ロボット側が知るのは『指令値(理想)』のみ
        return (forward, dtheta)

    # ------------------------------------------------------------------
    # 動的障害物の注入 (椅子が突然動く 等)
    # ------------------------------------------------------------------
    def set_cell(self, ix, iy, kind):
        if 0 <= ix < self.size and 0 <= iy < self.size:
            self.grid[iy][ix] = kind
