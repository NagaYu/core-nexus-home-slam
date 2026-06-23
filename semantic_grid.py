"""
semantic_grid.py
================
障害物のバウンディングボックス(サイズ・形状特徴)から、それが
  * 椅子 / テーブル (人間用家具)   … 通行コストを上げたい
  * ペットの皿 / おもちゃ (危険物) … 絶対進入禁止
かを自動分類し、確率的(ベイズ更新)に意味地図へ書き込む。

分類の手がかり (組込みでも計算できる単純特徴):
  * 占有セル数 area (大きい=家具、小さい=ペット用品)
  * 縦横比 / コンパクトさ
ノイズで1回の観測では誤分類しうるため、観測のたびにベイズ更新で
クラス事後確率 P(class | obs) を更新し、頑健にする。
"""

from collections import deque

from room_simulator import FREE, WALL, HUMAN_FURNITURE, PET_ZONE

# 意味ラベル
LABEL_UNKNOWN = 0
LABEL_HUMAN = 1     # 椅子・テーブル
LABEL_PET = 2       # ペットの皿・おもちゃ (危険)


class SemanticGrid:
    def __init__(self, size):
        self.size = size
        # 各セルのクラス事後確率 [P(human), P(pet)] (未観測は None)
        self.belief = [[None for _ in range(size)] for _ in range(size)]
        self.label = [[LABEL_UNKNOWN for _ in range(size)] for _ in range(size)]

    # ------------------------------------------------------------------
    # 観測された障害物セル集合からクラスタを抽出し分類・ベイズ更新
    # ------------------------------------------------------------------
    def observe(self, occupied_cells):
        """
        occupied_cells : set[(ix, iy)] — 今回 LiDAR で『可動障害物』と
                         判定された(壁ではない)占有セル群。
        """
        clusters = self._cluster(occupied_cells)
        for cells in clusters:
            area = len(cells)
            xs = [c[0] for c in cells]
            ys = [c[1] for c in cells]
            w = max(xs) - min(xs) + 1
            h = max(ys) - min(ys) + 1
            likelihood = self._class_likelihood(area, w, h)
            for (ix, iy) in cells:
                self._bayes_update(ix, iy, likelihood)

    def _cluster(self, occupied):
        """4近傍連結成分でクラスタリング。"""
        occupied = set(occupied)
        seen = set()
        clusters = []
        for cell in occupied:
            if cell in seen:
                continue
            comp = []
            q = deque([cell])
            seen.add(cell)
            while q:
                cx, cy = q.popleft()
                comp.append((cx, cy))
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nb = (cx + dx, cy + dy)
                    if nb in occupied and nb not in seen:
                        seen.add(nb)
                        q.append(nb)
            clusters.append(comp)
        return clusters

    def _class_likelihood(self, area, w, h):
        """
        観測特徴から尤度 P(obs | class) を返す。
        ヒューリスティック:
          - area >= 3 セル(約 0.75m 以上の広がり) → 家具らしい
          - area <= 1 かつコンパクト → ペット用品(小物)らしい
        """
        # P(obs | human), P(obs | pet)
        if area >= 3:
            return (0.85, 0.15)
        elif area == 2:
            return (0.6, 0.4)
        else:  # area == 1 : 小さな単独物 = ペット用品の可能性が高い
            return (0.2, 0.8)

    def _bayes_update(self, ix, iy, likelihood):
        l_h, l_p = likelihood
        prior = self.belief[iy][ix]
        if prior is None:
            prior = (0.5, 0.5)  # 一様事前分布
        ph, pp = prior
        # ベイズ則: 事後 ∝ 尤度 × 事前
        ph2 = l_h * ph
        pp2 = l_p * pp
        z = ph2 + pp2
        if z == 0:
            return
        ph2, pp2 = ph2 / z, pp2 / z
        self.belief[iy][ix] = (ph2, pp2)
        # MAP 推定でラベル確定
        self.label[iy][ix] = LABEL_HUMAN if ph2 >= pp2 else LABEL_PET

    # ------------------------------------------------------------------
    def confidence(self, ix, iy):
        b = self.belief[iy][ix]
        if b is None:
            return 0.0
        return max(b)

    def get_label(self, ix, iy):
        if 0 <= ix < self.size and 0 <= iy < self.size:
            return self.label[iy][ix]
        return LABEL_UNKNOWN
