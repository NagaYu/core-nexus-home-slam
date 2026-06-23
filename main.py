"""
main.py
=======
全モジュールを統合し、ロボットがスタートからゴールまで
  * SLAM で自己位置を推定しながら
  * ペットの皿を完璧に迂回し
  * 走行中に椅子が突然進路を塞いだら 10ms 以内で動的リプランニングし
ゴールへ到達するまでの軌跡をターミナルにアスキーアートで実時間描画する。

実行: python3 main.py
"""

import math
import time
import sys

from room_simulator import (
    RoomSimulator, GRID_SIZE, CELL_METERS,
    FREE, WALL, HUMAN_FURNITURE, PET_ZONE,
)
from slam_core import SlamCore
from semantic_grid import SemanticGrid, LABEL_HUMAN, LABEL_PET
from path_planner import CostmapPlanner, INF

# ====== 描画記号 ======
GLYPH = {
    'wall': '█',
    'free': '·',
    'human': 'H',     # 椅子・テーブル
    'pet': 'P',       # ペットの皿(危険)
    'pet_zone': '×',  # ペット皿の進入禁止膨張域
    'path': '*',      # 計画経路
    'robot': '◎',     # ロボット現在地(推定)
    'trail': 'o',     # 走行済み軌跡
    'goal': 'G',
    'start': 'S',
}


def known_wall_map(sim):
    """SLAM に渡す事前地図 = 外壁・固定壁のみ(家具は含めない)。"""
    grid = [[FREE for _ in range(sim.size)] for _ in range(sim.size)]
    for y in range(sim.size):
        for x in range(sim.size):
            if sim.grid[y][x] == WALL:
                grid[y][x] = WALL
    return grid


def perceive_occupancy(sim):
    """
    知覚レイヤ: ロボットが『現在そこにある』と認識する障害物配置を返す。
    (本シミュレーションでは真のマップを観測できるものとして簡略化。
     可動家具の動的変化もここに反映される。)
    戻り値: occupancy(2Dリスト), movable_cells(set)
    """
    n = sim.size
    occ = [[sim.grid[y][x] for x in range(n)] for y in range(n)]
    movable = set()
    for y in range(n):
        for x in range(n):
            if sim.grid[y][x] in (HUMAN_FURNITURE, PET_ZONE):
                movable.add((x, y))
    return occ, movable


def render(sim, planner, path, est_pose, goal, start, trail, step_info):
    """現在状態をアスキーアートで構築して文字列を返す。"""
    n = sim.size
    cm = planner.costmap
    path_set = set(path) if path else set()
    rx, ry = int(round(est_pose[0])), int(round(est_pose[1]))
    trail_set = set(trail)

    lines = []
    for y in range(n):
        row = []
        for x in range(n):
            k = sim.grid[y][x]
            ch = GLYPH['free']
            if k == WALL:
                ch = GLYPH['wall']
            elif k == PET_ZONE:
                ch = GLYPH['pet']
            elif k == HUMAN_FURNITURE:
                ch = GLYPH['human']
            else:
                # 自由空間: コストマップ由来の意味を重ねる
                if cm and cm[y][x] == INF:
                    ch = GLYPH['pet_zone']    # 進入禁止膨張域(主にペット周り)
            # レイヤ重ね順: 経路 < 軌跡 < start/goal < ロボット
            if (x, y) in path_set and k == FREE:
                ch = GLYPH['path']
            if (x, y) in trail_set:
                ch = GLYPH['trail']
            if (x, y) == start:
                ch = GLYPH['start']
            if (x, y) == goal:
                ch = GLYPH['goal']
            if (x, y) == (rx, ry):
                ch = GLYPH['robot']
            row.append(ch)
        lines.append(' '.join(row))

    header = "  CORE NEXUS HOME SLAM  —  軽量SLAM + セマンティクス経路探索"
    legend = ("  凡例: █=壁 H=椅子/机 P=ペット皿 ×=進入禁止 *=計画経路 "
              "o=軌跡 ◎=ロボット S=出発 G=目的地")
    out = [header, legend, "  " + "-" * (n * 2 - 1)]
    out += ["  " + ln for ln in lines]
    out += ["  " + "-" * (n * 2 - 1)]
    out.append("  " + step_info)
    return "\n".join(out)


# 対話端末ならアニメーション、パイプならスナップショット出力
INTERACTIVE = sys.stdout.isatty()


def main():
    sim = RoomSimulator(seed=7)
    n = sim.size
    last_frame = ""

    start = (2, 2)
    goal = (17, 17)
    sim.true_pose = (float(start[0]), float(start[1]), 0.0)

    slam = SlamCore(known_wall_map(sim), init_pose=(float(start[0]), float(start[1]), 0.0))
    semantic = SemanticGrid(n)
    planner = CostmapPlanner(n)

    # --- 初期知覚 → セマンティクス分類 → コストマップ → 経路計画 ---
    occ, movable = perceive_occupancy(sim)
    semantic.observe(movable)
    planner.build_costmap(occ, semantic)
    path = planner.plan(start, goal)

    print("\n================ 初期計画 ================")
    print(f"  A* 初期経路長 = {len(path)} セル, 計画時間 = {planner.last_plan_ms:.3f} ms")
    # セマンティクス分類のログ
    humans = sum(1 for y in range(n) for x in range(n)
                 if semantic.get_label(x, y) == LABEL_HUMAN)
    pets = sum(1 for y in range(n) for x in range(n)
               if semantic.get_label(x, y) == LABEL_PET)
    print(f"  セマンティクス分類: 人間用家具セル={humans}, ペット危険物セル={pets}")

    if not path:
        print("  [ERROR] 初期経路が見つかりませんでした。")
        return

    trail = []
    replan_log = []
    dynamic_event_done = False

    max_steps = 120
    path_idx = 1  # path[0] は現在地

    for step in range(max_steps):
        est = slam.est_pose
        rx, ry = int(round(est[0])), int(round(est[1]))

        # ---- 動的イベント: 経路の中盤で椅子が突然出現し進路を塞ぐ ----
        if not dynamic_event_done and len(trail) >= max(3, len(path) // 3):
            # ロボットの数歩先のセルに椅子を出現させる
            if path_idx + 2 < len(path):
                bx, by = path[path_idx + 2]
                if sim.grid[by][bx] == FREE:
                    sim.set_cell(bx, by, HUMAN_FURNITURE)
                    print("\n  >>> [動的イベント] "
                          f"椅子が突然 ({bx},{by}) に移動し進路を封鎖!")
                    dynamic_event_done = True

        # ---- 再知覚 → 動的リプランニング ----
        occ, movable = perceive_occupancy(sim)
        semantic.observe(movable)
        remaining = path[path_idx:] if path_idx < len(path) else []
        new_path, replanned, ms = planner.replan_if_blocked(
            remaining, occ, semantic, (rx, ry), goal)
        if replanned:
            ok = "成功" if new_path else "失敗(到達不可)"
            replan_log.append((step, ms, len(new_path)))
            print(f"  >>> [動的リプランニング] step={step}: "
                  f"再計算 {ms:.3f} ms → {ok}, 新経路長={len(new_path)}")
            if new_path:
                path = new_path
                path_idx = 1

        # ---- ゴール判定 ----
        if (rx, ry) == goal:
            print("\n  ★ ゴール到達!")
            break

        # ---- 次の経路点へ向かう運動指令を生成 ----
        if path_idx >= len(path):
            target = goal
        else:
            target = path[path_idx]
        tx, ty = target
        dx, dy = tx - est[0], ty - est[1]
        desired_theta = math.atan2(dy, dx)
        # 角度差 [-pi, pi]
        dth = (desired_theta - est[2] + math.pi) % (2 * math.pi) - math.pi
        forward = min(0.9, math.hypot(dx, dy))
        u = (forward, dth)

        # ---- 真の運動(誤差付き) → LiDAR → SLAM 補正 ----
        odom = sim.apply_true_motion(u)
        scan = sim.lidar_scan()
        slam.update(odom, scan)

        # 経路点に十分近づいたら次へ
        nest = slam.est_pose
        if math.hypot(target[0] - nest[0], target[1] - nest[1]) < 0.6:
            path_idx = min(path_idx + 1, len(path))

        trail.append((int(round(nest[0])), int(round(nest[1]))))

        # ---- 描画 ----
        err = slam.pose_error(sim.true_pose)
        info = (f"step={step:3d}  推定=({nest[0]:5.2f},{nest[1]:5.2f},"
                f"{math.degrees(nest[2]):6.1f}°)  "
                f"SLAM誤差={err:4.2f}セル  補正={slam.last_correction[0]:+.2f},"
                f"{slam.last_correction[1]:+.2f}  経路残={max(0,len(path)-path_idx)}")
        frame = render(sim, planner, path, nest, goal, start, trail, info)
        if INTERACTIVE:
            # 画面更新(ターミナルクリア) — 実時間アニメーション
            sys.stdout.write("\033[H\033[J")  # カーソル原点 + クリア
            sys.stdout.write(frame + "\n")
            sys.stdout.flush()
            time.sleep(0.12)
        else:
            # 非対話(パイプ)時: 数ステップごとにスナップショットを出力
            if step % 5 == 0:
                print(f"\n----- スナップショット step={step} -----")
                print(frame)
        last_frame = frame
    else:
        print("\n  [WARN] 最大ステップに達しました。")

    # 最終フレーム(ゴール到達状態)を表示
    print("\n================ 最終マップ ================")
    print(last_frame)

    # ====== サマリ ======
    print("\n================ 走行サマリ ================")
    print(f"  総ステップ数        : {len(trail)}")
    final_err = slam.pose_error(sim.true_pose)
    print(f"  最終SLAM自己位置誤差 : {final_err:.3f} セル "
          f"({final_err * CELL_METERS * 100:.1f} cm)")
    print(f"  動的リプランニング回数: {len(replan_log)}")
    for (st, ms, length) in replan_log:
        within = "✓ 10ms以内" if ms < 10 else "✗ 10ms超過"
        print(f"     - step {st:3d}: {ms:6.3f} ms  ({within}), 新経路長={length}")
    # ペット皿の侵入チェック
    invaded = False
    for (tx, ty) in trail:
        if 0 <= tx < n and 0 <= ty < n and sim.grid[ty][tx] == PET_ZONE:
            invaded = True
    print(f"  ペット皿への侵入     : {'発生(NG)' if invaded else 'なし(完璧に迂回)'}")
    print("==========================================\n")


if __name__ == "__main__":
    main()
