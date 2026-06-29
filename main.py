"""
main.py
=======
Integrates all modules and, as the robot goes from start to goal:
  * estimates its pose with SLAM,
  * perfectly detours around the pet bowl,
  * dynamically re-plans within 10 ms if a chair suddenly blocks the path,
renders the trajectory as ASCII art in the terminal in real time.

Run: python3 main.py
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

# ====== rendering glyphs ======
GLYPH = {
    'wall': '█',
    'free': '·',
    'human': 'H',     # chairs / tables
    'pet': 'P',       # pet bowl (hazard)
    'pet_zone': '×',  # pet-bowl keep-out inflation zone
    'path': '*',      # planned path
    'robot': '◎',     # robot's current (estimated) location
    'trail': 'o',     # traveled trail
    'goal': 'G',
    'start': 'S',
}


def known_wall_map(sim):
    """The prior map handed to SLAM = outer/fixed walls only (no furniture)."""
    grid = [[FREE for _ in range(sim.size)] for _ in range(sim.size)]
    for y in range(sim.size):
        for x in range(sim.size):
            if sim.grid[y][x] == WALL:
                grid[y][x] = WALL
    return grid


def perceive_occupancy(sim):
    """
    Perception layer: return the obstacle layout the robot believes is
    'currently there'. (Simplified here so the true map is observable; dynamic
    changes to movable furniture are reflected here too.)
    Returns: occupancy (2D list), movable_cells (set)
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
    """Build the current state as ASCII art and return it as a string."""
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
                # free space: overlay meaning derived from the costmap
                if cm and cm[y][x] == INF:
                    ch = GLYPH['pet_zone']    # keep-out inflation (mostly around pets)
            # layer order: path < trail < start/goal < robot
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

    header = "  CORE NEXUS HOME SLAM  —  lightweight SLAM + semantic path planning"
    legend = ("  legend: █=wall H=chair/table P=pet bowl ×=keep-out *=planned path "
              "o=trail ◎=robot S=start G=goal")
    out = [header, legend, "  " + "-" * (n * 2 - 1)]
    out += ["  " + ln for ln in lines]
    out += ["  " + "-" * (n * 2 - 1)]
    out.append("  " + step_info)
    return "\n".join(out)


# Animate on an interactive terminal; print snapshots when piped
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

    # --- initial perception -> semantic classification -> costmap -> planning ---
    occ, movable = perceive_occupancy(sim)
    semantic.observe(movable)
    planner.build_costmap(occ, semantic)
    path = planner.plan(start, goal)

    print("\n================ initial plan ================")
    print(f"  A* initial path length = {len(path)} cells, planning time = {planner.last_plan_ms:.3f} ms")
    # semantic classification log
    humans = sum(1 for y in range(n) for x in range(n)
                 if semantic.get_label(x, y) == LABEL_HUMAN)
    pets = sum(1 for y in range(n) for x in range(n)
               if semantic.get_label(x, y) == LABEL_PET)
    print(f"  semantic classification: human-furniture cells={humans}, pet-hazard cells={pets}")

    if not path:
        print("  [ERROR] no initial path was found.")
        return

    trail = []
    replan_log = []
    dynamic_event_done = False

    max_steps = 120
    path_idx = 1  # path[0] is the current location

    for step in range(max_steps):
        est = slam.est_pose
        rx, ry = int(round(est[0])), int(round(est[1]))

        # ---- dynamic event: midway, a chair suddenly appears and blocks the path ----
        if not dynamic_event_done and len(trail) >= max(3, len(path) // 3):
            # make a chair appear a few steps ahead of the robot
            if path_idx + 2 < len(path):
                bx, by = path[path_idx + 2]
                if sim.grid[by][bx] == FREE:
                    sim.set_cell(bx, by, HUMAN_FURNITURE)
                    print("\n  >>> [dynamic event] "
                          f"a chair suddenly moved to ({bx},{by}) and blocked the path!")
                    dynamic_event_done = True

        # ---- re-perceive -> dynamic re-planning ----
        occ, movable = perceive_occupancy(sim)
        semantic.observe(movable)
        remaining = path[path_idx:] if path_idx < len(path) else []
        new_path, replanned, ms = planner.replan_if_blocked(
            remaining, occ, semantic, (rx, ry), goal)
        if replanned:
            ok = "success" if new_path else "failed (unreachable)"
            replan_log.append((step, ms, len(new_path)))
            print(f"  >>> [dynamic re-planning] step={step}: "
                  f"recomputed in {ms:.3f} ms -> {ok}, new path length={len(new_path)}")
            if new_path:
                path = new_path
                path_idx = 1

        # ---- goal check ----
        if (rx, ry) == goal:
            print("\n  ★ goal reached!")
            break

        # ---- generate a motion command toward the next path point ----
        if path_idx >= len(path):
            target = goal
        else:
            target = path[path_idx]
        tx, ty = target
        dx, dy = tx - est[0], ty - est[1]
        desired_theta = math.atan2(dy, dx)
        # angle difference in [-pi, pi]
        dth = (desired_theta - est[2] + math.pi) % (2 * math.pi) - math.pi
        forward = min(0.9, math.hypot(dx, dy))
        u = (forward, dth)

        # ---- true motion (with error) -> LiDAR -> SLAM correction ----
        odom = sim.apply_true_motion(u)
        scan = sim.lidar_scan()
        slam.update(odom, scan)

        # advance to the next point once close enough
        nest = slam.est_pose
        if math.hypot(target[0] - nest[0], target[1] - nest[1]) < 0.6:
            path_idx = min(path_idx + 1, len(path))

        trail.append((int(round(nest[0])), int(round(nest[1]))))

        # ---- render ----
        err = slam.pose_error(sim.true_pose)
        info = (f"step={step:3d}  est=({nest[0]:5.2f},{nest[1]:5.2f},"
                f"{math.degrees(nest[2]):6.1f}°)  "
                f"SLAM err={err:4.2f} cells  corr={slam.last_correction[0]:+.2f},"
                f"{slam.last_correction[1]:+.2f}  path left={max(0,len(path)-path_idx)}")
        frame = render(sim, planner, path, nest, goal, start, trail, info)
        if INTERACTIVE:
            # screen refresh (clear terminal) — real-time animation
            sys.stdout.write("\033[H\033[J")  # cursor home + clear
            sys.stdout.write(frame + "\n")
            sys.stdout.flush()
            time.sleep(0.12)
        else:
            # when piped: print a snapshot every few steps
            if step % 5 == 0:
                print(f"\n----- snapshot step={step} -----")
                print(frame)
        last_frame = frame
    else:
        print("\n  [WARN] reached the maximum number of steps.")

    # show the final frame (goal-reached state)
    print("\n================ final map ================")
    print(last_frame)

    # ====== summary ======
    print("\n================ run summary ================")
    print(f"  total steps              : {len(trail)}")
    final_err = slam.pose_error(sim.true_pose)
    print(f"  final SLAM pose error    : {final_err:.3f} cells "
          f"({final_err * CELL_METERS * 100:.1f} cm)")
    print(f"  dynamic re-planning count: {len(replan_log)}")
    for (st, ms, length) in replan_log:
        within = "✓ within 10ms" if ms < 10 else "✗ over 10ms"
        print(f"     - step {st:3d}: {ms:6.3f} ms  ({within}), new path length={length}")
    # check for pet-bowl intrusion
    invaded = False
    for (tx, ty) in trail:
        if 0 <= tx < n and 0 <= ty < n and sim.grid[ty][tx] == PET_ZONE:
            invaded = True
    print(f"  pet-bowl intrusion       : {'occurred (NG)' if invaded else 'none (perfect detour)'}")
    print("==========================================\n")


if __name__ == "__main__":
    main()
