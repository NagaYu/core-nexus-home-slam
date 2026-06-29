# Core Nexus Home SLAM — Lightweight SLAM Educational Simulation

An educational demo built around a small robot moving through a home, illustrating the full pipeline of **self-localization (SLAM), semantic costmaps, A\* path planning, and dynamic re-planning** — implemented in **pure Python (standard library only)**, with no external frameworks (ROS, numpy, etc.). The robot's trajectory is rendered as ASCII art in the terminal in real time.

---

> ### ⚠️ This is an educational simulation (please read first)
>
> - **It is not a product or a research result.** It is unaffiliated with any specific company or product.
> - Strictly speaking this is *localization against a known wall map*. It is not full SLAM (it does not build a map from scratch or perform loop closure).
> - The perception layer is simplified and reads the simulator's ground-truth map (it is not real raw-sensor processing).
> - All figures in this README and in the logs (error, processing time, CPU usage) are **reference values measured inside this simulator on one specific machine**, not validated numbers from real hardware or real environments.
> - "Lightweight / embedded-oriented" describes the design intent; it is not a performance guarantee on real hardware.

---

## Features

- **20×20 grid virtual room** (outer walls, an L-shaped partition, chairs/tables, pet bowls/toys)
- **Simple LiDAR** (72-beam / 5° raycast point cloud with sensor noise)
- **Lightweight SLAM**: injects slip and rotation error into odometry, then corrects the pose with correlative scan matching that minimizes raycast residuals against the known wall map (coarse-to-fine two-stage search)
- **Semantic grid**: classifies obstacles as "human furniture" vs. "pet items (hazard)" from area/shape, with Bayesian updates
- **Intent-aware costmap + A\***: makes a 50 cm radius around pet bowls non-traversable and raises cost around chairs to generate the route
- **Dynamic re-planning**: instantly recomputes the route when a chair suddenly blocks the path mid-run

## Usage

```bash
python3 main.py
```

- No dependencies (Python 3.8+ / standard library only)
- **Interactive terminal**: real-time animation via screen clearing
- **When piped**: prints a snapshot every few steps

```bash
python3 main.py | less    # review the snapshots together
```

## Example output

```
  █ · S · · · · · · · · · · · · · · · · █     S start   G goal
  █ · o · H H · · × · · · · · · · · · · █     o trail    ◎ robot
  █ · o o · · × × P × × · · · · · · · · █     H chair/table   P pet bowl
  █ · · · o o o · × · · · · · · · · · · █     × keep-out zone (50cm inflation)
  █ · · · · · · o o o o o o o · · · · · █     █ wall
  ...
>>> [Dynamic event] A chair suddenly moved to (5,11) and blocked the path!
>>> [Dynamic re-planning] step=7: recomputed in 0.18 ms -> success, new path length=18
★ Goal reached!
```

## File structure

| File | Role |
|---|---|
| `room_simulator.py` | Virtual room, simple LiDAR, odometry error injection (ground truth) |
| `slam_core.py`      | Self-localization via likelihood field + raycast-residual scan matching |
| `semantic_grid.py`  | Furniture/hazard probabilistic classification via connected-component clustering + Bayesian updates |
| `path_planner.py`   | Semantic costmap generation, A\* path planning, dynamic re-planning |
| `main.py`           | Integration of all modules, ASCII-art rendering, run summary |

## Algorithm notes

- **Scan matching**: minimizes, per bearing, the residual between the LiDAR measured range and the "expected range" obtained by raycasting from a candidate pose against the known wall map. Beams that hit obstacles absent from the map (e.g., furniture) are excluded from the residual. Because it matches per bearing, the "corner-snapping" bias is largely avoided.
- **Likelihood field**: distance transform from wall cells, precomputed with BFS (integer arithmetic only).
- **Semantic costmap**: pet-bowl cells plus a 2-cell radius (≈50 cm) are made non-traversable; areas around chairs/tables are given higher cost.
- **A\***: 8-connected, no diagonal corner-cutting, octile-distance heuristic.

## Known limitations (deliberate simplifications for teaching)

- Assumes a known wall map (mapping of unknown environments and loop closure are not implemented)
- Fixed 20×20 grid; cell-based rather than continuous space
- Scan matching is local (brute-force) search and does not recover from global localization loss (kidnapped robot)
- Semantic classification is a simple model based mainly on area heuristics

## License

MIT License (see `LICENSE`)
