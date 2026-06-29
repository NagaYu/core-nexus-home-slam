"""
semantic_grid.py
================
From an obstacle's bounding box (size / shape features), automatically classify
whether it is
  * a chair / table (human furniture)   ... we want to raise the traversal cost
  * a pet bowl / toy (hazard)           ... strictly off-limits
and write the meaning into the map probabilistically (Bayesian update).

Classification cues (simple features computable even on embedded hardware):
  * occupied cell count `area` (large = furniture, small = pet item)
  * aspect ratio / compactness
A single observation can be misclassified due to noise, so on every observation
we update the class posterior P(class | obs) with a Bayesian update to stay
robust.
"""

from collections import deque

from room_simulator import FREE, WALL, HUMAN_FURNITURE, PET_ZONE

# semantic labels
LABEL_UNKNOWN = 0
LABEL_HUMAN = 1     # chairs / tables
LABEL_PET = 2       # pet bowls / toys (hazard)


class SemanticGrid:
    def __init__(self, size):
        self.size = size
        # per-cell class posterior [P(human), P(pet)] (None if unobserved)
        self.belief = [[None for _ in range(size)] for _ in range(size)]
        self.label = [[LABEL_UNKNOWN for _ in range(size)] for _ in range(size)]

    # ------------------------------------------------------------------
    # Extract clusters from the observed obstacle cells, classify, Bayes update
    # ------------------------------------------------------------------
    def observe(self, occupied_cells):
        """
        occupied_cells : set[(ix, iy)] — the occupied cells (not walls) judged
                         to be 'movable obstacles' by LiDAR this time.
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
        """Cluster by 4-neighbor connected components."""
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
        Return the likelihood P(obs | class) from observed features.
        Heuristic:
          - area >= 3 cells (a spread of ~0.75 m or more) -> likely furniture
          - area <= 1 and compact -> likely a pet item (small object)
        """
        # P(obs | human), P(obs | pet)
        if area >= 3:
            return (0.85, 0.15)
        elif area == 2:
            return (0.6, 0.4)
        else:  # area == 1 : a small standalone object = likely a pet item
            return (0.2, 0.8)

    def _bayes_update(self, ix, iy, likelihood):
        l_h, l_p = likelihood
        prior = self.belief[iy][ix]
        if prior is None:
            prior = (0.5, 0.5)  # uniform prior
        ph, pp = prior
        # Bayes' rule: posterior proportional to likelihood x prior
        ph2 = l_h * ph
        pp2 = l_p * pp
        z = ph2 + pp2
        if z == 0:
            return
        ph2, pp2 = ph2 / z, pp2 / z
        self.belief[iy][ix] = (ph2, pp2)
        # fix the label by MAP estimate
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
