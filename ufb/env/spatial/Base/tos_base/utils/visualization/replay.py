from typing import List, Dict, Tuple, Optional
import random, io, os, glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import imageio

from ...core.room import Room
from ...core.object import Agent
from ...actions.actions import ActionSequence
from ...managers.exploration_manager import ExplorationManager
from ...managers.spatial_solver import SpatialSolver
from ..room_utils import RoomPlotter




# ==================== icon helpers (merged & simplified) ====================

def _trim_transparent(icon: np.ndarray, thr: int = 10) -> np.ndarray:
    """Trim transparent margins from RGBA icons."""
    if icon.ndim == 3 and icon.shape[2] == 4:
        a = icon[..., 3] > thr
        if a.any():
            ys, xs = np.where(a)
            return icon[ys.min():ys.max()+1, xs.min():xs.max()+1]
    return icon

def _load_icon(name: str, files: List[str]) -> Optional[np.ndarray]:
    """Pick a best-matching file for the name, read and trim."""
    if not files: return None
    lname = name.lower().replace(' ', '_')
    # exact match
    for f in files:
        base = os.path.splitext(os.path.basename(f))[0].lower()
        if base == lname:
            return _trim_transparent(imageio.v2.imread(f))
    # category fallback
    if 'door' in lname:
        for f in files:
            base = os.path.splitext(os.path.basename(f))[0].lower()
            if base == 'door':
                return _trim_transparent(imageio.v2.imread(f))
    # final fallback
    return _trim_transparent(imageio.v2.imread(random.choice(files)))

def _dominant_rgb(icon: np.ndarray) -> Tuple[float, float, float]:
    """Coarse dominant color under alpha; clamps extremes for legibility."""
    arr = icon.astype(np.float32) / 255.0
    if arr.ndim < 3:
        arr = np.dstack([arr, arr, arr, np.ones_like(arr)])
    alpha = arr[..., 3] if arr.shape[2] == 4 else np.ones(arr.shape[:2], dtype=np.float32)
    rgb = arr[..., :3]
    mask = alpha > 0.1
    if not mask.any(): 
        return tuple(np.clip(rgb.mean(axis=(0, 1)), 0, 1))
    rgb = rgb[mask]
    q = np.clip((rgb * 255).astype(np.uint8) >> 3, 0, 31)
    uq, counts = np.unique(q, axis=0, return_counts=True)
    deq = (uq.astype(np.float32) + 0.5) / 32.0
    # prefer mid-luminance bins to avoid too-dark/too-bright
    lum = (0.2126 * deq[:, 0] + 0.7152 * deq[:, 1] + 0.0722 * deq[:, 2])
    prefer = (lum >= 0.20) & (lum <= 0.90)
    dom = deq[np.argmax(counts * prefer if prefer.any() else counts)]
    L = (0.2126*dom[0] + 0.7152*dom[1] + 0.0722*dom[2])
    if   L < 0.18: dom = dom * 0.6 + 0.4
    elif L > 0.92: dom = dom * 0.8
    return tuple(np.clip(dom, 0, 1))

def _add_icon(ax, icon: Optional[np.ndarray], x: float, y: float, size_points: float = 56.0, xycoords: str = 'data'):
    """Add icon centered at (x, y); size by approximate area in points^2."""
    if icon is None: return
    dpi = float(ax.figure.get_dpi())
    area_px = float(size_points) * (dpi / 72.0) ** 2
    diam_px = float(np.sqrt(max(1e-6, 4.0 * area_px / np.pi)))
    dim = float(max(1, int(max(icon.shape[0], icon.shape[1]))))
    zoom = max(0.02, min(0.6, 0.95 * diam_px / dim))
    ab = AnnotationBbox(OffsetImage(icon, zoom=zoom), (x, y), frameon=False, xycoords=xycoords)
    ab.set_clip_on(False)
    ax.add_artist(ab)


# ==================== palette & color tools ====================

_DOPAMINE_HEX = [
    '#FF6B6B', '#FFD166', '#06D6A0', '#118AB2', '#9B5DE5', '#F15BB5',
    '#00F5D4', '#F4A261', '#3A86FF', '#8338EC', '#FF006E', '#8AC926',
    '#2EC4B6', '#FFBE0B', '#EF476F', '#4ECDC4'
]

def _hex_to_rgb(h: str) -> tuple:
    h = h.lstrip('#')
    return tuple(int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4))

def _rgb_to_hex(rgb: Tuple[float, float, float]) -> str:
    r, g, b = [int(max(0, min(1, c)) * 255) for c in rgb]
    return '#%02x%02x%02x' % (r, g, b)

def _rgb_to_lab(rgb: Tuple[float, float, float]) -> Tuple[float, float, float]:
    def f(u): return u / 12.92 if u <= 0.04045 else ((u + 0.055) / 1.055) ** 2.4
    r, g, b = [f(c) for c in rgb]
    x = r*0.4124564 + g*0.3575761 + b*0.1804375
    y = r*0.2126729 + g*0.7151522 + b*0.0721750
    z = r*0.0193339 + g*0.1191920 + b*0.9503041
    xr, yr, zr = x/0.95047, y/1.00000, z/1.08883
    def gfun(t): return t ** (1/3) if t > 0.008856 else (7.787 * t + 16/116)
    fx, fy, fz = gfun(xr), gfun(yr), gfun(zr)
    L = 116*fy - 16; a = 500*(fx - fy); b2 = 200*(fy - fz)
    return (L, a, b2)

def _dist_lab(a, b): return np.linalg.norm(np.subtract(a, b))

def _assign_identity_colors(names: List[str]) -> Dict[str, str]:
    """Maximally spread palette assignment in CIE Lab; stable over names."""
    names = sorted(names)
    base = [_hex_to_rgb(h) for h in _DOPAMINE_HEX]
    base_lab = [_rgb_to_lab(c) for c in base]
    order, remaining = [0], list(range(1, len(base)))
    while remaining:
        best, best_d = None, -1.0
        for idx in remaining:
            d = min(_dist_lab(base_lab[idx], base_lab[j]) for j in order)
            if d > best_d: best, best_d = idx, d
        order.append(best); remaining.remove(best)
    pal = [_rgb_to_hex(base[i]) for i in order]
    if len(names) > len(pal):  # extend if needed
        import colorsys
        for i in range(len(names) - len(pal)):
            h, s, v = (i * 0.61803398875) % 1.0, 0.70, 0.95
            pal.append(_rgb_to_hex(colorsys.hsv_to_rgb(h, s, v)))
    return {n: pal[i % len(pal)] for i, n in enumerate(names)}


# ==================== numeric helpers ====================
def _full_domain_size(grid_size: int) -> int:
    """Calculates the total number of cells in a square domain."""
    g = int(grid_size)
    return (2 * g + 1) * (2 * g + 1)

def _gaussian_kernel(sigma: float = 0.3) -> np.ndarray:
    """
    Creates a 2D Gaussian kernel.
    NOTE: Default sigma has been reduced from 0.5 to 0.3 for a tighter kernel, as requested.
    """
    s = max(1, int(np.ceil(2.2 * sigma)))
    ax = np.arange(-s, s + 1)
    xx, yy = np.meshgrid(ax, ax)
    ker = np.exp(-(xx * xx + yy * yy) / (2.0 * sigma * sigma))
    ker /= ker.sum() if ker.sum() > 0 else 1.0
    return ker.astype(np.float32)

def _accumulate_gaussians(grid: np.ndarray, positions: List[Tuple[int, int]], grid_size: int, sigma: float = 0.5) -> np.ndarray:
    """
    Discrete accumulation of Gaussians on a grid.
    NOTE: Default sigma has been reduced from 1.0 to 0.5 for tighter heatmaps.
    """
    ker = _gaussian_kernel(sigma)
    ks, r, g = ker.shape[0], ker.shape[0] // 2, int(grid_size)
    H, W = grid.shape
    for (x, y) in positions:
        cx, cy = int(x) + g, int(y) + g
        x0, x1 = max(0, cx - r), min(W, cx + r + 1)
        y0, y1 = max(0, cy - r), min(H, cy + r + 1)
        kx0, kx1 = (0 if cx - r >= 0 else r - cx), (ks if cx + r + 1 <= W else r + (W - cx))
        ky0, ky1 = (0 if cy - r >= 0 else r - cy), (ks if cy + r + 1 <= H else r + (H - cy))
        grid[y0:y1, x0:x1] += ker[ky0:ky1, kx0:kx1]
    return grid

def _accumulate_gaussians_continuous(positions: List[Tuple[int, int]], XX: np.ndarray, YY: np.ndarray, sigma: float) -> np.ndarray:
    """Continuous accumulation on a high-res mesh (used for smooth contours & main heat)."""
    heat = np.zeros_like(XX, dtype=np.float32)
    if sigma <= 0: sigma = 1e-6
    inv2s2 = 1.0 / (2.0 * sigma * sigma)
    for (x, y) in positions:
        dx, dy = (XX - float(x)), (YY - float(y))
        heat += np.exp(-(dx * dx + dy * dy) * inv2s2).astype(np.float32)
    return heat

# ==================== sketchy dashed outline ====================

def _draw_sketchy_outline(ax,
                          positions: List[Tuple[int, int]],
                          color: str,
                          small: bool = False,
                          perturb_strength: float = 0.15):
    """
    Draws a smooth, sketch-like dashed outline around a set of discrete positions.

    Visibility tweaks:
      • Thicker stroke + subtle light halo (path effect) to pop over heatmaps.
      • Everything else left unchanged for minimal impact.
    """
    if not positions:
        return

    # Local imports to keep the outer module untouched.
    from matplotlib.path import Path
    from matplotlib.patches import PathPatch
    from matplotlib import patheffects as pe
    from collections import defaultdict

    positions = np.array(positions, dtype=int)
    # Single-candidate: no outline.
    uniq = np.unique(positions, axis=0)
    if uniq.shape[0] <= 1:
        return
    positions = uniq

    # Controls (unchanged except linewidth)
    _CHAIKIN_ITERS = 5
    _TAUBIN_ITERS  = 1
    _TAUBIN_LAMBDA = 0.33
    _TAUBIN_MU     = -0.34
    _JITTER_RATIO_OF_MEDLEN = 0.35
    dash_len = 7.0 if not small else 4.0
    gap_len  = 0.6 * dash_len

    # 1) Minimal binary grid enclosing all points (+1 margin).
    x_min, y_min = positions.min(axis=0) - 1
    x_max, y_max = positions.max(axis=0) + 1
    grid = np.zeros((y_max - y_min + 1, x_max - x_min + 1), dtype=np.int8)

    # 2) Mark occupied cells.
    grid[positions[:, 1] - y_min, positions[:, 0] - x_min] = 1

    # 3) Boundary transitions.
    v_diffs = np.diff(grid, axis=0)
    h_edge_rows, h_edge_cols = np.where(v_diffs != 0)

    h_diffs = np.diff(grid, axis=1)
    v_edge_rows, v_edge_cols = np.where(h_diffs != 0)

    # 4) Assemble segments in data coordinates.
    segments = []
    for r, c in zip(v_edge_rows, v_edge_cols):  # vertical edges
        x = c + x_min + 0.5
        segments.append(((x, r + y_min - 0.5), (x, r + y_min + 0.5)))
    for r, c in zip(h_edge_rows, h_edge_cols):  # horizontal edges
        y = r + y_min + 0.5
        segments.append(((c + x_min - 0.5, y), (c + x_min + 0.5, y)))
    if not segments:
        return

    # 5) Stitch into closed loops.
    SCALE = 2
    def _k(p): return (int(round(p[0] * SCALE)), int(round(p[1] * SCALE)))

    adj = defaultdict(list)
    for p1, p2 in segments:
        k1, k2 = _k(p1), _k(p2)
        adj[k1].append(k2); adj[k2].append(k1)

    loops, visited = [], set()
    for start in list(adj.keys()):
        if start in visited or len(adj[start]) < 2:
            visited.add(start); continue
        loop, prev, cur = [start], None, start
        max_steps = len(segments) * 4
        for _ in range(max_steps):
            nbrs = adj[cur]
            nxt = nbrs[0] if nbrs[0] != prev else (nbrs[1] if len(nbrs) > 1 else None)
            if nxt is None: loop = []; break
            prev, cur = cur, nxt
            if cur == start: break
            loop.append(cur)
        visited.update(loop)
        if len(loop) >= 3 and cur == start:
            loops.append(np.array(loop, dtype=np.int32))
    if not loops:
        return

    # 6) Helpers.
    def _to_float_coords(loop_int: np.ndarray) -> np.ndarray:
        return loop_int.astype(np.float32) / float(SCALE)

    def _chaikin_closed(points: np.ndarray, n: int) -> np.ndarray:
        P = points
        for _ in range(n):
            Q = np.empty((2 * len(P), 2), dtype=np.float32)
            Q[0::2] = 0.75 * P + 0.25 * np.roll(P, -1, axis=0)
            Q[1::2] = 0.25 * P + 0.75 * np.roll(P, -1, axis=0)
            P = Q
        return P

    def _taubin_closed(points: np.ndarray, lam: float, mu: float, n: int) -> np.ndarray:
        P = points
        for _ in range(n):
            L = 0.5 * (np.roll(P, -1, axis=0) + np.roll(P, 1, axis=0)) - P
            P = P + lam * L
            L = 0.5 * (np.roll(P, -1, axis=0) + np.roll(P, 1, axis=0)) - P
            P = P + mu * L
        return P

    def _add_sketch_jitter(points: np.ndarray, strength: float, cap_ratio: float) -> np.ndarray:
        if strength <= 0 or len(points) < 3: return points
        seg = np.roll(points, -1, axis=0) - points
        med_len = np.median(np.linalg.norm(seg, axis=1))
        amp = float(min(strength, cap_ratio * med_len))
        if amp <= 0: return points
        tang = np.roll(points, -1, axis=0) - np.roll(points, 1, axis=0)
        nrm = np.empty_like(tang)
        nrm[:, 0] = -tang[:, 1]; nrm[:, 1] = tang[:, 0]
        nrm /= (np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-8)
        noise = (np.random.randn(points.shape[0], 1).astype(np.float32)) * amp
        return points + nrm * noise

    # 7) Render (thicker + halo).
    lw = (1.35 if not small else 0.55)  # thicker for stronger presence

    for loop_int in loops:
        pts = _to_float_coords(loop_int)
        pts = _chaikin_closed(pts, _CHAIKIN_ITERS)
        pts = _taubin_closed(pts, _TAUBIN_LAMBDA, _TAUBIN_MU, _TAUBIN_ITERS)
        pts = _add_sketch_jitter(pts, perturb_strength, _JITTER_RATIO_OF_MEDLEN)

        verts = np.vstack([pts, pts[0]])
        codes = np.empty(len(verts), dtype=np.uint8)
        codes[0], codes[1:-1], codes[-1] = Path.MOVETO, Path.LINETO, Path.CLOSEPOLY

        patch = PathPatch(
            Path(verts, codes),
            facecolor='none',
            edgecolor=color,
            linewidth=lw,
            capstyle='round',
            joinstyle='round',
            antialiased=True,
            zorder=38  # slightly above heat
        )
        patch.set_linestyle((np.random.uniform(0, dash_len), (dash_len, gap_len)))
        # Light halo to increase contrast over busy backgrounds.
        patch.set_path_effects([
            pe.Stroke(linewidth=lw * 1.8, foreground=(1, 1, 1, 0.85)),
            pe.Normal()
        ])
        ax.add_patch(patch)




# ==================== main helper class ====================

class ReplayHelper:
    """Replay utilities for heatmaps with clean sketch style."""

    def __init__(self, room: 'Room', agent: 'Agent', grid_size: Optional[int] = None):
        self.room = room.copy()
        self.agent = agent.copy()
        g = (max(self.room.mask.shape) if getattr(self.room, 'mask', None) is not None else 10)
        self.grid_size = int(g if grid_size is None else grid_size)

    # -------- public API --------
    def plot_observation_heatmaps(
        self,
        action_results: List,
        max_positions: int = 500,
        sigma: float = 1.0,
        out_dir: Optional[str] = None,
        fps: int = 2,
        axes: bool = True,
        use_icons: bool = False,
        icons_dir: Optional[str] = None,
        use_icon_colors: bool = False,
        bg: float = 0.96,
        small_bg: float = 0.08
    ) -> List[np.ndarray]:
        """
        Render per-observation heatmaps. The icon/color legend is saved as a
        separate sheet when icons are enabled (icons_legend.png).
        """
        mgr = ExplorationManager(self.room, self.agent)  # kept for parity if external code relies on it
        _ = mgr  # avoid linter on the local variable (not otherwise used directly here)

        # 1) Prepare solver + collect snapshots
        snapshots = self._collect_snapshots(action_results)

        # 2) Colors/icons
        obj_names = [o.name for o in self.room.all_objects if o.name != 'initial_pos']
        color_by_name, icon_by_name, agent_icon = self._build_colors_and_icons(
            obj_names, use_icons, icons_dir, use_icon_colors
        )

        # 3) Bounds shared by all frames
        bounds = self._compute_bounds(snapshots)

        # 4) Render frames
        frames = [
            self._render_frame(
                domains=doms, bounds=bounds, max_positions=max_positions, sigma=sigma,
                color_by_name=color_by_name, icon_by_name=icon_by_name, agent_icon=agent_icon,
                axes=axes, bg=bg, small_bg=small_bg
            )
            for doms in snapshots
        ]

        # 5) Save outputs (frames + legend)
        if out_dir and frames:
            os.makedirs(out_dir, exist_ok=True)
            for i, img in enumerate(frames):
                imageio.v2.imwrite(f"{out_dir.rstrip('/')}/heatmap_{i:03d}.png", img)
            imageio.mimsave(f"{out_dir.rstrip('/')}/heatmaps.gif", frames, duration=(1.0 / max(1, int(fps))))
            legend_img = self._render_icon_legend_sheet(
                obj_names, color_by_name, icon_by_name, bg=bg
            ) if (use_icons and icon_by_name) else None
            if legend_img is not None:
                imageio.v2.imwrite(f"{out_dir.rstrip('/')}/icons_legend.png", legend_img)

        return frames

    # -------- internals: IO / colors / icons --------
    def _build_colors_and_icons(
        self,
        obj_names: List[str],
        use_icons: bool,
        icons_dir: Optional[str],
        use_icon_colors: bool
    ) -> Tuple[Dict[str, str], Dict[str, Optional[np.ndarray]], Optional[np.ndarray]]:
        """Build color map, object icons, and agent icon."""
        color_by_name = _assign_identity_colors(obj_names)

        icon_by_name: Dict[str, Optional[np.ndarray]] = {}
        agent_icon = None
        if use_icons:
            idir = icons_dir or os.path.join(os.path.dirname(__file__), 'icons')
            files = sorted(glob.glob(os.path.join(idir, '*.png')))
            if files:
                # agent icon (best effort)
                agent_path = os.path.join(idir, 'agent.png')
                agent_icon = _trim_transparent(imageio.v2.imread(agent_path)) if os.path.exists(agent_path) else None
                # per-object icons
                for n in obj_names:
                    icon_by_name[n] = _load_icon(n, files)

                if use_icon_colors:
                    for n in obj_names:
                        ic = icon_by_name.get(n)
                        if ic is not None:
                            color_by_name[n] = _rgb_to_hex(_dominant_rgb(ic))

        return color_by_name, icon_by_name, agent_icon

    def _render_icon_legend_sheet(
        self,
        obj_names: List[str],
        color_by_name: Dict[str, str],
        icon_by_name: Dict[str, Optional[np.ndarray]],
        bg: float = 0.96
    ) -> Optional[np.ndarray]:
        """Standalone icon + color legend; returns image array or None."""
        if not obj_names:
            return None

        ICON_S = 60.0  # slightly larger than old main legend
        n = len(obj_names)
        cols = min(9, max(4, int(np.ceil(np.sqrt(n)))))
        rows = int(np.ceil(n / cols))

        fig = plt.figure(figsize=(min(12, 1.2 * cols), max(1.8, 0.95 * rows)), dpi=160)
        fig.patch.set_facecolor((bg, bg, bg))
        ax = fig.add_axes([0.02, 0.02, 0.96, 0.96]); ax.axis('off')

        cw, rh = 1.0 / cols, 0.50 / rows
        pad_x, pad_y = 0.06 * cw, 0.12 * rh

        for i, name in enumerate(obj_names):
            r, c = i // cols, i % cols
            x0 = c * cw + pad_x
            y0 = 1.0 - (r + 1) * rh + pad_y

            # color swatch
            ax.add_patch(patches.Rectangle((x0, y0), 0.12 * cw, 0.50 * rh,
                                           transform=ax.transAxes,
                                           color=color_by_name.get(name, '#E69F00'),
                                           ec='k', lw=0.35))

            # icon (if any)
            ic = icon_by_name.get(name) if icon_by_name else None
            if ic is not None:
                cx, cy = x0 + 0.18 * cw, y0 + 0.25 * rh
                _add_icon(ax, ic, cx, cy, size_points=ICON_S, xycoords='axes fraction')

            # label
            # ax.text(x0, y0 - 0.10 * rh, name, fontsize=8,
            #         transform=ax.transAxes, ha='left', va='top', color='#333333')

        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=fig.dpi, bbox_inches='tight')
        plt.close(fig); buf.seek(0)
        return imageio.v2.imread(buf)

    # -------- internals: observations & bounds --------
    def _collect_snapshots(self, action_results: List) -> List[Dict[str, set]]:
        """Run the spatial solver over observation-like results to produce domain snapshots."""
        names = [o.name for o in self.room.all_objects] + ['initial_pos']
        solver = SpatialSolver(names, grid_size=self.grid_size)
        solver.set_initial_position('initial_pos', (0, 0))

        snapshots: List[Dict[str, set]] = []
        for res in action_results:
            at = getattr(res, 'action_type', None)
            if at in ('observe', 'query'):
                triples = res.data.get('relation_triples', []) if hasattr(res, 'data') else []
                if triples:
                    solver.add_observation(triples)
                    rel_sets = solver.get_possible_relations()
                    for (a, b), rels in rel_sets.items():
                        if len(rels) <= 20: print(a, b, rels)
                    positions = solver.get_possible_positions()
                    for name, pos in positions.items():
                        if len(pos) <= 50: print(name, pos)
                    print("-" * 100)
                    snapshots.append(solver.get_possible_positions())
        return snapshots

    def _compute_bounds(self, snapshots: List[Dict[str, set]]) -> Tuple[float, float, float, float]:
        """Tight bounds around all valid domains; always include agent at (0,0)."""
        g = int(self.grid_size); full_size = _full_domain_size(self.grid_size)
        if not snapshots:
            return (-g, g, -g, g)

        coords = []
        for d in snapshots:
            for n, dom in d.items():
                if n == 'initial_pos' or (not dom) or len(dom) >= full_size:
                    continue
                coords.extend(list(dom))

        # Ensure the agent (assumed at origin) is always visible.
        coords.append((0, 0))

        if coords:
            xs_all = [c[0] for c in coords]; ys_all = [c[1] for c in coords]
            return (min(xs_all) - 1, max(xs_all) + 1, min(ys_all) - 1, max(ys_all) + 1)
        return (-g, g, -g, g)


    # -------- internals: rendering --------
    def _render_frame(
        self,
        domains: Dict[str, set],
        bounds: Tuple[float, float, float, float],
        max_positions: int,
        sigma: float,
        color_by_name: Dict[str, str],
        icon_by_name: Dict[str, Optional[np.ndarray]],
        agent_icon: Optional[np.ndarray],
        axes: bool,
        bg: float,
        small_bg: float
    ) -> np.ndarray:
        """
        Render one frame image for a given domains snapshot.

        Fixes implemented (minimal change):
          1) Pixel-centering for the main heatmap (and strips) by using half-cell padded
             image extents so icons/markers at integer (x, y) sit exactly over heat peaks.
          2) "Fade margin" automatically expands x/y limits slightly beyond the tight
             content bounds (clipped to the image extent) so Gaussian tails fade out
             before reaching the axes border—eliminating visible hard boundaries.
        """
        g = int(self.grid_size)
        H = W = 2 * g + 1

        # --- Pixel-centered extents: put pixel centers on integer coordinates ---
        extent_img = (-g - 0.5, g + 0.5, -g - 0.5, g + 0.5)

        fig = plt.figure(figsize=(7.0, 7.4), dpi=140)
        fig.patch.set_facecolor((bg, bg, bg))
        main_left, main_bottom, main_width = 0.08, 0.30, 0.84
        top = 0.93

        ax = fig.add_axes([main_left, main_bottom, main_width, max(0.20, top - main_bottom - 0.02)])
        ax.set_facecolor((bg, bg, bg))  # Avoid seams if view exceeds image extent
        if axes:
            ax.set_xlabel('x'); ax.set_ylabel('y')
            ax.grid(True, linestyle=':', linewidth=0.35, alpha=0.33)
        else:
            ax.set_xticks([]); ax.set_yticks([])
        ax.set_aspect('equal')
        for sp in ax.spines.values(): sp.set_visible(False)

        # Tight bounds around content then expand with a fade margin (clipped to image extent)
        x_min, x_max, y_min, y_max = bounds
        fade = max(0.5, 3.0 * max(float(sigma), 0.08))  # ~3σ fade-out; min 0.5 cell for stability
        ax.set_xlim(max(extent_img[0], x_min - fade), min(extent_img[1], x_max + fade))
        ax.set_ylim(max(extent_img[2], y_min - fade), min(extent_img[3], y_max + fade))

        # High-res mesh aligned to integer centers
        xs = np.linspace(-g, g, W); ys = np.linspace(-g, g, H)
        XX, YY = np.meshgrid(xs, ys)

        # Winner-take-all composite canvas (no color fusion on overlaps)
        canvas = np.zeros((H, W, 3), dtype=np.float32)
        best_strength = np.zeros((H, W), dtype=np.float32)
        alpha_map = np.zeros((H, W), dtype=np.float32)
        hdr_queue: List[Tuple[List[Tuple[int,int]], float, str, bool, Optional[Tuple[float,float]]]] = []

        # Appearance constants
        SIGMA_SINGLE = 0.08
        T_MULTI, T_SINGLE = 0.30, 0.93
        SHARP_MULTI, SHARP_SINGLE = 0.55, 2.10
        COUNT_REF, COUNT_GAMMA = 6.0, 0.85
        W_MIN, W_MAX = 0.90, 2.40
        R_SINGLE_MAX, R_EDGE = 0.80, 0.08

        full_size = _full_domain_size(self.grid_size)

        for obj in self.room.all_objects:
            name = obj.name
            if name == 'initial_pos':
                continue

            dom = list(domains.get(name, set()))
            if not dom or len(dom) >= full_size:
                continue
            if len(dom) > max_positions:
                dom = random.sample(dom, max_positions)
            n = max(1, len(dom))

            sigma_local = (SIGMA_SINGLE if n == 1 else sigma)
            heat_raw = _accumulate_gaussians_continuous(dom, XX, YY, sigma_local)
            s = float(heat_raw.sum())
            if s <= 0:
                continue

            heat_prob = heat_raw / s
            h = heat_prob / (float(heat_prob.max()) + 1e-12)
            t = T_SINGLE if n == 1 else T_MULTI
            h = np.clip((h - t) / (1.0 - t), 0.0, 1.0)
            h = np.power(h, (SHARP_SINGLE if n == 1 else SHARP_MULTI))

            w = (COUNT_REF / n) ** COUNT_GAMMA
            if n == 1: w *= 1.30
            w = float(np.clip(w, W_MIN, W_MAX))
            boost = np.clip(h * w, 0.0, 1.0)

            if n == 1 and len(dom) == 1:  # compact single-domain cap
                (cx, cy) = dom[0]
                rr = np.sqrt((XX - cx) ** 2 + (YY - cy) ** 2)
                cap = 1.0 / (1.0 + np.exp((rr - R_SINGLE_MAX) / max(1e-6, R_EDGE)))
                boost *= cap

            rgb = np.array(_hex_to_rgb(color_by_name.get(name, '#E69F00')), dtype=np.float32)

            # Winner-take-all write
            mask = boost > best_strength
            if np.any(mask):
                canvas[mask] = (rgb * boost[mask, None]).astype(np.float32)
                best_strength[mask] = boost[mask]
            alpha_map = np.maximum(alpha_map, boost)

            # Schedule outline
            ctr_center = dom[0] if (n == 1 and len(dom) == 1) else None
            hdr_queue.append((dom, sigma_local, color_by_name.get(name, '#E69F00'), (n == 1), ctr_center))

            # Single-domain mark (icon or dot) — now perfectly centered due to pixel-centered extent
            if n == 1:
                (x, y) = dom[0]
                ic = icon_by_name.get(name) if icon_by_name else None
                if ic is not None:
                    _add_icon(ax, ic, x, y, size_points=56.0, xycoords='data')
                else:
                    ax.scatter([x], [y], s=68, c=color_by_name.get(name, '#E69F00'),
                               marker='o', edgecolors='k', linewidths=0.4, zorder=45)

        # Slightly reduce heat opacity to help dashed outlines read clearly
        HEAT_ALPHA_FACTOR = 0.88
        alpha = np.clip(alpha_map, 0, 1) * HEAT_ALPHA_FACTOR
        final = (1.0 - alpha)[..., None] * np.array([bg, bg, bg], dtype=np.float32) + alpha[..., None] * canvas
        ax.imshow(final, origin='lower', extent=extent_img, interpolation='bicubic', zorder=0)

        # Smooth dashed outlines (thicker + halo)
        for dom, sig, color, is_single, ctr in hdr_queue:
            _draw_sketchy_outline(ax, dom, color, small=False, perturb_strength=0.05)

        # Agent mark (origin guaranteed to be included by bounds + fade)
        if agent_icon is not None:
            _add_icon(ax, agent_icon, 0, 0, size_points=70.0, xycoords='data')
        else:
            ax.scatter([0], [0], s=88, c='#000000', marker='*', linewidths=0.55, zorder=48)

        # Simple text legend if icons are disabled
        if not icon_by_name:
            handles = [patches.Patch(color=color_by_name.get(o.name, '#E69F00'), label=o.name)
                       for o in self.room.all_objects if o.name != 'initial_pos']
            by_label = {h.get_label(): h for h in handles}
            ax.legend(handles=list(by_label.values()), loc='upper right', fontsize=8, framealpha=0.85)

        # Bottom per-object strips (kept intact; apply same pixel-centering + fade to avoid edges)
        names_local = sorted({o.name for o in self.room.all_objects} | {n for n in domains.keys()})
        names_local = [n for n in names_local if n != 'initial_pos']; nobj = len(names_local)

        if nobj > 0:
            gap, height, bottom = 0.006, 0.16, 0.11
            width = (main_width - (nobj - 1) * gap) / nobj
            width = max(0.04, min(0.15, width))
            total = nobj * width + (nobj - 1) * gap
            left0 = main_left + (main_width - total) * 0.5

            STRIP_ICON_S = 60.0
            for i, name in enumerate(names_local):
                lx = left0 + i * (width + gap)
                ax_s = fig.add_axes([lx, bottom, width, height])
                ax_s.set_xticks([]); ax_s.set_yticks([])
                ax_s.set_aspect('equal')
                for sp in ax_s.spines.values(): sp.set_visible(False)

                # Background tile
                ax_s.imshow(np.full((H, W), small_bg), origin='lower', extent=extent_img, cmap='gray', vmin=0, vmax=1)
                ax_s.set_xlim(max(extent_img[0], x_min - fade), min(extent_img[1], x_max + fade))
                ax_s.set_ylim(max(extent_img[2], y_min - fade), min(extent_img[3], y_max + fade))

                dom = list(domains.get(name, set()))
                if not dom or len(dom) >= _full_domain_size(self.grid_size):
                    ic = icon_by_name.get(name) if icon_by_name else None
                    if ic is not None:
                        _add_icon(ax_s, ic, 0.5, 1.065, size_points=STRIP_ICON_S, xycoords='axes fraction')
                    else:
                        ax_s.set_title(name, fontsize=7, color=color_by_name.get(name, '#444444'), pad=1)
                    circ = patches.Ellipse(((x_min + x_max) / 2.0, (y_min + y_max) / 2.0),
                                           0.08 * (x_max - x_min), 0.08 * (y_max - y_min),
                                           fill=False, edgecolor='#C7C7C7', linestyle='--', linewidth=0.20, zorder=10)
                    ax_s.add_patch(circ); continue

                if len(dom) > max_positions:
                    dom = random.sample(dom, max_positions)

                sigma_local = (0.08 if len(dom) == 1 else sigma)
                heat = _accumulate_gaussians(np.zeros((H, W), dtype=np.float32), dom, self.grid_size, sigma=sigma_local)
                ssum = float(heat.sum())
                if ssum > 0:
                    prob = heat / ssum
                    vmax = (prob.max() + 1e-9)
                    h = np.clip(prob / vmax, 0, 1)
                    t = 0.93 if len(dom) == 1 else 0.30
                    h = np.clip((h - t) / (1.0 - t), 0, 1)
                    rgb = np.array(_hex_to_rgb(color_by_name.get(name, '#E69F00')), dtype=np.float32)
                    bg_rgb = np.array([small_bg, small_bg, small_bg], dtype=np.float32)
                    h = np.power(h, 0.55)
                    final_small = bg_rgb + (rgb - bg_rgb) * np.clip(h * 1.2, 0, 1)[..., None]
                    ax_s.imshow(final_small, origin='lower', extent=extent_img, interpolation='bicubic', zorder=0)

                _draw_sketchy_outline(ax_s, dom, color_by_name.get(name, '#E69F00'), small=True, perturb_strength=0.05)

                if len(dom) == 1:
                    (x, y) = dom[0]
                    ax_s.scatter([x], [y], s=34, c=color_by_name.get(name, '#E69F00'),
                                 marker='o', edgecolors='k', linewidths=0.33, zorder=45)
                ic = icon_by_name.get(name) if icon_by_name else None
                if ic is not None:
                    _add_icon(ax_s, ic, 0.5, 1.065, size_points=STRIP_ICON_S, xycoords='axes fraction')
                else:
                    ax_s.set_title(name, fontsize=7, color=color_by_name.get(name, '#E69F00'), pad=1)

        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=fig.dpi, bbox_inches='tight')
        plt.close(fig); buf.seek(0)
        return imageio.v2.imread(buf)





    def animate_agent_trajectory(self, action_results: List, out_path: Optional[str] = None, fps: int = 2) -> str:
        mgr = ExplorationManager(self.room, self.agent); frames: List[np.ndarray] = []
        for res in action_results:
            act = ActionSequence._parse_single_action(res.action_command) if res.action_command else None
            if act is not None: _ = mgr.execute_success_action(act)
            frame = RoomPlotter.plot_to_image(mgr.exploration_room, mgr.agent, observe=(res.action_type in ('observe',)), dpi=120)
            frames.append(frame)
        out_file = out_path or 'trajectory.gif'
        imageio.mimsave(out_file, frames, duration=(1.0 / max(1, int(fps))))
        return out_file

    @staticmethod
    def flatten_turns(turns: List) -> List:
        seq = [];  [seq.extend(list(getattr(t, 'actions', []) or [])) for t in turns];  return seq


if __name__ == "__main__":
    from ...managers.agent_proxy import OracleAgentProxy, InquisitorAgentProxy, AnalystAgentProxy, StrategistAgentProxy
    from ...utils.room_utils import RoomGenerator
    from ...core.constant import ObjectInfo
    # candidate_objects = [
    #     ObjectInfo(name='basket', has_orientation=True),
    #     ObjectInfo(name='chair', has_orientation=True),
    #     ObjectInfo(name='office-chair', has_orientation=True),
    #     ObjectInfo(name='printer', has_orientation=True),
    #     ObjectInfo(name='backpack', has_orientation=True),
    #     ObjectInfo(name='table-lamp', has_orientation=True),
    #     ObjectInfo(name='table', has_orientation=True),
    #     ObjectInfo(name='tv', has_orientation=True),
    #     ObjectInfo(name='bookshelf', has_orientation=True),
    #     ObjectInfo(name='cabinet', has_orientation=True),
    #     ObjectInfo(name='floor-lamp', has_orientation=True),
    #     # ObjectInfo(name='laptop', has_orientation=True),
    #     # ObjectInfo(name='keyboard', has_orientation=True),
    # ]
    # room, agent = RoomGenerator.generate_room(
    #     room_size=[15, 15], n_objects=5, np_random=np.random.default_rng(2),
    #     level=1, main=5, candidate_objects=candidate_objects
    # )
    # room.objects = [o for o in room.objects if o.name not in ['printer', 'table']]
    candidate_objects = [
        ObjectInfo(name='basket', has_orientation=True),
        ObjectInfo(name='chair', has_orientation=True),
        ObjectInfo(name='office-chair', has_orientation=True),
        ObjectInfo(name='printer', has_orientation=True),
        ObjectInfo(name='cabinet', has_orientation=True),
        ObjectInfo(name='table-lamp', has_orientation=True),
        ObjectInfo(name='table', has_orientation=True),
        ObjectInfo(name='tv', has_orientation=True),
        ObjectInfo(name='bookshelf', has_orientation=True),
        ObjectInfo(name='backpack', has_orientation=True),
        ObjectInfo(name='floor-lamp', has_orientation=True),
        # ObjectInfo(name='laptop', has_orientation=True),
        # ObjectInfo(name='keyboard', has_orientation=True),
    ]
    room, agent = RoomGenerator.generate_room(
        room_size=[15, 15], n_objects=3, np_random=np.random.default_rng(13),
        level=1, main=5, candidate_objects=candidate_objects
    )
    room.objects = [o for o in room.objects if o.name not in ['printer', 'table']]
    room.get_object_by_name('floor-lamp').pos = np.array([9, 6])
    # room.get_object_by_name('backpack').pos = np.array([4, 6])

    print(room, agent)
    RoomPlotter.plot(room, agent, save_path='room.png', mode='img')
    # proxy = StrategistAgentProxy(room, agent)
    proxy = AnalystAgentProxy(room, agent, delegate='observer_analyst', observer_delegate='strategist')
    proxy.run()
    print(proxy.to_text())
    action_results = ReplayHelper.flatten_turns(proxy.turns)
    replay = ReplayHelper(room, agent)
    replay.plot_observation_heatmaps(action_results, out_dir='heatmaps', fps=1, axes=True, use_icons=True, use_icon_colors=True)
    replay.animate_agent_trajectory(action_results, out_path='trajectory.gif', fps=1)