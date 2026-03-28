import numpy as np
from typing import Optional, Tuple, List, Dict, Any
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from ..core.room import Room, BaseRoom
from ..core.constant import CANDIDATE_OBJECTS, ObjectInfo
from ..core.object import Object, Agent, Gate
from .generate_room_layout import generate_room_layout
from ..core.relationship import PairwiseRelationship
from ..actions.base import BaseAction

# Descriptive gate names to replace generic "door_X"
GATE_NAMES = [
    "red door", "blue door", "green door", "yellow door", "purple door",
    "orange door", "black door", "white door", "brown door", "gray door"
]


class RoomGenerator:
    """Room generator.
    1. Coordinates:
        - World coordinates are (x, y). Mask indexing is [x, y]. That is, rows map to x and columns map to y.
        - In mask indexing, row increasing means x increasing; column increasing means y increasing.
        - Neighbor convention on masks: "up" is y+1 and "down" is y-1; "right" is x+1 and "left" is x-1.
        - For visualization, we render with origin at bottom-left to keep y increasing upwards in world space.
    2. Room id:
        - Room id is an integer in [1, 99].
        - 0 for wall, 1 for main room, 100 for north-south door, 101 for east-west door.
    """
    
    @staticmethod
    def _validate_rotation_tasks(room: 'Room', agent: 'Agent', eval_tasks: list) -> bool:
        """Validate by attempting to create rotation tasks."""
        from ..evaluation.task_types import EvalTaskType
        try:
            for task in eval_tasks:
                if task.get('task_type') in ['rot', 'rot_dual']:
                    EvalTaskType.create_task(task['task_type'], np.random.default_rng(42), room, agent, task.get('task_kwargs', {})).generate_question()
            return True
        except:
            return False

    @staticmethod
    def _generate_objects_and_agent(mask, n_objects, fix_object_n, np_random, candidate_list=CANDIDATE_OBJECTS, gates= None, room_name=""):
        """Generate objects and agent for a room layout."""
        objects = RoomGenerator._gen_objects(
            n=n_objects,
            random_generator=np_random,
            room_size=[mask.shape[0], mask.shape[1]],
            perspective_taking=True,
            candidate_list=candidate_list,
            mask=mask,
            fix_object_n=fix_object_n,
        )
        
        # Find agent position
        valid_positions = RoomGenerator._get_valid_positions(mask, room_id=1)
        agent_pos = np_random.choice(valid_positions)
        while any(np.allclose(agent_pos, obj.pos) for obj in objects):
            agent_pos = np_random.choice(valid_positions)
            
        agent = Agent(name='agent', pos=agent_pos)
        agent.room_id = 1
        agent.init_room_id = 1
        
        room = Room(objects=objects, name=room_name, mask=mask.copy(), gates=gates)
        
        return room, agent
    @staticmethod
    def _default_mask(room_size: tuple[int, int]) -> np.ndarray:
        x_size, y_size = int(room_size[0]), int(room_size[1])
        # mask shape follows mask[x, y] convention: first axis = rows = x, second axis = cols = y
        mask = np.ones((x_size + 1, y_size + 1), dtype=np.int8)
        mask[[0, -1], :] = -1
        mask[:, [0, -1]] = -1
        return mask

    @staticmethod
    def _get_valid_positions(mask: np.ndarray, room_id: int | None = None) -> List[Tuple[int, int]]:
        """Get valid positions in the mask."""
        if room_id is not None:
            valid = np.argwhere(mask == int(room_id))
        else:
            valid = np.argwhere((mask >= 1) & (mask < 100))
        return [(int(pos[0]), int(pos[1])) for pos in valid]
    
    @staticmethod
    def _gen_gates_from_mask(msk: np.ndarray) -> List[Gate]:
        gates: List[Gate] = []
        h, w = msk.shape
        cnt = 0
        # vertical doors (100, go through horizontally): look up and down for room ids
        xs, ys = np.where(msk == 100)
        for x, y in zip(xs.tolist(), ys.tolist()):
            up, down = int(msk[x - 1, y]), int(msk[x + 1, y]) # NOTE up and down are with respect mask indexing
            if 1 <= up < 100 and 1 <= down < 100 and up != down:
                gate_name = GATE_NAMES[cnt % len(GATE_NAMES)] if cnt < len(GATE_NAMES) else f"door_{cnt}"
                g = Gate(
                    name=gate_name,
                    pos=np.array([x, y], dtype=int),
                    ori=np.array([1, 0], dtype=int),
                    room_id=[int(up), int(down)],
                    ori_by_room={int(up): np.array([-1, 0], dtype=int), int(down): np.array([1, 0], dtype=int)},
                )
                gates.append(g); cnt += 1
        # horizontal doors (101, go through vertically): look left and right for room ids
        xs, ys = np.where(msk == 101)
        for x, y in zip(xs.tolist(), ys.tolist()):
            left, right = int(msk[x, y - 1]), int(msk[x, y + 1])
            if 1 <= left < 100 and 1 <= right < 100 and left != right:
                gate_name = GATE_NAMES[cnt % len(GATE_NAMES)] if cnt < len(GATE_NAMES) else f"door_{cnt}"
                g = Gate(
                    name=gate_name,
                    pos=np.array([x, y], dtype=int),
                    ori=np.array([0, 1], dtype=int),
                    room_id=[int(left), int(right)],
                    ori_by_room={int(left): np.array([0, -1], dtype=int), int(right): np.array([0, 1], dtype=int)},
                )
                gates.append(g); cnt += 1
        return gates

    @staticmethod
    def generate_room(
        room_size: Tuple[int, int],
        n_objects: int,
        np_random: np.random.Generator,
        room_name: str = 'room',
        candidate_objects: List[ObjectInfo] = CANDIDATE_OBJECTS,
        level: int = 0,
        main: Optional[int] = None,
        fixed_mask: bool = False,
        **kwargs
    ) -> Tuple[Room, Agent]:
        """Generate a multi-room layout, gates, objects, and agent.
        - Mask is generated via generate_room_layout; gates derived from mask.
        - Agent is sampled from main room (room id = 1).
        - Validates layout for rotation tasks and retries if needed.
        - If fixed_mask is True, uses seed=42 for mask generation while preserving original randomness for object/agent placement.
        """
        eval_tasks = kwargs.get('eval_tasks', [])
        max_retries = kwargs.get('max_retries', 10)
        fix_room_size = kwargs.get('fix_room_size', None)
        same_room_size = kwargs.get('same_room_size', False)
        
        # Store original random state for reproducibility
        original_state = np_random.bit_generator.state
        
        # Get the original seed to generate deterministic sub-seeds
        temp_random = np.random.default_rng()
        temp_random.bit_generator.state = original_state
        base_seed = temp_random.integers(0, 2**32 - 1)
        
        for attempt in range(max_retries + 1):
            try:
                # Use deterministic sub-seed based on base seed and attempt number
                sub_seed = (base_seed + attempt * 1000007) % (2**32)  # Use prime number to avoid patterns
                attempt_random = np.random.default_rng(sub_seed)
                n = int(max(room_size[0], room_size[1]))
                # Generate layout - use seed=42 for mask if fixed_mask is True, otherwise use attempt_random
                if fixed_mask:
                    # Use fixed seed=42 for mask generation to get consistent room layout
                    mask_random = np.random.default_rng(42)
                    mask = generate_room_layout(
                        n=n, level=int(level), main=main, 
                        np_random=mask_random, fix_room_size=fix_room_size,
                        same_room_size=same_room_size
                    )
                else:
                    # Use original random behavior
                    mask = generate_room_layout(
                        n=n, level=int(level), main=main, 
                        np_random=attempt_random, fix_room_size=fix_room_size,
                        same_room_size=same_room_size
                    )

                gates = RoomGenerator._gen_gates_from_mask(mask)
                
                # Object distribution strategies
                fix_object_n = kwargs.get('fix_object_n', None)
                proportional_to_area = kwargs.get('proportional_to_area', False)
                
                if proportional_to_area and not fix_object_n:
                    # Calculate proportional distribution
                    total_area = np.sum((mask >= 1) & (mask < 100))
                    room_areas = []
                    num_rooms = level + 1
                    for room_id in range(1, num_rooms + 1):
                        room_area = np.sum(mask == room_id)
                        room_areas.append(room_area)
                    
                    # Distribute n_objects proportionally
                    if sum(room_areas) > 0:
                        proportions = [area / sum(room_areas) for area in room_areas]
                        fix_object_n = [max(1, round(n_objects * prop)) for prop in proportions]
                        # Adjust to ensure sum equals n_objects
                        diff = n_objects - sum(fix_object_n)
                        if diff != 0:
                            # Add/subtract from largest room
                            max_idx = room_areas.index(max(room_areas))
                            fix_object_n[max_idx] += diff
                
                room, agent = RoomGenerator._generate_objects_and_agent(
                    mask, n_objects, fix_object_n, attempt_random, candidate_objects, gates, room_name
                )
                
                # Validate layout for rotation tasks
                if RoomGenerator._validate_rotation_tasks(room, agent, eval_tasks):
                    return room, agent
                else:
                    if attempt == max_retries:
                        print(f"Warning: Failed to generate valid layout after {max_retries + 1} attempts. "
                              f"Using layout that may have insufficient angular separation for rotation tasks.")
                        return room, agent
                    # Continue to next attempt
                    
            except Exception as e:
                if attempt == max_retries:
                    raise e
                # Continue to next attempt
                
        # This should not be reached, but just in case
        raise RuntimeError(f"Failed to generate room after {max_retries + 1} attempts")

    @staticmethod
    def generate_base_room(
        room_size: Tuple[int, int],
        n_objects: int,
        np_random: np.random.Generator,
        room_name: str = 'room',
        candidate_objects: List[ObjectInfo] = CANDIDATE_OBJECTS,
    ) -> Tuple[BaseRoom, Agent]:
        """Generate a BaseRoom: no mask, no gates; set all room_id to 1."""
        # create a dummy mask just for sampling object positions within a bounding box
        x_size, y_size = int(room_size[0]), int(room_size[1])
        mask = RoomGenerator._default_mask((x_size, y_size))
        objects = RoomGenerator._gen_objects(
            n=n_objects,
            random_generator=np_random,
            room_size=[x_size, y_size],
            perspective_taking=True,
            candidate_list=candidate_objects,
            mask=mask,
        )
        # assign room_id=1 to all objects
        for o in objects:
            o.room_id = 1
        base = BaseRoom(objects=objects, name=room_name)
        # sample agent not colliding with objects
        def _rand_pt():
            xs = np_random.integers(0, x_size + 1)
            ys = np_random.integers(0, y_size + 1)
            return np.array([int(xs), int(ys)], dtype=int)
        agent_pos = _rand_pt()
        while any(np.allclose(agent_pos, obj.pos) for obj in objects):
            agent_pos = _rand_pt()
        agent = Agent(name='agent', pos=agent_pos)
        agent.room_id = 1
        agent.init_room_id = 1
        return base, agent
    
    @staticmethod
    def _gen_objects(
        n: int,
        random_generator: np.random.Generator,
        room_size: list[int] = [5, 5],
        perspective_taking: bool = False,
        candidate_list: list[ObjectInfo] = CANDIDATE_OBJECTS,
        mask: Optional[np.ndarray] = None,
        fix_object_n: Optional[List[int]] = None,
    ) -> List[Object]:
        """Sample objects (names, orientations) and positions from mask."""
        if mask is None:
            mask = RoomGenerator._default_mask((room_size[0], room_size[1]))
        
        objects = []
        ori_vectors = {0: [0, 1], 1: [1, 0], 2: [0, -1], 3: [-1, 0]}
        
        # Generate positions based on distribution strategy
        if fix_object_n is not None:
            positions = []
            for room_id, num_objects in enumerate(fix_object_n, start=1):
                if num_objects > 0:
                    room_positions = RoomGenerator._get_valid_positions(mask, room_id=room_id)
                    if len(room_positions) < num_objects:
                        raise ValueError(f"Room {room_id} needs {num_objects} objects but only has {len(room_positions)} positions")
                    random_generator.shuffle(room_positions)
                    positions.extend(room_positions[:num_objects])
        else:
            all_positions = RoomGenerator._get_valid_positions(mask)
            if len(all_positions) < n:
                raise ValueError(f"Need {n} objects but only {len(all_positions)} positions available")
            random_generator.shuffle(all_positions)
            positions = all_positions[:n]

        # Generate objects with selected positions
        indices = random_generator.choice(len(candidate_list), len(positions), replace=False)
        orientations = random_generator.integers(0, 4, len(positions))

        for idx, pos, ori_idx in zip(indices, positions, orientations):
            obj_info = candidate_list[idx]
            ori = np.array(ori_vectors[int(ori_idx)]) if obj_info.has_orientation and perspective_taking else np.array([0, 1])
            objects.append(Object(name=obj_info.name.replace('_', ' '), pos=np.array(pos, dtype=int), ori=ori, has_orientation=obj_info.has_orientation))
        
        return objects


class RoomPlotter:
    @staticmethod
    def plot(room: Room, agent: Agent | None, mode: str = 'text', save_path: str | None = None):
        """Render room.
        - mode='text': print ASCII.
        - mode='img': save image to save_path if provided; otherwise just draw (no return).
        Use plot_to_image() when a numpy frame is needed.
        """
        has_mask = getattr(room, 'mask', None) is not None and isinstance(room, Room)
        # gate labels: show gate index and connected rooms, e.g. G0[1-3]
        gate_labels = {}
        if room.gates:
            gate_labels = {g.name: g.name for i, g in enumerate(room.gates)}
        if mode == 'text':
            # helpers
            arrow = lambda v: { (0,1): '↑', (1,0): '→', (0,-1): '↓', (-1,0): '←' }.get(tuple(v), '•')
            col = lambda s,c: f"\033[{c}m{s}\033[0m"
            BLUE, RED, GRN = 34, 31, 32
            # grid from mask if available
            if has_mask:
                h, w = room.mask.shape  # h: number of rows (x), w: number of cols (y)
                min_x, max_x, min_y, max_y = 0, h - 1, 0, w - 1
                # Build grid as rows over world y (vertical), cols over world x (horizontal)
                grid = [[' '] * h for _ in range(w)]
                # draw background: rooms '.', walls '#', doors '+'
                for wy in range(w):
                    for wx in range(h):
                        v = int(room.mask[wx, wy])
                        if v == -1:
                            ch = ' '
                        elif v == 0:
                            ch = '#'
                        elif v in (100, 101):
                            ch = '+'
                        else:
                            ch = '·'
                        grid[w - 1 - wy][wx] = ch
                # room labels at centers
                rids = sorted(int(r) for r in np.unique(room.mask) if 1 <= int(r) < 100)
                for rid in rids:
                    xs, ys = np.where(room.mask == rid)
                    if len(xs) == 0:
                        continue
                    cx, cy = int(np.mean(xs)), int(np.mean(ys))
                    grid[w - 1 - cy][cx] = str(rid % 10)
            else:
                # bounds/grid from objects if no mask
                min_x, max_x, min_y, max_y = room.get_boundary()
                min_x, max_x, min_y, max_y = int(min_x)-1, int(max_x)+1, int(min_y)-1, int(max_y)+1
                w, h = max_x - min_x + 1, max_y - min_y + 1
                grid = [['·'] * w for _ in range(h)]
            # objects + gates with orientation
            for obj in room.all_objects:
                x, y = int(obj.pos[0]) - min_x, max_y - int(obj.pos[1])
                # grid may be (w rows, h cols) when has_mask, or (h rows, w cols) when no mask
                height, width = len(grid), len(grid[0]) if grid else (0)
                if 0 <= y < height and 0 <= x < width:
                    is_gate = isinstance(obj, Gate) if hasattr(room, 'gates') else False
                    ch = 'D' if is_gate else (arrow(obj.ori) if getattr(obj, 'has_orientation', True) else '●')
                    grid[y][x] = col(ch, RED if is_gate else BLUE)
                    if is_gate:
                        # write compact gate tag next to door cell when possible
                        tag = gate_labels.get(obj.name, '')
                        if x+1 < width:
                            grid[y][x+1] = col(tag, RED)
            # agent (current + init)
            if agent is not None:
                ax, ay = int(agent.pos[0]) - min_x, max_y - int(agent.pos[1])
                height, width = len(grid), len(grid[0]) if grid else (0)
                if 0 <= ay < height and 0 <= ax < width:
                    grid[ay][ax] = col(arrow(agent.ori), GRN)
                iax, iay = int(agent.init_pos[0]) - min_x, max_y - int(agent.init_pos[1])
                if 0 <= iay < height and 0 <= iax < width and not (iax == ax and iay == ay):
                    grid[iay][iax] = col('×', GRN)
            print(f"--- {room.name} ---")
            for yy in range(len(grid)):
                print(' '.join(grid[yy]))
            # gate legend
            if gate_labels:
                order = [f"{gate_labels[g.name]}:{g.name}" for g in room.gates]
                print("Gates:", ", ".join(order))
            if agent is not None:
                def name_ori(v):
                    m = {(0,1): 'N', (1,0): 'E', (0,-1): 'S', (-1,0): 'W'}
                    return m.get(tuple(v), '?')
                print(f"A:{tuple(agent.pos)} {name_ori(agent.ori)}  a(init):{tuple(agent.init_pos)} {name_ori(getattr(agent,'init_ori',agent.ori))}")
            return
        elif mode == 'img':

            fig, ax = plt.subplots(figsize=(6, 6))
            ax.set_facecolor('white')
            RoomPlotter._draw_img(ax, room, agent)
            if save_path:
                plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close(); return
        else:
            raise ValueError('mode must be text or img')

    @staticmethod
    def _draw_img(ax, room: Room, agent: Agent | None):
        has_mask = getattr(room, 'mask', None) is not None and isinstance(room, Room)
        gate_labels = {}
        if room.gates:
            gate_labels = {g.name: g.name for i, g in enumerate(room.gates)}
        if has_mask:
            h, w = room.mask.shape  # h: rows=x, w: cols=y
            min_x, max_x, min_y, max_y = 0, h - 1, 0, w - 1
            # build label map for background coloring
            label = np.zeros_like(room.mask, dtype=int)
            label[room.mask == 0] = 1
            label[(room.mask == 100) | (room.mask == 101)] = 2
            rids = sorted(int(r) for r in np.unique(room.mask) if 1 <= int(r) < 100)
            for i, rid in enumerate(rids, start=3):
                label[room.mask == rid] = i
            colors = ['#111111', '#888888', '#ffcc33', '#4e79a7', '#59a14f', '#f28e2b', '#e15759', '#76b7b2', '#edc949', '#af7aa1']
            # repeat colors if many rooms
            while len(colors) <= label.max():
                colors += colors[3:]
            
            # transpose so that imshow's x-axis corresponds to world x (rows)
            ax.imshow(label.T, origin='lower', cmap=ListedColormap(colors[:label.max()+1]),
                      extent=(min_x-0.5, max_x+0.5, min_y-0.5, max_y+0.5), interpolation='nearest', zorder=0, alpha=0.18)
            # room id labels
            for rid in rids:
                xs, ys = np.where(room.mask == rid)
                if len(xs) == 0:
                    continue
                cx, cy = float(np.mean(xs)), float(np.mean(ys)) # x, y
                ax.text(cx, cy, str(rid), color='white', ha='center', va='center', fontsize=8, zorder=1, alpha=0.85)
        else:
            min_x, max_x, min_y, max_y = room.get_boundary()
        # grid
        ax.set_xlim(min_x-0.5, max_x+0.5); ax.set_ylim(min_y-0.5, max_y+0.5)
        ax.set_xticks(np.arange(int(min_x), int(max_x)+1)); ax.set_yticks(np.arange(int(min_y), int(max_y)+1))
        ax.grid(True, color='#bdbdbd', linewidth=0.2)
        for s in ax.spines.values(): s.set_visible(False)
        # draw objects/gates + orientation (distinct markers/colors; legend, no text labels)
        palette = plt.get_cmap('tab10').colors
        markers = ['o','s','D','P','X','h','H','*','p','d']
        name_to_idx, seen_labels = {}, set()
        for obj in room.all_objects:
            x, y = float(obj.pos[0]), float(obj.pos[1])
            if isinstance(obj, Gate) and hasattr(room, 'gates'):
                color, marker, label = 'crimson', 'D', ('gate' if 'gate' not in seen_labels else None)
                seen_labels.add('gate')
            else:
                i = name_to_idx.setdefault(obj.name, len(name_to_idx))
                color, marker = palette[i % len(palette)], markers[i % len(markers)]
                label = obj.name if obj.name not in seen_labels else None
                seen_labels.add(obj.name)
            ax.scatter(x, y, c=[color], marker=marker, s=64, edgecolors='white', linewidths=0.7, zorder=3, label=label)
            if getattr(obj, 'has_orientation', True):
                dx, dy = float(obj.ori[0])*0.4, float(obj.ori[1])*0.4
                ax.quiver(x, y, dx, dy, angles='xy', scale_units='xy', scale=0.5, color='grey', width=0.005)
            if isinstance(obj, Gate) and gate_labels:
                ax.text(x+0.10, y+0.12, gate_labels.get(obj.name, ''), color='crimson', fontsize=8, zorder=4)
        if agent is not None:
            ax.scatter(agent.pos[0], agent.pos[1], c='green', marker='s', s=70, edgecolors='white', linewidths=0.7, label='agent', zorder=5)
            ax.scatter(agent.init_pos[0], agent.init_pos[1], c='green', marker='x', s=60, label='agent_init', zorder=4)
            dx, dy = float(agent.ori[0])*0.4, float(agent.ori[1])*0.4
            idx, idy = float(agent.init_ori[0])*0.4, float(agent.init_ori[1])*0.4
            ax.quiver(agent.pos[0], agent.pos[1], dx, dy, angles='xy', scale_units='xy', scale=0.5, color='grey', width=0.005)
            ax.quiver(agent.init_pos[0], agent.init_pos[1], idx, idy, angles='xy', scale_units='xy', scale=0.5, color='grey', width=0.005)
        h,l = ax.get_legend_handles_labels(); 
        if l:
            d = dict(zip(l,h)); ax.legend(d.values(), d.keys(), loc='upper right', frameon=False, fontsize=9)
        ax.set_aspect('equal'); ax.set_title(room.name)

    @staticmethod
    def plot_to_image(room: Room, agent: Agent | None, observe: bool = False, dpi: int = 120):
        import io, imageio
        from matplotlib import patches
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_facecolor('white')
        RoomPlotter._draw_img(ax, room, agent)
        if observe and agent is not None:
            # Draw FOV boundary lines centered on agent orientation
            fov = float(BaseAction.get_field_of_view())
            theta = float(np.arctan2(float(agent.ori[1]), float(agent.ori[0])))
            half = np.deg2rad(fov / 2.0)
            lengths = 0.9
            for ang in (theta - half, theta + half):
                dx, dy = lengths * np.cos(ang), lengths * np.sin(ang)
                ax.plot([float(agent.pos[0]), float(agent.pos[0]) + dx], [float(agent.pos[1]), float(agent.pos[1]) + dy], c='tab:green', lw=2.0, alpha=0.9)
        buf = io.BytesIO()
        plt.tight_layout(); plt.savefig(buf, format='png', dpi=int(dpi))
        plt.close(fig)
        buf.seek(0)
        return imageio.v2.imread(buf)


def get_topdown_info(room: Room, agent: Agent) -> str:
    mapping = {(0, 1): "north", (0, -1): "south", (1, 0): "east", (-1, 0): "west"}
    lines = [f"Agent at ({agent.pos[0]}, {agent.pos[1]}) facing {mapping[tuple(agent.ori)]}"]
    for obj in room.all_objects:
        if isinstance(obj, Gate):
            rs = obj.room_id if isinstance(obj.room_id, (list, tuple)) else [obj.room_id]
            rooms = [int(r) for r in rs]
            gate_ori_info = f'on east-west wall' if mapping[tuple(obj.ori)] in ('north', 'south') else f'on north-south wall'
            obj_info = f"Gate: {obj.name} at ({obj.pos[0]}, {obj.pos[1]}) {gate_ori_info}, connects rooms {rooms}"
        else:
            obj_info = f"Object: {obj.name} at ({obj.pos[0]}, {obj.pos[1]}) facing {mapping[tuple(obj.ori)]}"
        lines.append(obj_info)
    return "\n".join(lines)


def get_room_description(room: Room, agent: Agent, with_topdown: bool = False) -> str:
    # Get room information
    if hasattr(room, 'mask') and room.mask is not None:
        room_ids = [int(rid) for rid in np.unique(room.mask) if 1 <= int(rid) < 100]
        num_rooms = len(room_ids)
        room_names = [f"room {rid}" for rid in sorted(room_ids)]
    else:
        num_rooms = 1
        room_names = ["room 1"]

    room_type = f"{num_rooms} rooms connected by doors" if room.gates else "a single room"
    assert isinstance(agent.room_id, int), f"Agent room id must be an integer, got {agent.room_id}"

    # Separate objects and gates for clearer description
    # objects = [o.label + ": " + o.name + "(has facing direction)" if o.has_orientation else "" for o in room.all_objects if not isinstance(o, Gate)]
    # TODO, currently only for cognitive map
    objects = [((o.label + ": ") if o.label else "") + o.name  for o in room.all_objects if not isinstance(o, Gate)]
    # objects = [o.name for o in sorted([o for o in room.all_objects if not isinstance(o, Gate)], key=lambda x: x.label) ]
    gates = [o.name for o in room.all_objects if isinstance(o, Gate)]

    desc = f"Imagine {room_type}: {', '.join(room_names)}. You are currently in room {agent.room_id}. You face north."
    desc += f"\nObjects: {', '.join(objects)}" if objects else ""
    desc += f"\nDoors: {', '.join(gates)}" if gates else ""

    if with_topdown:
        desc += "\n" + get_topdown_info(room, agent)
    return desc

def initialize_room_from_json(json_data: Dict[str, Any]) ->  Tuple[Room, Agent]:
    """
    Initialize a Room from your metadata JSON, which now has:
      - objects: list of {oid, model, pos:{x,y,z}, rot:{x,y,z}, size:[w,h]}
      - cameras: list of {id, label, position:{x,y,z}, rotation:{y}}
      - room_size, screen_size, etc.
    """
    # Rotation to orientation vector mapping
    rotation_map = {0: np.array([0, 1]), 90: np.array([1, 0]), 180: np.array([0, -1]), 270: np.array([-1, 0])}
    offset = np.array(json_data['offset'])
    # 1) Parse all objects
    objects = []
    for obj in json_data['objects']:
        if 'door' not in obj['name']:
            objects.append(Object(
                name=obj['name'].replace('_', ' '),
                pos=np.array([obj["pos"]["x"], obj["pos"]["z"]]),
                ori=rotation_map.get(obj["rot"]["y"]) if obj["attributes"]["has_orientation"] else np.array([1, 0]),
                has_orientation=obj["attributes"]["has_orientation"],
                label=obj['label'],
            ))
        
    # 2) Room size metadata
    room_name = json_data.get("name", "room_from_json")

    # 3) Build and return
    agent_pos = [camera['position'] for camera in json_data['cameras'] if camera['id'] == 'agent'][0]
    agent_pos = np.array([agent_pos["x"], agent_pos["z"]])
    for obj in objects:
        obj.pos +=offset
    agent_pos +=offset
    mask = np.array(json_data['mask'])
    gates = RoomGenerator._gen_gates_from_mask(mask)

    # Update gate room_ids to match the connected rooms from JSON data
    door_objects = [obj for obj in json_data['objects'] if 'door' in obj['name']]
    for gate in gates:
        for door in door_objects:
            if set(gate.room_id) == set(door["attributes"]['connected_rooms']):
                gate.name = door['name'].replace('_', ' ')

    return Room(objects=objects, mask=mask, name=room_name, gates=gates), Agent(pos=agent_pos,room_id=1,init_room_id=1)


if __name__ == '__main__':
    np_random = np.random.default_rng(42)
    
    # # Test 3 object placement strategies
    # room1, _ = RoomGenerator.generate_room(room_size=[15, 15], level=2, n_objects=9, np_random=np_random)
    # room2, _ = RoomGenerator.generate_room(room_size=[15, 15], level=2, n_objects=9, proportional_to_area=True, np_random=np_random)
    # room3, _ = RoomGenerator.generate_room(room_size=[15, 15], level=2, n_objects=9, fix_object_n=[4, 3, 2], np_random=np_random)
    
    # RoomPlotter.plot(room1, None, mode='img', save_path='room1.png')
    # RoomPlotter.plot(room2, None, mode='img', save_path='room2.png')
    # RoomPlotter.plot(room3, None, mode='img', save_path='room3.png')

    # Test room layout generation
    room1, _ = RoomGenerator.generate_room(room_size=[20, 20], level=2, n_objects=9, np_random=np_random, same_room_size=True)
    
    RoomPlotter.plot(room1, None, mode='img', save_path='room1.png')
