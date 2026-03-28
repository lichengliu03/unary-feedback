import numpy as np
from typing import Tuple, List
import random

def generate_rooms(n: int, level: int, main: int = None, seed: int = None, debug: bool = False,
                  topology: str = "tree") -> np.ndarray:
    """
    Function to generate room layout

    Args:
        n: Grid size (n x n)
        level: Complexity level, level=0 means 1 room, level=1 means 2 rooms, and so on
        main: Main room size, if specified the first room will be main×main size
        seed: Random seed
        debug: Debug mode flag
        topology: Room connection topology, options:
            - "tree": Minimum spanning tree (default, no cycles)
            - "line": Linear connection (1→2→3→4)
            - "star": Star-like connection (1→2, 1→3, 1→4)

    Returns:
        n x n numpy array where:
        - 1 to level+1: Room ID
        - 0: Wall
        - -1: Impassable area (outside rooms)
        - 100: North-south door
        - 101: East-west door
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    num_rooms = level + 1

    # Initialize grid, all as impassable area
    grid = np.full((n, n), -1, dtype=int)

    if num_rooms == 1:
        return _generate_single_room(grid, n, main)

    # Multiple attempts to generate valid room layout
    max_attempts = 100
    for attempt in range(max_attempts):
        # Reset grid
        grid = np.full((n, n), -1, dtype=int)

        # Generate room layout based on topology
        if topology == "line":
            rooms = _generate_line_room_layout(n, num_rooms, main)
        elif topology == "star":
            rooms = _generate_star_room_layout(n, num_rooms, main)
        elif topology == "tree":
            rooms = _generate_tree_room_layout(n, num_rooms, main)
        else:
            rooms = _generate_room_layout(n, num_rooms, main)
        if not rooms or len(rooms) != num_rooms:
            continue

        # Generate room connections based on topology
        connections = _generate_connections(rooms, topology)

        # Place rooms in grid
        for i, room in enumerate(rooms):
            room_id = i + 1
            x1, y1, x2, y2 = room
            grid[y1:y2+1, x1:x2+1] = room_id

        # Add walls
        _add_walls_around_rooms(grid, rooms)

        # Add doors
        doors_added = _add_doors_between_rooms(grid, rooms, connections)

        # Check if all doors were successfully added and connected
        if len(doors_added) == len(connections) and _verify_connectivity(grid, num_rooms):
            # Final check: ensure no rooms reach the boundary
            if _verify_no_rooms_at_boundary(grid, num_rooms):
                return grid

    # If multiple attempts fail, return single room
    return _generate_single_room(grid, n, main)


def generate_rooms_auto(level: int, main: int = None, seed: int = None, debug: bool = False,
                         topology: str = "tree", canvas_n: int = None) -> np.ndarray:
    """
    Automatically generate room layout: no need to explicitly pass n, returns minimal m×n mask.

    Constraints:
    - The outermost four edges should not all be -1 (cropping to minimal bounding rectangle containing non-(-1) satisfies this)
    - Walls are single-layer (using existing wall logic, cropping won't thicken walls)

    Args:
        level: Complexity (number of rooms is level+1)
        main: Main room side length (optional)
        seed: Random seed (optional)
        debug: Debug flag (optional)
        topology: Topology (tree/line/star)
        canvas_n: Internal initial canvas size (optional); if not provided, estimated based on level/main

    Returns:
        Minimally cropped m×n numpy array
    """
    # Estimate a safe initial canvas size for calling existing generate_rooms; then crop to minimal size
    num_rooms = level + 1

    def _suggest_canvas_size() -> int:
        # Rough estimation based on main room and room count, prefer larger rather than smaller, will crop later
        base_room = max(6, (main or 6))

        if topology == "line":
            # Linear layout: rooms arranged in a line, space requirement is relatively predictable
            est = base_room + (num_rooms - 1) * 7 + 6  # Each room + wall + door
        elif topology == "star":
            # Star layout: main room in center, other rooms around it, moderate space requirement
            est = base_room * 3 + 8  # Main room + surrounding rooms + walls
        else:  # tree topology
            # Tree layout: rooms randomly distributed, need more space to ensure finding suitable positions and connections
            # Add more space to improve success rate
            est = max(30, base_room * 2 + num_rooms * 8 + 10)

        return max(15, min(100, est))  # Increase minimum value and upper limit

    n = canvas_n or _suggest_canvas_size()

    # First generate with fixed canvas
    full_grid = generate_rooms(n, level, main=main, seed=seed, debug=debug, topology=topology)

    # Then crop to minimal bounding rectangle containing all non-(-1)
    cropped = _crop_to_minimal_mask(full_grid)

    # Fallback: if exception causes all to be -1, return full_grid directly
    if cropped is None:
        return full_grid

    return cropped


def _crop_to_minimal_mask(grid: np.ndarray) -> np.ndarray:
    """
    Crop grid to minimal bounding rectangle: the rectangle contains all non-(-1) cells (rooms/walls/doors).
    - Crop after four edges must contain at least one non-(-1) value (satisfies "not all -1" on edges).
    - Does not change internal wall thickness (maintains single layer).

    Returns:
        Cropped grid; if grid is all -1, returns None
    """
    h, w = grid.shape

    # Find all rows and columns with non-(-1) content
    rows_with_content = [i for i in range(h) if np.any(grid[i, :] != -1)]
    cols_with_content = [j for j in range(w) if np.any(grid[:, j] != -1)]

    if not rows_with_content or not cols_with_content:
        return None

    r1, r2 = rows_with_content[0], rows_with_content[-1]
    c1, c2 = cols_with_content[0], cols_with_content[-1]

    return grid[r1:r2+1, c1:c2+1]


def _generate_single_room(grid: np.ndarray, n: int, main: int = None) -> np.ndarray:
    """Generate single room, occupying center area"""
    # Calculate room size, ensure space for surrounding walls
    if main is not None:
        # If main parameter is specified, use main×main as room size
        room_size = main
        if room_size > n - 4:  # Ensure room is not too large
            room_size = n - 4
        if room_size < 4:  # Ensure room is not too small
            room_size = 4
    else:
        # Default room size calculation
        min_room_size = 4
        max_room_size = n - 4  # Leave space for walls
        room_size = max(min_room_size, max_room_size)

    # Calculate room position (centered)
    start = (n - room_size) // 2
    end = start + room_size

    # Ensure room doesn't reach boundary, must have space for surrounding walls
    if end >= n - 1:  # Leave at least 1 cell for wall
        end = n - 2
        start = end - room_size + 1
    if start < 1:  # Leave at least 1 cell for wall
        start = 1
        end = start + room_size - 1

    # Place room
    grid[start:end+1, start:end+1] = 1

    # Add surrounding walls
    wall_start_x = max(0, start - 1)
    wall_end_x = min(n - 1, end + 1)
    wall_start_y = max(0, start - 1)
    wall_end_y = min(n - 1, end + 1)

    # First set wall area
    grid[wall_start_y:wall_end_y+1, wall_start_x:wall_end_x+1] = 0
    # Then set room area
    grid[start:end+1, start:end+1] = 1

    return grid

def _generate_line_room_layout(n: int, num_rooms: int, main: int = None) -> List[Tuple[int, int, int, int]]:
    """
    Generate room layout for line topology:
    - Connection relationship is chain-like: 1→2→3→4→5
    - Room physical positions don't need to be in a straight line, can be more flexible (like L-shape, Z-shape, etc.)
    - Uses incremental placement: each new room only needs to be adjacent to the previous room
    """
    if num_rooms <= 0:
        return []

    if num_rooms == 1:
        # Single room, use regular layout
        return _generate_room_layout(n, num_rooms, main)

    rooms: List[Tuple[int, int, int, int]] = []

    # Main room size - fixed size, doesn't change with seed
    if main is not None:
        main_size = max(4, min(main, n - 4))
    else:
        main_size = max(6, min(8, n // 4))

    # Other room sizes - add random variation, but must be smaller than main room
    base_other_size = max(4, min(main_size - 1, n // 6))
    other_size_variation = np.random.randint(-2, 4)
    other_size = max(4, min(base_other_size + other_size_variation, main_size - 1, n // 6))

    # First room (main room) placed near center - main room is always square
    mw = mh = main_size

    # Add random offset to main room position
    center_offset_x = np.random.randint(-2, 3)
    center_offset_y = np.random.randint(-2, 3)
    cx = max(mw//2 + 2, min(n - mw//2 - 2, n // 2 + center_offset_x))
    cy = max(mh//2 + 2, min(n - mh//2 - 2, n // 2 + center_offset_y))

    x1 = max(1, cx - mw // 2)
    y1 = max(1, cy - mh // 2)
    x2 = min(n - 2, x1 + mw - 1)
    y2 = min(n - 2, y1 + mh - 1)
    # Adjust position to ensure correct size
    x1 = x2 - mw + 1
    y1 = y2 - mh + 1

    main_room = (x1, y1, x2, y2)
    rooms.append(main_room)

    # Chain placement of other rooms: each new room only adjacent to the previous room
    for room_idx in range(1, num_rooms):
        placed = False
        attempts = 0
        max_attempts = 50

        # New room only adjacent to previous room (rooms[room_idx-1])
        prev_room = rooms[room_idx-1]
        px1, py1, px2, py2 = prev_room

        while not placed and attempts < max_attempts:
            # Try placing new room in four directions around previous room
            directions = ["left", "right", "top", "bottom"]
            np.random.shuffle(directions)

            for direction in directions:
                # New room size (add larger variation and rectangle possibility), but must be smaller than main room
                size_variation = np.random.randint(-1, 4)
                base_w = max(4, min(other_size + size_variation, main_size - 1))
                base_h = max(4, min(other_size + size_variation, main_size - 1))

                # 70% probability to generate rectangular room (line topology prefers rectangles)
                if np.random.random() < 0.7:
                    if np.random.random() < 0.5:
                        # Wide room
                        rw = min(base_w + np.random.randint(1, 4), main_size - 1)
                        rh = max(4, base_h - np.random.randint(0, 2))
                    else:
                        # Tall room
                        rw = max(4, base_w - np.random.randint(0, 2))
                        rh = min(base_h + np.random.randint(1, 4), main_size - 1)
                else:
                    # Square room
                    rw = rh = base_w

                if direction == "left":
                    # New room on the left side of previous room
                    rx2 = px1 - 2  # Leave 1 cell for wall/door
                    rx1 = rx2 - rw + 1
                    # y position: overlap with previous room to place door
                    ry1 = py1 + np.random.randint(0, max(1, (py2 - py1 + 1) - rh + 1))
                    ry2 = ry1 + rh - 1

                elif direction == "right":
                    # New room on the right side of previous room
                    rx1 = px2 + 2
                    rx2 = rx1 + rw - 1
                    ry1 = py1 + np.random.randint(0, max(1, (py2 - py1 + 1) - rh + 1))
                    ry2 = ry1 + rh - 1

                elif direction == "top":
                    # New room above previous room
                    ry2 = py1 - 2
                    ry1 = ry2 - rh + 1
                    rx1 = px1 + np.random.randint(0, max(1, (px2 - px1 + 1) - rw + 1))
                    rx2 = rx1 + rw - 1

                else:  # bottom
                    # New room below previous room
                    ry1 = py2 + 2
                    ry2 = ry1 + rh - 1
                    rx1 = px1 + np.random.randint(0, max(1, (px2 - px1 + 1) - rw + 1))
                    rx2 = rx1 + rw - 1

                new_room = (rx1, ry1, rx2, ry2)

                # Check if new room is within boundaries and doesn't overlap with existing rooms
                if (rx1 >= 1 and ry1 >= 1 and rx2 <= n - 2 and ry2 <= n - 2 and
                    not _rooms_overlap_with_walls(new_room, rooms)):
                    rooms.append(new_room)
                    placed = True
                    break

            attempts += 1

        if not placed:
            # If unable to place, return already placed rooms (may be fewer than required)
            break

    return rooms


def _generate_star_room_layout(n: int, num_rooms: int, main: int = None) -> List[Tuple[int, int, int, int]]:
    """
    Generate layout for star topology:
    - The 1st room is the main room, placed in center;
    - Other rooms are placed close to the four sides of main room (left/right/top/bottom), with 1 cell reserved between them and main room (for wall/door).
    - Ensure overlap interval exists with main room in adjacent direction, to place doors (_find_door_between_rooms needs overlap).
    If placement fails, return empty list for upper layer to retry.
    """
    if num_rooms <= 0:
        return []

    rooms: List[Tuple[int, int, int, int]] = []

    # Main room size - fixed size, doesn't change with seed
    if main is not None:
        main_size = max(4, min(main, n - 4))
    else:
        main_size = max(6, min(10, n // 3))

    # Main room placed in center, leave boundary 1 for outer walls - main room is always square
    mw = mh = main_size

    # Add small random offset to main room position
    center_offset_x = np.random.randint(-2, 3)
    center_offset_y = np.random.randint(-2, 3)
    cx = max(mw//2 + 2, min(n - mw//2 - 2, n // 2 + center_offset_x))
    cy = max(mh//2 + 2, min(n - mh//2 - 2, n // 2 + center_offset_y))

    x1 = max(1, cx - mw // 2)
    y1 = max(1, cy - mh // 2)
    x2 = min(n - 2, x1 + mw - 1)
    y2 = min(n - 2, y1 + mh - 1)
    # If truncated by boundary causing size change, correct top-left
    x1 = x2 - mw + 1
    y1 = y2 - mh + 1

    main_room = (x1, y1, x2, y2)
    rooms.append(main_room)

    # Other room size baseline - add random variation, but must be smaller than main room
    base_other = max(4, min(main_size - 1, max(4, n // (num_rooms + 2))))
    other_size_variation = np.random.randint(-2, 4)
    base_other = max(4, min(base_other + other_size_variation, main_size - 1, n // (num_rooms + 2)))

    # Sides in sequential cycle: left, right, top, bottom
    sides = ["left", "right", "top", "bottom"]

    def fits(room: Tuple[int,int,int,int]) -> bool:
        x1, y1, x2, y2 = room
        if x1 < 1 or y1 < 1 or x2 > n - 2 or y2 > n - 2:
            return False
        return not _rooms_overlap_with_walls(room, rooms)

    # Try to find positions for remaining num_rooms-1 rooms around main room
    idx = 0
    attempts_per_room = 20
    for i in range(1, num_rooms):
        side = sides[idx % len(sides)]
        idx += 1

        # For vertical sides (left/right) prioritize height not exceeding main room height, to ensure overlap;
        # For horizontal sides (top/bottom) prioritize width not exceeding main room width.
        # Add random variation and rectangle possibility to room sizes
        if side in ("left", "right"):
            base_h = min(base_other, (y2 - y1 + 1))
            base_w = base_other
        else:
            base_w = min(base_other, (x2 - x1 + 1))
            base_h = base_other

        # Add rectangle variation, but must be smaller than main room
        if np.random.random() < 0.6:  # 60% probability to generate rectangle
            if side in ("left", "right"):
                # Vertical sides: can adjust width
                w = max(4, min(base_w + np.random.randint(-1, 3), main_size - 1))
                h = base_h
            else:
                # Horizontal sides: can adjust height
                w = base_w
                h = max(4, min(base_h + np.random.randint(-1, 3), main_size - 1))
        else:
            w, h = base_w, base_h

        placed = False
        # If default size doesn't fit, gradually shrink
        for size_shrink in range(0, 4):
            ww = max(4, min(w - size_shrink, main_size - 1))
            hh = max(4, min(h - size_shrink, main_size - 1))

            # Calculate initial position (aligned with main room center) and allow sliding along contact edge
            if side == "left":
                rx2 = x1 - 2  # 1 cell distance from main room's left side
                rx1 = rx2 - ww + 1
                # y center alignment
                ry1_init = y1 + ((y2 - y1 + 1) - hh) // 2
                # Sliding range: maintain overlap with main room in vertical direction
                y_min = y1
                y_max = y2 - hh + 1
                candidates = list(range(ry1_init, ry1_init + 1)) + list(range(y_min, y_max + 1))
                for ry1 in candidates:
                    ry2 = ry1 + hh - 1
                    room = (rx1, ry1, rx2, ry2)
                    if fits(room):
                        rooms.append(room)
                        placed = True
                        break

            elif side == "right":
                rx1 = x2 + 2
                rx2 = rx1 + ww - 1
                ry1_init = y1 + ((y2 - y1 + 1) - hh) // 2
                y_min = y1
                y_max = y2 - hh + 1
                candidates = list(range(ry1_init, ry1_init + 1)) + list(range(y_min, y_max + 1))
                for ry1 in candidates:
                    ry2 = ry1 + hh - 1
                    room = (rx1, ry1, rx2, ry2)
                    if fits(room):
                        rooms.append(room)
                        placed = True
                        break

            elif side == "top":
                ry2 = y1 - 2
                ry1 = ry2 - hh + 1
                rx1_init = x1 + ((x2 - x1 + 1) - ww) // 2
                x_min = x1
                x_max = x2 - ww + 1
                candidates = list(range(rx1_init, rx1_init + 1)) + list(range(x_min, x_max + 1))
                for rx1 in candidates:
                    rx2 = rx1 + ww - 1
                    room = (rx1, ry1, rx2, ry2)
                    if fits(room):
                        rooms.append(room)
                        placed = True
                        break

            elif side == "bottom":
                ry1 = y2 + 2
                ry2 = ry1 + hh - 1
                rx1_init = x1 + ((x2 - x1 + 1) - ww) // 2
                x_min = x1
                x_max = x2 - ww + 1
                candidates = list(range(rx1_init, rx1_init + 1)) + list(range(x_min, x_max + 1))
                for rx1 in candidates:
                    rx2 = rx1 + ww - 1
                    room = (rx1, ry1, rx2, ry2)
                    if fits(room):
                        rooms.append(room)
                        placed = True
                        break

            if placed:
                break

        if not placed:
            # Placement failed, return empty list for upper layer to retry
            return []

    return rooms


def _generate_tree_room_layout(n: int, num_rooms: int, main: int = None) -> List[Tuple[int, int, int, int]]:
    """
    Generate room layout for tree topology:
    - Uses incremental placement: first place main room, then place other rooms one by one, ensuring each new room is adjacent to existing rooms
    - This generated layout is naturally suitable for tree connections, because each room can be adjacent to at least one existing room
    """
    if num_rooms <= 0:
        return []

    rooms: List[Tuple[int, int, int, int]] = []

    # Main room size - fixed size, doesn't change with seed
    if main is not None:
        main_size = max(4, min(main, n - 4))
    else:
        main_size = max(6, min(8, n // 4))

    # Other room sizes - add larger random variation range, but must be smaller than main room
    base_other_size = max(4, min(main_size - 1, n // 6))
    other_size_variation = np.random.randint(-2, 4)
    other_size = max(4, min(base_other_size + other_size_variation, main_size - 1, n // 6))

    # First room (main room) placed near center - main room is always square
    mw = mh = main_size

    # Add random offset to main room position
    center_offset_x = np.random.randint(-3, 4)
    center_offset_y = np.random.randint(-3, 4)
    cx = max(mw//2 + 2, min(n - mw//2 - 2, n // 2 + center_offset_x))
    cy = max(mh//2 + 2, min(n - mh//2 - 2, n // 2 + center_offset_y))

    x1 = max(1, cx - mw // 2)
    y1 = max(1, cy - mh // 2)
    x2 = min(n - 2, x1 + mw - 1)
    y2 = min(n - 2, y1 + mh - 1)
    # Adjust position to ensure correct size
    x1 = x2 - mw + 1
    y1 = y2 - mh + 1

    main_room = (x1, y1, x2, y2)
    rooms.append(main_room)

    # Incremental placement of other rooms
    for room_idx in range(1, num_rooms):
        placed = False
        attempts = 0
        max_attempts = 50

        while not placed and attempts < max_attempts:
            # Randomly select an existing room as "neighbor"
            neighbor_idx = np.random.randint(0, len(rooms))
            neighbor = rooms[neighbor_idx]
            nx1, ny1, nx2, ny2 = neighbor

            # Try placing new room in four directions around neighbor room
            directions = ["left", "right", "top", "bottom"]
            np.random.shuffle(directions)

            for direction in directions:
                # New room size (add larger variation and rectangle possibility), but must be smaller than main room
                size_variation = np.random.randint(-1, 4)  # Expand variation range
                base_w = max(4, min(other_size + size_variation, main_size - 1))
                base_h = max(4, min(other_size + size_variation, main_size - 1))

                # 60% probability to generate rectangular room
                if np.random.random() < 0.6:
                    if np.random.random() < 0.5:
                        # Wide room
                        rw = min(base_w + np.random.randint(1, 4), main_size - 1)
                        rh = max(4, base_h - np.random.randint(0, 2))
                    else:
                        # Tall room
                        rw = max(4, base_w - np.random.randint(0, 2))
                        rh = min(base_h + np.random.randint(1, 4), main_size - 1)
                else:
                    # Square room
                    rw = rh = base_w

                if direction == "left":
                    # New room on the left side of neighbor
                    rx2 = nx1 - 2  # Leave 1 cell for wall/door
                    rx1 = rx2 - rw + 1
                    # y position: overlap with neighbor to place door
                    ry1 = ny1 + np.random.randint(0, max(1, (ny2 - ny1 + 1) - rh + 1))
                    ry2 = ry1 + rh - 1

                elif direction == "right":
                    # New room on the right side of neighbor
                    rx1 = nx2 + 2
                    rx2 = rx1 + rw - 1
                    ry1 = ny1 + np.random.randint(0, max(1, (ny2 - ny1 + 1) - rh + 1))
                    ry2 = ry1 + rh - 1

                elif direction == "top":
                    # New room above neighbor
                    ry2 = ny1 - 2
                    ry1 = ry2 - rh + 1
                    rx1 = nx1 + np.random.randint(0, max(1, (nx2 - nx1 + 1) - rw + 1))
                    rx2 = rx1 + rw - 1

                else:  # bottom
                    # New room below neighbor
                    ry1 = ny2 + 2
                    ry2 = ry1 + rh - 1
                    rx1 = nx1 + np.random.randint(0, max(1, (nx2 - nx1 + 1) - rw + 1))
                    rx2 = rx1 + rw - 1

                new_room = (rx1, ry1, rx2, ry2)

                # Check if new room is within boundaries and doesn't overlap with existing rooms
                if (rx1 >= 1 and ry1 >= 1 and rx2 <= n - 2 and ry2 <= n - 2 and
                    not _rooms_overlap_with_walls(new_room, rooms)):
                    rooms.append(new_room)
                    placed = True
                    break

            attempts += 1

        if not placed:
            # If unable to place, return already placed rooms (may be fewer than required)
            break

    return rooms

def _generate_room_layout(n: int, num_rooms: int, main: int = None) -> List[Tuple[int, int, int, int]]:
    """Generate room layout, return list of room coordinates (x1, y1, x2, y2)"""
    rooms = []

    # Adjust room size based on number of rooms, increase differentiation
    if num_rooms == 1:
        # Single room, can be larger
        min_size = max(4, n // 4)
        max_size = max(min_size, n - 4)
    else:
        # Multiple rooms, increase size differentiation
        min_size = 4
        # Dynamically adjust maximum size based on grid size and number of rooms
        if n >= 20:
            max_size = max(min_size, min(10, n // max(2, num_rooms - 1)))
        else:
            max_size = max(min_size, min(8, n // (num_rooms + 1)))

    max_attempts = 1000

    for i in range(num_rooms):
        attempts = 0
        placed = False

        while attempts < max_attempts and not placed:
            # Generate different sizes for each room, increase differentiation
            if i == 0 and main is not None:
                # First room uses specified main×main size
                width = main
                height = main
                # Ensure main room is not too large (based on grid size, not max_size)
                max_possible_size = n - 4  # Leave space for walls
                if width > max_possible_size:
                    width = max_possible_size
                if height > max_possible_size:
                    height = max_possible_size
                # Ensure main room is not too small
                if width < min_size:
                    width = min_size
                if height < min_size:
                    height = min_size
            elif num_rooms > 1:
                # Use different size preferences for different rooms, increase larger differences
                if i == 0:
                    # First room is larger (if main is not specified)
                    room_min = min_size
                    room_max = max_size
                elif i == 1:
                    # Second room is medium size
                    room_min = min_size
                    room_max = max(min_size, max_size - random.randint(1, 2))
                elif i == 2:
                    # Third room is smaller
                    room_min = min_size
                    room_max = max(min_size, max_size - random.randint(2, 3))
                else:
                    # Other rooms have random size variation
                    size_variation = random.choice([0, 1, 2, 3])
                    room_min = min_size
                    room_max = max(min_size, max_size - size_variation)

                # Generate room size, width and height can be different
                width = random.randint(room_min, room_max)
                height = random.randint(room_min, room_max)

                # Increase probability of rectangular rooms with larger differences
                if random.random() < 0.6:  # 60% probability to generate rectangle
                    if random.random() < 0.5:
                        # Increase width
                        width = min(max_size, width + random.randint(1, 3))
                    else:
                        # Increase height
                        height = min(max_size, height + random.randint(1, 3))
            else:
                # Single room case
                room_min = min_size
                room_max = max_size
                width = random.randint(room_min, room_max)
                height = random.randint(room_min, room_max)

            # Ensure room doesn't reach boundary, must have space for surrounding walls
            max_x = n - width - 1  # Leave at least 1 cell for wall
            max_y = n - height - 1  # Leave at least 1 cell for wall

            if max_x < 1 or max_y < 1:
                # Not enough space, shrink room
                width = max(min_size, n - 2)  # Leave 1 cell on each side
                height = max(min_size, n - 2)  # Leave 1 cell on each side
                max_x = n - width - 1
                max_y = n - height - 1

                if max_x < 1 or max_y < 1:
                    break

            # Randomly generate room position (starting from 1, ensure space for surrounding walls)
            x1 = random.randint(1, max_x)
            y1 = random.randint(1, max_y)
            x2 = x1 + width - 1
            y2 = y1 + height - 1

            new_room = (x1, y1, x2, y2)

            # Check if it overlaps with existing rooms
            if not _rooms_overlap_with_walls(new_room, rooms):
                rooms.append(new_room)
                placed = True

            attempts += 1

        if not placed:
            # Try to force place a smaller room
            forced_room = _force_place_small_room(n, rooms)
            if forced_room:
                rooms.append(forced_room)
            else:
                # If even forced placement fails, stop generating more rooms
                break

    return rooms

def _rooms_overlap_with_walls(new_room: Tuple[int, int, int, int],
                             existing_rooms: List[Tuple[int, int, int, int]]) -> bool:
    """Check if rooms overlap (including space for walls and doors)"""
    x1, y1, x2, y2 = new_room

    for ex1, ey1, ex2, ey2 in existing_rooms:
        # Check if they can be adjacent (allow shared walls)
        # If rooms can be adjacent, only need 1 cell spacing for wall
        wall_margin = 1
        if not (x2 + wall_margin < ex1 or x1 > ex2 + wall_margin or
                y2 + wall_margin < ey1 or y1 > ey2 + wall_margin):
            return True
    return False

def _force_place_small_room(n: int, existing_rooms: List[Tuple[int, int, int, int]]) -> Tuple[int, int, int, int]:
    """Force place a small room"""
    min_size = 4
    max_size = min(6, n // 4)

    for size in range(min_size, max_size + 1):
        for x in range(1, n - size - 1):  # Ensure room doesn't reach boundary
            for y in range(1, n - size - 1):  # Ensure room doesn't reach boundary
                room = (x, y, x + size - 1, y + size - 1)
                if not _rooms_overlap_with_walls(room, existing_rooms):
                    return room

    # Last resort: place minimum room (4x4)
    if n >= 6:  # Ensure enough space for 4x4 room plus walls
        return (1, 1, 4, 4)
    return None

def _generate_connections(rooms: List[Tuple[int, int, int, int]], topology: str = "tree") -> List[Tuple[int, int]]:
    """Generate connections between rooms based on specified topology"""
    if len(rooms) <= 1:
        return []

    if topology == "line":
        return _generate_line_connections(rooms)
    elif topology == "star":
        return _generate_star_connections(rooms)
    else:  # Default to tree topology
        return _generate_tree_connections(rooms)

def _generate_line_connections(rooms: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int]]:
    """Generate linear connections between rooms (1→2→3→4)"""
    connections = []
    for i in range(len(rooms) - 1):
        connections.append((i, i + 1))
    return connections

def _generate_star_connections(rooms: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int]]:
    """Generate star-like connections between rooms (1→2, 1→3, 1→4)"""
    connections = []
    # Connect room 0 (first room) to all other rooms
    for i in range(1, len(rooms)):
        connections.append((0, i))
    return connections

def _generate_tree_connections(rooms: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int]]:
    """Generate tree structure connections between rooms (no cycles), using minimum spanning tree algorithm"""
    if len(rooms) <= 1:
        return []

    # Calculate room center points
    centers = []
    for x1, y1, x2, y2 in rooms:
        centers.append(((x1 + x2) // 2, (y1 + y2) // 2))

    # Use Prim's algorithm to generate minimum spanning tree, ensure no cycles
    visited = [False] * len(rooms)
    visited[0] = True  # Start from first room
    connections = []

    while len(connections) < len(rooms) - 1:
        min_dist = float('inf')
        best_edge = None

        # Find shortest connection from visited rooms to unvisited rooms
        for i in range(len(rooms)):
            if not visited[i]:
                continue
            for j in range(len(rooms)):
                if visited[j]:
                    continue

                # Calculate Manhattan distance
                cx1, cy1 = centers[i]
                cx2, cy2 = centers[j]
                dist = abs(cx1 - cx2) + abs(cy1 - cy2)

                if dist < min_dist:
                    min_dist = dist
                    best_edge = (i, j)

        if best_edge:
            i, j = best_edge
            visited[j] = True
            connections.append(best_edge)
        else:
            break  # Cannot find more connections

    return connections

def _add_walls_around_rooms(grid: np.ndarray, rooms: List[Tuple[int, int, int, int]]):
    """Add walls around all rooms, rooms don't reach boundary so there's always space for walls"""
    n = grid.shape[0]

    for i, room in enumerate(rooms):
        x1, y1, x2, y2 = room
        current_room_id = i + 1

        # Since rooms don't reach boundary, can always add walls around them
        # Top wall
        for x in range(max(0, x1-1), min(n, x2+2)):
            pos_value = grid[y1-1, x]
            # Set wall in impassable area or other room boundaries
            if pos_value == -1 or (1 <= pos_value <= len(rooms) and pos_value != current_room_id):
                grid[y1-1, x] = 0

        # Bottom wall
        for x in range(max(0, x1-1), min(n, x2+2)):
            pos_value = grid[y2+1, x]
            if pos_value == -1 or (1 <= pos_value <= len(rooms) and pos_value != current_room_id):
                grid[y2+1, x] = 0

        # Left wall
        for y in range(max(0, y1-1), min(n, y2+2)):
            pos_value = grid[y, x1-1]
            if pos_value == -1 or (1 <= pos_value <= len(rooms) and pos_value != current_room_id):
                grid[y, x1-1] = 0

        # Right wall
        for y in range(max(0, y1-1), min(n, y2+2)):
            pos_value = grid[y, x2+1]
            if pos_value == -1 or (1 <= pos_value <= len(rooms) and pos_value != current_room_id):
                grid[y, x2+1] = 0

    # Second pass: ensure all rooms are surrounded by walls or doors, cannot be directly adjacent to -1
    _ensure_rooms_surrounded_by_walls_or_doors(grid, rooms)

def _ensure_rooms_surrounded_by_walls_or_doors(grid: np.ndarray, rooms: List[Tuple[int, int, int, int]]):
    """Ensure all rooms are surrounded by walls or doors, rooms cannot be directly adjacent to -1"""
    n = grid.shape[0]

    for i, room in enumerate(rooms):
        x1, y1, x2, y2 = room
        current_room_id = i + 1

        # Check each boundary position of the room
        for y in range(y1, y2 + 1):
            for x in range(x1, x2 + 1):
                if grid[y, x] == current_room_id:
                    # Check four directions of room cell
                    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # right, left, down, up

                    for dy, dx in directions:
                        ny, nx = y + dy, x + dx

                        # If adjacent position is within grid
                        if 0 <= ny < n and 0 <= nx < n:
                            neighbor_value = grid[ny, nx]

                            # If room is directly adjacent to -1, need to add wall in between
                            if neighbor_value == -1:
                                grid[ny, nx] = 0  # Set as wall

def _add_doors_between_rooms(grid: np.ndarray, rooms: List[Tuple[int, int, int, int]],
                            connections: List[Tuple[int, int]]) -> List[Tuple[int, int, int]]:
    """Add doors between connected rooms, return list of successfully added doors"""
    doors_added = []

    for room1_idx, room2_idx in connections:
        room1 = rooms[room1_idx]
        room2 = rooms[room2_idx]

        # Find door position between two rooms
        door_pos = _find_door_between_rooms(grid, room1, room2)
        if door_pos:
            x, y, door_type = door_pos
            grid[y, x] = door_type
            doors_added.append(door_pos)

    return doors_added

def _find_door_between_rooms(grid: np.ndarray, room1: Tuple[int, int, int, int],
                            room2: Tuple[int, int, int, int]) -> Tuple[int, int, int]:
    """Find door position between two rooms"""
    x1_1, y1_1, x2_1, y2_1 = room1
    x1_2, y1_2, x2_2, y2_2 = room2

    # Check horizontal adjacency (east-west door)
    if abs(x2_1 + 1 - x1_2) <= 1:  # room1 on left, room2 on right
        door_x = x2_1 + 1 if x2_1 + 1 == x1_2 else (x2_1 + x1_2) // 2
        y_overlap_start = max(y1_1, y1_2)
        y_overlap_end = min(y2_1, y2_2)

        if y_overlap_end >= y_overlap_start:
            # Avoid placing door in corner
            if y_overlap_end > y_overlap_start:
                door_y = random.randint(y_overlap_start, y_overlap_end)
            else:
                door_y = y_overlap_start

            if 0 <= door_x < grid.shape[1] and 0 <= door_y < grid.shape[0]:
                return (door_x, door_y, 101)  # East-west door

    elif abs(x2_2 + 1 - x1_1) <= 1:  # room2 on left, room1 on right
        door_x = x2_2 + 1 if x2_2 + 1 == x1_1 else (x2_2 + x1_1) // 2
        y_overlap_start = max(y1_1, y1_2)
        y_overlap_end = min(y2_1, y2_2)

        if y_overlap_end >= y_overlap_start:
            if y_overlap_end > y_overlap_start:
                door_y = random.randint(y_overlap_start, y_overlap_end)
            else:
                door_y = y_overlap_start

            if 0 <= door_x < grid.shape[1] and 0 <= door_y < grid.shape[0]:
                return (door_x, door_y, 101)  # East-west door

    # Check vertical adjacency (north-south door)
    if abs(y2_1 + 1 - y1_2) <= 1:  # room1 on top, room2 on bottom
        door_y = y2_1 + 1 if y2_1 + 1 == y1_2 else (y2_1 + y1_2) // 2
        x_overlap_start = max(x1_1, x1_2)
        x_overlap_end = min(x2_1, x2_2)

        if x_overlap_end >= x_overlap_start:
            if x_overlap_end > x_overlap_start:
                door_x = random.randint(x_overlap_start, x_overlap_end)
            else:
                door_x = x_overlap_start

            if 0 <= door_x < grid.shape[1] and 0 <= door_y < grid.shape[0]:
                return (door_x, door_y, 100)  # North-south door

    elif abs(y2_2 + 1 - y1_1) <= 1:  # room2 on top, room1 on bottom
        door_y = y2_2 + 1 if y2_2 + 1 == y1_1 else (y2_2 + y1_1) // 2
        x_overlap_start = max(x1_1, x1_2)
        x_overlap_end = min(x2_1, x2_2)

        if x_overlap_end >= x_overlap_start:
            if x_overlap_end > x_overlap_start:
                door_x = random.randint(x_overlap_start, x_overlap_end)
            else:
                door_x = x_overlap_start

            if 0 <= door_x < grid.shape[1] and 0 <= door_y < grid.shape[0]:
                return (door_x, door_y, 100)  # North-south door

    return None

def _verify_connectivity(grid: np.ndarray, num_rooms: int) -> bool:
    """Verify if all rooms are connected"""
    if num_rooms <= 1:
        return True

    # Find positions of all rooms
    room_positions = {}
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if 1 <= grid[i, j] <= num_rooms:
                room_id = grid[i, j]
                if room_id not in room_positions:
                    room_positions[room_id] = []
                room_positions[room_id].append((i, j))

    # Check if all rooms exist
    if len(room_positions) != num_rooms:
        return False

    # Use BFS to check if all other rooms can be reached from room 1
    visited_rooms = set()
    start_pos = room_positions[1][0]  # Start from first position of room 1
    queue = [start_pos]
    visited_positions = set([start_pos])

    directions = [(0, 1), (0, -1), (-1, 0), (1, 0)]  # up, down, left, right

    while queue:
        y, x = queue.pop(0)
        current_value = grid[y, x]

        # If current position is a room, record visited room
        if 1 <= current_value <= num_rooms:
            visited_rooms.add(current_value)

        # Explore adjacent positions
        for dy, dx in directions:
            ny, nx = y + dy, x + dx

            # Check boundaries
            if (0 <= ny < grid.shape[0] and 0 <= nx < grid.shape[1] and
                (ny, nx) not in visited_positions):

                next_value = grid[ny, nx]

                # Can pass through rooms or doors
                if (1 <= next_value <= num_rooms) or next_value == 100 or next_value == 101:
                    visited_positions.add((ny, nx))
                    queue.append((ny, nx))

    # Check if all rooms were visited
    return len(visited_rooms) == num_rooms

def _verify_no_rooms_at_boundary(grid: np.ndarray, num_rooms: int) -> bool:
    """Verify no rooms reach grid boundary"""
    n = grid.shape[0]

    # Check four boundaries
    for i in range(n):
        # Top and bottom boundaries
        if 1 <= grid[0, i] <= num_rooms or 1 <= grid[n-1, i] <= num_rooms:
            return False
        # Left and right boundaries
        if 1 <= grid[i, 0] <= num_rooms or 1 <= grid[i, n-1] <= num_rooms:
            return False

    return True

def _grid_to_emoji(grid: np.ndarray) -> str:
    """Convert grid to emoji display"""
    emoji_map = {
        -1: "⬛",  # Impassable area - black square
        0: "🧱",   # Wall - brick
        100: "🚪", # North-south door - door
        101: "🚪", # East-west door - door
    }

    # Rooms use different colored squares
    room_emojis = ["🟦", "🟩", "🟨", "🟪", "🟧", "🟫", "⬜", "🟥"]

    result = []
    for row in grid:
        emoji_row = []
        for val in row:
            if val == -1:
                emoji_row.append("⬛")
            elif val == 0:
                emoji_row.append("🧱")
            elif val == 100 or val == 101:
                emoji_row.append("🚪")
            elif 1 <= val <= len(room_emojis):
                emoji_row.append(room_emojis[val - 1])
            else:
                emoji_row.append("❓")  # Unknown value
        result.append("".join(emoji_row))

    return "\n".join(result)

# Test function
if __name__ == "__main__":
    # Test different topologies
    topologies = ["tree", "line", "star"]

    for topology in topologies:
        print(f"\n{'='*50}")
        print(f"🏗️  Testing {topology.upper()} topology")
        print(f"{'='*50}")

        for level in range(1, 5):  # Test levels 1-3 for multiple rooms
            print(f"\n🔸 Level {level} ({level+1} rooms) - {topology} topology")

            # Use consistent seed for comparison
            seed = 43
            main_size = 8 if level > 0 else None

            grid = generate_rooms_auto(level, main=main_size, seed=seed, topology=topology)

            if main_size:
                print(f"🎯 Specified main room size: {main_size}×{main_size}")

            unique_values = sorted(set(grid.flatten()))
            print(f"📊 Unique values in grid: {unique_values}")

            # Count rooms and doors
            room_count = sum(1 for v in unique_values if v > 0 and v < 100)
            door_count = sum(1 for v in unique_values if v >= 100)
            print(f"🏠 Actual room count: {room_count}, 🚪 Door count: {door_count}")

            # Print emoji version
            print("🎨 Emoji version:")
            print(_grid_to_emoji(grid))
