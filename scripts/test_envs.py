"""Quick smoke test for countdown, sokoban, frozen_lake environments."""

import traceback

def test_countdown():
    from ufb.env.countdown.env import CountdownEnv
    from ufb.env.countdown.config import CountdownEnvConfig

    config = CountdownEnvConfig()
    env = CountdownEnv(config)

    obs = env.reset(seed=42)
    print(f"  reset obs: {obs}")

    # Give a wrong answer
    obs, reward, done, info = env.step("1 + 2 + 3")
    print(f"  step(wrong): obs={obs}, reward={reward}, done={done}, info={info}")

    # Reset and try with correct numbers
    obs = env.reset(seed=42)
    data = env.data[env.index]
    print(f"  target={data['target']}, nums={data['nums']}")

    # Try a simple equation using the actual numbers
    nums = data['nums']
    equation = " + ".join(str(n) for n in nums)
    obs, reward, done, info = env.step(equation)
    print(f"  step({equation}): obs={obs}, reward={reward}, done={done}, info={info}")
    print(f"  render: {env.render()}")


def test_sokoban():
    from ufb.env.sokoban.env import SokobanEnv
    from ufb.env.sokoban.config import SokobanEnvConfig

    config = SokobanEnvConfig(dim_x=6, dim_y=6, num_boxes=1, max_steps=100)
    env = SokobanEnv(config)

    obs = env.reset(seed=42)
    print(f"  reset obs:\n{obs}")

    # Take a few steps
    for action_name, action_id in [("Up", 1), ("Down", 2), ("Left", 3), ("Right", 4)]:
        obs, reward, done, info = env.step(action_id)
        print(f"  step({action_name}): reward={reward}, done={done}, info={info}")
        if done:
            break

    print(f"  final render:\n{env.render()}")


def test_frozen_lake():
    from ufb.env.frozen_lake.env import FrozenLakeEnv
    from ufb.env.frozen_lake.config import FrozenLakeEnvConfig

    config = FrozenLakeEnvConfig(size=4, is_slippery=True)
    env = FrozenLakeEnv(config)

    obs = env.reset(seed=42)
    print(f"  reset obs:\n{obs}")

    # Take a few steps: Left=1, Down=2, Right=3, Up=4
    for action_name, action_id in [("Down", 2), ("Right", 3), ("Down", 2), ("Right", 3)]:
        obs, reward, done, info = env.step(action_id)
        print(f"  step({action_name}): reward={reward}, done={done}, info={info}")
        if done:
            break

    print(f"  final render:\n{env.render()}")


if __name__ == "__main__":
    tests = [
        ("Countdown", test_countdown),
        ("Sokoban", test_sokoban),
        ("FrozenLake", test_frozen_lake),
    ]

    for name, test_fn in tests:
        print(f"\n{'='*50}")
        print(f"Testing {name}")
        print('='*50)
        try:
            test_fn()
            print(f"[PASS] {name}")
        except Exception as e:
            print(f"[FAIL] {name}: {e}")
            traceback.print_exc()
