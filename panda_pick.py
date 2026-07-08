"""Run a small robosuite Panda Lift demo.

This script is intentionally simple: it is a smoke test for the environment and
an entry point for learning how actions, observations, rewards, and rendering
fit together in robosuite.
"""

import argparse
import time



def parse_args():
    parser = argparse.ArgumentParser(description="Run a robosuite Panda Lift demo.")
    parser.add_argument("--steps", type=int, default=500, help="Number of control steps to run.")
    parser.add_argument(
        "--print-every",
        type=int,
        default=50,
        help="Print observation and reward information every N steps.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.02,
        help="Seconds to sleep after each step when rendering.",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Run without the interactive robosuite renderer.",
    )
    parser.add_argument(
        "--reset-on-done",
        action="store_true",
        help="Reset and continue if the environment reports done before --steps.",
    )
    return parser.parse_args()


def make_env(has_renderer):
    import robosuite as suite

    return suite.make(
        env_name="Lift",
        robots="Panda",
        has_renderer=has_renderer,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        control_freq=20,
    )


def scripted_action(action_dim, step):
    """Return a simple hand-written action for smoke testing.

    Panda's default action uses arm control dimensions followed by one gripper
    command. This action is not a real pick policy; it just verifies that control
    commands change the simulation.
    """
    import numpy as np

    action = np.zeros(action_dim)

    if step > 100:
        action[-1] = -1.0

    if 200 < step < 300:
        action[0] = 0.2

    return action


def print_step_summary(step, obs, reward):
    print(f"\nStep {step}")

    if "robot0_eef_pos" in obs:
        print("EEF position:", obs["robot0_eef_pos"])

    if "cube_pos" in obs:
        print("Cube position:", obs["cube_pos"])

    print("Reward:", reward)


def main():
    args = parse_args()
    env = make_env(has_renderer=not args.no_render)

    try:
        obs = env.reset()

        print("==============================")
        print("Environment loaded")
        print("Action dimension:", env.action_dim)
        print("Render enabled:", not args.no_render)
        print("Steps:", args.steps)
        print("==============================")

        print("\nObservation keys:")
        for key in obs.keys():
            print("-", key)

        for step in range(args.steps):
            action = scripted_action(env.action_dim, step)
            obs, reward, done, info = env.step(action)

            if not args.no_render:
                env.render()

            if args.print_every > 0 and step % args.print_every == 0:
                print_step_summary(step, obs, reward)

            if done:
                print(f"\nEnvironment returned done at step {step}.")
                if not args.reset_on_done:
                    break
                obs = env.reset()

            if not args.no_render and args.sleep > 0:
                time.sleep(args.sleep)
    finally:
        env.close()


if __name__ == "__main__":
    main()
