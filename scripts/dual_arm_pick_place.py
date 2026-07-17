import argparse
import csv
import time
from dataclasses import dataclass
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = REPO_ROOT / "models" / "wheeled_dual_arm" / "dual_arm_pick_place_scene.xml"

BASE_Z = 0.5595
BASE_QUAT = np.array([0.707107, -0.707107, 0.0, 0.0])
WHEEL_RADIUS = 0.0825

WAIST_HOME = {
    "joint_waist1": 0.0,
    "j_wasit2": -1.5708,
    "j_wasit3": 1.5708,
}

WAIST_CTRL = {
    "waist1_pos": 0.0,
    "waist2_pos": -1.5708,
    "waist3_pos": 1.5708,
}

LEFT_ARM = ["jl1_pos", "jl2_pos", "jl3_pos", "jl4_pos", "jl5_pos", "jl6_pos", "jl7_pos"]
RIGHT_ARM = ["jr1_pos", "jr2_pos", "jr3_pos", "jr4_pos", "jr5_pos", "jr6_pos", "jr7_pos"]
LEFT_JOINTS = [name[:-4] for name in LEFT_ARM]
RIGHT_JOINTS = [name[:-4] for name in RIGHT_ARM]

OPEN_GRIPPER = -0.014
CLOSED_GRIPPER = -0.002

PICK_POS = np.array([0.0, 0.92, 0.511])
LIFT_POS = np.array([0.0, 0.90, 0.548])
PLACE_POS = np.array([0.0, 0.96, 0.548])
PLACE_SETTLE_POS = np.array([0.0, 0.96, 0.511])

TRAY_POS = np.array([0.0, 0.905, 0.465])
TRAY_TARGET_LOCAL = np.array([0.0, 0.055, 0.046])
LEFT_GRIP_WORKPIECE_OFFSET = np.array([0.061, 0.011, -0.032])
RIGHT_GRIP_TRAY_OFFSET = np.array([-0.0055, 0.1438, -0.0369])
PLACEMENT_SUCCESS_THRESHOLD = 0.03
BASE_SUCCESS_THRESHOLD = 0.03


@dataclass(frozen=True)
class Stage:
    name: str
    duration: float
    base_y: float
    left: np.ndarray
    right: np.ndarray
    left_gripper: float
    right_gripper: float
    workpiece: np.ndarray
    workpiece_attached: bool
    tray_attached: bool


def arr(values):
    return np.array(values, dtype=float)


# Joint targets are small offsets from the launch_fixed_arms.py visual zero pose.
# They use the measured joint directions: left jl1 negative and right jr1 positive both
# move the grippers forward in world +Y. The arms stay on their own side of X=0.
DRIVE_READY_L = arr([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
DRIVE_READY_R = arr([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
APPROACH_L = arr([-0.38, 0.0, 0.0, 0.50, 0.0, 0.08, 0.0])
APPROACH_R = arr([0.38, 0.0, 0.0, -0.50, 0.0, -0.08, 0.0])
GRASP_L = arr([-0.38, 0.0, 0.0, 0.50, 0.0, 0.08, 0.0])
GRASP_R = arr([0.38, 0.0, 0.0, -0.50, 0.0, -0.08, 0.0])
LIFT_L = arr([-0.36, -0.04, 0.0, 0.46, 0.0, 0.06, 0.0])
LIFT_R = arr([0.36, 0.04, 0.0, -0.46, 0.0, -0.06, 0.0])
PLACE_L = arr([-0.459, 0.1213, 0.0, 0.8316, 0.0, 0.3195, 0.0])
TRAY_GRASP_R = arr([0.15, 0.25, 0.0, -0.2, 0.0, 0.0, 0.0])
TRAY_HOLD_R = arr([0.15, -0.05, 0.0, -0.575, 0.0, 0.15, 0.0])
PLACE_R = TRAY_HOLD_R
RELEASE_L = PLACE_L
RELEASE_R = PLACE_R
RETREAT_L = DRIVE_READY_L
RETREAT_R = DRIVE_READY_R

STAGES = [
    Stage("start", 0.5, -0.20, DRIVE_READY_L, DRIVE_READY_R, OPEN_GRIPPER, OPEN_GRIPPER, PICK_POS, False, False),
    Stage("drive_forward", 2.0, 0.25, DRIVE_READY_L, DRIVE_READY_R, OPEN_GRIPPER, OPEN_GRIPPER, PICK_POS, False, False),
    Stage("approach_objects", 1.4, 0.25, APPROACH_L, TRAY_GRASP_R, OPEN_GRIPPER, OPEN_GRIPPER, PICK_POS, False, False),
    Stage("close_grippers", 0.6, 0.25, GRASP_L, TRAY_GRASP_R, CLOSED_GRIPPER, CLOSED_GRIPPER, PICK_POS, False, False),
    Stage("lift_objects", 1.1, 0.25, LIFT_L, TRAY_HOLD_R, CLOSED_GRIPPER, CLOSED_GRIPPER, LIFT_POS, True, True),
    Stage("align_over_tray", 1.5, 0.25, PLACE_L, PLACE_R, CLOSED_GRIPPER, CLOSED_GRIPPER, PLACE_POS, True, True),
    Stage("lower_into_tray", 0.8, 0.25, PLACE_L, PLACE_R, CLOSED_GRIPPER, CLOSED_GRIPPER, PLACE_SETTLE_POS, True, True),
    Stage("release_workpiece", 0.7, 0.25, RELEASE_L, PLACE_R, OPEN_GRIPPER, CLOSED_GRIPPER, PLACE_SETTLE_POS, False, True),
    Stage("left_retreat", 1.3, 0.25, RETREAT_L, PLACE_R, OPEN_GRIPPER, CLOSED_GRIPPER, PLACE_SETTLE_POS, False, True),
]


def smoothstep(x):
    x = float(np.clip(x, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


def build_maps(model):
    joint_qpos = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i): model.jnt_qposadr[i]
        for i in range(model.njnt)
    }
    actuator_id = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i): i
        for i in range(model.nu)
    }
    return joint_qpos, actuator_id


def site_pos(model, data, name):
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    return data.site_xpos[site_id].copy()


def set_base_pose(model, data, joint_qpos, actuator_id, base_y, wheel_angle):
    qadr = joint_qpos["base_footprint_joint"]
    data.qpos[qadr : qadr + 3] = [0.0, base_y, BASE_Z]
    data.qpos[qadr + 3 : qadr + 7] = BASE_QUAT
    dofadr = model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "base_footprint_joint")]
    data.qvel[dofadr : dofadr + 6] = 0.0

    for joint_name in ("joint_left_wheel", "joint_right_wheel"):
        data.qpos[joint_qpos[joint_name]] = wheel_angle
    data.ctrl[actuator_id["left_wheel_motor"]] = 0.0
    data.ctrl[actuator_id["right_wheel_motor"]] = 0.0


def set_joint_pose(data, joint_qpos, actuator_id, left, right, left_gripper, right_gripper):
    for joint_name, value in WAIST_HOME.items():
        data.qpos[joint_qpos[joint_name]] = value
    for actuator_name, value in WAIST_CTRL.items():
        data.ctrl[actuator_id[actuator_name]] = value

    for actuator_name, joint_name, value in zip(LEFT_ARM, LEFT_JOINTS, left):
        data.ctrl[actuator_id[actuator_name]] = value
        data.qpos[joint_qpos[joint_name]] = value
    for actuator_name, joint_name, value in zip(RIGHT_ARM, RIGHT_JOINTS, right):
        data.ctrl[actuator_id[actuator_name]] = value
        data.qpos[joint_qpos[joint_name]] = value

    for actuator_name in ("left_finger_r_pos", "left_finger_l_pos"):
        data.ctrl[actuator_id[actuator_name]] = left_gripper
    for joint_name in ("joint_left_finger_r", "joint_left_finger_l"):
        data.qpos[joint_qpos[joint_name]] = left_gripper
    for actuator_name in ("right_finger_r_pos", "right_finger_l_pos"):
        data.ctrl[actuator_id[actuator_name]] = right_gripper
    for joint_name in ("joint_right_finger_r", "joint_right_finger_l"):
        data.qpos[joint_qpos[joint_name]] = right_gripper


def set_workpiece_pose(model, data, joint_qpos, pos):
    qadr = joint_qpos["workpiece_free"]
    data.qpos[qadr : qadr + 3] = pos
    data.qpos[qadr + 3 : qadr + 7] = [1.0, 0.0, 0.0, 0.0]
    dofadr = model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "workpiece_free")]
    data.qvel[dofadr : dofadr + 6] = 0.0


def set_tray_pose(model, data, joint_qpos, pos):
    qadr = joint_qpos["tray_free"]
    data.qpos[qadr : qadr + 3] = pos
    data.qpos[qadr + 3 : qadr + 7] = [1.0, 0.0, 0.0, 0.0]
    dofadr = model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "tray_free")]
    data.qvel[dofadr : dofadr + 6] = 0.0


def sample_stage(t):
    elapsed = 0.0
    for i in range(1, len(STAGES)):
        prev = STAGES[i - 1]
        curr = STAGES[i]
        if t <= elapsed + curr.duration:
            alpha = smoothstep((t - elapsed) / curr.duration)
            base_y = (1.0 - alpha) * prev.base_y + alpha * curr.base_y
            left = (1.0 - alpha) * prev.left + alpha * curr.left
            right = (1.0 - alpha) * prev.right + alpha * curr.right
            left_gripper = (1.0 - alpha) * prev.left_gripper + alpha * curr.left_gripper
            right_gripper = (1.0 - alpha) * prev.right_gripper + alpha * curr.right_gripper
            workpiece = (1.0 - alpha) * prev.workpiece + alpha * curr.workpiece
            return curr.name, base_y, left, right, left_gripper, right_gripper, workpiece, curr.workpiece_attached, curr.tray_attached
        elapsed += curr.duration
    last = STAGES[-1]
    return last.name, last.base_y, last.left, last.right, last.left_gripper, last.right_gripper, last.workpiece, last.workpiece_attached, last.tray_attached


def make_log_row(model, data, joint_qpos, t, stage_name, tray_target):
    base_qadr = joint_qpos["base_footprint_joint"]
    tray_qadr = joint_qpos["tray_free"]
    workpiece_qadr = joint_qpos["workpiece_free"]

    base_pos = data.qpos[base_qadr : base_qadr + 3].copy()
    tray_pos = data.qpos[tray_qadr : tray_qadr + 3].copy()
    workpiece_pos = data.qpos[workpiece_qadr : workpiece_qadr + 3].copy()
    left_grip = site_pos(model, data, "left_grip_center")
    right_grip = site_pos(model, data, "right_grip_center")
    placement_error = float(np.linalg.norm(workpiece_pos - tray_target))

    row = {
        "time": t,
        "stage": stage_name,
        "base_x": base_pos[0],
        "base_y": base_pos[1],
        "base_z": base_pos[2],
        "left_gripper_x": left_grip[0],
        "left_gripper_y": left_grip[1],
        "left_gripper_z": left_grip[2],
        "right_gripper_x": right_grip[0],
        "right_gripper_y": right_grip[1],
        "right_gripper_z": right_grip[2],
        "tray_x": tray_pos[0],
        "tray_y": tray_pos[1],
        "tray_z": tray_pos[2],
        "workpiece_x": workpiece_pos[0],
        "workpiece_y": workpiece_pos[1],
        "workpiece_z": workpiece_pos[2],
        "target_x": tray_target[0],
        "target_y": tray_target[1],
        "target_z": tray_target[2],
        "placement_error": placement_error,
    }
    return row


def summarize_rows(rows):
    if not rows:
        return {
            "success": False,
            "final_placement_error": float("inf"),
            "max_workpiece_height": 0.0,
            "final_base_error": float("inf"),
            "final_stage": "",
        }

    final = rows[-1]
    final_placement_error = float(final["placement_error"])
    final_base_error = abs(float(final["base_y"]) - STAGES[-1].base_y)
    max_workpiece_height = max(float(row["workpiece_z"]) for row in rows)
    success = (
        final_placement_error <= PLACEMENT_SUCCESS_THRESHOLD
        and final_base_error <= BASE_SUCCESS_THRESHOLD
        and final["stage"] == STAGES[-1].name
    )
    return {
        "success": success,
        "final_placement_error": final_placement_error,
        "max_workpiece_height": max_workpiece_height,
        "final_base_error": final_base_error,
        "final_stage": final["stage"],
    }


def write_log(log_file, rows):
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run(headless=False, realtime=1.0, log_file=None, log_every=10):
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    joint_qpos, actuator_id = build_maps(model)
    total_time = sum(stage.duration for stage in STAGES[1:])
    rows = []
    step_count = 0
    log_every = max(1, log_every)

    def step_once(t):
        nonlocal step_count
        stage_name, base_y, left, right, left_gripper, right_gripper, workpiece, attached, tray_attached = sample_stage(t)
        wheel_angle = -(base_y - STAGES[0].base_y) / WHEEL_RADIUS
        set_base_pose(model, data, joint_qpos, actuator_id, base_y, wheel_angle)
        set_joint_pose(data, joint_qpos, actuator_id, left, right, left_gripper, right_gripper)
        mujoco.mj_forward(model, data)

        left_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "left_grip_center")
        right_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_grip_center")
        tray_pos = data.site_xpos[right_site] + RIGHT_GRIP_TRAY_OFFSET if tray_attached else TRAY_POS
        tray_target = tray_pos + TRAY_TARGET_LOCAL

        if attached:
            workpiece = data.site_xpos[left_site] + LEFT_GRIP_WORKPIECE_OFFSET
        elif stage_name in ("lower_into_tray", "release_workpiece", "left_retreat"):
            workpiece = tray_target

        set_tray_pose(model, data, joint_qpos, tray_pos)
        set_workpiece_pose(model, data, joint_qpos, workpiece)
        mujoco.mj_forward(model, data)
        mujoco.mj_step(model, data)
        step_count += 1

        should_log = (headless or log_file is not None) and (step_count % log_every == 0 or t >= total_time)
        if should_log:
            rows.append(make_log_row(model, data, joint_qpos, t, stage_name, tray_target))

    if headless:
        t = 0.0
        while t <= total_time:
            step_once(t)
            t += model.opt.timestep
        if log_file is not None and rows:
            write_log(Path(log_file), rows)
        return summarize_rows(rows)

    print("Wheeled Dual-Arm Pick-and-Place demo")
    print("Model:", MODEL_PATH)
    print("Sequence: the tray starts on the table; the robot drives forward, closes the left gripper on the workpiece and the right gripper on the tray, lifts both, then inserts the workpiece into the tray hole.")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        start = time.time()
        while viewer.is_running():
            sim_t = ((time.time() - start) * realtime) % total_time
            step_once(sim_t)
            viewer.sync()
            time.sleep(model.opt.timestep)

    if log_file is not None and rows:
        write_log(Path(log_file), rows)
    return summarize_rows(rows)


def main():
    parser = argparse.ArgumentParser(description="Wheeled Dual-Arm Pick-and-Place demo.")
    parser.add_argument("--headless", action="store_true", help="Run one sequence without opening the viewer.")
    parser.add_argument("--realtime", type=float, default=1.0, help="Playback speed multiplier for the viewer.")
    parser.add_argument("--log-file", type=Path, help="Write per-step diagnostics to a CSV file.")
    parser.add_argument("--log-every", type=int, default=10, help="Record one row every N simulation steps.")
    args = parser.parse_args()
    result = run(headless=args.headless, realtime=args.realtime, log_file=args.log_file, log_every=args.log_every)
    if args.headless:
        print(
            "headless_ok success={success} final_placement_error={final_placement_error:.4f} "
            "max_workpiece_height={max_workpiece_height:.4f} final_base_error={final_base_error:.4f}".format(
                **result
            )
        )


if __name__ == "__main__":
    main()
