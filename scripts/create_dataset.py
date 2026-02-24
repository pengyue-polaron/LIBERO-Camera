import argparse
import os
from pathlib import Path
import h5py
import numpy as np
import json
import math
import xml.etree.ElementTree as ET
import robosuite
import robosuite.utils.transform_utils as T
import robosuite.macros as macros

import init_path
import libero.libero.utils.utils as libero_utils
import cv2
from PIL import Image
from robosuite.utils import camera_utils

from libero.libero.envs import *
from libero.libero import get_libero_path


def _quat_normalize_wxyz(quat):
    quat = np.asarray(quat, dtype=np.float64)
    norm = np.linalg.norm(quat)
    return quat / norm


def _quat_multiply_wxyz(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def _euler_to_quat_wxyz(roll, pitch, yaw):
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    return np.array(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ],
        dtype=np.float64,
    )


def _extract_camera_pose_from_xml(xml_str, camera_name):
    if isinstance(xml_str, bytes):
        xml_str = xml_str.decode("utf-8")
    root = ET.fromstring(xml_str)
    for camera in root.iter("camera"):
        if camera.get("name") != camera_name:
            continue
        pos = np.fromstring(camera.get("pos", ""), sep=" ", dtype=np.float64)
        quat = np.fromstring(camera.get("quat", ""), sep=" ", dtype=np.float64)
        return pos, _quat_normalize_wxyz(quat)


def _sample_camera_variation_pose(base_pos, base_quat, seed, variation_id, translate_range, rot_range_deg):
    rng = np.random.default_rng(seed + variation_id)
    delta_pos = rng.uniform(-translate_range, translate_range, size=3)
    delta_rpy_deg = rng.uniform(-rot_range_deg, rot_range_deg, size=3)
    delta_quat = _euler_to_quat_wxyz(*np.deg2rad(delta_rpy_deg))
    pos = np.asarray(base_pos, dtype=np.float64) + delta_pos
    quat = _quat_normalize_wxyz(
        _quat_multiply_wxyz(np.asarray(base_quat, dtype=np.float64), delta_quat)
    )
    return {
        "variation_id": variation_id,
        "delta_pos": delta_pos,
        "delta_rpy_deg": delta_rpy_deg,
        "applied_pos": pos,
        "applied_quat": quat,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo-file", default="demo.hdf5")

    parser.add_argument(
        "--use-actions",
        action="store_true",
    )
    parser.add_argument("--use-camera-obs", action="store_true")
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="datasets/",
    )

    parser.add_argument(
        "--dataset-name",
        type=str,
        default="training_set",
    )

    parser.add_argument("--no-proprio", action="store_true")

    parser.add_argument(
        "--use-depth",
        action="store_true",
    )

    parser.add_argument("--camera-variation-count", type=int, default=0)
    parser.add_argument("--camera-variation-seed", type=int, default=0)
    parser.add_argument("--camera-variation-translate-range", type=float, default=0.05)
    parser.add_argument("--camera-variation-rot-range-deg", type=float, default=8.0)
    parser.add_argument("--camera-variation-output-dir", type=str, default=None)
    parser.add_argument("--camera-variation-name-prefix", type=str, default=None)

    args = parser.parse_args()

    if args.camera_variation_count < 0:
        parser.error("--camera-variation-count must be >= 0")
    if args.camera_variation_translate_range < 0:
        parser.error("--camera-variation-translate-range must be >= 0")
    if args.camera_variation_rot_range_deg < 0:
        parser.error("--camera-variation-rot-range-deg must be >= 0")
    if args.camera_variation_count > 0 and not args.use_camera_obs:
        parser.error("--camera-variation-count > 0 requires --use-camera-obs")
    hdf5_path = args.demo_file
    f = h5py.File(hdf5_path, "r")
    env_name = f["data"].attrs["env"]

    env_args = f["data"].attrs["env_info"]
    env_kwargs = json.loads(f["data"].attrs["env_info"])

    problem_info = json.loads(f["data"].attrs["problem_info"])
    problem_info["domain_name"]
    problem_name = problem_info["problem_name"]
    language_instruction = problem_info["language_instruction"]

    # list of all demonstrations episodes
    demos = list(f["data"].keys())

    bddl_file_name = f["data"].attrs["bddl_file_name"]

    bddl_file_dir = os.path.dirname(bddl_file_name)
    replace_bddl_prefix = "/".join(bddl_file_dir.split("bddl_files/")[:-1] + "bddl_files")

    default_hdf5_path = os.path.join(get_libero_path("datasets"), bddl_file_dir.split("bddl_files/")[-1].replace(".bddl", "_demo.hdf5"))

    camera_variation_specs = []
    if args.camera_variation_count > 0:

        first_model_xml = f["data/{}".format(demos[0])].attrs["model_file"]
        base_pos, base_quat = _extract_camera_pose_from_xml(
            first_model_xml, "agentview"
        )
        print(
            "[camera-variation] base pose",
            "camera=agentview",
            f"pos={base_pos.tolist()}",
            f"quat={base_quat.tolist()}",
        )

        output_dir = Path(args.camera_variation_output_dir) if args.camera_variation_output_dir else Path(default_hdf5_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        output_prefix = args.camera_variation_name_prefix or Path(default_hdf5_path).stem

        output_paths = []
        for variation_id in range(args.camera_variation_count):
            pose = _sample_camera_variation_pose(
                base_pos=base_pos,
                base_quat=base_quat,
                seed=args.camera_variation_seed,
                variation_id=variation_id,
                translate_range=args.camera_variation_translate_range,
                rot_range_deg=args.camera_variation_rot_range_deg,
            )
            output_path = output_dir / f"{output_prefix}_camvar_{variation_id:02d}.hdf5"
            output_paths.append(output_path)
            camera_variation_specs.append(
                {
                    "output_path": str(output_path),
                    "cameras_dict": {
                        "agentview": {
                            "pos": " ".join(
                                f"{float(v):.10f}" for v in pose["applied_pos"]
                            ),
                            "quat": " ".join(
                                f"{float(v):.10f}" for v in pose["applied_quat"]
                            ),
                        }
                    },
                    "pose": pose,
                    "base_pos": base_pos,
                    "base_quat": base_quat,
                }
            )
            print(
                "[camera-variation]",
                f"id={variation_id}",
                f"pos={pose['applied_pos'].tolist()}",
                f"quat={pose['applied_quat'].tolist()}",
                f"delta_pos={pose['delta_pos'].tolist()}",
                f"delta_rpy_deg={pose['delta_rpy_deg'].tolist()}",
            )

        collisions = [str(path) for path in output_paths if path.exists()]
        if collisions:
            raise FileExistsError(
                "Refusing to overwrite existing camera variation files:\n" + "\n".join(collisions)
            )
    else:
        output_parent_dir = Path(default_hdf5_path).parent
        output_parent_dir.mkdir(parents=True, exist_ok=True)
        camera_variation_specs.append(
            {
                "output_path": default_hdf5_path,
                "cameras_dict": {},
                "pose": None,
                "base_pos": None,
                "base_quat": None,
            }
        )

    libero_utils.update_env_kwargs(
        env_kwargs,
        bddl_file_name=bddl_file_name,
        has_renderer=not args.use_camera_obs,
        has_offscreen_renderer=args.use_camera_obs,
        ignore_done=True,
        use_camera_obs=args.use_camera_obs,
        camera_depths=args.use_depth,
        camera_names=[
            "robot0_eye_in_hand",
            "agentview",
        ],
        reward_shaping=True,
        control_freq=20,
        camera_heights=128,
        camera_widths=128,
        camera_segmentations=None,
    )

    env = TASK_MAPPING[problem_name](
        **env_kwargs,
    )

    for camera_variation_spec in camera_variation_specs:
        hdf5_path = camera_variation_spec["output_path"]
        cameras_dict = camera_variation_spec["cameras_dict"]
        pose = camera_variation_spec["pose"]
        base_pos = camera_variation_spec["base_pos"]
        base_quat = camera_variation_spec["base_quat"]

        h5py_f = h5py.File(hdf5_path, "w")

        grp = h5py_f.create_group("data")

        grp.attrs["env_name"] = env_name
        grp.attrs["problem_info"] = f["data"].attrs["problem_info"]
        grp.attrs["macros_image_convention"] = macros.IMAGE_CONVENTION

        grp.attrs["bddl_file_name"] = bddl_file_name
        grp.attrs["bddl_file_content"] = open(bddl_file_name, "r").read()
        print(grp.attrs["bddl_file_content"])

        if pose is not None:
            grp.attrs["camera_variation_enabled"] = True
            grp.attrs["camera_variation_id"] = int(pose["variation_id"])
            grp.attrs["camera_variation_seed"] = int(args.camera_variation_seed)
            grp.attrs["camera_variation_target_camera"] = "agentview"
            grp.attrs["camera_variation_base_pos"] = json.dumps(base_pos.tolist())
            grp.attrs["camera_variation_base_quat"] = json.dumps(base_quat.tolist())
            grp.attrs["camera_variation_applied_pos"] = json.dumps(pose["applied_pos"].tolist())
            grp.attrs["camera_variation_applied_quat"] = json.dumps(pose["applied_quat"].tolist())
            grp.attrs["camera_variation_translate_range"] = float(args.camera_variation_translate_range)
            grp.attrs["camera_variation_rot_range_deg"] = float(args.camera_variation_rot_range_deg)

        env_args = {
            "type": 1,
            "env_name": env_name,
            "problem_name": problem_name,
            "bddl_file": f["data"].attrs["bddl_file_name"],
            "env_kwargs": env_kwargs,
        }

        grp.attrs["env_args"] = json.dumps(env_args)
        print(grp.attrs["env_args"])
        total_len = 0
        demos = demos

        cap_index = 5

        for (i, ep) in enumerate(demos):
            print("Playing back random episode... (press ESC to quit)")

            # # select an episode randomly
            # read the model xml, using the metadata stored in the attribute for this episode
            model_xml = f["data/{}".format(ep)].attrs["model_file"]
            reset_success = False
            while not reset_success:
                try:
                    env.reset()
                    reset_success = True
                except:
                    continue

            if isinstance(model_xml, bytes):
                model_xml = model_xml.decode("utf-8")
            model_xml = libero_utils.postprocess_model_xml(model_xml, cameras_dict)

            if not args.use_camera_obs:
                env.viewer.set_camera(0)

            # load the flattened mujoco states
            states = f["data/{}/states".format(ep)][()]
            actions = np.array(f["data/{}/actions".format(ep)][()])

            num_actions = actions.shape[0]

            init_idx = 0
            env.reset_from_xml_string(model_xml)
            env.sim.reset()
            env.sim.set_state_from_flattened(states[init_idx])
            env.sim.forward()
            model_xml = env.sim.model.get_xml()

            ee_states = []
            gripper_states = []
            joint_states = []
            robot_states = []

            agentview_images = []
            eye_in_hand_images = []

            agentview_depths = []
            eye_in_hand_depths = []

            agentview_seg = {0: [], 1: [], 2: [], 3: [], 4: []}

            rewards = []
            dones = []

            valid_index = []

            for j, action in enumerate(actions):

                obs, reward, done, info = env.step(action)

                if j < num_actions - 1:
                    # ensure that the actions deterministically lead to the same recorded states
                    state_playback = env.sim.get_state().flatten()
                    # assert(np.all(np.equal(states[j + 1], state_playback)))
                    err = np.linalg.norm(states[j + 1] - state_playback)

                    if err > 0.01:
                        print(
                            f"[warning] playback diverged by {err:.2f} for ep {ep} at step {j}"
                        )

                # Skip recording because the force sensor is not stable in
                # the beginning
                if j < cap_index:
                    continue

                valid_index.append(j)

                if not args.no_proprio:
                    if "robot0_gripper_qpos" in obs:
                        gripper_states.append(obs["robot0_gripper_qpos"])

                    joint_states.append(obs["robot0_joint_pos"])

                    ee_states.append(
                        np.hstack(
                            (
                                obs["robot0_eef_pos"],
                                T.quat2axisangle(obs["robot0_eef_quat"]),
                            )
                        )
                    )

                robot_states.append(env.get_robot_state_vector(obs))

                if args.use_camera_obs:

                    if args.use_depth:
                        agentview_depths.append(obs["agentview_depth"])
                        eye_in_hand_depths.append(obs["robot0_eye_in_hand_depth"])

                    agentview_images.append(obs["agentview_image"])
                    eye_in_hand_images.append(obs["robot0_eye_in_hand_image"])
                else:
                    env.render()

            # end of one trajectory
            states = states[valid_index]
            actions = actions[valid_index]
            dones = np.zeros(len(actions)).astype(np.uint8)
            dones[-1] = 1
            rewards = np.zeros(len(actions)).astype(np.uint8)
            rewards[-1] = 1
            print(len(actions), len(agentview_images))
            assert len(actions) == len(agentview_images)
            print(len(actions))

            ep_data_grp = grp.create_group(f"demo_{i}")

            obs_grp = ep_data_grp.create_group("obs")
            if not args.no_proprio:
                obs_grp.create_dataset(
                    "gripper_states", data=np.stack(gripper_states, axis=0)
                )
                obs_grp.create_dataset("joint_states", data=np.stack(joint_states, axis=0))
                obs_grp.create_dataset("ee_states", data=np.stack(ee_states, axis=0))
                obs_grp.create_dataset("ee_pos", data=np.stack(ee_states, axis=0)[:, :3])
                obs_grp.create_dataset("ee_ori", data=np.stack(ee_states, axis=0)[:, 3:])

            obs_grp.create_dataset("agentview_rgb", data=np.stack(agentview_images, axis=0))
            obs_grp.create_dataset(
                "eye_in_hand_rgb", data=np.stack(eye_in_hand_images, axis=0)
            )
            if args.use_depth:
                obs_grp.create_dataset(
                    "agentview_depth", data=np.stack(agentview_depths, axis=0)
                )
                obs_grp.create_dataset(
                    "eye_in_hand_depth", data=np.stack(eye_in_hand_depths, axis=0)
                )

            ep_data_grp.create_dataset("actions", data=actions)
            ep_data_grp.create_dataset("states", data=states)
            ep_data_grp.create_dataset("robot_states", data=np.stack(robot_states, axis=0))
            ep_data_grp.create_dataset("rewards", data=rewards)
            ep_data_grp.create_dataset("dones", data=dones)
            ep_data_grp.attrs["num_samples"] = len(agentview_images)
            ep_data_grp.attrs["model_file"] = model_xml
            ep_data_grp.attrs["init_state"] = states[init_idx]
            total_len += len(agentview_images)

        grp.attrs["num_demos"] = len(demos)
        grp.attrs["total"] = total_len

        h5py_f.close()

        print("The created dataset is saved in the following path: ")
        print(hdf5_path)

    env.close()
    f.close()


if __name__ == "__main__":
    main()
