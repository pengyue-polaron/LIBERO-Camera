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
import camera_variation_config as camvar_cfg
import camera_visibility

from libero.libero.envs import *
from libero.libero import get_libero_path


def _as_text(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def _quat_normalize_wxyz(quat):
    quat = np.asarray(quat, dtype=np.float64)
    norm = np.linalg.norm(quat)
    return quat / norm


def _extract_camera_pose_from_xml(xml_str, camera_name):
    root = ET.fromstring(_as_text(xml_str))
    for camera in root.iter("camera"):
        if camera.get("name") != camera_name:
            continue
        pos = np.fromstring(camera.get("pos", ""), sep=" ", dtype=np.float64)
        quat = np.fromstring(camera.get("quat", ""), sep=" ", dtype=np.float64)
        return pos, _quat_normalize_wxyz(quat)


def _postprocess_model_xml_for_dataset(model_xml, cameras_dict):
    xml_str = libero_utils.postprocess_model_xml(_as_text(model_xml), cameras_dict)
    root = ET.fromstring(xml_str)
    local_assets_root = os.path.join(os.getcwd(), "libero", "libero", "assets")

    for tag in ("mesh", "texture"):
        for elem in root.iter(tag):
            file_path = elem.get("file")
            if not file_path:
                continue
            normalized = file_path.replace("\\", "/")
            marker = "/assets/"
            if marker not in normalized:
                continue
            suffix = normalized.split(marker, 1)[1]
            candidate = os.path.join(local_assets_root, suffix)
            if os.path.exists(candidate):
                elem.set("file", candidate)

    return ET.tostring(root, encoding="utf8").decode("utf8")


def _resolve_eef_target_pos_for_dataset(f, ep_name, target_request, env=None):
    ep_grp = f[f"data/{ep_name}"]
    offset = np.asarray(target_request["offset"], dtype=np.float64)
    state_index = int(target_request["state_index"])

    if "obs" in ep_grp and "ee_pos" in ep_grp["obs"]:
        ee_pos = ep_grp["obs"]["ee_pos"][()]
        if len(ee_pos) == 0:
            raise ValueError(f"Episode {ep_name} has empty obs/ee_pos")
        if state_index < 0:
            state_index += len(ee_pos)
        state_index = max(0, min(state_index, len(ee_pos) - 1))
        return np.asarray(ee_pos[state_index], dtype=np.float64) + offset

    if env is None:
        raise ValueError("orbit_lookat target source 'eef_pos' requires env fallback, but env is None")

    states = ep_grp["states"][()]
    if len(states) == 0:
        raise ValueError(f"Episode {ep_name} has empty states")
    if state_index < 0:
        state_index += len(states)
    state_index = max(0, min(state_index, len(states) - 1))

    model_xml = ep_grp.attrs["model_file"]
    xml_override = _postprocess_model_xml_for_dataset(model_xml, {})

    reset_success = False
    while not reset_success:
        try:
            env.reset()
            reset_success = True
        except Exception:
            continue
    env.reset_from_xml_string(xml_override)
    env.sim.reset()
    env.sim.set_state_from_flattened(states[state_index])
    env.sim.forward()
    env._post_process()
    env._update_observables(force=True)
    obs = env._get_observations()
    if "robot0_eef_pos" not in obs:
        raise ValueError("Cannot resolve eef target position from environment observations")
    return np.asarray(obs["robot0_eef_pos"], dtype=np.float64) + offset


def _reset_env_with_xml_and_state_for_dataset(env, model_xml, state, cameras_dict):
    xml_override = _postprocess_model_xml_for_dataset(model_xml, cameras_dict)
    reset_success = False
    while not reset_success:
        try:
            env.reset()
            reset_success = True
        except Exception:
            continue
    env.reset_from_xml_string(xml_override)
    env.sim.reset()
    env.sim.set_state_from_flattened(state)
    env.sim.forward()
    env._post_process()
    env._update_observables(force=True)
    return env._get_observations()


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
    parser.add_argument("--camera-variation-config", type=str, default=None)

    args = parser.parse_args()
    camera_variation_cfg = camvar_cfg.load_camera_variation_config(args.camera_variation_config)
    effective_camera_variation_count = camvar_cfg.get_effective_count(
        args.camera_variation_count, camera_variation_cfg
    )

    if args.camera_variation_count < 0:
        parser.error("--camera-variation-count must be >= 0")
    if args.camera_variation_translate_range < 0:
        parser.error("--camera-variation-translate-range must be >= 0")
    if args.camera_variation_rot_range_deg < 0:
        parser.error("--camera-variation-rot-range-deg must be >= 0")
    if effective_camera_variation_count > 0 and not args.use_camera_obs:
        parser.error("camera variation generation requires --use-camera-obs")
    hdf5_path = args.demo_file
    f = h5py.File(hdf5_path, "r")
    env_name = f["data"].attrs.get("env", f["data"].attrs.get("env_name"))

    env_info_raw = f["data"].attrs.get("env_info")
    env_kwargs = None
    if env_info_raw not in (None, ""):
        env_kwargs = json.loads(_as_text(env_info_raw))
    else:
        env_args_raw = f["data"].attrs.get("env_args")
        if env_args_raw in (None, ""):
            raise ValueError("HDF5 missing both env_info and env_args; cannot build env")
        env_args_json = json.loads(_as_text(env_args_raw))
        env_kwargs = env_args_json["env_kwargs"]
        if env_name is None:
            env_name = env_args_json.get("env_name")

    problem_info = json.loads(_as_text(f["data"].attrs["problem_info"]))
    problem_name = problem_info["problem_name"]

    # list of all demonstrations episodes
    demos = list(f["data"].keys())

    bddl_file_name = _as_text(f["data"].attrs["bddl_file_name"])

    bddl_file_dir = os.path.dirname(bddl_file_name)

    def _get_default_hdf5_path():
        return os.path.join(
            get_libero_path("datasets"),
            bddl_file_dir.split("bddl_files/")[-1].replace(".bddl", "_demo.hdf5"),
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

    env_args = {
        "type": 1,
        "env_name": env_name,
        "problem_name": problem_name,
        "bddl_file": _as_text(f["data"].attrs["bddl_file_name"]),
        "env_kwargs": env_kwargs,
    }

    camera_variation_specs = []
    if effective_camera_variation_count > 0:
        if len(demos) == 0:
            raise ValueError("Input demo file does not contain any episodes under /data")

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

        target_pos = None
        if camvar_cfg.needs_target_pos(camera_variation_cfg):
            target_request = camvar_cfg.get_target_request(camera_variation_cfg)
            if target_request["source"] == "eef_pos":
                target_pos = _resolve_eef_target_pos_for_dataset(
                    f, demos[0], target_request, env=env
                )
            elif target_request["source"] == "fixed_world":
                if target_request["position"] is None:
                    raise ValueError("camera variation config target.position is required for fixed_world")
                target_pos = np.asarray(target_request["position"], dtype=np.float64) + np.asarray(
                    target_request["offset"], dtype=np.float64
                )
            else:
                raise ValueError(f"Unsupported target source: {target_request['source']}")
            print("[camera-variation] target_pos", target_pos.tolist())

        poses = camvar_cfg.generate_camera_variation_poses(
            base_pos=base_pos,
            base_quat=base_quat,
            count=effective_camera_variation_count,
            seed=args.camera_variation_seed,
            translate_range=args.camera_variation_translate_range,
            rot_range_deg=args.camera_variation_rot_range_deg,
            cfg=camera_variation_cfg,
            target_pos=target_pos,
            validator=camera_visibility.make_pose_validator(
                env=env,
                model_xml=first_model_xml,
                states=f["data/{}".format(demos[0])]["states"][()],
                reset_fn=_reset_env_with_xml_and_state_for_dataset,
                cfg=camera_variation_cfg,
                camera_name="agentview",
            )
            if camera_visibility.constraints_enabled(camera_variation_cfg)
            else None,
        )

        if args.camera_variation_output_dir:
            output_dir = Path(args.camera_variation_output_dir)
        else:
            output_dir = Path(_get_default_hdf5_path()).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        output_prefix = args.camera_variation_name_prefix
        if output_prefix is None:
            if args.camera_variation_output_dir:
                output_prefix = Path(args.demo_file).stem
            else:
                output_prefix = Path(_get_default_hdf5_path()).stem

        output_paths = []
        for pose in poses:
            variation_id = int(pose["variation_id"])
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
                f"strategy={pose.get('strategy', 'random_local')}",
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
        default_hdf5_path = _get_default_hdf5_path()
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

        if pose is not None:
            grp.attrs["camera_variation_enabled"] = True
            grp.attrs["camera_variation_id"] = int(pose["variation_id"])
            grp.attrs["camera_variation_seed"] = int(args.camera_variation_seed)
            grp.attrs["camera_variation_target_camera"] = "agentview"
            if camera_variation_cfg is not None:
                grp.attrs["camera_variation_config"] = json.dumps(camera_variation_cfg)
            grp.attrs["camera_variation_base_pos"] = json.dumps(base_pos.tolist())
            grp.attrs["camera_variation_base_quat"] = json.dumps(base_quat.tolist())
            grp.attrs["camera_variation_applied_pos"] = json.dumps(pose["applied_pos"].tolist())
            grp.attrs["camera_variation_applied_quat"] = json.dumps(pose["applied_quat"].tolist())
            grp.attrs["camera_variation_translate_range"] = float(args.camera_variation_translate_range)
            grp.attrs["camera_variation_rot_range_deg"] = float(args.camera_variation_rot_range_deg)

        grp.attrs["env_args"] = json.dumps(env_args)
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

            model_xml = _postprocess_model_xml_for_dataset(model_xml, cameras_dict)

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
