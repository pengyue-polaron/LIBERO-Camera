import argparse
import json
import math
import os
from pathlib import Path
import xml.etree.ElementTree as ET

import h5py
import numpy as np
from PIL import Image, ImageDraw
import cv2

import init_path
import libero.libero.utils.utils as libero_utils
from libero.libero.envs import *


def _as_text(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def _quat_normalize_wxyz(quat):
    quat = np.asarray(quat, dtype=np.float64)
    norm = np.linalg.norm(quat)
    if norm < 1e-12:
        raise ValueError("Quaternion norm is too small")
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


def _float_list_to_str(values):
    return " ".join(f"{float(v):.10f}" for v in values)


def _extract_camera_pose_from_xml(xml_str, camera_name):
    root = ET.fromstring(_as_text(xml_str))
    for camera in root.iter("camera"):
        if camera.get("name") != camera_name:
            continue
        pos = np.fromstring(camera.get("pos", ""), sep=" ", dtype=np.float64)
        quat = np.fromstring(camera.get("quat", ""), sep=" ", dtype=np.float64)
        if pos.shape[0] != 3 or quat.shape[0] != 4:
            raise ValueError(f"Invalid pose for camera '{camera_name}' in XML")
        return {"pos": pos, "quat": _quat_normalize_wxyz(quat)}
    raise ValueError(f"Camera '{camera_name}' not found in XML")


def _sample_camera_pose(base_pose, seed, idx, translate_range, rot_range_deg):
    rng = np.random.default_rng(seed + idx)
    delta_pos = rng.uniform(-translate_range, translate_range, size=3)
    delta_rpy_rad = np.deg2rad(
        rng.uniform(-rot_range_deg, rot_range_deg, size=3)
    )
    delta_quat = _euler_to_quat_wxyz(*delta_rpy_rad)
    quat = _quat_normalize_wxyz(
        _quat_multiply_wxyz(np.asarray(base_pose["quat"]), delta_quat)
    )
    pos = np.asarray(base_pose["pos"]) + delta_pos
    return {
        "pos": pos,
        "quat": quat,
        "delta_pos": delta_pos,
        "delta_rpy_deg": np.rad2deg(delta_rpy_rad),
    }


def _build_env_from_demo_attrs(h5_file, args):
    data_attrs = h5_file["data"].attrs

    env_info_raw = data_attrs.get("env_info")
    env_args_raw = data_attrs.get("env_args")
    problem_info_raw = data_attrs.get("problem_info")

    env_kwargs = None
    problem_name = None

    if env_info_raw not in (None, ""):
        env_kwargs = json.loads(_as_text(env_info_raw))

    if problem_info_raw not in (None, ""):
        problem_info = json.loads(_as_text(problem_info_raw))
        problem_name = problem_info.get("problem_name")

    if (env_kwargs is None or problem_name is None) and env_args_raw not in (None, ""):
        env_args = json.loads(_as_text(env_args_raw))
        if env_kwargs is None:
            env_kwargs = env_args.get("env_kwargs")
        if problem_name is None:
            problem_name = env_args.get("problem_name")

    if env_kwargs is None or problem_name is None:
        raise ValueError("Cannot recover env kwargs / problem name from HDF5 attrs")

    bddl_file_name = _as_text(data_attrs.get("bddl_file_name"))
    if not bddl_file_name and env_args_raw not in (None, ""):
        env_args = json.loads(_as_text(env_args_raw))
        bddl_file_name = env_args.get("bddl_file")
    if args.bddl_file_override:
        bddl_file_name = args.bddl_file_override

    libero_utils.update_env_kwargs(
        env_kwargs,
        bddl_file_name=bddl_file_name,
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_depths=False,
        camera_names=["agentview", "robot0_eye_in_hand"],
        camera_heights=args.image_size,
        camera_widths=args.image_size,
        camera_segmentations=None,
        reward_shaping=True,
        control_freq=20,
        ignore_done=True,
    )
    return TASK_MAPPING[problem_name](**env_kwargs), problem_name, bddl_file_name


def _capture_image_at_state(env, model_xml, state, cameras_dict):
    xml_override = _postprocess_model_xml_for_replay(model_xml, cameras_dict)

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
    obs = env._get_observations()
    return obs["agentview_image"]


def _reset_env_with_xml_and_state(env, model_xml, state, cameras_dict):
    xml_override = _postprocess_model_xml_for_replay(model_xml, cameras_dict)

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


def _postprocess_model_xml_for_replay(model_xml, cameras_dict):
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


def _open_video_writer(path, frame_shape, fps):
    h, w = frame_shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, float(fps), (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {path}")
    return writer


def _write_rgb_frame(writer, rgb_frame):
    writer.write(cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR))


def _write_video_from_hdf5_frames(frames, video_path, fps, draw_label=None):
    if len(frames) == 0:
        raise ValueError("No frames found in HDF5 dataset")

    writer = _open_video_writer(video_path, frames[0].shape, fps)
    try:
        for frame in frames:
            out = frame.copy()
            if draw_label:
                cv2.putText(
                    out,
                    draw_label,
                    (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
            _write_rgb_frame(writer, out)
        return len(frames)
    finally:
        writer.release()


def _render_video_for_action_slice(
    env,
    model_xml,
    init_state,
    actions,
    cameras_dict,
    video_path,
    fps,
    draw_label=None,
):
    obs = _reset_env_with_xml_and_state(env, model_xml, init_state, cameras_dict)
    frame0 = obs["agentview_image"]
    writer = _open_video_writer(video_path, frame0.shape, fps)
    try:
        frame = frame0.copy()
        if draw_label:
            cv2.putText(
                frame,
                draw_label,
                (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        _write_rgb_frame(writer, frame)

        for step_idx, action in enumerate(actions):
            obs, _, _, _ = env.step(action)
            frame = obs["agentview_image"].copy()
            if draw_label:
                cv2.putText(
                    frame,
                    draw_label,
                    (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
            _write_rgb_frame(writer, frame)
        return len(actions) + 1
    finally:
        writer.release()


def _make_collage(images, labels, out_path):
    pil_imgs = [Image.fromarray(img) for img in images]
    w, h = pil_imgs[0].size
    cols = min(4, len(pil_imgs))
    rows = int(math.ceil(len(pil_imgs) / cols))
    title_h = 20
    canvas = Image.new("RGB", (cols * w, rows * (h + title_h)), color=(20, 20, 20))
    draw = ImageDraw.Draw(canvas)
    for idx, (img, label) in enumerate(zip(pil_imgs, labels)):
        r = idx // cols
        c = idx % cols
        x = c * w
        y = r * (h + title_h)
        canvas.paste(img, (x, y + title_h))
        draw.text((x + 4, y + 4), label, fill=(255, 255, 255))
    canvas.save(out_path)


def main():
    parser = argparse.ArgumentParser(
        description="Replay one demo and render camera-variation screenshots or replay videos."
    )
    parser.add_argument("--demo-file", required=True)
    parser.add_argument("--bddl-file-override", type=str, default=None)
    parser.add_argument("--episode", type=str, default="demo_0")
    parser.add_argument(
        "--state-index",
        type=int,
        default=None,
        help="Index into recorded states. Default uses --action-index+1 if provided, else final state.",
    )
    parser.add_argument(
        "--action-index",
        type=int,
        default=None,
        help="Optional action index for display / selecting state (uses state[action_index+1]).",
    )
    parser.add_argument("--camera-variation-count", type=int, default=8)
    parser.add_argument("--camera-variation-seed", type=int, default=0)
    parser.add_argument("--translate-range", type=float, default=0.05)
    parser.add_argument("--rot-range-deg", type=float, default=8.0)
    parser.add_argument("--target-camera", type=str, default="agentview")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument(
        "--mode",
        choices=["screenshot", "video", "dataset-video"],
        default="video",
        help=(
            "screenshot: one state across camera variations; "
            "video: replay action sequence for each camera variation; "
            "dataset-video: strictly encode stored HDF5 obs frames into video"
        ),
    )
    parser.add_argument(
        "--obs-key",
        type=str,
        default="agentview_rgb",
        help="Only used in dataset-video mode. Observation key under /data/<episode>/obs/",
    )
    parser.add_argument("--start-action-index", type=int, default=0)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=200,
        help="Only used in video mode. Number of actions to replay. <=0 means replay all remaining actions.",
    )
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--output-dir", type=str, default="camera_variation_replay")
    parser.add_argument("--output-prefix", type=str, default=None)
    args = parser.parse_args()

    if args.mode != "dataset-video" and args.camera_variation_count <= 0:
        parser.error("--camera-variation-count must be > 0")
    if args.mode != "dataset-video" and args.target_camera != "agentview":
        parser.error("v1 only supports --target-camera agentview")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.demo_file, "r") as f:
        if args.episode not in f["data"]:
            parser.error(f"Episode '{args.episode}' not found in /data")
        ep_grp = f[f"data/{args.episode}"]
        model_xml = _as_text(ep_grp.attrs["model_file"])
        states = ep_grp["states"][()]
        actions = ep_grp["actions"][()]

        if args.state_index is not None:
            state_index = args.state_index
        elif args.action_index is not None:
            state_index = args.action_index + 1
        else:
            state_index = len(states) - 1

        if state_index < 0:
            state_index += len(states)
        if state_index < 0 or state_index >= len(states):
            parser.error(f"--state-index out of range: {state_index}, len(states)={len(states)}")

        action_msg = ""
        if args.action_index is not None:
            if args.action_index < 0 or args.action_index >= len(actions):
                parser.error(
                    f"--action-index out of range: {args.action_index}, len(actions)={len(actions)}"
                )
            action_msg = f", action[{args.action_index}]={actions[args.action_index].tolist()}"

        print(
            f"[replay] episode={args.episode}, state_index={state_index}{action_msg}, "
            f"num_states={len(states)}, num_actions={len(actions)}, mode={args.mode}"
        )

        prefix = args.output_prefix or f"{Path(args.demo_file).stem}_{args.episode}"

        if args.mode == "dataset-video":
            if args.obs_key not in ep_grp["obs"]:
                parser.error(
                    f"obs key '{args.obs_key}' not found under /data/{args.episode}/obs"
                )

            frames = ep_grp["obs"][args.obs_key][()]
            if frames.ndim != 4 or frames.shape[-1] != 3:
                parser.error(
                    f"obs/{args.obs_key} must have shape (T,H,W,3), got {frames.shape}"
                )

            out_file = output_dir / f"{prefix}_{args.obs_key}.mp4"
            n_frames = _write_video_from_hdf5_frames(
                frames=frames, video_path=out_file, fps=args.fps, draw_label=args.obs_key
            )
            print(f"[replay] saved dataset-video {out_file} ({n_frames} frames)")
            return

        base_pose = _extract_camera_pose_from_xml(model_xml, args.target_camera)
        print(
            f"[replay] base camera {args.target_camera}: "
            f"pos={base_pose['pos'].tolist()}, quat={base_pose['quat'].tolist()}"
        )

        env, problem_name, bddl_file = _build_env_from_demo_attrs(f, args)
        print(f"[replay] problem={problem_name}, bddl={bddl_file}")

        try:
            poses = []
            for idx in range(args.camera_variation_count):
                pose = _sample_camera_pose(
                    base_pose=base_pose,
                    seed=args.camera_variation_seed,
                    idx=idx,
                    translate_range=args.translate_range,
                    rot_range_deg=args.rot_range_deg,
                )
                print(
                    f"[replay] camvar_{idx:02d} "
                    f"delta_pos={pose['delta_pos'].tolist()} "
                    f"delta_rpy_deg={pose['delta_rpy_deg'].tolist()}"
                )
                poses.append(pose)

            if args.mode == "screenshot":
                images = []
                labels = []
                state = states[state_index]
                for idx, pose in enumerate(poses):
                    cameras_dict = {
                        args.target_camera: {
                            "pos": _float_list_to_str(pose["pos"]),
                            "quat": _float_list_to_str(pose["quat"]),
                        }
                    }
                    img = _capture_image_at_state(
                        env=env,
                        model_xml=model_xml,
                        state=state,
                        cameras_dict=cameras_dict,
                    )
                    images.append(img)
                    labels.append(f"camvar_{idx:02d}")
                    out_file = output_dir / f"{prefix}_camvar_{idx:02d}.png"
                    Image.fromarray(img).save(out_file)
                    print(f"[replay] saved {out_file}")

                collage_path = output_dir / f"{prefix}_collage.png"
                _make_collage(images, labels, collage_path)
                print(f"[replay] saved collage: {collage_path}")
            else:
                start_action_idx = args.start_action_index
                if start_action_idx < 0:
                    start_action_idx += len(actions)
                if start_action_idx < 0 or start_action_idx >= len(actions):
                    parser.error(
                        f"--start-action-index out of range: {start_action_idx}, len(actions)={len(actions)}"
                    )

                if len(states) < len(actions):
                    parser.error(
                        "This replay script expects len(states) >= len(actions) for video mode"
                    )

                if args.max_steps <= 0:
                    end_action_idx = len(actions)
                else:
                    end_action_idx = min(len(actions), start_action_idx + args.max_steps)
                action_slice = actions[start_action_idx:end_action_idx]
                init_state = states[start_action_idx]
                print(
                    f"[replay] video action range=[{start_action_idx}, {end_action_idx}), "
                    f"frames={len(action_slice)+1}, fps={args.fps}"
                )

                for idx, pose in enumerate(poses):
                    cameras_dict = {
                        args.target_camera: {
                            "pos": _float_list_to_str(pose["pos"]),
                            "quat": _float_list_to_str(pose["quat"]),
                        }
                    }
                    out_file = output_dir / f"{prefix}_camvar_{idx:02d}.mp4"
                    n_frames = _render_video_for_action_slice(
                        env=env,
                        model_xml=model_xml,
                        init_state=init_state,
                        actions=action_slice,
                        cameras_dict=cameras_dict,
                        video_path=out_file,
                        fps=args.fps,
                        draw_label=f"camvar_{idx:02d}",
                    )
                    print(f"[replay] saved video {out_file} ({n_frames} frames)")
        finally:
            env.close()


if __name__ == "__main__":
    main()
