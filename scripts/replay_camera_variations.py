import argparse
import json
import math
import os
import re
from pathlib import Path
import xml.etree.ElementTree as ET

import h5py
import numpy as np
from PIL import Image, ImageDraw
import cv2
import robosuite.macros as macros

import init_path
import libero.libero.utils.utils as libero_utils
from libero.libero.envs import *
import camera_variation_config as camvar_cfg
import camera_visibility


def _as_text(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def _needs_vertical_flip(image_convention):
    if image_convention is None:
        return False
    s = _as_text(image_convention)
    if s is None:
        return False
    return str(s).strip().lower() == "opengl"


def _maybe_flip_rgb_vertical(rgb_image, flip_vertical):
    if not flip_vertical:
        return rgb_image
    return np.ascontiguousarray(np.flipud(rgb_image))


def _quat_normalize_wxyz(quat):
    quat = np.asarray(quat, dtype=np.float64)
    norm = np.linalg.norm(quat)
    if norm < 1e-12:
        raise ValueError("Quaternion norm is too small")
    return quat / norm


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


def _resolve_target_pos_for_replay(ep_grp, states, env, model_xml, target_request):
    source = target_request["source"]
    offset = np.asarray(target_request["offset"], dtype=np.float64)
    state_index = int(target_request["state_index"])

    if source == "fixed_world":
        if target_request["position"] is None:
            raise ValueError("camera variation config target.position is required for fixed_world")
        return np.asarray(target_request["position"], dtype=np.float64) + offset

    if source != "eef_pos":
        raise ValueError(f"Unsupported target source: {source}")

    if "obs" in ep_grp and "ee_pos" in ep_grp["obs"]:
        ee_pos = ep_grp["obs"]["ee_pos"][()]
        if len(ee_pos) == 0:
            raise ValueError("Episode obs/ee_pos is empty")
        if state_index < 0:
            state_index += len(ee_pos)
        state_index = max(0, min(state_index, len(ee_pos) - 1))
        return np.asarray(ee_pos[state_index], dtype=np.float64) + offset

    if state_index < 0:
        state_index += len(states)
    state_index = max(0, min(state_index, len(states) - 1))
    obs = _reset_env_with_xml_and_state(env, model_xml, states[state_index], {})
    if "robot0_eef_pos" not in obs:
        raise ValueError("Cannot resolve eef target position from environment observations")
    return np.asarray(obs["robot0_eef_pos"], dtype=np.float64) + offset


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


def _write_video_from_hdf5_frames(frames, video_path, fps, draw_label=None, flip_vertical=False):
    if len(frames) == 0:
        raise ValueError("No frames found in HDF5 dataset")

    writer = _open_video_writer(video_path, frames[0].shape, fps)
    try:
        for frame in frames:
            out = _maybe_flip_rgb_vertical(frame, flip_vertical).copy()
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


def _project_world_points_to_pixels(sim, camera_name, world_points, image_h, image_w):
    cam_id = sim.model.camera_name2id(camera_name)
    cam_pos = np.asarray(sim.data.cam_xpos[cam_id], dtype=np.float64)
    cam_rot = np.asarray(sim.data.cam_xmat[cam_id], dtype=np.float64).reshape(3, 3)

    rel = np.asarray(world_points, dtype=np.float64) - cam_pos[None, :]
    cam_pts = rel @ cam_rot  # world -> camera (R^T * rel), cam_rot rows are camera axes in world

    fovy_deg = float(sim.model.cam_fovy[cam_id])
    fy = 0.5 * float(image_h) / math.tan(math.radians(fovy_deg) * 0.5)
    fx = fy
    cx = (float(image_w) - 1.0) * 0.5
    cy = (float(image_h) - 1.0) * 0.5

    pixels = []
    visible = True
    for x_cam, y_cam, z_cam in cam_pts:
        depth = -z_cam  # MuJoCo camera looks along -Z
        if depth <= 1e-6:
            visible = False
            pixels.append(None)
            continue
        u = fx * (x_cam / depth) + cx
        v = cy - fy * (y_cam / depth)
        pixels.append((int(round(u)), int(round(v))))
    return pixels, visible


def _get_site_corners_world(sim, site_name):
    try:
        site_id = sim.model.site_name2id(site_name)
    except Exception:
        return None

    size = np.asarray(sim.model.site_size[site_id], dtype=np.float64)
    pos = np.asarray(sim.data.site_xpos[site_id], dtype=np.float64)
    rot = np.asarray(sim.data.site_xmat[site_id], dtype=np.float64).reshape(3, 3)

    hx = float(size[0]) if len(size) > 0 else 0.0
    hy = float(size[1]) if len(size) > 1 else 0.0
    hz = float(size[2]) if len(size) > 2 else 0.0
    z = hz
    local = np.array(
        [
            [-hx, -hy, z],
            [hx, -hy, z],
            [hx, hy, z],
            [-hx, hy, z],
        ],
        dtype=np.float64,
    )
    world = pos[None, :] + local @ rot.T
    return world


def _draw_bddl_region_overlay(env, rgb_image, camera_name="agentview", draw_labels=True):
    img = rgb_image.copy()
    parsed_problem = getattr(env, "parsed_problem", None)
    if not parsed_problem or "regions" not in parsed_problem:
        return img

    region_dict = parsed_problem["regions"]
    goal_state = parsed_problem.get("goal_state", [])
    h, w = img.shape[:2]
    colors = [
        (255, 80, 80),
        (80, 220, 80),
        (80, 160, 255),
        (255, 200, 80),
        (220, 80, 220),
        (80, 220, 220),
    ]

    def collect_goal_regions(node, out):
        if isinstance(node, (list, tuple)):
            for item in node:
                collect_goal_regions(item, out)
        elif isinstance(node, str) and node in region_dict:
            out.add(node)

    goal_regions = set()
    collect_goal_regions(goal_state, goal_regions)

    color_idx = 0
    for region_name, region_info in region_dict.items():
        world_corners = _get_site_corners_world(env.sim, region_name)
        if world_corners is None:
            continue

        pixels, visible = _project_world_points_to_pixels(
            env.sim, camera_name=camera_name, world_points=world_corners, image_h=h, image_w=w
        )
        if not visible or any(p is None for p in pixels):
            continue

        pts = np.array(pixels, dtype=np.int32).reshape(-1, 1, 2)
        is_goal_region = region_name in goal_regions
        color = (255, 255, 0) if is_goal_region else colors[color_idx % len(colors)]
        thickness = 3 if is_goal_region else 2
        color_idx += 1
        cv2.polylines(
            img, [pts], isClosed=True, color=color, thickness=thickness, lineType=cv2.LINE_AA
        )

        if draw_labels:
            label_pt = pts[0, 0].tolist()
            label = region_name
            if is_goal_region:
                label = f"{region_name} (goal)"
            cv2.putText(
                img,
                label,
                (int(label_pt[0]) + 4, int(label_pt[1]) - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                color,
                1,
                cv2.LINE_AA,
            )
    return img


def _render_video_for_action_slice(
    env,
    model_xml,
    init_state,
    actions,
    cameras_dict,
    video_path,
    fps,
    draw_label=None,
    flip_vertical=False,
    draw_bddl_regions=False,
    hide_region_labels=False,
    camera_name="agentview",
):
    obs = _reset_env_with_xml_and_state(env, model_xml, init_state, cameras_dict)
    frame0 = _maybe_flip_rgb_vertical(obs["agentview_image"], flip_vertical)
    if draw_bddl_regions:
        frame0 = _draw_bddl_region_overlay(
            env, frame0, camera_name=camera_name, draw_labels=not hide_region_labels
        )
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
            frame = _maybe_flip_rgb_vertical(obs["agentview_image"], flip_vertical).copy()
            if draw_bddl_regions:
                frame = _draw_bddl_region_overlay(
                    env, frame, camera_name=camera_name, draw_labels=not hide_region_labels
                )
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


_CAMVAR_FILE_RE = re.compile(
    r"^(?P<base>.+)_camvar_(?P<id>\d+)(?:_(?P<uid>[A-Za-z0-9]+))?\.hdf5$"
)


def _collect_variation_groups(root_dir):
    root_dir = Path(root_dir)
    groups = {}
    for path in root_dir.rglob("*.hdf5"):
        m = _CAMVAR_FILE_RE.match(path.name)
        if not m:
            continue
        rel_parent = path.parent.relative_to(root_dir)
        key = (str(rel_parent), m.group("base"))
        groups.setdefault(key, []).append((int(m.group("id")), m.group("uid"), path))
    for key in groups:
        groups[key] = sorted(groups[key], key=lambda x: (x[0], x[2].name))
    return groups


def _extract_hdf5_rgb_frame(hdf5_path, episode, obs_key, frame_index):
    with h5py.File(hdf5_path, "r") as f:
        if "data" not in f or episode not in f["data"]:
            raise KeyError(f"{hdf5_path}: episode '{episode}' not found")
        ep_grp = f[f"data/{episode}"]
        if "obs" not in ep_grp or obs_key not in ep_grp["obs"]:
            raise KeyError(f"{hdf5_path}: obs key '{obs_key}' not found in episode '{episode}'")
        frames = ep_grp["obs"][obs_key]
        if frames.ndim != 4 or frames.shape[-1] != 3:
            raise ValueError(f"{hdf5_path}: obs/{obs_key} shape {frames.shape} is not (T,H,W,3)")
        idx = frame_index
        if idx < 0:
            idx += len(frames)
        if idx < 0 or idx >= len(frames):
            raise IndexError(f"{hdf5_path}: frame_index {frame_index} out of range for len={len(frames)}")
        flip_vertical = _needs_vertical_flip(f["data"].attrs.get("macros_image_convention"))
        return _maybe_flip_rgb_vertical(frames[idx][()], flip_vertical)


def _extract_hdf5_rgb_frame_with_region_overlay(
    hdf5_path,
    episode,
    obs_key,
    frame_index,
    image_size_hint,
    draw_labels,
):
    with h5py.File(hdf5_path, "r") as f:
        if "data" not in f or episode not in f["data"]:
            raise KeyError(f"{hdf5_path}: episode '{episode}' not found")
        ep_grp = f[f"data/{episode}"]
        if "obs" not in ep_grp or obs_key not in ep_grp["obs"]:
            raise KeyError(f"{hdf5_path}: obs key '{obs_key}' not found in episode '{episode}'")

        frames = ep_grp["obs"][obs_key]
        if frames.ndim != 4 or frames.shape[-1] != 3:
            raise ValueError(f"{hdf5_path}: obs/{obs_key} shape {frames.shape} is not (T,H,W,3)")

        idx = frame_index
        if idx < 0:
            idx += len(frames)
        if idx < 0 or idx >= len(frames):
            raise IndexError(f"{hdf5_path}: frame_index {frame_index} out of range for len={len(frames)}")
        img = frames[idx][()]
        flip_vertical = _needs_vertical_flip(f["data"].attrs.get("macros_image_convention"))
        img = _maybe_flip_rgb_vertical(img, flip_vertical)

        states = ep_grp["states"][()]
        if len(states) == 0:
            return img
        state_idx = min(idx, len(states) - 1)
        model_xml = _as_text(ep_grp.attrs["model_file"])

        class _Args:
            bddl_file_override = None
            image_size = image_size_hint if image_size_hint is not None else int(img.shape[0])

        env, _, _ = _build_env_from_demo_attrs(f, _Args())
        try:
            _reset_env_with_xml_and_state(env, model_xml, states[state_idx], {})
            camera_name = "agentview"
            if obs_key == "eye_in_hand_rgb":
                camera_name = "robot0_eye_in_hand"
            img = _draw_bddl_region_overlay(
                env, img, camera_name=camera_name, draw_labels=draw_labels
            )
        finally:
            env.close()
        return img


def _render_dataset_folder_collages(
    variation_root,
    output_dir,
    episode,
    obs_key,
    frame_index,
    draw_bddl_regions=False,
    hide_region_labels=False,
    image_size_hint=None,
):
    groups = _collect_variation_groups(variation_root)
    if not groups:
        raise ValueError(f"No camera-variation HDF5 files found under {variation_root}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    for (rel_parent, base_name), items in sorted(groups.items()):
        images = []
        labels = []
        for camvar_id, camvar_uid, path in items:
            try:
                if draw_bddl_regions:
                    img = _extract_hdf5_rgb_frame_with_region_overlay(
                        path,
                        episode=episode,
                        obs_key=obs_key,
                        frame_index=frame_index,
                        image_size_hint=image_size_hint,
                        draw_labels=not hide_region_labels,
                    )
                else:
                    img = _extract_hdf5_rgb_frame(
                        path, episode=episode, obs_key=obs_key, frame_index=frame_index
                    )
            except Exception as e:
                print(f"[replay] skip {path}: {e}")
                continue
            images.append(img)
            label = f"camvar_{camvar_id:02d}"
            if camvar_uid:
                label = f"{label}_{camvar_uid}"
            labels.append(label)

        if not images:
            print(f"[replay] skip group {base_name}: no readable variation files")
            continue

        rel_parent_path = Path(rel_parent)
        out_group_dir = output_dir / rel_parent_path
        out_group_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_group_dir / f"{base_name}_{episode}_{obs_key}_f{frame_index}_collage.png"
        _make_collage(images, labels, out_file)
        saved.append(out_file)
        print(f"[replay] saved folder-collage {out_file} ({len(images)} tiles)")

    return saved


def main():
    parser = argparse.ArgumentParser(
        description="Replay one demo and render camera-variation screenshots or replay videos."
    )
    parser.add_argument("--demo-file", required=False)
    parser.add_argument(
        "--variation-root",
        type=str,
        default=None,
        help="Only used in dataset-folder-collage mode. Root directory containing camera-variation HDF5 files.",
    )
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
    parser.add_argument("--camera-variation-count", type=int, default=0)
    parser.add_argument("--camera-variation-seed", type=int, default=0)
    parser.add_argument("--translate-range", type=float, default=0.05)
    parser.add_argument("--rot-range-deg", type=float, default=8.0)
    parser.add_argument("--camera-variation-config", type=str, default=None)
    parser.add_argument("--target-camera", type=str, default="agentview")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument(
        "--mode",
        choices=["screenshot", "video", "dataset-video", "dataset-folder-collage"],
        default="video",
        help=(
            "screenshot: one state across camera variations; "
            "video: replay action sequence for each camera variation; "
            "dataset-video: strictly encode stored HDF5 obs frames into video; "
            "dataset-folder-collage: traverse a folder of camera-variation HDF5 files and make one collage per source"
        ),
    )
    parser.add_argument(
        "--obs-key",
        type=str,
        default="agentview_rgb",
        help="Used in dataset-video / dataset-folder-collage modes. Observation key under /data/<episode>/obs/",
    )
    parser.add_argument(
        "--frame-index",
        type=int,
        default=-1,
        help="Used in dataset-folder-collage mode. Frame index within obs/<obs-key> (default -1 = last frame).",
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
    parser.add_argument("--include-original-view", dest="include_original_view", action="store_true")
    parser.add_argument("--exclude-original-view", dest="include_original_view", action="store_false")
    parser.add_argument(
        "--draw-bddl-regions",
        action="store_true",
        help="Only used in screenshot mode. Overlay BDDL region rectangles (projected from env region sites).",
    )
    parser.add_argument(
        "--hide-region-labels",
        action="store_true",
        help="Only used with --draw-bddl-regions. Draw region boxes without text labels.",
    )
    parser.set_defaults(include_original_view=True)
    args = parser.parse_args()
    camera_variation_cfg = camvar_cfg.load_camera_variation_config(args.camera_variation_config)
    effective_camera_variation_count = camvar_cfg.get_effective_count(
        args.camera_variation_count, camera_variation_cfg
    )

    if args.mode in ("screenshot", "video") and effective_camera_variation_count <= 0:
        parser.error("--camera-variation-count must be > 0")
    if args.mode in ("screenshot", "video") and args.target_camera != "agentview":
        parser.error("v1 only supports --target-camera agentview")
    if args.mode == "dataset-folder-collage":
        if not args.variation_root:
            parser.error("--variation-root is required in dataset-folder-collage mode")
    else:
        if not args.demo_file:
            parser.error("--demo-file is required unless using dataset-folder-collage mode")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "dataset-folder-collage":
        saved = _render_dataset_folder_collages(
            variation_root=args.variation_root,
            output_dir=output_dir,
            episode=args.episode,
            obs_key=args.obs_key,
            frame_index=args.frame_index,
            draw_bddl_regions=args.draw_bddl_regions,
            hide_region_labels=args.hide_region_labels,
            image_size_hint=args.image_size,
        )
        print(f"[replay] folder-collage done, saved {len(saved)} images")
        return

    with h5py.File(args.demo_file, "r") as f:
        if args.episode not in f["data"]:
            parser.error(f"Episode '{args.episode}' not found in /data")
        ep_grp = f[f"data/{args.episode}"]
        model_xml = _as_text(ep_grp.attrs["model_file"])
        states = ep_grp["states"][()]
        actions = ep_grp["actions"][()]
        hdf5_flip_vertical = _needs_vertical_flip(f["data"].attrs.get("macros_image_convention"))
        env_flip_vertical = _needs_vertical_flip(getattr(macros, "IMAGE_CONVENTION", None))

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
                frames=frames,
                video_path=out_file,
                fps=args.fps,
                draw_label=args.obs_key,
                flip_vertical=hdf5_flip_vertical,
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
            target_pos = None
            if camvar_cfg.needs_target_pos(camera_variation_cfg):
                target_request = camvar_cfg.get_target_request(camera_variation_cfg)
                target_pos = _resolve_target_pos_for_replay(
                    ep_grp=ep_grp,
                    states=states,
                    env=env,
                    model_xml=model_xml,
                    target_request=target_request,
                )
                print(f"[replay] target_pos={target_pos.tolist()}")

            cfg_poses = camvar_cfg.generate_camera_variation_poses(
                base_pos=base_pose["pos"],
                base_quat=base_pose["quat"],
                count=effective_camera_variation_count,
                seed=args.camera_variation_seed,
                translate_range=args.translate_range,
                rot_range_deg=args.rot_range_deg,
                cfg=camera_variation_cfg,
                target_pos=target_pos,
                validator=camera_visibility.make_pose_validator(
                    env=env,
                    model_xml=model_xml,
                    states=states,
                    reset_fn=_reset_env_with_xml_and_state,
                    cfg=camera_variation_cfg,
                    camera_name=args.target_camera,
                )
                if camera_visibility.constraints_enabled(camera_variation_cfg)
                else None,
            )

            poses = []
            for cfg_pose in cfg_poses:
                idx = int(cfg_pose["variation_id"])
                uid = camvar_cfg.build_camera_variation_uid(args.demo_file, idx, cfg_pose)
                pose = {
                    "variation_id": idx,
                    "variation_uid": uid,
                    "pos": np.asarray(cfg_pose["applied_pos"], dtype=np.float64),
                    "quat": np.asarray(cfg_pose["applied_quat"], dtype=np.float64),
                    "delta_pos": np.asarray(cfg_pose["delta_pos"], dtype=np.float64),
                    "delta_rpy_deg": np.asarray(cfg_pose["delta_rpy_deg"], dtype=np.float64),
                }
                print(
                    f"[replay] {uid} "
                    f"strategy={cfg_pose.get('strategy', 'random_local')} "
                    f"delta_pos={pose['delta_pos'].tolist()} "
                    f"delta_rpy_deg={pose['delta_rpy_deg'].tolist()}"
                )
                poses.append(pose)

            if args.mode == "screenshot":
                images = []
                labels = []
                state = states[state_index]
                if args.include_original_view:
                    img = _capture_image_at_state(
                        env=env,
                        model_xml=model_xml,
                        state=state,
                        cameras_dict={},
                    )
                    img = _maybe_flip_rgb_vertical(img, env_flip_vertical)
                    if args.draw_bddl_regions:
                        img = _draw_bddl_region_overlay(
                            env,
                            img,
                            camera_name=args.target_camera,
                            draw_labels=not args.hide_region_labels,
                        )
                    images.append(img)
                    labels.append("original")
                    out_file = output_dir / f"{prefix}_original.png"
                    Image.fromarray(img).save(out_file)
                    print(f"[replay] saved {out_file}")
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
                    img = _maybe_flip_rgb_vertical(img, env_flip_vertical)
                    if args.draw_bddl_regions:
                        img = _draw_bddl_region_overlay(
                            env,
                            img,
                            camera_name=args.target_camera,
                            draw_labels=not args.hide_region_labels,
                        )
                    images.append(img)
                    label = pose["variation_uid"]
                    labels.append(label)
                    out_file = output_dir / f"{prefix}_camvar_{idx:02d}_{label.split('-')[0]}.png"
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
                        flip_vertical=env_flip_vertical,
                        draw_bddl_regions=args.draw_bddl_regions,
                        hide_region_labels=args.hide_region_labels,
                        camera_name=args.target_camera,
                    )
                    print(f"[replay] saved video {out_file} ({n_frames} frames)")
        finally:
            env.close()


if __name__ == "__main__":
    main()
