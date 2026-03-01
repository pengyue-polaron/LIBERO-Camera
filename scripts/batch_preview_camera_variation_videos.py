import argparse
from pathlib import Path

import h5py
import cv2
import numpy as np

import init_path
import camera_variation_config as camvar_cfg
import camera_visibility
import replay_camera_variations as replay_utils

DEFAULT_CAMERA_VARIATION_CONFIG = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "camera_variation"
    / "orbit_arm_8_pitch_yaw.json"
)


def _iter_hdf5_files(dataset_root, file_pattern, skip_camvar_outputs):
    dataset_root = Path(dataset_root)
    files = sorted(dataset_root.rglob(file_pattern))
    if skip_camvar_outputs:
        files = [p for p in files if replay_utils._CAMVAR_FILE_RE.match(p.name) is None]
    return files


def _ensure_episode(h5_file, episode):
    return "data" in h5_file and episode in h5_file["data"]


def _build_output_prefix(dataset_root, hdf5_path):
    rel = hdf5_path.relative_to(dataset_root)
    parent = rel.parent
    stem = hdf5_path.stem
    return parent, stem


def _render_frames_for_variation(
    env,
    model_xml,
    init_state,
    actions,
    cameras_dict,
    flip_vertical,
    draw_bddl_regions,
    hide_region_labels,
):
    obs = replay_utils._reset_env_with_xml_and_state(env, model_xml, init_state, cameras_dict)
    frame = replay_utils._maybe_flip_rgb_vertical(obs["agentview_image"], flip_vertical)
    if draw_bddl_regions:
        frame = replay_utils._draw_bddl_region_overlay(
            env, frame, camera_name="agentview", draw_labels=not hide_region_labels
        )
    frames = [frame.copy()]

    for action in actions:
        obs, _, _, _ = env.step(action)
        frame = replay_utils._maybe_flip_rgb_vertical(obs["agentview_image"], flip_vertical)
        if draw_bddl_regions:
            frame = replay_utils._draw_bddl_region_overlay(
                env, frame, camera_name="agentview", draw_labels=not hide_region_labels
            )
        frame = frame.copy()
        frames.append(frame)
    return frames


def _compose_grid_frames(frame_lists, labels, cell_size, cols=4, draw_labels=True):
    if not frame_lists:
        return []
    num_tiles = len(frame_lists)
    num_frames = len(frame_lists[0])
    for frames in frame_lists:
        if len(frames) != num_frames:
            raise ValueError("All frame lists must have the same length to compose a grid video")

    cols = min(cols, num_tiles)
    rows = int(np.ceil(num_tiles / cols))
    label_h = 20
    out_h = rows * (cell_size + label_h)
    out_w = cols * cell_size
    grid_frames = []

    for frame_idx in range(num_frames):
        canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        for tile_idx, frames in enumerate(frame_lists):
            row = tile_idx // cols
            col = tile_idx % cols
            x0 = col * cell_size
            y0 = row * (cell_size + label_h)
            tile = cv2.resize(frames[frame_idx], (cell_size, cell_size), interpolation=cv2.INTER_AREA)
            canvas[y0 + label_h : y0 + label_h + cell_size, x0 : x0 + cell_size] = tile
            if draw_labels:
                cv2.putText(
                    canvas,
                    labels[tile_idx],
                    (x0 + 4, y0 + 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
        grid_frames.append(canvas)
    return grid_frames


def _write_frames_to_video(frames, out_file, fps):
    writer = replay_utils._open_video_writer(out_file, frames[0].shape, fps)
    try:
        for frame in frames:
            replay_utils._write_rgb_frame(writer, frame)
    finally:
        writer.release()


def _render_variation_videos_for_file(hdf5_path, args):
    hdf5_path = Path(hdf5_path)
    with h5py.File(hdf5_path, "r") as f:
        if not _ensure_episode(f, args.episode):
            print(f"[batch-preview] skip {hdf5_path}: episode '{args.episode}' not found")
            return 0

        ep_grp = f[f"data/{args.episode}"]
        model_xml = replay_utils._as_text(ep_grp.attrs["model_file"])
        states = ep_grp["states"][()]
        actions = ep_grp["actions"][()]

        env_flip_vertical = replay_utils._needs_vertical_flip(
            getattr(replay_utils.macros, "IMAGE_CONVENTION", None)
        )
        base_pose = replay_utils._extract_camera_pose_from_xml(model_xml, "agentview")

        env, problem_name, bddl_file = replay_utils._build_env_from_demo_attrs(f, args)
        print(f"[batch-preview] file={hdf5_path.name} problem={problem_name} bddl={bddl_file}")
        try:
            target_pos = None
            if camvar_cfg.needs_target_pos(args.camera_variation_cfg):
                target_request = camvar_cfg.get_target_request(args.camera_variation_cfg)
                target_pos = replay_utils._resolve_target_pos_for_replay(
                    ep_grp=ep_grp,
                    states=states,
                    env=env,
                    model_xml=model_xml,
                    target_request=target_request,
                )
                print(f"[batch-preview] target_pos={target_pos.tolist()}")

            poses = camvar_cfg.generate_camera_variation_poses(
                base_pos=base_pose["pos"],
                base_quat=base_pose["quat"],
                count=args.camera_variation_count,
                seed=args.camera_variation_seed,
                translate_range=args.translate_range,
                rot_range_deg=args.rot_range_deg,
                cfg=args.camera_variation_cfg,
                target_pos=target_pos,
                validator=camera_visibility.make_pose_validator(
                    env=env,
                    model_xml=model_xml,
                    states=states,
                    reset_fn=replay_utils._reset_env_with_xml_and_state,
                    cfg=args.camera_variation_cfg,
                    camera_name="agentview",
                )
                if camera_visibility.constraints_enabled(args.camera_variation_cfg)
                else None,
            )

            start_action_idx = args.start_action_index
            if start_action_idx < 0:
                start_action_idx += len(actions)
            if start_action_idx < 0 or start_action_idx >= len(actions):
                raise ValueError(
                    f"{hdf5_path}: --start-action-index {args.start_action_index} out of range for len(actions)={len(actions)}"
                )

            if len(states) < len(actions):
                raise ValueError(
                    f"{hdf5_path}: expected len(states) >= len(actions), got {len(states)} < {len(actions)}"
                )

            if args.max_steps <= 0:
                end_action_idx = len(actions)
            else:
                end_action_idx = min(len(actions), start_action_idx + args.max_steps)
            action_slice = actions[start_action_idx:end_action_idx]
            init_state = states[start_action_idx]

            rel_parent, stem = _build_output_prefix(Path(args.dataset_root), hdf5_path)
            output_parent = Path(args.output_dir) / rel_parent
            output_parent.mkdir(parents=True, exist_ok=True)

            count = 0
            grid_frame_lists = []
            grid_labels = []

            if args.include_original_view:
                original_label = "original"
                original_frames = _render_frames_for_variation(
                    env=env,
                    model_xml=model_xml,
                    init_state=init_state,
                    actions=action_slice,
                    cameras_dict={},
                    flip_vertical=env_flip_vertical,
                    draw_bddl_regions=args.draw_bddl_regions,
                    hide_region_labels=args.hide_region_labels,
                )
                if args.grid_video:
                    grid_frame_lists.append(original_frames)
                    grid_labels.append(original_label)

                if not args.grid_only:
                    original_out_file = output_parent / f"{stem}_{args.episode}_original.mp4"
                    if args.hide_video_labels:
                        frames_for_original = [frame.copy() for frame in original_frames]
                    else:
                        frames_for_original = [
                            cv2.putText(
                                frame.copy(),
                                original_label,
                                (8, 20),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (255, 255, 255),
                                1,
                                cv2.LINE_AA,
                            )
                            for frame in original_frames
                        ]
                    _write_frames_to_video(frames_for_original, original_out_file, args.fps)
                    print(
                        f"[batch-preview] saved {original_out_file} "
                        f"frames={len(original_frames)} strategy=original"
                    )
                    count += 1

            for pose in poses:
                variation_id = int(pose["variation_id"])
                cameras_dict = {
                    "agentview": {
                        "pos": replay_utils._float_list_to_str(pose["applied_pos"]),
                        "quat": replay_utils._float_list_to_str(pose["applied_quat"]),
                    }
                }
                label = f"camvar_{variation_id:02d}"

                if args.grid_video:
                    frames = _render_frames_for_variation(
                        env=env,
                        model_xml=model_xml,
                        init_state=init_state,
                        actions=action_slice,
                        cameras_dict=cameras_dict,
                        flip_vertical=env_flip_vertical,
                        draw_bddl_regions=args.draw_bddl_regions,
                        hide_region_labels=args.hide_region_labels,
                    )
                    grid_frame_lists.append(frames)
                    grid_labels.append(label)
                    n_frames = len(frames)
                else:
                    n_frames = len(action_slice) + 1

                if not args.grid_only:
                    out_file = output_parent / f"{stem}_{args.episode}_camvar_{variation_id:02d}.mp4"
                    if args.grid_video:
                        if args.hide_video_labels:
                            frames_for_single = [frame.copy() for frame in frames]
                        else:
                            frames_for_single = [
                                cv2.putText(
                                    frame.copy(),
                                    label,
                                    (8, 20),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5,
                                    (255, 255, 255),
                                    1,
                                    cv2.LINE_AA,
                                )
                                for frame in frames
                            ]
                        _write_frames_to_video(frames_for_single, out_file, args.fps)
                    else:
                        replay_utils._render_video_for_action_slice(
                            env=env,
                            model_xml=model_xml,
                            init_state=init_state,
                            actions=action_slice,
                            cameras_dict=cameras_dict,
                            video_path=out_file,
                            fps=args.fps,
                            draw_label=None if args.hide_video_labels else label,
                            flip_vertical=env_flip_vertical,
                            draw_bddl_regions=args.draw_bddl_regions,
                            hide_region_labels=args.hide_region_labels,
                            camera_name="agentview",
                        )
                    print(
                        f"[batch-preview] saved {out_file} "
                        f"frames={n_frames} strategy={pose.get('strategy', 'random_local')}"
                    )
                    count += 1

            if args.grid_video and grid_frame_lists:
                grid_frames = _compose_grid_frames(
                    grid_frame_lists,
                    grid_labels,
                    cell_size=args.grid_cell_size,
                    cols=args.grid_cols,
                    draw_labels=not args.hide_tile_labels,
                )
                grid_out_file = output_parent / f"{stem}_{args.episode}_grid.mp4"
                _write_frames_to_video(grid_frames, grid_out_file, args.fps)
                print(
                    f"[batch-preview] saved {grid_out_file} "
                    f"frames={len(grid_frames)} tiles={len(grid_frame_lists)}"
                )
                count += 1
            return count
        finally:
            env.close()


def main():
    parser = argparse.ArgumentParser(
        description="Batch-render camera-variation preview videos from a datasets directory."
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--episode", type=str, default="demo_0")
    parser.add_argument("--file-pattern", type=str, default="*.hdf5")
    parser.add_argument("--skip-camvar-outputs", action="store_true")
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument(
        "--camera-variation-config",
        default=str(DEFAULT_CAMERA_VARIATION_CONFIG),
        help=f"Path to camera variation config JSON. Default: {DEFAULT_CAMERA_VARIATION_CONFIG}",
    )
    parser.add_argument("--camera-variation-count", type=int, default=0)
    parser.add_argument("--camera-variation-seed", type=int, default=0)
    parser.add_argument("--translate-range", type=float, default=0.05)
    parser.add_argument("--rot-range-deg", type=float, default=8.0)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--start-action-index", type=int, default=0)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="Number of actions to replay. <=0 means replay the full remaining trajectory.",
    )
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--grid-video", dest="grid_video", action="store_true")
    parser.add_argument("--no-grid-video", dest="grid_video", action="store_false")
    parser.add_argument("--grid-only", dest="grid_only", action="store_true")
    parser.add_argument("--no-grid-only", dest="grid_only", action="store_false")
    parser.add_argument("--grid-cell-size", type=int, default=128)
    parser.add_argument("--grid-cols", type=int, default=5)
    parser.add_argument("--include-original-view", dest="include_original_view", action="store_true")
    parser.add_argument("--exclude-original-view", dest="include_original_view", action="store_false")
    parser.add_argument("--hide-video-labels", dest="hide_video_labels", action="store_true")
    parser.add_argument("--show-video-labels", dest="hide_video_labels", action="store_false")
    parser.add_argument("--hide-tile-labels", action="store_true")
    parser.add_argument(
        "--draw-bddl-regions",
        dest="draw_bddl_regions",
        action="store_true",
        help="Overlay BDDL random-region ranges on video frames.",
    )
    parser.add_argument(
        "--no-draw-bddl-regions",
        dest="draw_bddl_regions",
        action="store_false",
    )
    parser.add_argument("--hide-region-labels", dest="hide_region_labels", action="store_true")
    parser.add_argument(
        "--show-region-labels",
        dest="hide_region_labels",
        action="store_false",
        help="Show region names next to boxes. Default is boxes only.",
    )
    parser.add_argument("--bddl-file-override", type=str, default=None)
    parser.set_defaults(
        include_original_view=True,
        hide_region_labels=True,
        hide_video_labels=True,
        draw_bddl_regions=True,
        grid_video=True,
        grid_only=True,
    )
    args = parser.parse_args()

    args.camera_variation_cfg = camvar_cfg.load_camera_variation_config(args.camera_variation_config)
    args.camera_variation_count = camvar_cfg.get_effective_count(
        args.camera_variation_count, args.camera_variation_cfg
    )
    if args.camera_variation_count <= 0:
        parser.error("camera variation count must be > 0 (set in config or CLI)")
    if args.grid_only and not args.grid_video:
        parser.error("--grid-only requires --grid-video")

    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = _iter_hdf5_files(
        dataset_root=dataset_root,
        file_pattern=args.file_pattern,
        skip_camvar_outputs=args.skip_camvar_outputs,
    )
    if args.max_files > 0:
        files = files[: args.max_files]
    if not files:
        parser.error(f"No HDF5 files found under {dataset_root} matching {args.file_pattern}")

    total_videos = 0
    for path in files:
        total_videos += _render_variation_videos_for_file(path, args)

    print(f"[batch-preview] done, rendered {total_videos} videos from {len(files)} files")


if __name__ == "__main__":
    main()
