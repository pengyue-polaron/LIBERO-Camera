import argparse
import subprocess
import sys
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "camera_variation" / "orbit_arm_8_pitch_yaw.json"
PREVIEW_SCRIPT = ROOT / "scripts" / "batch_preview_camera_variation_videos.py"
CREATE_SCRIPT = ROOT / "scripts" / "create_dataset.py"


def _iter_hdf5_files(dataset_root, file_pattern):
    return sorted(Path(dataset_root).rglob(file_pattern))


def _filter_skip_dirs(files, dataset_root, skip_dirs):
    if not skip_dirs:
        return files
    skip_set = {s.strip() for s in skip_dirs if s and s.strip()}
    if not skip_set:
        return files
    kept = []
    for p in files:
        rel_parts = set(p.relative_to(dataset_root).parts)
        if rel_parts.intersection(skip_set):
            continue
        kept.append(p)
    return kept


def _run(cmd):
    print("[pipeline] run:", " ".join(str(x) for x in cmd))
    subprocess.run(cmd, check=True)


def _preview_cmd(args):
    cmd = [
        sys.executable,
        str(PREVIEW_SCRIPT),
        "--dataset-root",
        str(args.dataset_root),
        "--output-dir",
        str(args.preview_output_dir),
        "--camera-variation-config",
        str(args.camera_variation_config),
        "--episode",
        args.episode,
        "--fps",
        str(args.fps),
        "--start-action-index",
        str(args.start_action_index),
        "--max-steps",
        str(args.max_steps),
        "--image-size",
        str(args.image_size),
        "--grid-cell-size",
        str(args.grid_cell_size),
        "--grid-cols",
        str(args.grid_cols),
    ]
    if args.draw_bddl_regions:
        cmd.append("--draw-bddl-regions")
    if args.hide_region_labels:
        cmd.append("--hide-region-labels")
    if args.hide_video_labels:
        cmd.append("--hide-video-labels")
    if args.grid_video:
        cmd.append("--grid-video")
    if args.grid_only:
        cmd.append("--grid-only")
    if args.exclude_original_view:
        cmd.append("--exclude-original-view")
    if args.skip_camvar_outputs:
        cmd.append("--skip-camvar-outputs")
    if args.resume:
        cmd.append("--resume")
    if args.continue_on_error:
        cmd.append("--continue-on-error")
    if args.preview_fail_log:
        cmd.extend(["--fail-log", str(args.preview_fail_log)])
    if args.file_pattern:
        cmd.extend(["--file-pattern", args.file_pattern])
    if args.skip_dirs:
        cmd.extend(["--skip-dirs", args.skip_dirs])
    if args.max_files > 0:
        cmd.extend(["--max-files", str(args.max_files)])
    return cmd


def _preview_cmd_for_file(args, src_hdf5):
    cmd = [
        sys.executable,
        str(PREVIEW_SCRIPT),
        "--dataset-root",
        str(args.dataset_root),
        "--output-dir",
        str(args.preview_output_dir),
        "--camera-variation-config",
        str(args.camera_variation_config),
        "--episode",
        args.episode,
        "--fps",
        str(args.fps),
        "--start-action-index",
        str(args.start_action_index),
        "--max-steps",
        str(args.max_steps),
        "--image-size",
        str(args.image_size),
        "--grid-cell-size",
        str(args.grid_cell_size),
        "--grid-cols",
        str(args.grid_cols),
        "--only-file",
        str(src_hdf5),
    ]
    if args.draw_bddl_regions:
        cmd.append("--draw-bddl-regions")
    if args.hide_region_labels:
        cmd.append("--hide-region-labels")
    if args.hide_video_labels:
        cmd.append("--hide-video-labels")
    if args.grid_video:
        cmd.append("--grid-video")
    if args.grid_only:
        cmd.append("--grid-only")
    if args.exclude_original_view:
        cmd.append("--exclude-original-view")
    if args.skip_camvar_outputs:
        cmd.append("--skip-camvar-outputs")
    if args.skip_dirs:
        cmd.extend(["--skip-dirs", args.skip_dirs])
    if args.resume:
        cmd.append("--resume")
    if args.continue_on_error:
        cmd.append("--continue-on-error")
    if args.preview_fail_log:
        cmd.extend(["--fail-log", str(args.preview_fail_log)])
    return cmd


def _create_cmd(args, src_hdf5, dst_dir, name_prefix):
    cmd = [
        sys.executable,
        str(CREATE_SCRIPT),
        "--demo-file",
        str(src_hdf5),
        "--use-camera-obs",
        "--camera-variation-config",
        str(args.camera_variation_config),
        "--camera-variation-output-dir",
        str(dst_dir),
        "--camera-variation-name-prefix",
        name_prefix,
    ]
    if args.camera_variation_uid_in_filename:
        cmd.append("--camera-variation-uid-in-filename")
    if args.resume:
        cmd.append("--resume")
    return cmd


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run the full camera-variation pipeline for a datasets folder: "
            "first preview videos, then generate dataset_camera HDF5 outputs."
        )
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--preview-output-dir", required=True)
    parser.add_argument("--dataset-camera-root", required=True)
    parser.add_argument("--camera-variation-config", default=str(DEFAULT_CONFIG))
    parser.add_argument(
        "--camera-variation-uid-in-filename",
        dest="camera_variation_uid_in_filename",
        action="store_true",
        help="Append a short stable UID to each generated camera-variation HDF5 filename.",
    )
    parser.add_argument(
        "--no-camera-variation-uid-in-filename",
        dest="camera_variation_uid_in_filename",
        action="store_false",
    )
    parser.add_argument("--episode", type=str, default="demo_0")
    parser.add_argument("--file-pattern", type=str, default="*.hdf5")
    parser.add_argument(
        "--skip-dirs",
        type=str,
        default="libero_90,libero90",
        help="Comma-separated directory names to skip under dataset-root.",
    )
    parser.add_argument("--skip-camvar-outputs", action="store_true")
    parser.add_argument("--resume", dest="resume", action="store_true", help="Skip existing outputs.")
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument(
        "--continue-on-error",
        dest="continue_on_error",
        action="store_true",
        help="Continue with remaining files if one file fails.",
    )
    parser.add_argument("--no-continue-on-error", dest="continue_on_error", action="store_false")
    parser.add_argument(
        "--fail-log",
        type=str,
        default=None,
        help="Optional path to write pipeline failures.",
    )
    parser.add_argument(
        "--preview-fail-log",
        type=str,
        default=None,
        help="Optional path to write preview-stage failures.",
    )
    parser.add_argument("--include-original-hdf5", dest="include_original_hdf5", action="store_true")
    parser.add_argument("--no-include-original-hdf5", dest="include_original_hdf5", action="store_false")
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--start-action-index", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--draw-bddl-regions", dest="draw_bddl_regions", action="store_true")
    parser.add_argument("--no-draw-bddl-regions", dest="draw_bddl_regions", action="store_false")
    parser.add_argument("--hide-region-labels", dest="hide_region_labels", action="store_true")
    parser.add_argument("--show-region-labels", dest="hide_region_labels", action="store_false")
    parser.add_argument("--hide-video-labels", dest="hide_video_labels", action="store_true")
    parser.add_argument("--show-video-labels", dest="hide_video_labels", action="store_false")
    parser.add_argument("--grid-video", dest="grid_video", action="store_true")
    parser.add_argument("--no-grid-video", dest="grid_video", action="store_false")
    parser.add_argument("--grid-only", dest="grid_only", action="store_true")
    parser.add_argument("--no-grid-only", dest="grid_only", action="store_false")
    parser.add_argument("--grid-cell-size", type=int, default=128)
    parser.add_argument("--grid-cols", type=int, default=4)
    parser.add_argument("--exclude-original-view", action="store_true")
    parser.set_defaults(
        draw_bddl_regions=True,
        hide_region_labels=True,
        hide_video_labels=True,
        grid_video=True,
        grid_only=True,
        include_original_hdf5=True,
        resume=True,
        continue_on_error=True,
        camera_variation_uid_in_filename=True,
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    preview_output_dir = Path(args.preview_output_dir)
    dataset_camera_root = Path(args.dataset_camera_root)
    preview_output_dir.mkdir(parents=True, exist_ok=True)
    dataset_camera_root.mkdir(parents=True, exist_ok=True)

    failures = []

    files = _iter_hdf5_files(dataset_root, args.file_pattern)
    skip_dirs = [x for x in args.skip_dirs.split(",")] if args.skip_dirs else []
    files = _filter_skip_dirs(files, dataset_root=dataset_root, skip_dirs=skip_dirs)
    if args.skip_camvar_outputs:
        files = [p for p in files if "_camvar_" not in p.name]
    if args.max_files > 0:
        files = files[: args.max_files]
    if not files:
        raise SystemExit(f"No hdf5 files found under {dataset_root}")

    for src_hdf5 in files:
        rel_parent = src_hdf5.relative_to(dataset_root).parent
        dst_dir = dataset_camera_root / rel_parent
        dst_dir.mkdir(parents=True, exist_ok=True)
        print(f"[pipeline] processing {src_hdf5}")

        try:
            _run(_preview_cmd_for_file(args, src_hdf5))
        except subprocess.CalledProcessError as exc:
            msg = f"preview {src_hdf5}: {exc}"
            if not args.continue_on_error:
                raise
            print(f"[pipeline] WARN {msg}")
            failures.append(msg)

        if args.include_original_hdf5:
            dst_original = dst_dir / src_hdf5.name
            if not dst_original.exists():
                shutil.copy2(src_hdf5, dst_original)
                print(f"[pipeline] copied original -> {dst_original}")
        try:
            _run(_create_cmd(args, src_hdf5, dst_dir, src_hdf5.stem))
        except subprocess.CalledProcessError as exc:
            msg = f"{src_hdf5}: {exc}"
            if not args.continue_on_error:
                raise
            print(f"[pipeline] WARN {msg}")
            failures.append(msg)

    if failures:
        fail_log = Path(args.fail_log) if args.fail_log else (dataset_camera_root / "pipeline_failures.txt")
        fail_log.parent.mkdir(parents=True, exist_ok=True)
        fail_log.write_text("\n".join(failures) + "\n")
        print(f"[pipeline] wrote fail log: {fail_log}")

    print(f"[pipeline] done (failed={len(failures)})")


if __name__ == "__main__":
    main()
