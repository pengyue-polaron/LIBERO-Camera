import argparse
import json
import math
from pathlib import Path

import cv2
import h5py
import numpy as np
import robosuite.macros as macros

import init_path
import replay_camera_variations as replay_utils


def _quat_normalize_wxyz(quat):
    quat = np.asarray(quat, dtype=np.float64)
    n = np.linalg.norm(quat)
    if n < 1e-12:
        raise ValueError("Quaternion norm too small")
    return quat / n


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


def _apply_camera_pose(env, camera_name, pos, quat):
    cam_id = env.sim.model.camera_name2id(camera_name)
    env.sim.model.cam_pos[cam_id] = np.asarray(pos, dtype=np.float64)
    env.sim.model.cam_quat[cam_id] = np.asarray(quat, dtype=np.float64)
    env.sim.forward()


def _draw_hud(frame, camera_name, step_idx, total_steps, captures, max_captures, trans_step, rot_step_deg):
    out = frame.copy()
    lines = [
        f"{camera_name} step {step_idx}/{total_steps}",
        f"captures {len(captures)}/{max_captures}",
        f"move {trans_step:.3f}m  rot {rot_step_deg:.1f}deg",
        "W/S:X  A/D:Y  Q/E:Z",
        "I/K:yaw  J/L:roll  U/O:pitch",
        "C:capture  X:undo  +/-:speed  ESC:quit",
    ]

    h, w = out.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1
    scale = 0.50
    while scale > 0.30:
        max_text_w = 0
        line_h = 0
        for line in lines:
            (tw, th), _ = cv2.getTextSize(line, font, scale, thickness)
            max_text_w = max(max_text_w, tw)
            line_h = max(line_h, th)
        if max_text_w <= (w - 16):
            break
        scale -= 0.02

    line_gap = max(4, int(line_h * 0.45))
    box_h = len(lines) * (line_h + line_gap) + 8
    box_w = min(w - 8, max_text_w + 12)
    overlay = out.copy()
    cv2.rectangle(overlay, (4, 4), (4 + box_w, 4 + box_h), (0, 0, 0), -1)
    out = cv2.addWeighted(overlay, 0.55, out, 0.45, 0.0)

    y = 4 + line_h + 2
    for line in lines:
        cv2.putText(
            out,
            line,
            (8, y),
            font,
            scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
        y += line_h + line_gap
    return out


def _build_config(captures):
    return {
        "strategy": "manual_poses",
        "count": len(captures),
        "manual_poses": captures,
        "selection": {"mode": "as_list"},
        "visibility_constraints": {"enabled": False},
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Replay one episode while you manually move agentview and capture "
            "camera poses for manual camera-variation generation."
        )
    )
    parser.add_argument("--demo-file", required=True)
    parser.add_argument("--episode", type=str, default="demo_0")
    parser.add_argument("--camera-name", type=str, default="agentview")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--capture-count", type=int, default=5)
    parser.add_argument("--output-config", required=True)
    parser.add_argument(
        "--resume-config",
        type=str,
        default=None,
        help="Optional existing manual config JSON to resume captures from.",
    )
    parser.add_argument("--bddl-file-override", type=str, default=None)
    parser.add_argument("--start-action-index", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--playback-delay-ms", type=int, default=20)
    args = parser.parse_args()

    if args.capture_count <= 0:
        parser.error("--capture-count must be > 0")

    with h5py.File(args.demo_file, "r") as f:
        if "data" not in f or args.episode not in f["data"]:
            raise ValueError(f"Episode {args.episode} not found in {args.demo_file}")

        ep_grp = f[f"data/{args.episode}"]
        model_xml = replay_utils._as_text(ep_grp.attrs["model_file"])
        states = ep_grp["states"][()]
        actions = ep_grp["actions"][()]
        base_pose = replay_utils._extract_camera_pose_from_xml(model_xml, args.camera_name)

        env_args = argparse.Namespace(
            bddl_file_override=args.bddl_file_override,
            image_size=args.image_size,
        )
        env, problem_name, bddl_file = replay_utils._build_env_from_demo_attrs(f, env_args)
        print(f"[manual-cam] problem={problem_name} bddl={bddl_file}")

        try:
            start_idx = args.start_action_index
            if start_idx < 0:
                start_idx += len(actions)
            if start_idx < 0 or start_idx >= len(actions):
                raise ValueError(f"start-action-index out of range: {args.start_action_index}")

            if args.max_steps <= 0:
                end_idx = len(actions)
            else:
                end_idx = min(len(actions), start_idx + args.max_steps)
            action_slice = actions[start_idx:end_idx]
            if len(action_slice) == 0:
                raise ValueError("No actions to replay")

            cam_pos = np.asarray(base_pose["pos"], dtype=np.float64).copy()
            cam_quat = np.asarray(base_pose["quat"], dtype=np.float64).copy()

            obs = replay_utils._reset_env_with_xml_and_state(
                env=env,
                model_xml=model_xml,
                state=states[start_idx],
                cameras_dict={},
            )
            _apply_camera_pose(env, args.camera_name, cam_pos, cam_quat)
            flip_vertical = replay_utils._needs_vertical_flip(
                getattr(macros, "IMAGE_CONVENTION", None)
            )

            captures = []
            if args.resume_config:
                resume_path = Path(args.resume_config)
                if resume_path.exists():
                    with open(resume_path, "r") as fp:
                        prev_cfg = json.load(fp)
                    prev_caps = prev_cfg.get("manual_poses", [])
                    for idx, cap in enumerate(prev_caps):
                        pos = np.asarray(cap.get("pos", []), dtype=np.float64)
                        quat = np.asarray(cap.get("quat", []), dtype=np.float64)
                        if pos.shape != (3,) or quat.shape != (4,):
                            print(f"[manual-cam] skip invalid resume pose at index {idx}")
                            continue
                        captures.append(
                            {
                                "pos": [float(v) for v in pos.tolist()],
                                "quat": [float(v) for v in _quat_normalize_wxyz(quat).tolist()],
                            }
                        )
                    if len(captures) > args.capture_count:
                        captures = captures[: args.capture_count]
                    print(
                        f"[manual-cam] resumed {len(captures)} capture(s) from {resume_path}"
                    )
                else:
                    print(f"[manual-cam] resume config not found, ignore: {resume_path}")
            trans_step = 0.01
            rot_step_deg = 3.0
            win = "Manual Camera Variation Recorder"
            cv2.namedWindow(win, cv2.WINDOW_NORMAL)

            step_idx = 0
            cycle_idx = 0
            exit_requested = False
            while True:
                if len(captures) >= args.capture_count:
                    print(f"[manual-cam] reached capture target: {len(captures)}")
                    break

                if step_idx >= len(action_slice):
                    cycle_idx += 1
                    step_idx = 0
                    replay_utils._reset_env_with_xml_and_state(
                        env=env,
                        model_xml=model_xml,
                        state=states[start_idx],
                        cameras_dict={},
                    )
                    _apply_camera_pose(env, args.camera_name, cam_pos, cam_quat)

                _apply_camera_pose(env, args.camera_name, cam_pos, cam_quat)
                obs, _, _, _ = env.step(action_slice[step_idx])
                frame = replay_utils._maybe_flip_rgb_vertical(
                    obs[f"{args.camera_name}_image"], flip_vertical
                )
                frame = _draw_hud(
                    frame,
                    camera_name=args.camera_name,
                    step_idx=step_idx + 1,
                    total_steps=len(action_slice),
                    captures=captures,
                    max_captures=args.capture_count,
                    trans_step=trans_step,
                    rot_step_deg=rot_step_deg,
                )
                cv2.putText(
                    frame,
                    f"loop={cycle_idx}",
                    (8, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                cv2.imshow(win, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                key = cv2.waitKey(max(1, int(args.playback_delay_ms))) & 0xFF

                if key == 27:  # ESC
                    print("[manual-cam] exit by ESC")
                    exit_requested = True
                    break
                elif key in (ord("w"), ord("W")):
                    cam_pos[0] += trans_step
                elif key in (ord("s"), ord("S")):
                    cam_pos[0] -= trans_step
                elif key in (ord("a"), ord("A")):
                    cam_pos[1] -= trans_step
                elif key in (ord("d"), ord("D")):
                    cam_pos[1] += trans_step
                elif key in (ord("q"), ord("Q")):
                    cam_pos[2] += trans_step
                elif key in (ord("e"), ord("E")):
                    cam_pos[2] -= trans_step
                elif key in (ord("i"), ord("I")):
                    dq = _euler_to_quat_wxyz(0.0, 0.0, math.radians(rot_step_deg))
                    cam_quat = _quat_normalize_wxyz(_quat_multiply_wxyz(cam_quat, dq))
                elif key in (ord("k"), ord("K")):
                    dq = _euler_to_quat_wxyz(0.0, 0.0, math.radians(-rot_step_deg))
                    cam_quat = _quat_normalize_wxyz(_quat_multiply_wxyz(cam_quat, dq))
                elif key in (ord("j"), ord("J")):
                    dq = _euler_to_quat_wxyz(math.radians(-rot_step_deg), 0.0, 0.0)
                    cam_quat = _quat_normalize_wxyz(_quat_multiply_wxyz(cam_quat, dq))
                elif key in (ord("l"), ord("L")):
                    dq = _euler_to_quat_wxyz(math.radians(rot_step_deg), 0.0, 0.0)
                    cam_quat = _quat_normalize_wxyz(_quat_multiply_wxyz(cam_quat, dq))
                elif key in (ord("u"), ord("U")):
                    dq = _euler_to_quat_wxyz(0.0, math.radians(rot_step_deg), 0.0)
                    cam_quat = _quat_normalize_wxyz(_quat_multiply_wxyz(cam_quat, dq))
                elif key in (ord("o"), ord("O")):
                    dq = _euler_to_quat_wxyz(0.0, math.radians(-rot_step_deg), 0.0)
                    cam_quat = _quat_normalize_wxyz(_quat_multiply_wxyz(cam_quat, dq))
                elif key in (ord("+"), ord("=")):
                    trans_step = min(0.10, trans_step * 1.2)
                    rot_step_deg = min(30.0, rot_step_deg * 1.2)
                elif key in (ord("-"), ord("_")):
                    trans_step = max(0.001, trans_step / 1.2)
                    rot_step_deg = max(0.2, rot_step_deg / 1.2)
                elif key in (ord("c"), ord("C")):
                    if len(captures) < args.capture_count:
                        captures.append(
                            {
                                "pos": [float(v) for v in cam_pos.tolist()],
                                "quat": [float(v) for v in cam_quat.tolist()],
                            }
                        )
                        print(
                            f"[manual-cam] captured {len(captures)}/{args.capture_count} "
                            f"pos={captures[-1]['pos']} quat={captures[-1]['quat']}"
                        )
                elif key in (ord("x"), ord("X")):
                    if captures:
                        captures.pop()
                        print(f"[manual-cam] removed last capture, now {len(captures)}")

                step_idx += 1

            cv2.destroyAllWindows()

            if len(captures) == 0:
                print("[manual-cam] no capture saved; nothing written")
                return

            if len(captures) < args.capture_count:
                print(
                    f"[manual-cam] warning: only captured {len(captures)} poses "
                    f"(requested {args.capture_count})"
                )
                if exit_requested:
                    print("[manual-cam] exited early by ESC")

            out_cfg = _build_config(captures)
            out_path = Path(args.output_config)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w") as fp:
                json.dump(out_cfg, fp, indent=2)
            print(f"[manual-cam] wrote config: {out_path}")
        finally:
            env.close()


if __name__ == "__main__":
    main()
