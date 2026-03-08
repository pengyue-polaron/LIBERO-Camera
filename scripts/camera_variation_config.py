import json
import math
import hashlib
from pathlib import Path

import numpy as np
import camera_visibility


def build_camera_variation_task_uid_prefix(source_id):
    digest = hashlib.sha1()
    digest.update(str(Path(source_id).resolve()).encode("utf-8"))
    return f"t{digest.hexdigest()[:6]}"


def build_camera_variation_uid(source_id, variation_id, pose=None):
    return f"{build_camera_variation_task_uid_prefix(source_id)}-{int(variation_id):02d}"


def load_camera_variation_config(path):
    if not path:
        return None
    with open(path, "r") as f:
        cfg = json.load(f)
    if not isinstance(cfg, dict):
        raise ValueError("camera variation config must be a JSON object")
    return cfg


def get_config_count(cfg):
    if not cfg:
        return None
    count = cfg.get("count")
    if count is None:
        return None
    count = int(count)
    if count <= 0:
        raise ValueError("camera variation config 'count' must be > 0")
    return count


def get_effective_count(cli_count, cfg):
    cfg_count = get_config_count(cfg)
    if cfg_count is None and (cfg or {}).get("strategy") == "manual_poses":
        manual_poses = (cfg or {}).get("manual_poses", [])
        if len(manual_poses) == 0:
            raise ValueError("manual_poses strategy requires non-empty 'manual_poses'")
        cfg_count = len(manual_poses)
    if cfg_count is None:
        return cli_count
    if cli_count not in (0, cfg_count):
        raise ValueError(
            f"CLI camera variation count ({cli_count}) conflicts with config count ({cfg_count})"
        )
    return cfg_count


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


def _rotmat_to_quat_wxyz(R):
    R = np.asarray(R, dtype=np.float64)
    trace = np.trace(R)
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    else:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
    return _quat_normalize_wxyz(np.array([w, x, y, z], dtype=np.float64))


def _lookat_quat_wxyz(camera_pos, target_pos, up_ref):
    camera_pos = np.asarray(camera_pos, dtype=np.float64)
    target_pos = np.asarray(target_pos, dtype=np.float64)
    up_ref = np.asarray(up_ref, dtype=np.float64)

    forward = target_pos - camera_pos
    forward_norm = np.linalg.norm(forward)
    if forward_norm < 1e-9:
        raise ValueError("camera_pos and target_pos are identical")
    forward = forward / forward_norm

    z_axis = -forward  # MuJoCo camera looks along -Z
    x_axis = np.cross(up_ref, z_axis)
    x_norm = np.linalg.norm(x_axis)
    if x_norm < 1e-9:
        fallback_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        x_axis = np.cross(fallback_up, z_axis)
        x_norm = np.linalg.norm(x_axis)
        if x_norm < 1e-9:
            fallback_up = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            x_axis = np.cross(fallback_up, z_axis)
            x_norm = np.linalg.norm(x_axis)
        if x_norm < 1e-9:
            raise ValueError("Cannot construct look-at camera basis")
    x_axis = x_axis / x_norm
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)

    R = np.stack([x_axis, y_axis, z_axis], axis=1)
    return _rotmat_to_quat_wxyz(R)


def _parse_schedule(spec, count, name):
    if isinstance(spec, (int, float)):
        return np.full((count,), float(spec), dtype=np.float64)

    if isinstance(spec, list):
        if len(spec) != count:
            raise ValueError(
                f"{name} list length ({len(spec)}) must equal count ({count})"
            )
        return np.asarray(spec, dtype=np.float64)

    if isinstance(spec, dict):
        schedule_type = spec.get("type", "linspace")
        if schedule_type == "linspace":
            start = float(spec["start"])
            stop = float(spec["stop"])
            endpoint = bool(spec.get("endpoint", True))
            return np.linspace(start, stop, count, endpoint=endpoint, dtype=np.float64)
        raise ValueError(f"Unsupported {name} schedule type: {schedule_type}")

    raise ValueError(f"Unsupported schedule format for {name}: {type(spec)}")


def needs_target_pos(cfg):
    if not cfg:
        return False
    return cfg.get("strategy", "random_local") == "orbit_lookat"


def get_target_request(cfg):
    target_cfg = (cfg or {}).get("target", {})
    source = target_cfg.get("source", "eef_pos")
    offset = np.asarray(target_cfg.get("offset", [0.0, 0.0, 0.0]), dtype=np.float64)
    state_index = int(target_cfg.get("state_index", 0))
    position = target_cfg.get("position")
    if position is not None:
        position = np.asarray(position, dtype=np.float64)
    return {
        "source": source,
        "offset": offset,
        "state_index": state_index,
        "position": position,
    }


def _can_resize_spec(spec):
    return not isinstance(spec, list)


def _sample_random_local_pose(base_pos, base_quat, seed, variation_id, translate_range, rot_range_deg):
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
        "strategy": "random_local",
    }


def _sample_orbit_lookat_poses(base_pos, count, cfg, target_pos):
    orbit_cfg = cfg.get("orbit", {})
    up_ref = np.asarray(orbit_cfg.get("up_ref", [0.0, 0.0, 1.0]), dtype=np.float64)
    relative = bool(orbit_cfg.get("angles_relative_to_base", True))

    yaw_vals_deg = _parse_schedule(orbit_cfg.get("yaw_deg", {"type": "linspace", "start": -35, "stop": 35}), count, "yaw_deg")
    pitch_vals_deg = _parse_schedule(orbit_cfg.get("pitch_deg", -15.0), count, "pitch_deg")
    radius_scale_vals = _parse_schedule(orbit_cfg.get("radius_scale", 1.0), count, "radius_scale")
    radius_offset_vals = _parse_schedule(orbit_cfg.get("radius_offset", 0.0), count, "radius_offset")
    radius_offset_per_abs_yaw_deg = float(orbit_cfg.get("radius_offset_per_abs_yaw_deg", 0.0))
    radius_offset_per_abs_pitch_deg = float(orbit_cfg.get("radius_offset_per_abs_pitch_deg", 0.0))

    base_vec = np.asarray(base_pos, dtype=np.float64) - np.asarray(target_pos, dtype=np.float64)
    base_r = np.linalg.norm(base_vec)
    if base_r < 1e-9:
        raise ValueError("Base camera is too close to target for orbit_lookat")
    base_yaw = math.degrees(math.atan2(base_vec[1], base_vec[0]))
    base_pitch = math.degrees(math.atan2(base_vec[2], np.linalg.norm(base_vec[:2])))

    poses = []
    for variation_id in range(count):
        yaw_deg = float(yaw_vals_deg[variation_id])
        pitch_deg = float(pitch_vals_deg[variation_id])
        if relative:
            yaw_world_deg = base_yaw + yaw_deg
            pitch_world_deg = base_pitch + pitch_deg
        else:
            yaw_world_deg = yaw_deg
            pitch_world_deg = pitch_deg

        radius = float(base_r * radius_scale_vals[variation_id] + radius_offset_vals[variation_id])
        radius += radius_offset_per_abs_yaw_deg * abs(yaw_deg)
        radius += radius_offset_per_abs_pitch_deg * abs(pitch_deg)
        if radius <= 1e-6:
            raise ValueError(f"Invalid orbit radius for variation {variation_id}: {radius}")

        yaw_rad = math.radians(yaw_world_deg)
        pitch_rad = math.radians(pitch_world_deg)
        xy = radius * math.cos(pitch_rad)
        offset = np.array(
            [
                xy * math.cos(yaw_rad),
                xy * math.sin(yaw_rad),
                radius * math.sin(pitch_rad),
            ],
            dtype=np.float64,
        )
        pos = np.asarray(target_pos, dtype=np.float64) + offset
        quat = _lookat_quat_wxyz(pos, target_pos, up_ref)

        poses.append(
            {
                "variation_id": variation_id,
                "applied_pos": pos,
                "applied_quat": quat,
                "delta_pos": pos - np.asarray(base_pos, dtype=np.float64),
                "delta_rpy_deg": np.array([0.0, pitch_deg, yaw_deg], dtype=np.float64),
                "strategy": "orbit_lookat",
                "orbit_yaw_deg": yaw_deg,
                "orbit_pitch_deg": pitch_deg,
                "orbit_radius": radius,
                "target_pos": np.asarray(target_pos, dtype=np.float64),
            }
        )
    return poses


def _generate_candidate_poses(
    base_pos,
    base_quat,
    count,
    seed,
    translate_range,
    rot_range_deg,
    cfg,
    target_pos,
):
    if not cfg:
        return [
            _sample_random_local_pose(
                base_pos=base_pos,
                base_quat=base_quat,
                seed=seed,
                variation_id=i,
                translate_range=translate_range,
                rot_range_deg=rot_range_deg,
            )
            for i in range(count)
        ]

    strategy = cfg.get("strategy", "random_local")
    if strategy == "random_local":
        rng_cfg = cfg.get("random_local", {})
        tr = float(rng_cfg.get("translate_range", translate_range))
        rr = float(rng_cfg.get("rot_range_deg", rot_range_deg))
        return [
            _sample_random_local_pose(
                base_pos=base_pos,
                base_quat=base_quat,
                seed=seed,
                variation_id=i,
                translate_range=tr,
                rot_range_deg=rr,
            )
            for i in range(count)
        ]

    if strategy == "orbit_lookat":
        if target_pos is None:
            raise ValueError("orbit_lookat config requires target_pos")
        return _sample_orbit_lookat_poses(base_pos=base_pos, count=count, cfg=cfg, target_pos=target_pos)

    if strategy == "manual_poses":
        manual = (cfg or {}).get("manual_poses", [])
        if len(manual) == 0:
            raise ValueError("manual_poses strategy requires non-empty 'manual_poses'")
        if len(manual) < count:
            raise ValueError(
                f"manual_poses only has {len(manual)} entries, but need {count}"
            )
        poses = []
        for i in range(count):
            p = manual[i]
            pos = np.asarray(p["pos"], dtype=np.float64)
            quat = _quat_normalize_wxyz(np.asarray(p["quat"], dtype=np.float64))
            if pos.shape != (3,) or quat.shape != (4,):
                raise ValueError(f"Invalid manual pose at index {i}")
            poses.append(
                {
                    "variation_id": i,
                    "delta_pos": pos - np.asarray(base_pos, dtype=np.float64),
                    "delta_rpy_deg": np.array([0.0, 0.0, 0.0], dtype=np.float64),
                    "applied_pos": pos,
                    "applied_quat": quat,
                    "strategy": "manual_poses",
                }
            )
        return poses

    raise ValueError(f"Unsupported camera variation strategy: {strategy}")


def _choose_evenly(accepted, target_count):
    if len(accepted) < target_count:
        raise ValueError(
            f"Only found {len(accepted)} valid camera poses, but need {target_count}"
        )
    if len(accepted) == target_count:
        return accepted
    idxs = np.linspace(0, len(accepted) - 1, target_count, dtype=int)
    selected = []
    used = set()
    for idx in idxs.tolist():
        while idx in used and idx + 1 < len(accepted):
            idx += 1
        used.add(idx)
        selected.append(accepted[idx])
    return selected


def _pose_feature_vector(pose):
    if "orbit_yaw_deg" in pose:
        return np.array(
            [
                float(pose.get("orbit_yaw_deg", 0.0)),
                float(pose.get("orbit_pitch_deg", 0.0)),
                float(pose.get("orbit_radius", 0.0)),
            ],
            dtype=np.float64,
        )
    delta_pos = np.asarray(pose.get("delta_pos", [0.0, 0.0, 0.0]), dtype=np.float64)
    delta_rpy = np.asarray(pose.get("delta_rpy_deg", [0.0, 0.0, 0.0]), dtype=np.float64)
    return np.concatenate([delta_pos, delta_rpy], axis=0)


def _choose_diverse_fps(accepted, target_count):
    if len(accepted) < target_count:
        raise ValueError(
            f"Only found {len(accepted)} valid camera poses, but need {target_count}"
        )
    if len(accepted) == target_count:
        return list(accepted)

    feats = np.stack([_pose_feature_vector(p) for p in accepted], axis=0)
    std = np.std(feats, axis=0)
    std[std < 1e-9] = 1.0
    feats = (feats - np.mean(feats, axis=0)) / std

    seed_idx = int(np.argmax(np.linalg.norm(feats, axis=1)))
    selected = [seed_idx]
    min_dists = np.linalg.norm(feats - feats[seed_idx : seed_idx + 1], axis=1)

    while len(selected) < target_count:
        min_dists[selected] = -1.0
        next_idx = int(np.argmax(min_dists))
        selected.append(next_idx)
        d = np.linalg.norm(feats - feats[next_idx : next_idx + 1], axis=1)
        min_dists = np.minimum(min_dists, d)

    selected = sorted(set(selected))
    while len(selected) > target_count:
        selected.pop()
    return [accepted[i] for i in selected]


def generate_camera_variation_poses(
    base_pos,
    base_quat,
    count,
    seed,
    translate_range,
    rot_range_deg,
    cfg=None,
    target_pos=None,
    validator=None,
):
    if count <= 0:
        return []

    if (cfg or {}).get("strategy") == "manual_poses":
        poses = _generate_candidate_poses(
            base_pos=base_pos,
            base_quat=base_quat,
            count=count,
            seed=seed,
            translate_range=translate_range,
            rot_range_deg=rot_range_deg,
            cfg=cfg,
            target_pos=target_pos,
        )
        for new_id, pose in enumerate(poses):
            pose["candidate_variation_id"] = int(pose["variation_id"])
            pose["variation_id"] = new_id
        return poses

    if validator is None or not camera_visibility.constraints_enabled(cfg):
        return _generate_candidate_poses(
            base_pos=base_pos,
            base_quat=base_quat,
            count=count,
            seed=seed,
            translate_range=translate_range,
            rot_range_deg=rot_range_deg,
            cfg=cfg,
            target_pos=target_pos,
        )

    candidate_count = camera_visibility.get_candidate_pool_size(cfg, count)
    strategy = (cfg or {}).get("strategy", "random_local")
    if strategy == "orbit_lookat":
        orbit_cfg = cfg.get("orbit", {})
        resizeable = all(
            _can_resize_spec(orbit_cfg.get(key, default))
            for key, default in (
                ("yaw_deg", {"type": "linspace", "start": -35, "stop": 35}),
                ("pitch_deg", -15.0),
                ("radius_scale", 1.0),
                ("radius_offset", 0.0),
            )
        )
        if not resizeable:
            candidate_count = count

    candidates = _generate_candidate_poses(
        base_pos=base_pos,
        base_quat=base_quat,
        count=candidate_count,
        seed=seed,
        translate_range=translate_range,
        rot_range_deg=rot_range_deg,
        cfg=cfg,
        target_pos=target_pos,
    )
    accepted = [pose for pose in candidates if validator(pose)]
    if len(accepted) < count:
        vc = camera_visibility.get_constraints(cfg)
        fallback_mode = str(vc.get("fallback", "unfiltered")).lower()
        if fallback_mode in ("none", "strict"):
            raise ValueError(
                f"Only found {len(accepted)} valid camera poses, but need {count}"
            )

        # Keep whatever passed visibility checks, then fill the rest from
        # deterministic unfiltered poses so batch processing doesn't fail hard.
        accepted_sorted = sorted(accepted, key=lambda x: int(x["variation_id"]))
        selected = list(accepted_sorted[:count])
        if len(selected) < count:
            fallback_candidates = _generate_candidate_poses(
                base_pos=base_pos,
                base_quat=base_quat,
                count=count,
                seed=seed,
                translate_range=translate_range,
                rot_range_deg=rot_range_deg,
                cfg=cfg,
                target_pos=target_pos,
            )
            used_ids = {int(p["variation_id"]) for p in selected}
            for pose in fallback_candidates:
                pose_id = int(pose["variation_id"])
                if pose_id in used_ids:
                    continue
                selected.append(pose)
                used_ids.add(pose_id)
                if len(selected) >= count:
                    break
        print(
            f"[camera-variation] visibility accepted {len(accepted)}/{candidate_count}, "
            f"fallback='{fallback_mode}', final={len(selected)}"
        )
    else:
        selection_cfg = (cfg or {}).get("selection", {})
        selection_mode = str(selection_cfg.get("mode", "diverse_fps")).lower()
        if selection_mode == "diverse_fps":
            selected = _choose_diverse_fps(accepted, count)
        elif selection_mode == "evenly":
            selected = _choose_evenly(accepted, count)
        else:
            raise ValueError(f"Unsupported selection mode: {selection_mode}")
    for new_id, pose in enumerate(selected):
        pose["candidate_variation_id"] = int(pose["variation_id"])
        pose["variation_id"] = new_id
    return selected


def default_example_config():
    return {
        "strategy": "orbit_lookat",
        "count": 7,
        "target": {
            "source": "eef_pos",
            "state_index": 0,
            "offset": [0.0, 0.0, 0.08],
        },
        "orbit": {
            "angles_relative_to_base": True,
            "yaw_deg": [-55.0, -36.0, -18.0, 0.0, 18.0, 36.0, 55.0],
            "pitch_deg": [8.0, 10.0, 13.0, 16.0, 13.0, 10.0, 8.0],
            "radius_scale": [1.18, 1.12, 1.06, 1.0, 1.06, 1.12, 1.18],
            "radius_offset": 0.07,
            "radius_offset_per_abs_yaw_deg": 0.0,
            "radius_offset_per_abs_pitch_deg": 0.0,
            "up_ref": [0.0, 0.0, 1.0],
        },
        "visibility_constraints": {
            "enabled": False,
        },
    }


def save_default_example_config(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(default_example_config(), f, indent=2)
