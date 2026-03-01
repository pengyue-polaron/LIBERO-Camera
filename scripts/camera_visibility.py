import math

import numpy as np


def constraints_enabled(cfg):
    return bool((cfg or {}).get("visibility_constraints", {}).get("enabled", False))


def get_constraints(cfg):
    return (cfg or {}).get("visibility_constraints", {})


def get_candidate_pool_size(cfg, target_count):
    vc = get_constraints(cfg)
    pool_factor = int(vc.get("candidate_pool_factor", 8))
    max_trials = int(vc.get("max_sampling_trials", max(target_count, target_count * pool_factor)))
    return max(target_count, max_trials)


def get_check_state_indices(num_states, cfg):
    vc = get_constraints(cfg)
    specs = vc.get("check_frames", [0.0, 0.5, 1.0])
    if num_states <= 0:
        return []

    indices = []
    for spec in specs:
        if isinstance(spec, float) and 0.0 <= spec <= 1.0:
            idx = int(round(spec * max(0, num_states - 1)))
        else:
            idx = int(spec)
            if idx < 0:
                idx += num_states
        idx = max(0, min(num_states - 1, idx))
        indices.append(idx)
    return sorted(set(indices))


def _project_world_points_to_pixels(sim, camera_name, world_points, image_h, image_w):
    cam_id = sim.model.camera_name2id(camera_name)
    cam_pos = np.asarray(sim.data.cam_xpos[cam_id], dtype=np.float64)
    cam_rot = np.asarray(sim.data.cam_xmat[cam_id], dtype=np.float64).reshape(3, 3)

    rel = np.asarray(world_points, dtype=np.float64) - cam_pos[None, :]
    cam_pts = rel @ cam_rot

    fovy_deg = float(sim.model.cam_fovy[cam_id])
    fy = 0.5 * float(image_h) / math.tan(math.radians(fovy_deg) * 0.5)
    fx = fy
    cx = (float(image_w) - 1.0) * 0.5
    cy = (float(image_h) - 1.0) * 0.5

    pixels = []
    visible = True
    for x_cam, y_cam, z_cam in cam_pts:
        depth = -z_cam
        if depth <= 1e-6:
            visible = False
            pixels.append(None)
            continue
        u = fx * (x_cam / depth) + cx
        v = cy - fy * (y_cam / depth)
        pixels.append((float(u), float(v)))
    return pixels, visible


def _point_in_image(pt, image_h, image_w, margin_px):
    x, y = pt
    return margin_px <= x < (image_w - margin_px) and margin_px <= y < (image_h - margin_px)


def _body_pos_if_exists(sim, body_name):
    try:
        body_id = sim.model.body_name2id(body_name)
    except Exception:
        return None
    return np.asarray(sim.data.body_xpos[body_id], dtype=np.float64)


def _site_corners_if_exists(sim, site_name):
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
    return pos[None, :] + local @ rot.T


def _collect_goal_regions(parsed_problem):
    region_dict = parsed_problem.get("regions", {})
    goal_state = parsed_problem.get("goal_state", [])
    goal_regions = set()

    def walk(node):
        if isinstance(node, (list, tuple)):
            for item in node:
                walk(item)
        elif isinstance(node, str) and node in region_dict:
            goal_regions.add(node)

    walk(goal_state)
    return goal_regions


def _build_cameras_dict(pose, camera_name="agentview"):
    return {
        camera_name: {
            "pos": " ".join(f"{float(v):.10f}" for v in pose["applied_pos"]),
            "quat": " ".join(f"{float(v):.10f}" for v in pose["applied_quat"]),
        }
    }


def make_pose_validator(
    env,
    model_xml,
    states,
    reset_fn,
    cfg,
    camera_name="agentview",
):
    vc = get_constraints(cfg)
    margin_px = int(vc.get("pixel_margin", 8))
    require_goal_region_visible = bool(vc.get("require_goal_region_visible", True))
    require_obj_of_interest_visible = bool(vc.get("require_obj_of_interest_visible", True))
    require_eef_visible = bool(vc.get("require_eef_visible", True))
    obj_names = list(getattr(env, "parsed_problem", {}).get("obj_of_interest", []))
    goal_regions = _collect_goal_regions(getattr(env, "parsed_problem", {}))
    state_indices = get_check_state_indices(len(states), cfg)

    def validator(pose):
        cameras_dict = _build_cameras_dict(pose, camera_name=camera_name)
        for state_idx in state_indices:
            obs = reset_fn(env, model_xml, states[state_idx], cameras_dict)
            image_h, image_w = obs["agentview_image"].shape[:2]

            if require_eef_visible:
                if "robot0_eef_pos" not in obs:
                    return False
                pixels, visible = _project_world_points_to_pixels(
                    env.sim,
                    camera_name,
                    np.asarray([obs["robot0_eef_pos"]], dtype=np.float64),
                    image_h,
                    image_w,
                )
                if not visible or pixels[0] is None or not _point_in_image(
                    pixels[0], image_h, image_w, margin_px
                ):
                    return False

            if require_obj_of_interest_visible:
                for obj_name in obj_names:
                    pos = _body_pos_if_exists(env.sim, obj_name)
                    if pos is None:
                        continue
                    pixels, visible = _project_world_points_to_pixels(
                        env.sim,
                        camera_name,
                        np.asarray([pos], dtype=np.float64),
                        image_h,
                        image_w,
                    )
                    if not visible or pixels[0] is None or not _point_in_image(
                        pixels[0], image_h, image_w, margin_px
                    ):
                        return False

            if require_goal_region_visible:
                for region_name in goal_regions:
                    world_corners = _site_corners_if_exists(env.sim, region_name)
                    if world_corners is None:
                        continue
                    pixels, visible = _project_world_points_to_pixels(
                        env.sim, camera_name, world_corners, image_h, image_w
                    )
                    if not visible or any(p is None for p in pixels):
                        return False
                    if not all(_point_in_image(p, image_h, image_w, margin_px) for p in pixels):
                        return False

        return True

    return validator
