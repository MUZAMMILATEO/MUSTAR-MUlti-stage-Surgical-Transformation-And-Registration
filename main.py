import os
import argparse
import datetime
import pathlib
import open3d as o3d
import sys
sys.path.append('./TEASER-plusplus/examples/teaser_python_ply/')
import numpy as np
import time
import cv2
import lietorch
import torch
import tqdm
import yaml
from mast3r_slam.global_opt import FactorGraph

from mast3r_slam.config import load_config, config, set_global_config
from mast3r_slam.dataloader import Intrinsics, load_dataset
import mast3r_slam.evaluate as eval
from mast3r_slam.frame import Mode, SharedKeyframes, SharedStates, create_frame, SharedPointCloud
from mast3r_slam.mast3r_utils import (
    load_mast3r,
    load_retriever,
    mast3r_inference_mono,
)
from mast3r_slam.multiprocess_utils import new_queue, try_get_msg
from mast3r_slam.tracker import FrameTracker
from mast3r_slam.visualization_sfm_teaser_double import WindowMsg, run_visualization
import torch.multiprocessing as mp

from io import BytesIO
from plyfile import PlyData, PlyElement
from teaser_python_ply_rt_mixed import register_SfM_SLAM
from teaser_pre_intra import pre_intra_reg, pre_intra_reg_refine

from scipy.spatial import cKDTree

def current_point_cloud(keyframes, c_conf_threshold):

    keyframe = keyframes.last_keyframe()
    if config["use_calib"]:
        X_canon = constrain_points_to_ray(
            keyframe.img_shape.flatten()[:2], keyframe.X_canon[None], keyframe.K
        )
        keyframe.X_canon = X_canon.squeeze(0)
    pW = keyframe.T_WC.act(keyframe.X_canon).cpu().numpy().reshape(-1, 3)
    color = (keyframe.uimg.cpu().numpy() * 255).astype(np.uint8).reshape(-1, 3)
    valid = (
        keyframe.get_average_conf().cpu().numpy().astype(np.float32).reshape(-1)
        > c_conf_threshold
    )
    npW = pW[valid]
    ncolor = color[valid]
    
    return current_point_cloud_helper(npW, ncolor)
    
def current_point_cloud_helper(points, colors):
    colors = colors.astype(np.uint8)
    # Combine XYZ and RGB into a structured array
    pcd = np.empty(
        len(points),
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )
    pcd["x"], pcd["y"], pcd["z"] = points.T
    pcd["red"], pcd["green"], pcd["blue"] = colors.T
    vertex_element = PlyElement.describe(pcd, "vertex")
    return PlyData([vertex_element], text=False)
    

def save_transformed_pre_cloud(static_pre_pcd, trans_pre_cloud, static_pre_pcd_with_interior, static_pre_pcd_with_interior_path):

    # Save the transformed point cloud data to the file.
    # Check if the point cloud contains color information
    if not static_pre_pcd.has_colors():
        print("Warning: The input file does not contain color information.")
        # You can choose to continue (writing only points) or raise an error.
    
    # Create a new point cloud with just the points and colors (if available)
    new_pre_pcd = o3d.geometry.PointCloud()
    new_pre_pcd.points = static_pre_pcd.points
    if static_pre_pcd.has_colors():
        new_pre_pcd.colors = static_pre_pcd.colors

    # Save the new point cloud to file
    o3d.io.write_point_cloud(trans_pre_cloud, new_pre_pcd)
    
    if static_pre_pcd_with_interior is not None:
        # Save the transformed point cloud data to the file.
        # Check if the point cloud contains color information
        if not static_pre_pcd_with_interior.has_colors():
            print("Warning: The input file does not contain color information.")
            # You can choose to continue (writing only points) or raise an error.
    
        # Create a new point cloud with just the points and colors (if available)
        new_pre_pcd_with_int = o3d.geometry.PointCloud()
        new_pre_pcd_with_int.points = static_pre_pcd_with_interior.points
        if static_pre_pcd_with_interior.has_colors():
            new_pre_pcd_with_int.colors = static_pre_pcd_with_interior.colors

        # Save the new point cloud to file
        o3d.io.write_point_cloud(static_pre_pcd_with_interior_path, new_pre_pcd_with_int)
        

def create_ply_from_keyframe(keyframe, c_conf_threshold, filename):
    # Optionally apply calibration if enabled.
    if config["use_calib"]:
        X_canon = constrain_points_to_ray(
            keyframe.img_shape.flatten()[:2], keyframe.X_canon[None], keyframe.K
        )
        keyframe.X_canon = X_canon.squeeze(0)
    
    # Transform keyframe points to world coordinates.
    pW = keyframe.T_WC.act(keyframe.X_canon).cpu().numpy().reshape(-1, 3)
    color = (keyframe.uimg.cpu().numpy() * 255).astype(np.uint8).reshape(-1, 3)
    
    # Determine valid points based on the confidence threshold.
    valid = keyframe.get_average_conf().cpu().numpy().astype(np.float32).reshape(-1) > 0 # c_conf_threshold
    pW_valid = pW[valid]
    color_valid = color[valid]
    
    # Combine the valid XYZ and RGB values into a structured array.
    pcd = np.empty(len(pW_valid),
                   dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"),
                          ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    pcd["x"], pcd["y"], pcd["z"] = pW_valid.T
    pcd["red"], pcd["green"], pcd["blue"] = color_valid.T
    
    # Create a PlyElement and PlyData structure.
    vertex_element = PlyElement.describe(pcd, "vertex")
    ply_data = PlyData([vertex_element], text=False)
    
    ply_data.write(filename)
    
def apply_registration_transform(static_pcd, keyframes, last_msg, pre_static_pcd, pre_static_pcd_with_interior=None):
    """
    Saves the static point cloud and the first keyframe's point cloud to temporary PLY files,
    calls the registration function, and then updates the static point cloud with the transformed
    (registered) source point cloud.

    Parameters:
        static_pcd: An object with attributes 'path_to_ply', 'lock', and a method 'get_point_cloud()'.
                    The point cloud data is expected to be a dict with a "points" key.
        keyframes:  A list of keyframe objects; the first keyframe is used for registration.
        last_msg:   An object containing the attribute 'C_conf_threshold' used for the keyframe PLY creation.

    Returns:
        The updated static_pcd object with its point cloud transformed.
    """
    import pathlib
    import numpy as np

    # Build the destination path for the keyframe PLY file.
    dst_path = pathlib.Path(static_pcd.path_to_ply).parent / "keyframe0.ply"
    
    # Get the first keyframe.
    f_keyframe = keyframes[0]
    
    # Create the keyframe PLY file using the confidence threshold.
    create_ply_from_keyframe(f_keyframe, last_msg.C_conf_threshold, dst_path)
    
    # Call the registration function to obtain transformation matrices, scaling factors, and the registered source.
    # Now the registration function returns:
    # T1, T2, S1, S2, FT, registered_source
    T1, T2, S1, S2, FT, registered_source, updated_static_pre_cloud, updated_pre_static_pcd_with_interior = register_SfM_SLAM(static_pcd.path_to_ply, dst_path, pre_static_pcd.path_to_ply, pre_static_pcd_with_interior)
    
    # Convert the registered source (an Open3D point cloud) to a NumPy array of points.
    registered_points = np.asarray(registered_source.points)
    updated_static_pre_cloud_points = np.asarray(updated_static_pre_cloud.points)
    
    if updated_pre_static_pcd_with_interior is not None:
        registered_points_with_interior_points = np.asarray(updated_pre_static_pcd_with_interior.points)
    
    # Update the static_pcd's point cloud safely.
    with static_pcd.lock:
        static_pcd.point_cloud["points"] = registered_points
        pre_static_pcd.point_cloud["points"] = updated_static_pre_cloud_points
        if updated_pre_static_pcd_with_interior is not None:
            pre_static_pcd_with_interior.point_cloud["points"] = registered_points_with_interior_points
        else:
            pre_static_pcd_with_interior = None
    
    return static_pcd, pre_static_pcd, pre_static_pcd_with_interior

def relocalization(frame, keyframes, factor_graph, retrieval_database):
    # we are adding and then removing from the keyframe, so we need to be careful.
    # The lock slows viz down but safer this way...
    with keyframes.lock:
        kf_idx = []
        retrieval_inds = retrieval_database.update(
            frame,
            add_after_query=False,
            k=config["retrieval"]["k"],
            min_thresh=config["retrieval"]["min_thresh"],
        )
        kf_idx += retrieval_inds
        successful_loop_closure = False
        if kf_idx:
            keyframes.append(frame)
            n_kf = len(keyframes)
            kf_idx = list(kf_idx)  # convert to list
            frame_idx = [n_kf - 1] * len(kf_idx)
            print("RELOCALIZING against kf ", n_kf - 1, " and ", kf_idx)
            if factor_graph.add_factors(
                frame_idx,
                kf_idx,
                config["reloc"]["min_match_frac"],
                is_reloc=config["reloc"]["strict"],
            ):
                retrieval_database.update(
                    frame,
                    add_after_query=True,
                    k=config["retrieval"]["k"],
                    min_thresh=config["retrieval"]["min_thresh"],
                )
                print("Success! Relocalized")
                successful_loop_closure = True
                keyframes.T_WC[n_kf - 1] = keyframes.T_WC[kf_idx[0]].clone()
            else:
                keyframes.pop_last()
                print("Failed to relocalize")

        if successful_loop_closure:
            if config["use_calib"]:
                factor_graph.solve_GN_calib()
            else:
                factor_graph.solve_GN_rays()
        return successful_loop_closure


def run_backend(states, keyframes):
    mode = states.get_mode()
    if mode == Mode.INIT or states.is_paused():
        return
    if mode == Mode.RELOC:
        frame = states.get_frame()
        success = relocalization(frame, keyframes, factor_graph, retrieval_database)
        if success:
            states.set_mode(Mode.TRACKING)
        states.dequeue_reloc()
        return
    idx = -1
    with states.lock:
        if len(states.global_optimizer_tasks) > 0:
            idx = states.global_optimizer_tasks[0]
    if idx == -1:
        return
    # Graph Construction
    kf_idx = []
    # k to previous consecutive keyframes
    n_consec = 1
    for j in range(min(n_consec, idx)):
        kf_idx.append(idx - 1 - j)
    frame = keyframes[idx]
    retrieval_inds = retrieval_database.update(
        frame,
        add_after_query=True,
        k=config["retrieval"]["k"],
        min_thresh=config["retrieval"]["min_thresh"],
    )
    kf_idx += retrieval_inds

    lc_inds = set(retrieval_inds)
    lc_inds.discard(idx - 1)
    if len(lc_inds) > 0:
        print("Database retrieval", idx, ": ", lc_inds)

    kf_idx = set(kf_idx)  # Remove duplicates by using set
    kf_idx.discard(idx)  # Remove current kf idx if included
    kf_idx = list(kf_idx)  # convert to list
    frame_idx = [idx] * len(kf_idx)
    if kf_idx:
        factor_graph.add_factors(
            kf_idx, frame_idx, config["local_opt"]["min_match_frac"]
        )

    with states.lock:
        states.edges_ii[:] = factor_graph.ii.cpu().tolist()
        states.edges_jj[:] = factor_graph.jj.cpu().tolist()

    if config["use_calib"]:
        factor_graph.solve_GN_calib()
    else:
        factor_graph.solve_GN_rays()

    with states.lock:
        if len(states.global_optimizer_tasks) > 0:
            idx = states.global_optimizer_tasks.pop(0)


if __name__ == "__main__":
    mp.set_start_method("spawn")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_grad_enabled(False)
    device = "cuda:0"
    save_frames = False
    datetime_now = str(datetime.datetime.now()).replace(" ", "_")
    repeat_reg = True
    PRE_CLOUD_PATH = "/home/khanm2204/MASt3R-SLAM/pre_cloud/kyoto_CT.ply"
    trans_pre_cloud = os.path.join(os.path.dirname(PRE_CLOUD_PATH), "kyoto_CT_transformed.ply")
    trans_pre_cloud_refined = os.path.join(os.path.dirname(PRE_CLOUD_PATH), "kyoto_CT_reformed.ply")
    trans_pre_cloud_with_interior = os.path.join(os.path.dirname(PRE_CLOUD_PATH), "kyoto_CT_with_interior.ply")
    trans_pre_cloud_with_interior_0 = os.path.join(os.path.dirname(PRE_CLOUD_PATH), "kyoto_CT_with_interior_transformed.ply")
    trans_pre_cloud_with_interior_1 = os.path.join(os.path.dirname(PRE_CLOUD_PATH), "kyoto_CT_with_interior_refined.ply")
    curr_point_cloud = None
    pre_static_pcd_with_interior = None
    registration_loss = 0

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="datasets/tum/rgbd_dataset_freiburg1_desk")
    parser.add_argument("--config", default="config/base.yaml")
    parser.add_argument("--save-as", default="default")
    parser.add_argument("--no-viz", action="store_true")
    parser.add_argument("--calib", default="")

    args = parser.parse_args()

    load_config(args.config)
    print(args.dataset)
    print(config)

    manager = mp.Manager()
    main2viz = new_queue(manager, args.no_viz)
    viz2main = new_queue(manager, args.no_viz)

    dataset = load_dataset(args.dataset)
    dataset.subsample(config["dataset"]["subsample"])
    h, w = dataset.get_img_shape()[0]

    if args.calib:
        with open(args.calib, "r") as f:
            intrinsics = yaml.load(f, Loader=yaml.SafeLoader)
        config["use_calib"] = True
        dataset.use_calibration = True
        dataset.camera_intrinsics = Intrinsics.from_calib(
            dataset.img_size,
            intrinsics["width"],
            intrinsics["height"],
            intrinsics["calibration"],
        )

    keyframes = SharedKeyframes(manager, h, w)
    states = SharedStates(manager, h, w)
    static_pcd = SharedPointCloud("/home/khanm2204/MASt3R-SLAM/sfm_pts/points3D.ply")
    _, _, _, _, _, static_pre_pcd, static_pre_pcd_with_interior = pre_intra_reg(static_pcd.path_to_ply, PRE_CLOUD_PATH, trans_pre_cloud_with_interior, 0.11, 0.009)    
    save_transformed_pre_cloud(static_pre_pcd, trans_pre_cloud, static_pre_pcd_with_interior, trans_pre_cloud_with_interior_0)
    
    # load the transformed preoperative and static_pcd point cloud
    
    
    static_pre_pcd, static_pre_pcd_with_interior = pre_intra_reg_refine(trans_pre_cloud, static_pcd.path_to_ply, 0.0, 0.05, trans_pre_cloud_with_interior_0) 
    save_transformed_pre_cloud(static_pre_pcd, trans_pre_cloud_refined, static_pre_pcd_with_interior, trans_pre_cloud_with_interior_1)
    
    pre_static_pcd = SharedPointCloud(trans_pre_cloud_refined)
    if static_pre_pcd_with_interior is not None:
        pre_static_pcd_with_interior = SharedPointCloud(trans_pre_cloud_with_interior_1)
    else:
        pre_static_pcd_with_interior = None

    last_msg = WindowMsg()

    if not args.no_viz:
        viz = mp.Process(
            target=run_visualization,
            args=(config, states, keyframes, main2viz, viz2main, static_pcd, pre_static_pcd, registration_loss, last_msg),
        )
        viz.start()

    model = load_mast3r(device=device)
    model.share_memory()

    has_calib = dataset.has_calib()
    use_calib = config["use_calib"]

    if use_calib and not has_calib:
        print("[Warning] No calibration provided for this dataset!")
        sys.exit(0)
    K = None
    if use_calib:
        K = torch.from_numpy(dataset.camera_intrinsics.K_frame).to(
            device, dtype=torch.float32
        )
        keyframes.set_intrinsics(K)

    # remove the trajectory from the previous run
    if dataset.save_results:
        save_dir, seq_name = eval.prepare_savedir(args, dataset)
        traj_file = save_dir / f"{seq_name}.txt"
        recon_file = save_dir / f"{seq_name}.ply"
        if traj_file.exists():
            traj_file.unlink()
        if recon_file.exists():
            recon_file.unlink()

    tracker = FrameTracker(model, keyframes, device)
    # last_msg = WindowMsg()

    factor_graph = FactorGraph(model, keyframes, K, device)
    retrieval_database = load_retriever(model)

    i = 0
    fps_timer = time.time()

    frames = []

    while True:
        mode = states.get_mode()
        msg = try_get_msg(viz2main)
        last_msg = msg if msg is not None else last_msg
        if last_msg.is_terminated:
            states.set_mode(Mode.TERMINATED)
            break

        if last_msg.is_paused and not last_msg.next:
            states.pause()
            time.sleep(0.01)
            continue

        if not last_msg.is_paused:
            states.unpause()

        if i == len(dataset):
            states.set_mode(Mode.TERMINATED)
            break

        timestamp, img = dataset[i]
        if save_frames:
            frames.append(img)

        # get frames last camera pose
        T_WC = (
            lietorch.Sim3.Identity(1, device=device)
            if i == 0
            else states.get_frame().T_WC
        )
        frame = create_frame(i, img, T_WC, img_size=dataset.img_size, device=device)

        if mode == Mode.INIT:
            # Initialize via mono inference, and encoded features neeed for database
            X_init, C_init = mast3r_inference_mono(model, frame)
            frame.update_pointmap(X_init, C_init)
            keyframes.append(frame)
            states.queue_global_optimization(len(keyframes) - 1)
            states.set_mode(Mode.TRACKING)
            states.set_frame(frame)
            i += 1
            continue

        if mode == Mode.TRACKING:
            add_new_kf, match_info, try_reloc = tracker.track(frame)
            if try_reloc:
                states.set_mode(Mode.RELOC)
            states.set_frame(frame)

        elif mode == Mode.RELOC:
            X, C = mast3r_inference_mono(model, frame)
            frame.update_pointmap(X, C)
            states.set_frame(frame)
            states.queue_reloc()
        else:
            raise Exception("Invalid mode")

        if add_new_kf:
            keyframes.append(frame)
            states.queue_global_optimization(len(keyframes) - 1)
            
            # Only when a new keyframe is added, compute the current point cloud.
            curr_point_cloud = current_point_cloud(keyframes, last_msg.C_conf_threshold)
            compute_registration_loss = True

        run_backend(states, keyframes)

        # log time
        if i % 30 == 0:
            FPS = i / (time.time() - fps_timer)
            print(f"FPS: {FPS}")
        i += 1
        
        if repeat_reg:
            # === BEGIN REGISTRATION CALL ===
            # This block saves the static point cloud and the first keyframe's point cloud
            # to temporary PLY files, then calls register_SfM_SLAM.
    
            static_pcd, pre_static_pcd, pre_static_pcd_with_interior = apply_registration_transform(static_pcd, keyframes, last_msg, pre_static_pcd, pre_static_pcd_with_interior)
            
            if pre_static_pcd_with_interior is None:
                # Create a payload with the updated static point cloud data:
                updated_data = {
                    "points": static_pcd.point_cloud["points"],
                    "colors": static_pcd.point_cloud["colors"],
                }
                # Send a tuple that includes both the command and the new data.
                main2viz.put(("update_static", updated_data))
            
                updated_pre = {
                    "points": pre_static_pcd.point_cloud["points"],
                    # Optionally include colors if available:
                    "colors": pre_static_pcd.point_cloud.get("colors", None)
                }
                main2viz.put(("update_pre_static", updated_pre))
            
                pre_points = pre_static_pcd.point_cloud["points"]  # pre_static_pcd is updated in the registration block.
                kdtree = cKDTree(pre_points)
            else:
                # Create a payload with the updated static point cloud data:
                updated_data = {
                    "points": static_pcd.point_cloud["points"],
                    "colors": static_pcd.point_cloud["colors"],
                }
                # Send a tuple that includes both the command and the new data.
                main2viz.put(("update_static", updated_data))
            
                updated_pre = {
                    "points": pre_static_pcd_with_interior.point_cloud["points"],
                    # Optionally include colors if available:
                    "colors": pre_static_pcd_with_interior.point_cloud.get("colors", None)
                }
                main2viz.put(("update_pre_static", updated_pre))
            
                pre_points = pre_static_pcd.point_cloud["points"]  # pre_static_pcd is updated in the registration block.
                kdtree = cKDTree(pre_points)
                
            
            repeat_reg = False

            # === END REGISTRATION CALL ===
            
        if (curr_point_cloud is not None) and (compute_registration_loss):
            vertices = curr_point_cloud["vertex"].data
            curr_points = np.column_stack((vertices["x"], vertices["y"], vertices["z"]))
            # Query the KDTree for the nearest neighbor distances.
            distances, _ = kdtree.query(curr_points, k=1)
            
            # Compute the registration loss metric as the mean squared distance.
            registration_loss = np.mean(distances**2)
            # Send the updated loss to the visualization process.
            main2viz.put(("update_registration_loss", registration_loss))
            compute_registration_loss = False

    if dataset.save_results:
        save_dir, seq_name = eval.prepare_savedir(args, dataset)
        eval.save_traj(save_dir, f"{seq_name}.txt", dataset.timestamps, keyframes)
        eval.save_reconstruction(
            save_dir,
            f"{seq_name}.ply",
            keyframes,
            last_msg.C_conf_threshold,
        )
        eval.save_keyframes(
            save_dir / "keyframes" / seq_name, dataset.timestamps, keyframes
        )
    if save_frames:
        savedir = pathlib.Path(f"logs/frames/{datetime_now}")
        savedir.mkdir(exist_ok=True, parents=True)
        for i, frame in tqdm.tqdm(enumerate(frames), total=len(frames)):
            frame = (frame * 255).clip(0, 255)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f"{savedir}/{i}.png", frame)

    print("done")
    if not args.no_viz:
        viz.join()
