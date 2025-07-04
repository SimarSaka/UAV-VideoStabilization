# UAV-VideoStabilization
import cv2
import numpy as np
import os
from scipy.interpolate import UnivariateSpline
from scipy.signal import medfilt
def load_frames_from_directory(frames_path):
    """Load frames from directory in sorted order"""
    frame_files = []
    for filename in sorted(os.listdir(frames_path)):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            frame_files.append(os.path.join(frames_path, filename))
    return frame_files

def detect_and_match_features_fast(img1, img2):
    # FAST keypoints + ORB descriptors for speed and robustness
    fast = cv2.FastFeatureDetector_create(threshold=60)
    orb = cv2.ORB_create(nfeatures=1000)
    kp1 = fast.detect(img1, None)
    kp2 = fast.detect(img2, None)
    kp1, des1 = orb.compute(img1, kp1)
    kp2, des2 = orb.compute(img2, kp2)
    if des1 is None or des2 is None:
        return np.array([]), np.array([])
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    if len(matches) < 4:
        return np.array([]), np.array([])
    good_matches = matches[:min(50, len(matches))]
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])
    return pts1, pts2

def filter_motion_outliers(pts1, pts2):
    motion_vectors = pts2 - pts1
    motion_magnitudes = np.linalg.norm(motion_vectors, axis=1)
    median_motion = np.median(motion_magnitudes)
    mad = np.median(np.abs(motion_magnitudes - median_motion))
    threshold = median_motion + 2 * mad
    static_mask = motion_magnitudes < threshold
    return pts1[static_mask], pts2[static_mask]

def create_water_mask(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_water = np.array([90, 0, 50])
    upper_water = np.array([130, 100, 200])
    water_mask = cv2.inRange(hsv, lower_water, upper_water)
    kernel = np.ones((5,5), np.uint8)
    water_mask = cv2.morphologyEx(water_mask, cv2.MORPH_CLOSE, kernel)
    return water_mask

def filter_features_by_region(pts1, pts2, frame):
    water_mask = create_water_mask(frame)
    valid_mask = []
    for pt in pts1:
        x, y = int(pt[0]), int(pt[1])
        if 0 <= x < water_mask.shape[1] and 0 <= y < water_mask.shape[0]:
            valid_mask.append(water_mask[y, x] == 0)
        else:
            valid_mask.append(True)
    valid_mask = np.array(valid_mask)
    return pts1[valid_mask], pts2[valid_mask]

def estimate_homography(pts1, pts2):
    if len(pts1) >= 4 and len(pts2) >= 4:
        H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
        if H is None:
            return np.eye(3)
        return H
    else:
        return np.eye(3)

def adaptive_trajectory_smoothing(trajectory, motion_threshold=10):
    trajectory = np.array(trajectory)
    smoothed = np.zeros_like(trajectory)
    for i in range(trajectory.shape[1]):
        component = trajectory[:, i]
        motion_magnitude = np.abs(np.diff(component))
        from scipy.signal import savgol_filter
        if np.any(motion_magnitude > motion_threshold):
            smoothed[:, i] = savgol_filter(component, window_length=5, polyorder=2)
        else:
            smoothed[:, i] = savgol_filter(component, window_length=15, polyorder=3)
    return smoothed

def improved_stabilization_from_frames(frames_path, output_path, fps=29.97):
    """
    Stabilization using pre-extracted frames from directory
    """
    # Load frame file paths
    frame_files = load_frames_from_directory(frames_path)
    total_frames = len(frame_files)
    
    if total_frames == 0:
        print("Error: No frames found in directory")
        return
    
    print(f"Found {total_frames} frames in directory")
    
    # Read first frame to get dimensions
    first_frame = cv2.imread(frame_files[0])
    if first_frame is None:
        print("Error: Could not read first frame")
        return
    
    h, w = first_frame.shape[:2]
    
    # Setup output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    
    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    transforms = []
    trajectory = [np.eye(3)]
    
    print("Phase 1: Computing transforms...")
    
    # Process frames from directory
    for i in range(1, total_frames):
        curr_frame = cv2.imread(frame_files[i])
        if curr_frame is None:
            print(f"Warning: Could not read frame {i}")
            transforms.append(np.eye(3))
            trajectory.append(trajectory[-1])
            continue
        
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        pts1, pts2 = detect_and_match_features_fast(prev_gray, curr_gray)
        
        if len(pts1) > 0:
            pts1, pts2 = filter_motion_outliers(pts1, pts2)
        if len(pts1) > 0:
            pts1, pts2 = filter_features_by_region(pts1, pts2, curr_frame)
        
        H = estimate_homography(pts1, pts2)
        transforms.append(H)
        trajectory.append(trajectory[-1] @ H)
        prev_gray = curr_gray
        
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{total_frames} frames")
    
    print("Phase 2: Smoothing trajectory...")
    
    # Extract parameters and smooth
    params = []
    for H in trajectory:
        dx = H[0, 2]
        dy = H[1, 2]
        da = np.arctan2(H[1, 0], H[0, 0])
        params.append([dx, dy, da])
    
    smoothed_params = adaptive_trajectory_smoothing(params, motion_threshold=15)
    
    # Calculate smoothed transforms
    new_transforms = []
    for i in range(len(smoothed_params)):
        dx, dy, da = smoothed_params[i]
        cos_a = np.cos(da)
        sin_a = np.sin(da)
        new_H = np.array([[cos_a, -sin_a, dx],
                          [sin_a,  cos_a, dy],
                          [0,      0,     1]], dtype=np.float32)
        new_transforms.append(new_H)
    
    print("Phase 3: Applying stabilization and writing video...")
    
    # Apply stabilization and write video
    for i in range(total_frames):
        frame = cv2.imread(frame_files[i])
        if frame is None:
            continue
        
        if i < len(new_transforms):
            H = new_transforms[i]
            if H is not None and not np.isnan(H).any() and not np.isinf(H).any():
                try:
                    frame_stabilized = cv2.warpPerspective(frame, H, (w, h), borderMode=cv2.BORDER_REFLECT)
                except:
                    frame_stabilized = frame
            else:
                frame_stabilized = frame
        else:
            frame_stabilized = frame
        
        out.write(frame_stabilized)
        
        if (i + 1) % 100 == 0:
            print(f"Stabilized and wrote {i + 1}/{total_frames} frames")
    
    out.release()
    print(f"Stabilized video saved at: {output_path}")

# Usage with your extracted frames:
frames_path = r"C:\Users\simar\OneDrive\Desktop\Python_stabilization\unzipped_video"
output_path = r"C:\Users\simar\OneDrive\Desktop\Python_stabilization\RegionBased_stabilized.mp4"
improved_stabilization_from_frames(frames_path, output_path, fps=29.97)
