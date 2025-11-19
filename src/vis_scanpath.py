import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

def draw_scanpath_mpl(ax, x, y, color='green', end_color='red', linewidth=2, marker_size=50, label=None):
    x = [float(coord) for coord in x]
    y = [float(coord) for coord in y]
    
    # Draw lines connecting points
    for i in range(len(x)-1):
        ax.plot([x[i], x[i+1]], [y[i], y[i+1]], color=color, linewidth=linewidth, label =label if i == 0 else "")
    
    # Draw start point
    ax.scatter(x[0], y[0], s=marker_size, c=end_color, zorder=5)
    
    return ax


def make_video_with_points(image, arrays, filename="output.mp4", fps=30, dot_color=(0,0,255), dot_radius=1, accumulate=True):

    h, w = image.shape[:2]

    # Ensure 3-channel BGR image
    background = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Convert times (ms) to frame indices
    ms_per_frame = 1000.0 / fps
    max_t = max(arr[2].max() for arr in arrays)
    total_frames = int(np.ceil(max_t / ms_per_frame))

    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, (w, h))

    # Pre-convert arrays: (x,y,t) -> (x,y,frame)
    processed = []
    for arr in arrays:
        x, y, t = arr
        frames = (t / ms_per_frame).astype(int)
        processed.append((x.astype(int), y.astype(int), frames))

    # Render frames
    accumulated_points = []
    for f in range(total_frames + 1):
        frame = background.copy()
        if accumulate:
            accumulated_points.extend([(xi, yi) for x, y, fr in processed for xi, yi, fi in zip(x, y, fr) if fi <= f])
            for (xi, yi) in accumulated_points:
                cv2.circle(frame, (xi, yi), dot_radius, dot_color, -1)
        else:
            for x, y, fr in processed:
                mask = (fr == f)
                for xi, yi in zip(x[mask], y[mask]):
                    cv2.circle(frame, (xi, yi), dot_radius, dot_color, -1)

        out.write(frame)

    out.release()
    print(f"Video saved to {filename}")


def plot_xy_on_image(image, arrays, size=5, colors="red", alpha=0.8):
    fig, ax = plt.subplots(figsize=(8,8))
    ax.imshow(image, cmap="gray")

    for arr, color in zip(arrays, colors):
        x, y = arr[0], arr[1]
        ax.scatter(x, y, s=size, c=color, alpha=alpha)

    ax.set_axis_off()
    plt.show()

def create_gaze_video(gaze_data_list, image_sources, output_path='scanpath_video', fps=60, gaze_radius=5, gaze_color=(0, 0, 255), trail_length=12, trail_color=(255, 165, 0)):
    """
    Creates a video by overlaying gaze data on a series of images.

    Args:
        gaze_data_list (list): A list of gaze trajectories. Each trajectory is a 
                               np.ndarray of shape (3, N) with [x, y, timestamp_ms].
        image_sources (list): A list of image file paths or loaded numpy array images.
        output_path (str): Path to save the output MP4 video file.
        fps (int): Frames per second for the output video. Should match the data's
                   sampling rate.
        gaze_radius (int): Radius of the circle representing the current gaze point.
        gaze_color (tuple): BGR color for the current gaze point (default is red).
        trail_length (int): Number of past points to show as a "trail".
        trail_color (tuple): BGR color for the gaze trail (default is cyan/blue).
    """
    # 2. --- Initialize Video Writer ---
    # Load the first image to determine video dimensions
    first_image_src = image_sources[0]
    if isinstance(first_image_src, str):
        if not os.path.exists(first_image_src):
            raise FileNotFoundError(f"Image file not found: {first_image_src}")
        frame = cv2.imread(first_image_src)
    else: # Assume it's a NumPy array
        frame = first_image_src
    
    if frame is None:
        raise IOError("Could not read the first image to set video dimensions.")
        
    height, width, _ = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for .mp4
    trial_idx = []
    # 3. --- Process Each Gaze Trajectory and Image ---
    for i, (gaze_trajectory, image_source) in enumerate(zip(gaze_data_list, image_sources)):
        
        video_writer = cv2.VideoWriter(f"{output_path}_{i}.mp4", fourcc, fps, (width, height))
        # Load the background image for the current trial
        base_image = cv2.cvtColor(image_source, cv2.COLOR_BGR2RGB)

        timestamps_ms = gaze_trajectory[2, :]
        coords = gaze_trajectory[:2, :].astype(int) # Use integer coordinates for drawing
        nan_mask = np.any(np.isnan(coords), axis = 0)
        x_mask = (coords[0] < 0) | (coords[0] > base_image.shape[1])
        y_mask = (coords[1] < 0) | (coords[1] > base_image.shape[0])
        invalid_mask = nan_mask | x_mask | y_mask
        # 4. --- Generate Frames for the Current Trial ---
        duration_ms = timestamps_ms[-1]
        num_frames_for_trial = int(duration_ms * fps / 1000)

        for frame_idx in range(num_frames_for_trial):
            # Create a fresh copy of the image for this frame
            current_frame = base_image.copy()
            
            # Find the gaze data point corresponding to the current video time
            current_time_ms = (frame_idx / fps) * 1000
            gaze_idx = np.searchsorted(timestamps_ms, current_time_ms, side="right") - 1
            if gaze_idx < 0: continue
            
            if invalid_mask[gaze_idx]:
                gaze_idx = trial_idx[-1]
            current_gaze_point = (int(coords[0, gaze_idx]), int(coords[1, gaze_idx]))
            if len(trial_idx) == trail_length:
                trial_idx.pop(0)
            trial_idx.append(gaze_idx)
            # Draw the gaze trail
            if gaze_idx > 0:
                trail_points = coords[:, trial_idx].T.reshape((-1, 1, 2))
                cv2.polylines(current_frame, [trail_points], isClosed=False, color=trail_color, thickness=2, lineType=cv2.LINE_AA)

            # Draw the current gaze point on top
            cv2.circle(current_frame, current_gaze_point, gaze_radius, gaze_color, -1) # -1 for a filled circle
            cv2.circle(current_frame, current_gaze_point, gaze_radius, (255,255,255), 2) # White outline for visibility

            # Write the completed frame to the video
            video_writer.write(current_frame)

        # 5. --- Finalize ---
        video_writer.release()
        print(f"✅ Video successfully created at: {output_path}_{i}.mp4")



def create_two_gaze_video(gaze_data_list,gaze_data_listG, image_sources, output_path='scanpath_video', fps=60, gaze_radius=5, gaze_color=(0, 0, 255), trail_length=12, trail_color=(255, 165, 0)):
    """
    Creates a video by overlaying gaze data on a series of images.

    Args:
        gaze_data_list (list): A list of gaze trajectories. Each trajectory is a 
                               np.ndarray of shape (3, N) with [x, y, timestamp_ms].
        image_sources (list): A list of image file paths or loaded numpy array images.
        output_path (str): Path to save the output MP4 video file.
        fps (int): Frames per second for the output video. Should match the data's
                   sampling rate.
        gaze_radius (int): Radius of the circle representing the current gaze point.
        gaze_color (tuple): BGR color for the current gaze point (default is red).
        trail_length (int): Number of past points to show as a "trail".
        trail_color (tuple): BGR color for the gaze trail (default is cyan/blue).
    """
    
    height, width, _ = image_sources[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for .mp4

    # 3. --- Process Each Gaze Trajectory and Image ---
    for i, (gaze_trajectory,gaze_trajectoryG, image_source) in enumerate(zip(gaze_data_list,gaze_data_listG, image_sources)):
        
        video_writer = cv2.VideoWriter(f"{output_path}_{i}.mp4", fourcc, fps, (width, height))
        # Load the background image for the current trial
        base_image = cv2.cvtColor(image_source, cv2.COLOR_BGR2RGB)

        timestamps_ms = gaze_trajectory[2, :]
        coords = gaze_trajectory[:2, :].astype(int) # Use integer coordinates for drawing
        coordsG = gaze_trajectoryG[:2, :].astype(int) # Use integer coordinates for drawing
        
        # 4. --- Generate Frames for the Current Trial ---
        duration_ms = timestamps_ms[-1]
        num_frames_for_trial = int(duration_ms * fps / 1000)

        for frame_idx in range(num_frames_for_trial):
            # Create a fresh copy of the image for this frame
            current_frame = base_image.copy()
            
            # Find the gaze data point corresponding to the current video time
            current_time_ms = (frame_idx / fps) * 1000
            gaze_idx = np.searchsorted(timestamps_ms, current_time_ms, side="right") - 1
            if gaze_idx < 0: continue

            # Define the start of the trail
            trail_start_idx = max(0, gaze_idx - trail_length)
            
            # Draw the gaze trail
            if gaze_idx > 0:
                trail_points = coords[:, trail_start_idx:gaze_idx+1].T.reshape((-1, 1, 2))
                cv2.polylines(current_frame, [trail_points], isClosed=False, color=trail_color, thickness=2, lineType=cv2.LINE_AA)

            # Draw the current gaze point on top
            current_gaze_point = (coords[0, gaze_idx], coords[1, gaze_idx])
            cv2.circle(current_frame, current_gaze_point, gaze_radius, gaze_color, -1) # -1 for a filled circle
            cv2.circle(current_frame, current_gaze_point, gaze_radius, (255,255,255), 2) # White outline for visibility

            if gaze_idx > 0:
                trail_points = coordsG[:, trail_start_idx:gaze_idx+1].T.reshape((-1, 1, 2))
                cv2.polylines(current_frame, [trail_points], isClosed=False, color=trail_color, thickness=2, lineType=cv2.LINE_AA)

            # Draw the current gaze point on top
            current_gaze_point = (coordsG[0, gaze_idx], coordsG[1, gaze_idx])
            cv2.circle(current_frame, current_gaze_point, gaze_radius, (0,255,0), -1) # -1 for a filled circle
            cv2.circle(current_frame, current_gaze_point, gaze_radius, (255,255,255), 2) # White outline for visibility

            # Write the completed frame to the video
            video_writer.write(current_frame)

        # 5. --- Finalize ---
        video_writer.release()
        print(f"✅ Video successfully created at: {output_path}_{i}.mp4")

