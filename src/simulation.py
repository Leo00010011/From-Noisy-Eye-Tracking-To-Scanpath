import numpy as np
from scipy.signal import lfilter


# Saccades

def get_saccade_amp(x,y, ptoa):
    p_dist = [np.sqrt(np.diff(xx)**2 + np.diff(yy)**2) for xx, yy in zip(x, y)]
    ang_dist = [d * ptoa for d in p_dist]
    return ang_dist

def get_saccade_durations(ang_dist):
    return [2.1*d + 21 for d in ang_dist]
        

def saccadic_displacement(s, normalized = True):
    if normalized:
        norm_factor = saccadic_displacement(1, normalized= False)
        return saccadic_displacement(s, normalized= False)/norm_factor
    else:
        return 1/10*s**5 + - 1/4*s**4 + 1/6*s**3

def gen_saccades(x, y, dur, time_step):
    saccades = []

    for i in range(1, len(x)):
        # duration of this saccade
        d = int(dur[i-1])
        n_samples = d // time_step  # ensure at least 2 points

        # time vector relative to this saccade
        ts = np.linspace(0, d, n_samples)

        # displacement percentage (custom function)
        disp_per = saccadic_displacement(ts / d)

        # interpolate
        xs = x[i-1] + (x[i] - x[i-1]) * disp_per
        ys = y[i-1] + (y[i] - y[i-1]) * disp_per

        saccades.append(np.stack([xs, ys, ts], axis=0))

    return saccades


# Fixations


def microsaccades_delta(n_samples,mean = 0, std_dev = 12/60, w0 = 0.85, alpha = 0.8):
    #TODO Option to not do the pink noise
    white_noise = np.random.normal(mean, std_dev, n_samples)
    
    # Pass through the white noise through the pink noise IIR filter
    a = [1, -alpha]
    b = [1, -w0]
    return lfilter(b, a, white_noise)

def gen_fixations(x,y,t, time_step, ptoa):
    t = np.array(t,np.uint16)
    samples_per_fixation = t//time_step
    microsaccades = microsaccades_delta(samples_per_fixation.sum()*2)
    microsaccade_x = microsaccades[:microsaccades.shape[0]//2] 
    microsaccade_y = microsaccades[microsaccades.shape[0]//2:] 
    fixations = []
    current = 0
    for i,n_samples in enumerate(samples_per_fixation):
        fixations.append(
            np.stack(
                [x[i] + microsaccade_x[current: current + n_samples]/ptoa,
                 y[i] + microsaccade_y[current: current + n_samples]/ptoa]
            )
        )
        current = current + n_samples
    return fixations




def process_scanpaths(data, indices, sampling_rate, get_fixation_mask = False, get_scanpath = False):
    """
    Processes a batch of scanpaths to generate continuous gaze data.

    Args:
        data: An object with a get_scanpath method.
        indices (list[int]): A list of indices for the scanpaths to process.

    Returns:
        list[np.ndarray]: A list of generated gaze trajectories.
    """
    batch_x, batch_y, batch_fix_durs = data.get_scanpath(indices)
    batch_sac_amp = get_saccade_amp(batch_x, batch_y, data.ptoa)
    batch_sac_durs = get_saccade_durations(batch_sac_amp)

    gaze_list, fm_list = [], []
    for i in range(len(indices)):
        gaze, fixation_mask = _generate_gaze_from_scanpath(
            batch_x.iloc[i],
            batch_y.iloc[i],
            batch_fix_durs.iloc[i],
            batch_sac_durs[i],
            sampling_rate
        )
        gaze_list.append(gaze)
        fm_list.append(fixation_mask)
    
    

    output = gaze_list
    if get_scanpath:
        fixations = []
        for x,y,d in zip(batch_x,batch_y,batch_fix_durs):
            fixations.append(np.vstack((x,y,d)))
        output = (output, fixations)
    
    if get_fixation_mask:
       output = (*output, fm_list)
    return output 

# --- Core Logic for a Single Scanpath ---

def _generate_gaze_from_scanpath(x_coords, y_coords, fixation_durs, saccade_durs, sampling_rate):
    """
    Generates a single continuous gaze trajectory from its event-based representation.
    
    Args:
        x_coords (pd.Series): X coordinates of fixations.
        y_coords (pd.Series): Y coordinates of fixations.
        fixation_durs (pd.Series): Durations of each fixation in ms.
        saccade_durs (list): Durations of each saccade in ms.

    Returns:
        np.ndarray: A (3, N) numpy array representing the gaze trajectory [x, y, time].
    """
    fixation_mask, sac_proportions, time_step = _segment_timeline(fixation_durs, saccade_durs,sampling_rate)

    gaze_trajectory = _synthesize_gaze_data(
        x_coords, y_coords, fixation_mask, sac_proportions, time_step
    )
    
    return gaze_trajectory, fixation_mask

# --- Helper Functions ---

def _segment_timeline(fixation_durs, saccade_durs, sampling_rate):
    """
    Segments the scanpath timeline into discrete samples for fixations and saccades.
    """
    time_step = 1000/sampling_rate
    total_duration = int(sum(fixation_durs) + sum(saccade_durs))
    total_samples = int(total_duration // time_step)
    fixation_mask = np.zeros(total_samples, dtype=np.uint8)
    sac_proportions = []
    
    time_offset = 0
    time_rem = 0.0  # Remainder from the previous time step

    # Interleave fixations and saccades
    num_events = len(fixation_durs) + len(saccade_durs)
    for event_idx in range(num_events):
        # Determine if the current event is a fixation or a saccade
        if event_idx % 2 == 0:  # Fixation event
            event_duration = fixation_durs[event_idx // 2]
            if event_duration < time_rem:
                time_rem -= event_duration
                continue

            # Calculate samples for this fixation
            adj_duration = event_duration - time_rem
            num_samples = int(np.ceil(adj_duration / time_step))
            
            fixation_id = event_idx // 2 + 1
            fixation_mask[time_offset : time_offset + num_samples] = fixation_id
            
            time_offset += num_samples
            time_rem = (num_samples * time_step) - adj_duration
        
        else:  # Saccade event
            event_duration = saccade_durs[event_idx // 2]
            if event_duration < time_rem:
                time_rem -= event_duration
                continue

            # Calculate the proportion of saccade completed at each sample point
            proportions = (np.arange(time_rem, event_duration, time_step)) / event_duration
            if proportions.size > 0:
                sac_proportions.append(proportions)
                time_offset += len(proportions)
                # Calculate remainder for the next event
                time_rem = time_step - (event_duration - time_rem) % time_step
                if time_rem == time_step: time_rem = 0


    return fixation_mask, sac_proportions, time_step

def _synthesize_gaze_data(x_coords, y_coords, fixation_mask, sac_proportions, time_step):
    """
    Constructs the final gaze coordinate array from the segmented timeline.
    """
    fixation_coords = np.stack([x_coords, y_coords])
    total_samples = len(fixation_mask)
    
    # Initialize gaze array with timestamps
    gaze = np.empty((3, total_samples), dtype=np.float32)
    gaze[2] = np.arange(1, total_samples + 1) * time_step

    # --- Populate Fixations ---
    is_fixation = fixation_mask > 0
    fixation_indices_in_mask = fixation_mask[is_fixation] - 1
    num_fix_samples = is_fixation.sum()
    
    # Add microsaccadic noise during fixations
    deltas = microsaccades_delta(num_fix_samples * 2).reshape(2, -1)
    gaze[:2, is_fixation] = fixation_coords[:, fixation_indices_in_mask] + deltas

    # --- Populate Saccades ---
    is_saccade = fixation_mask == 0
    if not sac_proportions: # Handle cases with no saccades
        return gaze

    proportions_flat = np.concatenate(sac_proportions)
    saccade_displacements = saccadic_displacement(proportions_flat)
    
    # Map each saccade sample to its corresponding start/end fixation
    saccade_ids = np.repeat(np.arange(len(sac_proportions)), [len(p) for p in sac_proportions])
    
    coord_deltas = fixation_coords[:, 1:] - fixation_coords[:, :-1]
    
    start_coords = fixation_coords[:, :-1][:, saccade_ids]
    delta_coords = coord_deltas[:, saccade_ids]
    
    gaze[:2, is_saccade] = start_coords + delta_coords * saccade_displacements

    return gaze


def downsampling_index(gaze, down_time_step = 200):
    count = []
    idx = []
    prev_idx = 0
    for t in np.arange(0,gaze[2,-1],down_time_step):
        new_idx = np.searchsorted(gaze[2],t,"right")
        count.append(new_idx - prev_idx)
        idx.append(new_idx)
        prev_idx = new_idx

    return idx, count

def downsample(gaze_list, down_time_step = 200):
    gaze_down = []
    for gaze in gaze_list:
        idx,_ = downsampling_index(gaze,down_time_step)
        new_gaze = np.empty((3,len(idx)))
        new_gaze = gaze[:,idx]
        gaze_down.append(new_gaze)
    return gaze_down

def downsampling_with_same_size(gaze_list, down_time_step = 200):
    gaze_down = []
    for gaze in gaze_list:
        idx,count = downsampling_index(gaze,down_time_step)
        new_gaze = np.empty((3,sum(count)))
        new_gaze[0] = np.repeat(gaze[0][idx],count)
        new_gaze[1] = np.repeat(gaze[1][idx],count)
        new_gaze[2] = gaze[2][:sum(count)]
        gaze_down.append(new_gaze)
    return gaze_down