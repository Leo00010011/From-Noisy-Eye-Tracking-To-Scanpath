import numpy as np
from scipy.special import ellipe
from numba import njit

AVG_WEB_GAZER_ERROR_ANG = 4.17
STD_WEB_GAZER_ERROR_ANG = 3.54
WEIGHTS = [0.66488757, 0.33511243]
NOISE_MEAN = np.asanyarray([[ 7.46979502,  2.85483476] , [37.52364778, 26.33031239]])
NOISE_COV = np.asanyarray([[[ 3453.83797427, 1818.69016266],
            [ 1818.69016266, 1272.58837166]],
            [[50520.7022505, 13483.93496482],
            [13483.93496482, 13085.09003597]]])
# gen web_gazer_noise
def add_gaussian_noise(gaze_list,ptoa, ang_mean = AVG_WEB_GAZER_ERROR_ANG):
    noisy_gaze = []
    # from relationship of the mean of the Rayleigh distribution and the std of the Normal
    std = (ang_mean/ptoa)/1.253
    noise = np.random.normal(0, std,(2,gaze.shape[1]))
    for gaze in gaze_list:
        new_gaze = gaze.copy()
        noise = np.random.normal(0, std,(2,gaze.shape[1]))
        new_gaze[:2] += noise
        noisy_gaze.append(
            new_gaze
        )
    return noisy_gaze

def gen_elliptical_params(a, r):
    # https://chatgpt.com/c/68e64e38-3bb4-832c-b5bd-7061c05a6e3c
    k_sqrd = 1 - r**2
    x_std = a/ellipe(k_sqrd)*((np.pi/2)**(1/2))
    y_std = x_std*np.sqrt(1 - k_sqrd)
    return x_std, y_std

def gen_bivariate_normal(n_samples, x_std,y_std):
    noise_ellipx = np.random.normal(0, x_std,(1,n_samples))
    noise_ellipy = np.random.normal(0, y_std,(1,n_samples))
    return np.vstack([noise_ellipx, noise_ellipy])


def add_elliptical_gaussian_noise(gaze_list,ptoa,r = 0.7, ang_mean = AVG_WEB_GAZER_ERROR_ANG):
    noisy_gaze = []
    a = ang_mean/ptoa
    x_std, y_std = gen_elliptical_params(a, r)
    for gaze in gaze_list:
        new_gaze = gaze.copy()
        noise_ellip = gen_bivariate_normal(gaze.shape[1], x_std, y_std)
        new_gaze[:2] += noise_ellip
        noisy_gaze.append(
            new_gaze
        )
    return noisy_gaze

def add_learned_gaussian_noise(gaze_list, scale):
    noisy_gaze = []
    # fitting a gaussian to vectors from our database
    noise_mean = np.asanyarray([17.42210613 ,10.59617855])
    noise_cov = np.asanyarray(
        [[18848.31266896, 5891.93623468]
        ,[ 5891.93623468, 5163.09455225]])
    noise_cov *= scale
    for gaze in gaze_list:
        new_gaze = gaze.copy()
        noise = np.random.multivariate_normal(noise_mean, noise_cov, size=gaze.shape[1])
        new_gaze[:2] += noise.T
        noisy_gaze.append(
            new_gaze
        )
    return noisy_gaze

def gen_from_mixture(n_samples, weights, means, cov):
    mixture_count = []
    for i in range(len(weights) - 1):
        mixture_count.append(round(n_samples*weights[i]))
    mixture_count.append(n_samples - sum(mixture_count))

    samples = []
    for i in range(len(weights)):
        samples.append(np.random.multivariate_normal(means[i],cov[i], mixture_count[i]))
    samples = np.concat(samples,axis = 0).T
    col_indices = np.arange(samples.shape[1])
    np.random.shuffle(col_indices)
    samples[:] = samples[:, col_indices]
    return samples

def add_learned_gaussian_noiseV2(gaze_list, scale = 1):
    # Obtaining the model of our data 
    # - Fitting a mixture of two gaussian to our database
    # Obtaining a model of clean movement
    # - Fitting a gaussian to CocoFreeView data
    # Obtaining a model of the Noise
    # - Substracting the mean and 90% of the cov of the clean to the fitting to our data
    weights = [0.66488757, 0.33511243]
    noise_mean = np.asanyarray([[ 7.46979502,  2.85483476] , [37.52364778, 26.33031239]])
    noise_cov = np.asanyarray([[[ 3453.83797427, 1818.69016266],
                [ 1818.69016266, 1272.58837166]],
                [[50520.7022505, 13483.93496482],
                [13483.93496482, 13085.09003597]]])
    noisy_gaze = []
    for gaze in gaze_list:
        new_gaze = gaze.copy()
        noise = gen_from_mixture(gaze.shape[1], weights, noise_mean,noise_cov)
        new_gaze[:2] += noise*scale
        noisy_gaze.append(
            new_gaze
        )
    return noisy_gaze
# @njit
# def correlate(noise, corr):
#     output = np.empty_like(noise)
#     output[:, 0] = noise[:, 0]
#     for i in range(1, noise.shape[1]):
#         output[:, i] = corr * output[:, i - 1] + noise[:, i]
        
#     return output

# def add_autoregressive_noise(gaze_list, scale, corr):
#     noisy_gaze = []
#     for gaze in gaze_list:
#         new_gaze = gaze.copy()
#         noise = gen_from_mixture(gaze.shape[1], WEIGHTS, NOISE_MEAN, NOISE_COV)
#         corr_noise = correlate(noise, corr)
#         new_gaze[:2] += corr_noise*scale
#         noisy_gaze.append(
#             new_gaze
#         )
#     return noisy_gaze


def generate_noisy_center_path(initial_center, num_samples, target_std, corr):
    """
    Generates a 2D correlated noise path for a moving center point.
    
    This simulates a smoothly drifting or jittering fixation point using an
    Ornstein-Uhlenbeck process, which tends to revert to the mean (initial_center).
    """
    
    # Calculate the required input noise std to achieve the target std in the output
    if corr**2 >= 1:
        input_std = 0
    else:
        # This formula ensures the final path has the desired standard deviation
        input_std = target_std * np.sqrt(1 - corr**2)
        
    # Generate the underlying white noise (random "kicks")
    white_noise = np.random.normal(
        loc=0.0,
        scale=input_std,
        size=(2, num_samples)
    )
    
    # Generate the correlated path
    path = np.zeros_like(white_noise)
    path[:, 0] = initial_center.flatten() # Start at the initial center
    
    for i in range(1, num_samples):
        # The process is pulled back towards the initial_center, creating a stable jitter
        previous_pos = path[:, i - 1].reshape(2, 1)
        path[:, i] = (initial_center + (previous_pos - initial_center) * corr + white_noise[:, i].reshape(2, 1)).flatten()
        
    return path


# The original correlate_magnitudes function remains the same
def correlate_magnitudes(magnitudes, corr):
    """
    Applies a standard AR(1) correlation to a 1D array of noise magnitudes.
    """
    output = np.empty_like(magnitudes)
    output[0] = magnitudes[0]
    for i in range(1, magnitudes.shape[0]):
        output[i] = output[i - 1] * corr + magnitudes[i]
    return output

def generate_correlated_radial_noise(samples, initial_center, ptoa, radial_corr,
                                        radial_avg_norm, radial_std,
                                        center_noise_std=0.0, center_corr=0.98):
    """
    Generates temporally correlated radial noise with an optionally dynamic center.
    
    Args:
        samples: The (2, N) array of gaze points.
        initial_center: The (2,) initial center point for the noise.
        ptoa: Pixels-to-angle conversion factor.
        radial_corr: Correlation factor for the radial noise magnitude (0 to 1).
        radial_avg_norm: The average magnitude of the radial noise in degrees.
        radial_std: The standard deviation of the radial noise magnitude in degrees.
        center_noise_std (optional): The std deviation of the center's jitter in pixels. 
                                     If 0, the center is static. Defaults to 0.
        center_corr (optional): The correlation factor for the center's movement.
    """
    num_samples = samples.shape[1]
    
    # 1. Generate the center path (either static or dynamic)
    if center_noise_std > 0:
        center_path = generate_noisy_center_path(
            initial_center, num_samples, center_noise_std, center_corr
        )
    else:
        # Create a static path by repeating the initial center
        center_path = np.asarray(initial_center).reshape(2, 1)
    
    # --- The rest of the function generates the radial noise around this path ---
    
    # 2. Define target statistics for the radial noise magnitudes
    target_mean = radial_avg_norm / ptoa
    target_std = radial_std / ptoa

    # 3. Calculate input stats for the uncorrelated magnitude noise
    input_mean = target_mean * (1 - radial_corr)
    if radial_corr**2 >= 1:
        input_std = 0
    else:
        input_std = target_std * np.sqrt(1 - radial_corr**2)
        
    # 4. Generate and correlate the radial magnitudes
    uncorrelated_magnitudes = np.random.normal(input_mean, input_std, num_samples)
    correlated_magnitudes = correlate_magnitudes(uncorrelated_magnitudes, radial_corr)
    
    # 5. Determine the radial direction from the (now possibly dynamic) center
    # The subtraction works element-wise for each time step
    radial_vectors = samples - center_path
    
    # 6. Normalize to get unit direction vectors
    norms = np.linalg.norm(radial_vectors, ord=2, axis=0)
    norms[norms == 0] = 1 # Avoid division by zero
    unit_vectors = radial_vectors / norms
    
    # 7. Create the final noise by scaling unit vectors by correlated magnitudes
    final_noise = unit_vectors * correlated_magnitudes
    
    return final_noise

def add_correlated_radial_noise(gaze_list, initial_center, ptoa,
                                   radial_corr, 
                                   radial_avg_norm = AVG_WEB_GAZER_ERROR_ANG,
                                   radial_std = STD_WEB_GAZER_ERROR_ANG,
                                   center_noise_std=0.0, 
                                   center_corr=0.98):
    """
    Adds correlated radial noise with an optionally dynamic center to a list of gaze data.
    """
    noisy_gaze = []
    initial_center = np.asarray(initial_center).reshape(2, 1)
    for gaze in gaze_list:
        new_gaze = gaze.copy()
        
        # Use the new, more capable function to generate the noise
        noise = generate_correlated_radial_noise(
            samples=new_gaze[:2],
            initial_center=initial_center,
            ptoa=ptoa,
            radial_corr=radial_corr,
            radial_avg_norm=radial_avg_norm,
            radial_std=radial_std,
            center_noise_std=center_noise_std,
            center_corr=center_corr
        )
        
        new_gaze[:2] += noise
        noisy_gaze.append(new_gaze)
        
    return noisy_gaze

def add_random_center_correlated_radial_noise(gaze_list, initial_center, ptoa,
                                   radial_corr, 
                                   radial_avg_norm = AVG_WEB_GAZER_ERROR_ANG,
                                   radial_std = STD_WEB_GAZER_ERROR_ANG,
                                   center_noise_std=0.0, 
                                   center_corr=0.98,
                                   center_delta_norm = 0,
                                   center_delta_r = 0):
    """
    Adds correlated radial noise with an optionally dynamic center to a list of gaze data.
    """
    x_std, y_std = gen_elliptical_params(center_delta_norm,center_delta_r)
    center_delta = gen_bivariate_normal(len(gaze_list),x_std,y_std)
    print(center_delta.shape)
    initial_center = np.asarray(initial_center).reshape(2, 1)
    initial_center = center_delta + initial_center
    noisy_gaze = []
    for i, gaze in enumerate(gaze_list):
        new_gaze = gaze.copy()
        # Use the new, more capable function to generate the noise
        noise = generate_correlated_radial_noise(
            samples=new_gaze[:2],
            initial_center=initial_center[:,i].reshape(2, 1),
            ptoa=ptoa,
            radial_corr=radial_corr,
            radial_avg_norm=radial_avg_norm,
            radial_std=radial_std,
            center_noise_std=center_noise_std,
            center_corr=center_corr
        )
        
        new_gaze[:2] += noise
        noisy_gaze.append(new_gaze)
        
    return noisy_gaze, initial_center

    