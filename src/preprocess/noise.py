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
    '''
    Method that compute the needed x_std and y_std for a multivariate normal to have 
        **a**: y_std/x_std
        **r**: expected norm 
    '''
    # https://chatgpt.com/c/68e64e38-3bb4-832c-b5bd-7061c05a6e3c
    k_sqrd = 1 - r**2
    x_std = a/ellipe(k_sqrd)*((np.pi/2)**(1/2))
    y_std = x_std*np.sqrt(1 - k_sqrd)
    return x_std, y_std
@njit
def gen_bivariate_normal(n_samples, x_std,y_std):
    # diagonal covariance bivariate normal can be simulated as two independent gaussians
    """
    Numba-jitted version of bivariate normal generation.
    Generates two independent normal distributions and stacks them.
    """
    # Numba's np.random.normal creates 1D arrays
    noise_ellipx = np.random.normal(0.0, x_std, n_samples)
    noise_ellipy = np.random.normal(0.0, y_std, n_samples)
    
    # Manually stack them into a (2, n_samples) array
    output = np.empty((2, n_samples), dtype=np.float64) # Specify dtype for clarity
    output[0, :] = noise_ellipx
    output[1, :] = noise_ellipy
    return output


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

@njit
def generate_noisy_center_path(initial_center_1d, num_samples, target_std, corr):
    """
    Generates a 2D correlated noise path for a moving center point.
    
    This simulates a smoothly drifting or jittering fixation point using an
    Ornstein-Uhlenbeck process, which tends to revert to the mean (initial_center).
    """
    
    # 1. Calculate input noise std
    if corr >= 1.0:
        input_std = 0.0
    else:
        input_std = target_std * np.sqrt(1.0 - corr**2)
        
    # 2. Generate underlying white noise
    # Numba's np.random.normal supports positional args and a size tuple
    white_noise = np.random.normal(0.0, input_std, (2, num_samples))
    
    # 3. Generate the correlated path
    path = np.zeros_like(white_noise)
    
    # Set initial position (explicitly indexing)
    path[0, 0] = initial_center_1d[0]
    path[1, 0] = initial_center_1d[1]
    
    # Use explicit loops for x (j=0) and y (j=1)
    for i in range(1, num_samples):
        # Get the (2,) slice of the previous position
        prev_pos_x = path[0, i - 1]
        prev_pos_y = path[1, i - 1]
        
        # Get the (2,) slice of the current noise sample
        noise_x = white_noise[0, i]
        noise_y = white_noise[1, i]
        
        # Calculate the new position for x and y coordinates
        # path[j, i] = (initial_center[j] + (prev_pos[j] - initial_center[j]) * corr + noise[j])
        path[0, i] = initial_center_1d[0] + (prev_pos_x - initial_center_1d[0]) * corr + noise_x
        path[1, i] = initial_center_1d[1] + (prev_pos_y - initial_center_1d[1]) * corr + noise_y
            
    return path


# The original correlate_magnitudes function remains the same
@njit
def correlate_magnitudes(magnitudes, corr):
    """
    Applies a standard AR(1) correlation to a 1D array of noise magnitudes.
    """
    output = np.empty_like(magnitudes)
    output[0] = magnitudes[0]
    for i in range(1, magnitudes.shape[0]):
        output[i] = output[i - 1] * corr + magnitudes[i]
    return output

@njit
def generate_correlated_radial_noise_numba(
    samples, initial_center_1d, ptoa, radial_corr,
    radial_avg_norm, radial_std,
    center_noise_std, center_corr
):
    num_samples = samples.shape[1]
    
    # 1. Generate the center path (either static or dynamic)
    center_path = np.empty((2, num_samples))
    if center_noise_std > 0:
        # Call the jitted function
        center_path = generate_noisy_center_path(
            initial_center_1d, num_samples, center_noise_std, center_corr
        )
    else:
        # Create a static path by repeating the initial center
        # Use explicit loops for Numba-friendliness
        for i in range(num_samples):
            center_path[0, i] = initial_center_1d[0]
            center_path[1, i] = initial_center_1d[1]
            
    # 2. Define target statistics
    target_mean = radial_avg_norm / ptoa
    target_std = radial_std / ptoa

    # 3. Calculate input stats
    input_mean = target_mean * (1.0 - radial_corr)
    if radial_corr >= 1.0:
        input_std = 0.0
    else:
        input_std = target_std * np.sqrt(1.0 - radial_corr**2)
        
    # 4. Generate and correlate the radial magnitudes
    uncorrelated_magnitudes = np.random.normal(input_mean, input_std, num_samples)
    correlated_magnitudes = correlate_magnitudes(uncorrelated_magnitudes, radial_corr)
    
    # 5. Create final noise array
    final_noise = np.empty_like(samples)
    
    # 6. Main processing loop (replaces steps 5-8 from original)
    # This single loop calculates the radial vector, norm, and final noise
    # for each sample, avoiding large intermediate arrays.
    for i in range(num_samples):
        
        # Step 5: Determine the radial vector
        rad_x = samples[0, i] - center_path[0, i]
        rad_y = samples[1, i] - center_path[1, i]
        
        # Step 6: Normalize (replaces np.linalg.norm and boolean indexing)
        norm = np.sqrt(rad_x**2 + rad_y**2)
        
        # Avoid division by zero
        if norm == 0.0:
            norm = 1.0 # Set to 1 to avoid NaN; noise magnitude will be 0 if mag is 0
            
        # Step 7: Create unit vectors
        unit_x = rad_x / norm
        unit_y = rad_y / norm
        
        # Step 8: Scale unit vectors by correlated magnitudes
        mag = correlated_magnitudes[i]
        final_noise[0, i] = unit_x * mag
        final_noise[1, i] = unit_y * mag
        
    return final_noise

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
    
    # Prepare initial_center for Numba
    initial_center_1d = initial_center.flatten()
    if initial_center_1d.shape[0] != 2:
        raise ValueError("initial_center must be a 2-element array.")
        
    # Call the jitted function with all arguments positionally
    return generate_correlated_radial_noise_numba(
        samples, initial_center_1d, ptoa, radial_corr,
        radial_avg_norm, radial_std,
        center_noise_std, center_corr
    )

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

def add_random_center_correlated_radial_noise(gaze, initial_center, ptoa,
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
    boxed = False
    if type(gaze) != list:
        boxed = True
        gaze = [gaze]
    
    x_std, y_std = gen_elliptical_params(center_delta_norm,center_delta_r)
    initial_center = np.asarray(initial_center).reshape(2, 1)
    center_delta = gen_bivariate_normal(len(gaze),x_std,y_std)
    initial_center = center_delta + initial_center
    noisy_gaze = []
    for i, single_gaze in enumerate(gaze):
        # Use the new, more capable function to generate the noise
        noise = generate_correlated_radial_noise(
            samples=single_gaze[:2],
            initial_center=initial_center[:,i].reshape(2, 1),
            ptoa=ptoa,
            radial_corr=radial_corr,
            radial_avg_norm=radial_avg_norm,
            radial_std=radial_std,
            center_noise_std=center_noise_std,
            center_corr=center_corr
        )

        single_gaze[:2] += noise
        noisy_gaze.append(single_gaze)

    if boxed:
        noisy_gaze = noisy_gaze[0]
    return noisy_gaze, initial_center


def discretization_noise(image_shape, gaze):
    gaze[0,:] = (np.clip(np.round((gaze[0,:]/image_shape[1])*100),0,100) + 0.5)*(image_shape[1]/100)
    gaze[1,:] = (np.clip(np.round((gaze[1,:]/image_shape[0])*100),0,100) + 0.5)*(image_shape[0]/100)
    return gaze
