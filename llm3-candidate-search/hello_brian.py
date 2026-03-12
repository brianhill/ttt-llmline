import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.ndimage import maximum_filter, shift, uniform_filter
from astropy.modeling import models, fitting

# Clear all previous figures so only current run's plots are visible
plt.close('all')

# Corrected directory containing the NumPy files (relative path)
data_dir = "../llm-2-photometric-rescaling"

# File paths
template_path = os.path.join(data_dir, "median_template.npy")
rescaled_science_path = os.path.join(data_dir, "rescaled_aligned_science.npy")

# Check if files exist
if not os.path.exists(template_path):
    raise FileNotFoundError(f"Template file not found: {template_path}")
if not os.path.exists(rescaled_science_path):
    raise FileNotFoundError(f"Rescaled science file not found: {rescaled_science_path}")

# Load the NumPy arrays
print("Loading NumPy arrays...")
template_img = np.load(template_path)
rescaled_science_img = np.load(rescaled_science_path)

print(f"Template image loaded: shape {template_img.shape}, dtype {template_img.dtype}")
print(f"Rescaled science image loaded: shape {rescaled_science_img.shape}, dtype {rescaled_science_img.dtype}")

# Basic statistics
print("\nTemplate image statistics:")
print(f"  Min: {np.min(template_img):.1f}, Max: {np.max(template_img):.1f}, Mean: {np.mean(template_img):.1f}")

# ────────────────────────────────────────────────
# Find and report the single brightest pixel in the template
# ────────────────────────────────────────────────
print("\nSearching for the brightest pixel in the median template image...")

if template_img.size == 0:
    print("  Template image is empty — cannot find brightest pixel.")
    x_max = y_max = None
else:
    max_value = np.max(template_img)
    max_coords = np.unravel_index(np.argmax(template_img), template_img.shape)
    y_max, x_max = max_coords  # numpy: (row=y, column=x)

    print(f"  Brightest pixel value: {max_value:.1f} counts")
    print(f"  Location (x, y): ({x_max}, {y_max})")
    print(f"  Location (y, x — numpy order): ({y_max}, {x_max})")

    half = 2
    y0 = max(0, y_max - half)
    y1 = min(template_img.shape[0], y_max + half + 1)
    x0 = max(0, x_max - half)
    x1 = min(template_img.shape[1], x_max + half + 1)

    print(f"  5×5 neighborhood around brightest pixel (values):")
    print(template_img[y0:y1, x0:x1])

print("\nRescaled science image statistics:")
print(
    f"  Min: {np.min(rescaled_science_img):.1f}, Max: {np.max(rescaled_science_img):.1f}, Mean: {np.mean(rescaled_science_img):.1f}")

# ────────────────────────────────────────────────
# INJECT ARTIFICIAL GAUSSIAN SOURCE FOR TESTING
# ────────────────────────────────────────────────
print("\nInjecting artificial 2D Gaussian source with total flux = 100 for testing...")

# Make a working copy
science_with_injection = rescaled_science_img.copy()

# Gaussian parameters
fwhm = 3.0
sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))  # ≈ 1.274 pixels
total_flux = 100.0

# Patch size (odd number)
patch_size = 11
half_patch = patch_size // 2

# Grid for the patch (relative coordinates)
yy, xx = np.mgrid[-half_patch:half_patch + 1, -half_patch:half_patch + 1]

# Normalized 2D Gaussian (integral over all space = 1)
gauss = np.exp(- (xx ** 2 + yy ** 2) / (2 * sigma ** 2))
gauss_integral = gauss.sum()  # approximate integral over 11×11
gauss_normalized = gauss / gauss_integral

# Scale to desired total flux
gauss_patch = gauss_normalized * total_flux

print(f"  Gaussian patch sum (≈ total flux): {gauss_patch.sum():.2f}")

# Central 2000×2000 region
ny, nx = template_img.shape
center_y, center_x = ny // 2, nx // 2
margin = half_patch + 10  # safety margin
cy_min = center_y - 1000 + margin
cy_max = center_y + 1000 - margin
cx_min = center_x - 1000 + margin
cx_max = center_x + 1000 - margin

# Random integer position
np.random.seed()  # remove or set fixed value for reproducibility
inj_y = np.random.randint(cy_min, cy_max + 1)
inj_x = np.random.randint(cx_min, cx_max + 1)

print(f"  Injected Gaussian at (x, y) = ({inj_x}, {inj_y})")

# Add the patch to the science image (handle edges)
y0 = max(0, inj_y - half_patch)
y1 = min(ny, inj_y + half_patch + 1)
x0 = max(0, inj_x - half_patch)
x1 = min(nx, inj_x + half_patch + 1)

patch_y0 = half_patch - (inj_y - y0)
patch_y1 = patch_y0 + (y1 - y0)
patch_x0 = half_patch - (inj_x - x0)
patch_x1 = patch_x0 + (x1 - x0)

science_with_injection[y0:y1, x0:x1] += gauss_patch[patch_y0:patch_y1, patch_x0:patch_x1]

# Replace original science image with injected version
rescaled_science_img = science_with_injection

print("  Artificial Gaussian source injected successfully.")

# ────────────────────────────────────────────────
# Continue with the rest of the pipeline unchanged
# ────────────────────────────────────────────────

# Compute difference: rescaled science - template
difference = rescaled_science_img - template_img

print("\nDifference (rescaled science − template) statistics:")
print(
    f"  Min: {np.min(difference):.1f}, Max: {np.max(difference):.1f}, Mean: {np.mean(difference):.1f}, Std: {np.std(difference):.1f}")

# Absolute value version of the difference
abs_difference = np.abs(difference)
abs_median = np.median(abs_difference)
print(f"\nAbsolute difference median: {abs_median:.1f} counts")

# 3×3 boxcar sum-smoothed difference
print("\nCreating 3×3 boxcar sum-smoothed difference image...")
smoothed_sum_difference = uniform_filter(difference, size=3) * 9

# Absolute value of the sum-smoothed difference
abs_smoothed_sum = np.abs(smoothed_sum_difference)

# Median of absolute sum-smoothed values
smoothed_abs_median = np.median(abs_smoothed_sum)
print(f"Median of absolute sum-smoothed difference: {smoothed_abs_median:.1f} counts")

# Median multiplier and threshold
median_multiplier = 10
threshold = smoothed_abs_median * median_multiplier
print(f"Median multiplier: {median_multiplier}")
print(f"Threshold (median × {median_multiplier}): {threshold:.1f} counts")

# Constants for star finding and culling
saturation_threshold = 60000.0
aperture_radius = 5.0
aperture_max_threshold = 10000.0
edge_margin = 200
duplicate_distance_threshold = 3.0
bright_star_exclusion_radius = 100.0


# Function to find star centroids and FWHM
def find_star_centroids_and_fwhms(image_data, num_stars=20):
    background = np.median(image_data)
    data_sub = image_data - background

    local_max = maximum_filter(data_sub, size=15) == data_sub
    peaks = np.where(local_max & (data_sub > 5 * np.std(data_sub)))

    if len(peaks[0]) == 0:
        raise ValueError("No bright sources detected.")

    star_data = []
    box_half_size = 15
    fitter = fitting.LevMarLSQFitter()

    intensities = data_sub[peaks[0], peaks[1]]
    sorted_indices = np.argsort(-intensities)
    sorted_peaks_y = peaks[0][sorted_indices]
    sorted_peaks_x = peaks[1][sorted_indices]

    existing_centroids = []
    y_grid, x_grid = np.indices(image_data.shape)
    ny, nx = image_data.shape

    for i in range(min(len(sorted_peaks_y), num_stars * 5)):
        y = sorted_peaks_y[i]
        x = sorted_peaks_x[i]

        y0, y1 = max(0, y - box_half_size), min(image_data.shape[0], y + box_half_size + 1)
        x0, x1 = max(0, x - box_half_size), min(image_data.shape[1], x + box_half_size + 1)
        cutout = data_sub[y0:y1, x0:x1]

        if cutout.size < 100:
            continue

        yy, xx = np.mgrid[:cutout.shape[0], :cutout.shape[1]]
        y_center = y - y0
        x_center = x - x0

        g_init = models.Gaussian2D(
            amplitude=cutout.max(),
            x_mean=x_center,
            y_mean=y_center,
            x_stddev=3.0,
            y_stddev=3.0
        )

        try:
            g_fit = fitter(g_init, xx, yy, cutout)
            if g_fit.x_stddev.value > 0 and g_fit.y_stddev.value > 0:
                fwhm_mean = np.mean([g_fit.x_stddev.value * 2.355, g_fit.y_stddev.value * 2.355])
                full_x = x0 + g_fit.x_mean.value
                full_y = y0 + g_fit.y_mean.value
                peak_pixel = image_data[int(np.round(full_y)), int(np.round(full_x))]

                if peak_pixel > saturation_threshold:
                    continue

                dist_sq = (x_grid - full_x) ** 2 + (y_grid - full_y) ** 2
                aperture_mask = dist_sq <= aperture_radius ** 2
                if np.max(image_data[aperture_mask]) > aperture_max_threshold:
                    continue

                if (full_x < edge_margin or full_x > nx - edge_margin or
                        full_y < edge_margin or full_y > ny - edge_margin):
                    continue

                too_close = False
                for ex_x, ex_y in existing_centroids:
                    dist = np.sqrt((full_x - ex_x) ** 2 + (full_y - ex_y) ** 2)
                    if dist < duplicate_distance_threshold:
                        too_close = True
                        break
                if too_close:
                    continue

                star_data.append((full_x, full_y, fwhm_mean, peak_pixel))
                existing_centroids.append((full_x, full_y))
        except Exception:
            continue

        if len(star_data) >= num_stars:
            break

    if len(star_data) < num_stars:
        print(f"Warning: Only found {len(star_data)} unique non-saturated non-edge star fits (requested {num_stars}).")

    return star_data


# Characterize PSF in the median template image
print("\nCharacterizing PSF in the median template image...")
template_psf_data = find_star_centroids_and_fwhms(template_img, num_stars=50)

if len(template_psf_data) == 0:
    print("No suitable stars found for PSF characterization in template.")
else:
    fwhms = [fwhm for _, _, fwhm, _ in template_psf_data]
    median_fwhm = np.median(fwhms)
    mean_fwhm = np.mean(fwhms)
    std_fwhm = np.std(fwhms)
    print(f"PSF characterization using {len(template_psf_data)} isolated, non-saturated stars:")
    print(f"  Median FWHM: {median_fwhm:.2f} pixels")
    print(f"  Mean FWHM: {mean_fwhm:.2f} pixels (± {std_fwhm:.2f} pixels)")

# Create master empirical PSF by stacking cutouts from PSF stars
print("\nCreating master empirical PSF by stacking cutouts from PSF stars...")
cutout_size = 41
half_cut = cutout_size // 2
psf_cutouts = []

ny, nx = template_img.shape

for full_x, full_y, _, _ in template_psf_data:
    x0 = int(np.floor(full_x - half_cut))
    x1 = int(np.floor(full_x + half_cut + 1))
    y0 = int(np.floor(full_y - half_cut))
    y1 = int(np.floor(full_y + half_cut + 1))

    pad_left = max(0, -x0)
    pad_right = max(0, x1 - nx)
    pad_top = max(0, -y0)
    pad_bottom = max(0, y1 - ny)

    cutout = template_img[max(0, y0):min(ny, y1), max(0, x0):min(nx, x1)]
    cutout = np.pad(cutout, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values=0)

    shift_y = half_cut + pad_top - (full_y - max(0, y0))
    shift_x = half_cut + pad_left - (full_x - max(0, x0))
    cutout = shift(cutout, (shift_y, shift_x), order=3, mode='constant', cval=0.0)

    total_flux = np.sum(cutout)
    if total_flux > 0:
        cutout /= total_flux

    psf_cutouts.append(cutout)

if len(psf_cutouts) > 0:
    master_psf = np.median(psf_cutouts, axis=0)
    print(
        f"Master empirical PSF created from median stack of {len(psf_cutouts)} normalized cutouts ({cutout_size}×{cutout_size}).")

    # ────────────────────────────────────────────────
    # Report FWHM of the master_psf
    # ────────────────────────────────────────────────
    print("\nEstimating FWHM of the master empirical PSF...")

    # Use central region of master_psf for fit (e.g. 21×21)
    psf_center_size = 21
    psf_half_center = psf_center_size // 2
    cy = cutout_size // 2
    cx = cutout_size // 2
    psf_center = master_psf[cy - psf_half_center:cy + psf_half_center + 1,
                 cx - psf_half_center:cx + psf_half_center + 1]

    if psf_center.shape[0] < 10 or psf_center.shape[1] < 10:
        print("  Master PSF too small to estimate FWHM reliably.")
        master_fwhm = np.nan
    else:
        yy, xx = np.mgrid[0:psf_center.shape[0], 0:psf_center.shape[1]]
        amplitude_init = psf_center.max()
        x_mean_init = (psf_center.shape[1] - 1) / 2
        y_mean_init = (psf_center.shape[0] - 1) / 2

        g_init = models.Gaussian2D(amplitude=amplitude_init,
                                   x_mean=x_mean_init,
                                   y_mean=y_mean_init,
                                   x_stddev=1.5,
                                   y_stddev=1.5)

        fitter = fitting.LevMarLSQFitter()
        try:
            g_fit = fitter(g_init, xx, yy, psf_center)
            if g_fit.x_stddev.value > 0 and g_fit.y_stddev.value > 0:
                fwhm_x = g_fit.x_stddev.value * 2.355
                fwhm_y = g_fit.y_stddev.value * 2.355
                master_fwhm = (fwhm_x + fwhm_y) / 2
                print(f"  Estimated FWHM of master PSF: {master_fwhm:.2f} pixels")
                print(f"  (x_stddev = {g_fit.x_stddev.value:.3f}, y_stddev = {g_fit.y_stddev.value:.3f})")
            else:
                print("  Gaussian fit failed to produce positive stddev.")
                master_fwhm = np.nan
        except Exception as e:
            print(f"  Gaussian fit failed: {e}")
            master_fwhm = np.nan

    if np.isnan(master_fwhm):
        print("  Could not reliably estimate FWHM of master PSF.")
else:
    master_psf = np.zeros((cutout_size, cutout_size))
    print("No cutouts available for master PSF.")
    master_fwhm = np.nan

# Display the master PSF
plt.figure(figsize=(8, 8))
plt.imshow(master_psf, cmap='viridis', origin='lower', vmin=0)
plt.title('Master Empirical PSF\n(Median stack of normalized 41×41 cutouts)')
plt.colorbar(label='Normalized flux')
plt.tight_layout()
plt.show(block=False)

# Save master PSF
master_psf_path = os.path.join(os.getcwd(), "master_psf.npy")
np.save(master_psf_path, master_psf)
print(f"Master PSF saved to {master_psf_path}")

# Initial candidates for transients
print("\nFinding initial candidate pixels exceeding threshold...")
y_indices, x_indices = np.where(abs_smoothed_sum > threshold)
candidates = list(zip(x_indices, y_indices))
values = abs_smoothed_sum[y_indices, x_indices].tolist()

print(f"Found {len(candidates)} initial candidate pixels")

# Edge culling
print("\nEdge culling: eliminating candidates within 50 pixels of image edges...")
ny, nx = abs_smoothed_sum.shape
edge_margin_candidates = 50
edge_culled = []
for i in range(len(candidates)):
    x, y = candidates[i]
    val = values[i]
    if (x >= edge_margin_candidates and x <= nx - edge_margin_candidates and
            y >= edge_margin_candidates and y <= ny - edge_margin_candidates):
        edge_culled.append(((x, y), val))

candidates = [coord for coord, _ in edge_culled]
values = [val for _, val in edge_culled]

print(f"After edge culling: {len(candidates)} candidates remain")

# Adjacency culling
print("\nAdjacency culling: keeping brightest within 5 pixels...")
culled_candidates = []

values_array = np.array(values)
sorted_indices = np.argsort(-values_array)
sorted_candidates = [candidates[i] for i in sorted_indices]

for i, (x, y) in enumerate(sorted_candidates):
    too_close = False
    for cx, cy in culled_candidates:
        dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        if dist <= 5.0:
            too_close = True
            break
    if not too_close:
        culled_candidates.append((x, y))

candidates = culled_candidates

print(f"After adjacency culling: {len(candidates)} unique bright candidates remain")

# ──────────────────────────────────────────────────────────────
# Exclude candidates within 100 pixels of the brightest pixel
# ──────────────────────────────────────────────────────────────
print(
    f"\nExcluding candidates within {bright_star_exclusion_radius} pixels of the brightest pixel ({x_max}, {y_max})...")

if x_max is None or y_max is None:
    print("  Brightest pixel not found → no exclusion applied.")
    excluded_count = 0
else:
    kept_candidates = []
    excluded_count = 0

    for x, y in candidates:
        dist = np.sqrt((x - x_max) ** 2 + (y - y_max) ** 2)
        if dist <= bright_star_exclusion_radius:
            excluded_count += 1
        else:
            kept_candidates.append((x, y))

    candidates = kept_candidates

print(f"  Removed {excluded_count} candidates within {bright_star_exclusion_radius} pixels of brightest pixel.")
print(f"  Remaining candidates: {len(candidates)}")

# 3×3 boxcar sum-smoothed template for Poisson culling
print("\nCreating 3×3 boxcar sum-smoothed template image...")
smoothed_template = uniform_filter(template_img, size=3) * 9

# Secondary Poisson culling
poisson_multiplier = 5
print(f"\nSecondary culling with poisson_multiplier = {poisson_multiplier}")

final_candidates = []
for x, y in candidates:
    smoothed_template_val = smoothed_template[y, x]
    effective_val = max(smoothed_template_val, 1.0)
    poisson_threshold = poisson_multiplier * np.sqrt(effective_val)
    candidate_val = abs_smoothed_sum[y, x]
    if candidate_val >= poisson_threshold:
        final_candidates.append((x, y))

candidates = final_candidates

print(f"After secondary Poisson culling: {len(candidates)} candidates remain")

# Fit each final candidate to master PSF + local background over 5x5 region (background >= 0)
print("\nFitting final candidates to master PSF + local background (5x5 region, background >= 0)...")

# Crop master PSF to central 5x5 for fitting
psf_fit_size = 5
psf_half = psf_fit_size // 2
master_psf_central = master_psf[half_cut - psf_half:half_cut + psf_half + 1,
                     half_cut - psf_half:half_cut + psf_half + 1]

# Flatten for fitting
psf_vec = master_psf_central.flatten()
psf_sum = np.sum(psf_vec)

fit_results = []
for i, (x, y) in enumerate(candidates):
    # Extract 5x5 cutout from rescaled science image
    x0 = int(x - psf_half)
    x1 = int(x + psf_half + 1)
    y0 = int(y - psf_half)
    y1 = int(y + psf_half + 1)

    cutout = rescaled_science_img[y0:y1, x0:x1]
    data_vec = cutout.flatten()

    # Design matrix: PSF vector and constant for background
    design = np.vstack([psf_vec, np.ones(psf_fit_size ** 2)]).T

    # Unconstrained least squares fit
    params = np.linalg.lstsq(design, data_vec, rcond=None)[0]
    amplitude, background = params

    # Enforce background >= 0
    if background < 0:
        # Fix background to 0 and refit amplitude only
        design_fixed = psf_vec[:, np.newaxis]
        amplitude = np.linalg.lstsq(design_fixed, data_vec, rcond=None)[0][0]
        background = 0.0

    # Total flux
    flux = amplitude * psf_sum

    # Final model and chi-squared
    model = amplitude * psf_vec + background
    chi2 = np.sum((data_vec - model) ** 2)
    dof = psf_fit_size ** 2 - 2
    chi2_dof = chi2 / dof if dof > 0 else np.nan

    fit_results.append((flux, background, chi2_dof))

# Tertiary culling: remove candidates with chi2/dof > 30
print("\nTertiary culling: eliminating candidates with chi²/dof > 30...")
good_fit_candidates = []
good_fit_results = []

for i in range(len(candidates)):
    chi2_dof = fit_results[i][2]
    if chi2_dof <= 30.0:
        good_fit_candidates.append(candidates[i])
        good_fit_results.append(fit_results[i])

candidates = good_fit_candidates
fit_results = good_fit_results

print(f"After chi²/dof culling: {len(candidates)} candidates remain")

# Print revised table with fit parameters
if len(candidates) > 0:
    print("Revised final candidates with PSF fit parameters (5x5 region, background >= 0):")
    print("  #    (x, y)      residual    flux       background   chi2/dof")
    for i in range(len(candidates)):
        x, y = candidates[i]
        residual = abs_smoothed_sum[y, x]
        flux, bg, chi2_dof = fit_results[i]
        print(f"  {i + 1:2d}: ({x:4d}, {y:4d})  {residual:8.1f}  {flux:8.1f}  {bg:8.1f}  {chi2_dof:6.2f}")
else:
    print("No candidates remain after chi²/dof culling.")

# ────────────────────────────────────────────────
# Display 30×30 postage stamps for each final candidate
# ────────────────────────────────────────────────
print("\nGenerating 30×30 postage stamps around final candidates (from rescaled science image)...")

stamp_size = 30
half_stamp = stamp_size // 2

for i, (x, y) in enumerate(candidates):
    plt.figure(figsize=(6, 6))

    x_c = int(np.round(x))
    y_c = int(np.round(y))

    y0 = max(0, y_c - half_stamp)
    y1 = min(rescaled_science_img.shape[0], y_c + half_stamp + 1)
    x0 = max(0, x_c - half_stamp)
    x1 = min(rescaled_science_img.shape[1], x_c + half_stamp + 1)

    cutout = rescaled_science_img[y0:y1, x0:x1]

    plt.imshow(cutout, cmap='gray', origin='lower',
               vmin=np.percentile(rescaled_science_img, 1),
               vmax=np.percentile(rescaled_science_img, 99))

    plt.plot(x - x0 + 0.5, y - y0 + 0.5, marker='+', color='red',
             markersize=16, markeredgewidth=2.5, label='Candidate position')

    flux_val, bg_val, chi2dof_val = fit_results[i]
    plt.title(f"Candidate {i + 1}   ({x:.1f}, {y:.1f})\n"
              f"flux = {flux_val:.1f}    χ²/dof = {chi2dof_val:.2f}")
    plt.xlabel("pixel offset")
    plt.ylabel("pixel offset")
    plt.legend(loc='upper right', fontsize=10)
    plt.colorbar(shrink=0.78, label="counts")
    plt.tight_layout()
    plt.show(block=False)

    print(f"  Displayed stamp {i + 1}/{len(candidates)} at ({x:.1f}, {y:.1f})")

print(f"All {len(candidates)} final candidates shown as individual 30×30 stamps.")

# Display FULL rescaled science image with red circles (diameter 20 px) around final candidates
print("\nDisplaying FULL rescaled science image with red circles (diameter 20 px) on all final candidates...")
plt.figure(figsize=(20, 20))
plt.imshow(rescaled_science_img, cmap='gray', origin='lower',
           vmin=np.percentile(rescaled_science_img, 1),
           vmax=np.percentile(rescaled_science_img, 99))

for x, y in candidates:
    circ = Circle((x, y), radius=10, fill=False, color='red', linewidth=2)
    plt.gca().add_patch(circ)

plt.title(
    f'Full Rescaled Aligned Science Image\nRed circles (diameter 20 px) on all {len(candidates)} final candidates')
plt.xlabel('X pixel')
plt.ylabel('Y pixel')
plt.colorbar(shrink=0.8, label='Counts')
plt.tight_layout()
plt.show(block=False)

print(f"All {len(candidates)} final candidates are marked on the full image.")

# Pre-compute central 2000×2000 regions for other displays
half_large = 1000
ny, nx = template_img.shape
center_y_large, center_x_large = ny // 2, nx // 2

template_central_large = template_img[center_y_large - half_large:center_y_large + half_large,
                         center_x_large - half_large:center_x_large + half_large]
science_central_large = rescaled_science_img[center_y_large - half_large:center_y_large + half_large,
                        center_x_large - half_large:center_x_large + half_large]
diff_central_large = difference[center_y_large - half_large:center_y_large + half_large,
                     center_x_large - half_large:center_x_large + half_large]
abs_diff_central_large = abs_difference[center_y_large - half_large:center_y_large + half_large,
                         center_x_large - half_large:center_x_large + half_large]

# 2×2 grid layout for large central regions
plt.figure(figsize=(24, 24))

# Template (top-left)
plt.subplot(2, 2, 1)
plt.imshow(template_central_large, cmap='gray', origin='lower',
           vmin=np.percentile(template_central_large, 1),
           vmax=np.percentile(template_central_large, 99))
plt.title('Central 2000×2000 of Median Template')
plt.xlabel('X pixel')
plt.ylabel('Y pixel')
plt.colorbar(shrink=0.8)

# Rescaled science (top-right)
plt.subplot(2, 2, 2)
plt.imshow(science_central_large, cmap='gray', origin='lower',
           vmin=np.percentile(science_central_large, 1),
           vmax=np.percentile(science_central_large, 99))
plt.title('Central 2000×2000 of Rescaled Aligned Science Image')
plt.xlabel('X pixel')
plt.ylabel('Y pixel')
plt.colorbar(shrink=0.8)

# Difference (bottom-left)
vmax = np.std(diff_central_large) * 3
plt.subplot(2, 2, 3)
plt.imshow(diff_central_large, cmap='RdBu', origin='lower',
           vmin=-vmax, vmax=vmax)
plt.title('Central 2000×2000 Difference\n(Rescaled Science − Template)')
plt.xlabel('X pixel')
plt.ylabel('Y pixel')
plt.colorbar(label='Difference (counts)', shrink=0.8)

# Absolute difference (bottom-right)
vmax_abs = np.percentile(abs_diff_central_large, 99.5)
plt.subplot(2, 2, 4)
plt.imshow(abs_diff_central_large, cmap='viridis', origin='lower',
           vmin=0, vmax=vmax_abs)
plt.title(f'Central 2000×2000 Absolute Difference\nMedian: {abs_median:.1f} counts')
plt.xlabel('X pixel')
plt.ylabel('Y pixel')
plt.colorbar(label='Absolute Difference (counts)', shrink=0.8)

plt.tight_layout()
plt.show(block=False)

# 100×100 zooms of the two difference images centered on (x=1699, y=2154)
print("\nDisplaying 100×100 zoom around (1699, 2154)...")
zoom_center_x = 1699
zoom_center_y = 2154
zoom_half = 50

diff_zoom = difference[zoom_center_y - zoom_half:zoom_center_y + zoom_half,
            zoom_center_x - zoom_half:zoom_center_x + zoom_half]
abs_diff_zoom = abs_difference[zoom_center_y - zoom_half:zoom_center_y + zoom_half,
                zoom_center_x - zoom_half:zoom_center_x + zoom_half]

plt.figure(figsize=(24, 12))

# Difference zoom
vmax_zoom = np.std(diff_zoom) * 3
plt.subplot(1, 2, 1)
plt.imshow(diff_zoom, cmap='RdBu', origin='lower',
           vmin=-vmax_zoom, vmax=vmax_zoom)
plt.title(f'100×100 Zoom of Difference\nCentered at ({zoom_center_x}, {zoom_center_y})')
plt.xlabel('X pixel')
plt.ylabel('Y pixel')
plt.colorbar(label='Difference (counts)', shrink=0.8)

# Absolute difference zoom
vmax_abs_zoom = np.percentile(abs_diff_zoom, 99.5)
plt.subplot(1, 2, 2)
plt.imshow(abs_diff_zoom, cmap='viridis', origin='lower',
           vmin=0, vmax=vmax_abs_zoom)
plt.title(
    f'100×100 Zoom of Absolute Difference\nCentered at ({zoom_center_x}, {zoom_center_y})\nMedian (whole image): {abs_median:.1f} counts')
plt.xlabel('X pixel')
plt.ylabel('Y pixel')
plt.colorbar(label='Absolute Difference (counts)', shrink=0.8)

plt.tight_layout()
plt.show(block=False)

# Histogram of the absolute difference values (central region) with logarithmic y-axis
print("\nDisplaying histogram of the absolute difference values (central 2000×2000 region)...")
plt.figure(figsize=(12, 8))
plt.hist(abs_diff_central_large.flatten(), bins=200, color='skyblue', alpha=0.7,
         edgecolor='black')
plt.yscale('log')  # Logarithmic vertical axis
plt.axvline(abs_median, color='red', linestyle='--', linewidth=2, label=f'Median: {abs_median:.2f}')
plt.axvline(np.mean(abs_diff_central_large), color='orange', linestyle=':', linewidth=2,
            label=f'Mean: {np.mean(abs_diff_central_large):.2f}')
plt.title('Histogram of Absolute Difference Values\n(Central 2000×2000, Logarithmic Y-Axis)')
plt.xlabel('Absolute Difference (counts)')
plt.ylabel('Number of pixels (log scale)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()  # last plot can be blocking

print(f"Absolute difference median (whole image): {abs_median:.1f} counts")
