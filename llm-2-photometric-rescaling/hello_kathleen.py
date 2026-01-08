import os
import glob
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import maximum_filter, shift
from astropy.modeling import models, fitting
from scipy.spatial import KDTree
import pandas as pd

# Your original configuration (kept exactly as provided)
cwd = os.getcwd()
image_directory_path = "/Volumes/Astronomy Data/2025 TTT Targets/ttt-00167/bin3x3"
prefix = "TTT1_QHY411-1_2025-04-26-23-0"
suffix = ".fits"

# Alignment precision (fixed value)
target_alignment_accuracy = 0.1  # pixels
num_stars_for_alignment = 20
num_images_for_template = 7  # Number of first images to use for median template
saturation_threshold = 60000.0  # Dismiss stars with peak > this value (likely saturated)
duplicate_distance_threshold = 3.0  # Pixels; skip if centroid within this distance of existing
aperture_radius = 5.0  # Radius for aperture checks
aperture_max_threshold = 10000.0  # Dismiss stars if any pixel in aperture > this value
edge_margin = 200  # Exclude stars within this many pixels of image edges


###############################################################################
# IMAGE LOADING BLOCK
#
# This section reads the calibrated FITS images from disk.
#
# Key points:
# - The files are assumed to be fully calibrated science frames
#   (bias, dark, flat-field corrections already applied).
# - Many astronomical cameras (including QHY series) store raw or calibrated
#   pixel data as unsigned 16-bit integers (UINT16 / BITPIX=16, BZERO=32768, BSCALE=1).
# - astropy.io.fits automatically handles the FITS scaling keywords (BSCALE/BZERO)
#   and returns data as signed integers or floats, correctly mapping the unsigned
#   range 0–65535 to positive values.
# - However, to guarantee safe handling regardless of header keywords and to
#   prevent any risk of negative values due to signed interpretation, we:
#     1. Open with uint=True → forces astropy to treat the raw pixel data as
#        unsigned integers (no sign interpretation).
#     2. Immediately convert to np.float32 → preserves the exact original values
#        (0–65535) as positive floats and allows safe arithmetic without overflow.
#     3. After conversion, any remaining negative values (which should not occur
#        with proper calibration but can appear due to over-subtraction) are
#        replaced with 32768.0 — the conventional "zero" level for unsigned 16-bit
#        data (corresponding to BZERO in FITS headers).
#     4. Subtract the whole-image median to create background-subtracted data.
#        This is used for all subsequent processing (star detection, alignment,
#        template creation, photometry, differences).
#
# Benefits:
# - Large UINT16 values (e.g., >32767) will never become negative.
# - Background is normalized to ~0 in each frame.
# - All subsequent operations are performed on background-subtracted data with
#   full floating-point precision.
#
# Potential sources of artifacts:
# - If calibration produced negative values that were clipped to zero, or if
#   BZERO/BSCALE were incorrect, residual patterns may remain.
# - Residual flat-field gradients or illumination mismatches can persist.
#
# Recommendation: Always verify a few frames visually and check header keywords
# (BITPIX, BZERO, BSCALE) to confirm data integrity.
###############################################################################
def load_matching_fits_images(dir_path, file_prefix, file_suffix):
    pattern = os.path.join(dir_path, file_prefix + "*" + file_suffix)
    matching_files = sorted(glob.glob(pattern))

    if len(matching_files) == 0:
        raise FileNotFoundError(f"No files found matching pattern: {pattern}")
    if len(matching_files) < 2:
        raise ValueError(f"Only {len(matching_files)} file(s) found. Need at least 2 for differencing.")

    print(f"Found {len(matching_files)} matching FITS files:")
    for f in matching_files:
        print(f"  - {os.path.basename(f)}")

    image_arrays = []
    for filepath in matching_files:
        # Open FITS with uint=True to correctly interpret unsigned integer data
        with fits.open(filepath, uint=True) as hdul:
            data = hdul[0].data
            if data is None:
                raise ValueError(f"No image data in primary HDU of {filepath}")
            # Convert to float32 to preserve exact values and enable safe arithmetic
            float_data = np.array(data, dtype=np.float32)
            # Fix any negative values (from over-subtraction) to 32768.0 (UINT16 zero level)
            float_data[float_data < 0] = 32768.0
            # Compute and report whole-image median background
            background = np.median(float_data)
            print(f"  Background (median) for {os.path.basename(filepath)}: {background:.1f} counts")
            # Subtract background - all further processing uses background-subtracted data
            float_data -= background
            image_arrays.append(float_data)

    return image_arrays, matching_files


def find_star_centroids_and_fwhms(image_data, num_stars=20):
    # Note: image_data is already background-subtracted from loader
    # The additional subtraction here is redundant but harmless (median ~0)
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

    # List to track existing centroids for de-duplication
    existing_centroids = []

    # Pre-compute coordinate grids for aperture checks
    y_grid, x_grid = np.indices(image_data.shape)

    # Image dimensions for edge filtering
    ny, nx = image_data.shape

    for i in range(min(len(sorted_peaks_y), num_stars * 5)):  # Try more to fill after filtering
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
                # Peak pixel in background-subtracted image (net above background)
                peak_pixel = image_data[int(np.round(full_y)), int(np.round(full_x))]

                # Dismiss saturated stars (peak pixel)
                if peak_pixel > saturation_threshold:
                    continue

                # Additional check: dismiss if any pixel in 5-pixel aperture > 10000
                dist_sq = (x_grid - full_x) ** 2 + (y_grid - full_y) ** 2
                aperture_mask = dist_sq <= aperture_radius ** 2
                if np.max(image_data[aperture_mask]) > aperture_max_threshold:
                    continue

                # Eliminate stars within 200 pixels of image edges
                if (full_x < edge_margin or full_x > nx - edge_margin or
                        full_y < edge_margin or full_y > ny - edge_margin):
                    continue

                # De-duplicate: check distance to existing centroids
                too_close = False
                for ex_x, ex_y in existing_centroids:
                    dist = np.sqrt((full_x - ex_x) ** 2 + (full_y - ex_y) ** 2)
                    if dist < duplicate_distance_threshold:
                        too_close = True
                        break
                if too_close:
                    continue

                # Accept this star
                star_data.append((full_x, full_y, fwhm_mean, peak_pixel))
                existing_centroids.append((full_x, full_y))
        except Exception:
            continue

        if len(star_data) >= num_stars:
            break

    if len(star_data) < num_stars:
        print(f"Warning: Only found {len(star_data)} unique non-saturated non-edge star fits (requested {num_stars}).")

    return star_data


def estimate_fwhm(star_data):
    if len(star_data) == 0:
        raise ValueError("No star data for FWHM estimation.")
    fwhms = [s[2] for s in star_data]
    return np.median(fwhms)


# Uses stars from the entire image (no central restriction for zoom star)
def find_suitable_bright_star(star_data, image_shape):
    if len(star_data) < 2:
        raise ValueError("Fewer than 2 stars found in the image.")

    # Sort by brightness (peak pixel) descending
    sorted_stars = sorted(star_data, key=lambda s: s[3], reverse=True)

    # Select the SECOND brightest star
    selected = sorted_stars[1]
    full_x, full_y, _, selected_peak = selected

    return full_y, full_x, selected_peak


def extract_local_patch(image, center_y, center_x, radius=5):
    y0 = int(np.floor(center_y - radius))
    y1 = int(np.floor(center_y + radius + 1))
    x0 = int(np.floor(center_x - radius))
    x1 = int(np.floor(center_x + radius + 1))

    ny, nx = image.shape
    patch = image[max(0, y0):min(ny, y1), max(0, x0):min(nx, x1)]
    return patch


def align_images(template, target, template_star_data, num_stars=20):
    targ_star_data = find_star_centroids_and_fwhms(target, num_stars)

    if len(template_star_data) < 5 or len(targ_star_data) < 5:
        raise ValueError("Insufficient stars for alignment.")

    template_pos = np.array([[x, y] for x, y, _, _ in template_star_data])
    targ_pos = np.array([[x, y] for x, y, _, _ in targ_star_data])

    tree = KDTree(targ_pos)
    max_dist = 20.0

    matched_data = []
    deltas = []
    for i, pos_template in enumerate(template_pos):
        dist, idx = tree.query(pos_template)
        if dist < max_dist:
            pos_targ = targ_pos[idx]
            delta = pos_targ - pos_template
            deltas.append(delta)
            matched_data.append({
                'template_x': pos_template[0],
                'template_y': pos_template[1],
                'targ_x': pos_targ[0],
                'targ_y': pos_targ[1]
            })

    if len(deltas) < 5:
        raise ValueError(f"Only {len(deltas)} matched stars found.")

    mean_delta = np.mean(deltas, axis=0)
    shift_x, shift_y = mean_delta
    aligned_shift_y, aligned_shift_x = -shift_y, -shift_x

    # ==============================================================================
    # KEY STEP: Sub-pixel image displacement via interpolation
    #
    # The following line performs the actual alignment of the target image to the
    # template by shifting it by the computed (sub-pixel) offsets.
    #
    # - shift() from scipy.ndimage uses spline interpolation.
    # - order=5 specifies 5th-order (quintic) spline interpolation for high accuracy.
    # - mode='constant' fills areas outside the original image with a constant value.
    # - cval=np.median(target) uses the median of the target image as the fill value
    #   to avoid introducing bright or dark edges.
    #
    # This interpolation allows precise sub-pixel shifts without resampling artifacts
    #   that would degrade the subtraction quality.
    # ==============================================================================
    aligned = shift(target, (aligned_shift_y, aligned_shift_x), order=5, mode='constant', cval=np.median(target))

    for entry in matched_data:
        entry['aligned_x'] = entry['targ_x'] + aligned_shift_x
        entry['aligned_y'] = entry['targ_y'] + aligned_shift_y

    print(f"  Matched {len(deltas)} stars")
    print(f"  Applied shift: Δy = {aligned_shift_y:+.3f} px, Δx = {aligned_shift_x:+.3f} px")
    print(f"  (Target alignment accuracy: ±{target_alignment_accuracy:.2f} pixel)")

    return aligned, matched_data


# Main execution
if __name__ == "__main__":
    try:
        images_data, image_files = load_matching_fits_images(
            image_directory_path, prefix, suffix
        )

        print(f"\nTotal number of images in the sequence: {len(images_data)}")

        # Use last image as target
        targ_img = images_data[-1]
        file2 = image_files[-1]

        print(f"\nTarget (last in sequence): {os.path.basename(file2)}")

        ############################################################################
        # TEMPLATE GENERATION BLOCK
        #
        # This block creates a high-quality median template from the first
        # num_images_for_template images in the sequence.
        #
        # 1. The very first image is used as the alignment base.
        # 2. Star centroids are measured on this base image.
        # 3. Each subsequent image (2nd through 7th) is aligned to the base using
        #    the same centroid-matching and sub-pixel shift method as the main
        #    alignment.
        # 4. A pixel-by-pixel median is computed across the 7 aligned images.
        #
        # Benefits:
        #   - Cosmic rays and transient artifacts are rejected by the median.
        #   - Signal-to-noise is improved.
        #   - The template remains photometrically consistent with the individual
        #     frames.
        #
        # The resulting template_img is used for all further analysis and subtraction.
        ############################################################################
        print(f"\nCreating median template from the first {num_images_for_template} images...")
        first_images = images_data[:num_images_for_template]
        template_base = first_images[0]
        template_star_data = find_star_centroids_and_fwhms(template_base, num_stars_for_alignment)

        aligned_first_images = [template_base]  # First image is already aligned to itself
        for img in first_images[1:]:
            aligned_img, _ = align_images(template_base, img, template_star_data, num_stars_for_alignment)
            aligned_first_images.append(aligned_img)

        template_img = np.median(aligned_first_images, axis=0)
        print("Median template created.")

        # Compute and print final template background (should be near 0)
        template_final_background = np.median(template_img)
        print(
            f"  Median template final background (after subtraction and combination): {template_final_background:.1f} counts (expected near 0)")

        # Display central 1000×1000 of the template
        ny, nx = template_img.shape
        half = 500
        center_y_t, center_x_t = ny // 2, nx // 2
        template_central = template_img[center_y_t - half:center_y_t + half, center_x_t - half:center_x_t + half]

        plt.figure(figsize=(10, 10))
        plt.imshow(template_central, cmap='gray', origin='lower', vmin=np.percentile(template_central, 1),
                   vmax=np.percentile(template_central, 99))
        plt.title(
            f'Central 1000×1000 pixels of the Median Template\nCenter pixel: ({center_x_t:.1f}, {center_y_t:.1f})')
        plt.xlabel('X pixel')
        plt.ylabel('Y pixel')
        plt.colorbar(label='Counts', shrink=0.8)
        plt.tight_layout()
        plt.show()

        # Display central 1000×1000 of all seven aligned images used for the template
        print("\nDisplaying central 1000×1000 pixels of the seven aligned images used to build the template...")
        fig, axs = plt.subplots(2, 4, figsize=(20, 10))
        axs = axs.ravel()

        for i, aligned_img in enumerate(aligned_first_images):
            ny_i, nx_i = aligned_img.shape
            cy_i, cx_i = ny_i // 2, nx_i // 2
            central_i = aligned_img[cy_i - half:cy_i + half, cx_i - half:cx_i + half]
            im = axs[i].imshow(central_i, cmap='gray', origin='lower',
                               vmin=np.percentile(central_i, 1), vmax=np.percentile(central_i, 99))
            axs[i].set_title(f'Image {i + 1} (aligned)\nCenter: ({cx_i:.1f}, {cy_i:.1f})')
            axs[i].set_xlabel('X pixel')
            axs[i].set_ylabel('Y pixel')
            fig.colorbar(im, ax=axs[i], shrink=0.8)

        # Hide the empty 8th subplot
        axs[7].axis('off')

        plt.suptitle('Central 1000×1000 pixels of the 7 aligned images used for the median template')
        plt.tight_layout()
        plt.show()

        # Display central 1000×1000 of the original (unaligned) seven images
        print("\nDisplaying central 1000×1000 pixels of the original (unaligned) seven images used for the template...")
        fig_un, axs_un = plt.subplots(2, 4, figsize=(20, 10))
        axs_un = axs_un.ravel()

        for i, orig_img in enumerate(first_images):
            ny_i, nx_i = orig_img.shape
            cy_i, cx_i = ny_i // 2, nx_i // 2
            central_orig = orig_img[cy_i - half:cy_i + half, cx_i - half:cx_i + half]
            im_un = axs_un[i].imshow(central_orig, cmap='gray', origin='lower',
                                     vmin=np.percentile(central_orig, 1), vmax=np.percentile(central_orig, 99))
            axs_un[i].set_title(f'Image {i + 1} (original, unaligned)\nCenter: ({cx_i:.1f}, {cy_i:.1f})')
            axs_un[i].set_xlabel('X pixel')
            axs_un[i].set_ylabel('Y pixel')
            fig_un.colorbar(im_un, ax=axs_un[i], shrink=0.8)

        # Hide the empty 8th subplot
        axs_un[7].axis('off')

        plt.suptitle('Central 1000×1000 pixels of the 7 original (unaligned) images used for the median template')
        plt.tight_layout()
        plt.show()

        fwhm_pixels = estimate_fwhm(template_star_data)
        print(f"Estimated median FWHM (from first image): {fwhm_pixels:.2f} pixels")

        print("\nSelecting a different bright star for zoom...")
        center_y, center_x, template_star_peak = find_suitable_bright_star(template_star_data, template_img.shape)
        print(f"Selected star at (x, y) = ({center_x:.2f}, {center_y:.2f})")
        print(f"Net peak value above background of this template star: {template_star_peak:.1f} counts")
        zoom_half_size = int(5 * fwhm_pixels)

        # Compute differences
        unaligned_diff = template_img - targ_img

        print("\nAligning target image to template using star centroids...")
        aligned_target, matched_centroids = align_images(template_img, targ_img, template_star_data,
                                                         num_stars_for_alignment)

        aligned_diff = template_img - aligned_target

        # Aperture photometry table for the stars
        print("\nComputing aperture photometry for the alignment stars...")
        photometry_data = []
        y_grid, x_grid = np.indices(template_img.shape)

        for i, (full_x, full_y, fwhm, peak) in enumerate(template_star_data):
            dist_sq = (x_grid - full_x) ** 2 + (y_grid - full_y) ** 2
            aperture_mask = dist_sq <= aperture_radius ** 2

            template_sum = np.sum(template_img[aperture_mask])
            science_sum = np.sum(aligned_target[aperture_mask])

            photometry_data.append({
                'Star': i + 1,
                'X': round(full_x, 2),
                'Y': round(full_y, 2),
                'Template Net Counts': round(template_sum),
                'Science Net Counts': round(science_sum)
            })

        df_phot = pd.DataFrame(photometry_data)
        print("\n" + "=" * 80)
        print("APERTURE PHOTOMETRY TABLE (5-pixel radius, net above background)")
        print("=" * 80)
        print(df_phot.to_string(index=False))
        print("=" * 80)

        # Photometric scaling of the science image to match template
        avg_template = np.mean(df_phot['Template Net Counts'])
        avg_science = np.mean(df_phot['Science Net Counts'])
        if avg_template == 0:
            print("Warning: Average template counts is zero - cannot scale.")
            scale_ratio = 1.0
        else:
            scale_ratio = avg_science / avg_template
        print(f"\nAverage template net counts: {avg_template:.0f}")
        print(f"Average science net counts (aligned): {avg_science:.0f}")
        print(f"Photometric scale ratio (science / template): {scale_ratio:.4f}")

        # Scale the aligned science image to match template flux
        scaled_aligned_target = aligned_target / scale_ratio

        # Recompute photometry on the rescaled science image
        print("\nRecomputing photometry on the rescaled science image...")
        for row in photometry_data:
            full_x = row['X']
            full_y = row['Y']
            dist_sq = (x_grid - full_x) ** 2 + (y_grid - full_y) ** 2
            aperture_mask = dist_sq <= aperture_radius ** 2
            rescaled_sum = np.sum(scaled_aligned_target[aperture_mask])
            row['Rescaled Science Net Counts'] = round(rescaled_sum)

        df_phot = pd.DataFrame(photometry_data)
        print("\n" + "=" * 100)
        print("APERTURE PHOTOMETRY TABLE WITH RESCALED SCIENCE VALUES (5-pixel radius, net above background)")
        print("=" * 100)
        print(df_phot.to_string(index=False))
        print("=" * 100)

        avg_rescaled = np.mean(df_phot['Rescaled Science Net Counts'])
        print(f"\nAverage rescaled science net counts: {avg_rescaled:.0f}")

        # Recompute the final difference with scaled science image
        scaled_aligned_diff = template_img - scaled_aligned_target

        # Local min/max now on the scaled difference
        local_diff_patch = extract_local_patch(scaled_aligned_diff, center_y, center_x, radius=5)
        local_diff_min = np.min(local_diff_patch)
        local_diff_max = np.max(local_diff_patch)

        # NEW: Save the template and rescaled science images as NumPy arrays
        template_npy_path = os.path.join(cwd, "median_template.npy")
        rescaled_science_npy_path = os.path.join(cwd, "rescaled_aligned_science.npy")
        np.save(template_npy_path, template_img)
        np.save(rescaled_science_npy_path, scaled_aligned_target)
        print(f"\nSaved template image as NumPy array: {template_npy_path}")
        print(f"Saved rescaled science image as NumPy array: {rescaled_science_npy_path}")

        # Save the scaled difference as the final result
        scaled_aligned_output_path = os.path.join(cwd, "scaled_aligned_template_last_difference.fits")
        fits.PrimaryHDU(scaled_aligned_diff).writeto(scaled_aligned_output_path, overwrite=True)

        print(f"\nScaled aligned difference saved to: {scaled_aligned_output_path}")

        # Update aligned_diff for display
        aligned_diff = scaled_aligned_diff

        # Local max in template star (original image)
        local_template_patch = extract_local_patch(template_img, center_y, center_x, radius=5)
        local_template_max = np.max(local_template_patch)

        # Save differences
        unaligned_output_path = os.path.join(cwd, "unaligned_template_last_difference.fits")
        aligned_output_path = os.path.join(cwd, "aligned_template_last_difference.fits")
        fits.PrimaryHDU(unaligned_diff).writeto(unaligned_output_path, overwrite=True)
        fits.PrimaryHDU(aligned_diff).writeto(aligned_output_path, overwrite=True)

        # Also save the template image
        template_output_path = os.path.join(cwd, "median_template.fits")
        fits.PrimaryHDU(template_img).writeto(template_output_path, overwrite=True)

        print("\nUnaligned difference (global):")
        print(f"  Mean: {np.mean(unaligned_diff):.4f}   Std: {np.std(unaligned_diff):.4f}")
        print("\nScaled aligned difference (final result):")
        print(f"  Global mean: {np.mean(aligned_diff):.4f}   Global std: {np.std(aligned_diff):.4f}")
        print(
            f"  Local (within 5 px of template star at ({center_x:.1f}, {center_y:.1f})) minimum: {local_diff_min:.1f}")
        print(
            f"  Local (within 5 px of template star at ({center_x:.1f}, {center_y:.1f})) maximum: {local_diff_max:.1f}")
        print(
            f"\nNote: Local template star max in original image: {local_template_max:.1f} counts (should match peak above)")
        print(f"\nImages saved to:")
        print(f"  Median template: {template_output_path}")
        print(f"  Unaligned difference: {unaligned_output_path}")
        print(f"  Scaled aligned difference: {scaled_aligned_output_path}")

        # Centroid table
        print("\n" + "=" * 100)
        print("CENTROID TABLE FOR MATCHED STARS")
        print("=" * 100)
        df_centroids = pd.DataFrame(matched_centroids)
        df_centroids = df_centroids[['template_x', 'template_y', 'targ_x', 'targ_y', 'aligned_x', 'aligned_y']]
        df_centroids.columns = ['Template X', 'Template Y', 'Targ X', 'Targ Y', 'Aligned Targ X', 'Aligned Targ Y']
        df_centroids = df_centroids.round(3)
        print(df_centroids.to_string(index=False))
        print("=" * 100)

        # Display zoomed differences
        fig, axs = plt.subplots(1, 2, figsize=(20, 8), sharey=True)

        # Define relative coordinates from -15 to +15 around the selected star
        rel_extent = [-15.5, 15.5, -15.5, 15.5]  # half-pixel offset for correct centering

        im0 = axs[0].imshow(unaligned_diff, cmap='RdBu', origin='lower',
                            vmin=-np.std(unaligned_diff) * 3, vmax=np.std(unaligned_diff) * 3,
                            extent=rel_extent)
        axs[0].set_title(f'Unaligned Difference (template − last)\nMedian template − {os.path.basename(file2)}\n'
                         f'FWHM ≈ {fwhm_pixels:.2f} px | Center: ({center_x:.1f}, {center_y:.1f})')
        axs[0].set_xlabel('ΔX from center (pixels)')
        axs[0].set_ylabel('ΔY from center (pixels)')
        axs[0].set_xlim(-15, 15)
        axs[0].set_ylim(-15, 15)
        fig.colorbar(im0, ax=axs[0], label='Difference (counts)', shrink=0.8)

        im1 = axs[1].imshow(aligned_diff, cmap='RdBu', origin='lower',
                            vmin=-np.std(aligned_diff) * 3, vmax=np.std(aligned_diff) * 3,
                            extent=rel_extent)
        axs[1].set_title(f'Scaled Aligned Difference (template − last)\nMedian template − {os.path.basename(file2)}\n'
                         f'Alignment target: ±{target_alignment_accuracy:.2f} px | Center: ({center_x:.1f}, {center_y:.1f})')
        axs[1].set_xlabel('ΔX from center (pixels)')
        axs[1].set_xlim(-15, 15)
        axs[1].set_ylim(-15, 15)
        fig.colorbar(im1, ax=axs[1], label='Difference (counts)', shrink=0.8)

        plt.tight_layout()
        plt.show()

    except FileNotFoundError as e:
        print(f"\nFile error: {e}")
    except ValueError as e:
        print(f"\nProcessing error: {e}")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
