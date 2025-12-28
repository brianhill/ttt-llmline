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
        with fits.open(filepath) as hdul:
            data = hdul[0].data
            if data is None:
                raise ValueError(f"No image data in primary HDU of {filepath}")
            image_arrays.append(np.array(data, dtype=np.float32))

    return image_arrays, matching_files


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

    for i in range(min(len(sorted_peaks_y), num_stars * 2)):
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
                star_data.append((full_x, full_y, fwhm_mean, peak_pixel))
        except Exception:
            continue

        if len(star_data) >= num_stars:
            break

    if len(star_data) < num_stars:
        print(f"Warning: Only found {len(star_data)} good star fits (requested {num_stars}).")

    return star_data


def estimate_fwhm(star_data):
    if len(star_data) == 0:
        raise ValueError("No star data for FWHM estimation.")
    fwhms = [s[2] for s in star_data]
    return np.median(fwhms)


def find_suitable_bright_star(star_data, image_shape, max_dist_from_center=2000.0):
    ny, nx = image_shape
    cy, cx = ny / 2.0, nx / 2.0

    candidates = []
    for x, y, _, peak in star_data:
        dist_to_center = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        if dist_to_center <= max_dist_from_center:
            candidates.append((dist_to_center, peak, x, y))

    if len(candidates) < 2:
        raise ValueError("Fewer than 2 stars found within 2000 px of the image center.")

    candidates.sort(key=lambda t: (t[0], -t[1]))

    # Select the SECOND candidate
    _, _, best_x, best_y = candidates[1]

    for entry in star_data:
        if abs(entry[0] - best_x) < 0.1 and abs(entry[1] - best_y) < 0.1:
            selected_peak = entry[3]
            break
    else:
        selected_peak = 0.0

    return best_y, best_x, selected_peak


def extract_local_patch(image, center_y, center_x, radius=5):
    y0 = int(np.floor(center_y - radius))
    y1 = int(np.floor(center_y + radius + 1))
    x0 = int(np.floor(center_x - radius))
    x1 = int(np.floor(center_x + radius + 1))

    ny, nx = image.shape
    patch = image[max(0, y0):min(ny, y1), max(0, x0):min(nx, x1)]
    return patch


def align_images(reference, target, ref_star_data, num_stars=20):
    targ_star_data = find_star_centroids_and_fwhms(target, num_stars)

    if len(ref_star_data) < 5 or len(targ_star_data) < 5:
        raise ValueError("Insufficient stars for alignment.")

    ref_pos = np.array([[x, y] for x, y, _, _ in ref_star_data])
    targ_pos = np.array([[x, y] for x, y, _, _ in targ_star_data])

    tree = KDTree(targ_pos)
    max_dist = 20.0

    matched_data = []
    deltas = []
    for i, pos_ref in enumerate(ref_pos):
        dist, idx = tree.query(pos_ref)
        if dist < max_dist:
            pos_targ = targ_pos[idx]
            delta = pos_targ - pos_ref
            deltas.append(delta)
            matched_data.append({
                'ref_x': pos_ref[0],
                'ref_y': pos_ref[1],
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
    # reference by shifting it by the computed (sub-pixel) offsets.
    #
    # - shift() from scipy.ndimage uses spline interpolation.
    # - order=5 specifies 5th-order (quintic) spline interpolation for high accuracy.
    # - mode='constant' fills areas outside the original image with a constant value.
    # - cval=np.median(target) uses the median of the target image as the fill value
    #   to avoid introducing bright or dark edges.
    #
    # This interpolation allows precise sub-pixel shifts without resampling artifacts
    # that would degrade the subtraction quality.
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

        # NEW: Print total number of images loaded
        print(f"\nTotal number of images in the sequence: {len(images_data)}")

        # Use first and last images
        ref_img = images_data[0]  # First image in sequence
        targ_img = images_data[-1]  # Last image in sequence
        file1 = image_files[0]
        file2 = image_files[-1]

        print(f"\nUsing images:")
        print(f"  Reference (first in sequence): {os.path.basename(file1)}")
        print(f"  Target (last in sequence): {os.path.basename(file2)}")

        print("\nFinding stars in reference image...")
        ref_star_data = find_star_centroids_and_fwhms(ref_img, num_stars_for_alignment)
        fwhm_pixels = estimate_fwhm(ref_star_data)
        print(f"Estimated median FWHM: {fwhm_pixels:.2f} pixels")

        print("\nSelecting a different bright star (within 2000 px of center) for zoom...")
        center_y, center_x, reference_star_peak = find_suitable_bright_star(ref_star_data, ref_img.shape)
        print(f"Selected star at (x, y) = ({center_x:.2f}, {center_y:.2f})")
        print(f"Maximum value at the peak pixel of this reference star: {reference_star_peak:.1f} counts")
        zoom_half_size = int(5 * fwhm_pixels)

        # Compute differences
        unaligned_diff = ref_img - targ_img

        print("\nAligning target image to reference using star centroids...")
        aligned_target, matched_centroids = align_images(ref_img, targ_img, ref_star_data, num_stars_for_alignment)

        aligned_diff = ref_img - aligned_target

        # Local max in reference star (original image)
        local_ref_patch = extract_local_patch(ref_img, center_y, center_x, radius=5)
        local_ref_max = np.max(local_ref_patch)

        # Local min/max in aligned difference around the same star
        local_diff_patch = extract_local_patch(aligned_diff, center_y, center_x, radius=5)
        local_diff_min = np.min(local_diff_patch)
        local_diff_max = np.max(local_diff_patch)

        # Save differences
        unaligned_output_path = os.path.join(cwd, "unaligned_first_last_difference.fits")
        aligned_output_path = os.path.join(cwd, "aligned_first_last_difference.fits")
        fits.PrimaryHDU(unaligned_diff).writeto(unaligned_output_path, overwrite=True)
        fits.PrimaryHDU(aligned_diff).writeto(aligned_output_path, overwrite=True)

        print("\nUnaligned difference (global):")
        print(f"  Mean: {np.mean(unaligned_diff):.4f}   Std: {np.std(unaligned_diff):.4f}")
        print("\nAligned difference (final result):")
        print(f"  Global mean: {np.mean(aligned_diff):.4f}   Global std: {np.std(aligned_diff):.4f}")
        print(f"  Local (within 5 px of reference star) minimum: {local_diff_min:.1f}")
        print(f"  Local (within 5 px of reference star) maximum: {local_diff_max:.1f}")
        print(
            f"\nNote: Local reference star max in original image: {local_ref_max:.1f} counts (should match peak above)")
        print(f"\nDifference images saved to:\n  {unaligned_output_path}\n  {aligned_output_path}")

        # Centroid table
        print("\n" + "=" * 100)
        print("CENTROID TABLE FOR MATCHED STARS")
        print("=" * 100)
        df_centroids = pd.DataFrame(matched_centroids)
        df_centroids = df_centroids[['ref_x', 'ref_y', 'targ_x', 'targ_y', 'aligned_x', 'aligned_y']]
        df_centroids.columns = ['Ref X', 'Ref Y', 'Targ X', 'Targ Y', 'Aligned Targ X', 'Aligned Targ Y']
        df_centroids = df_centroids.round(3)
        print(df_centroids.to_string(index=False))
        print("=" * 100)

        # Display zoomed differences
        fig, axs = plt.subplots(1, 2, figsize=(20, 8), sharey=True)

        im0 = axs[0].imshow(unaligned_diff, cmap='RdBu', origin='lower',
                            vmin=-np.std(unaligned_diff) * 3, vmax=np.std(unaligned_diff) * 3)
        axs[0].set_title(f'Unaligned Difference (first − last)\n{os.path.basename(file1)} − {os.path.basename(file2)}\n'
                         f'FWHM ≈ {fwhm_pixels:.2f} px | Zoom on star at ({center_x:.1f}, {center_y:.1f})')
        axs[0].set_xlabel('X pixel')
        axs[0].set_ylabel('Y pixel')
        axs[0].set_xlim(center_x - zoom_half_size, center_x + zoom_half_size)
        axs[0].set_ylim(center_y - zoom_half_size, center_y + zoom_half_size)
        fig.colorbar(im0, ax=axs[0], label='Difference (counts)', shrink=0.8)

        im1 = axs[1].imshow(aligned_diff, cmap='RdBu', origin='lower',
                            vmin=-np.std(aligned_diff) * 3, vmax=np.std(aligned_diff) * 3)
        axs[1].set_title(f'Aligned Difference (first − last)\n{os.path.basename(file1)} − {os.path.basename(file2)}\n'
                         f'Alignment target: ±{target_alignment_accuracy:.2f} px')
        axs[1].set_xlabel('X pixel')
        axs[1].set_xlim(center_x - zoom_half_size, center_x + zoom_half_size)
        axs[1].set_ylim(center_y - zoom_half_size, center_y + zoom_half_size)
        fig.colorbar(im1, ax=axs[1], label='Difference (counts)', shrink=0.8)

        plt.tight_layout()
        plt.show()

    except FileNotFoundError as e:
        print(f"\nFile error: {e}")
    except ValueError as e:
        print(f"\nProcessing error: {e}")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
