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
    """
    Finds and loads FITS images matching the prefix and suffix.
    Returns list of image data (as float32) and list of filenames.
    """
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
    """
    Finds bright, isolated sources and fits 2D Gaussians to get sub-pixel centroids and FWHMs.
    Returns list of (x, y, fwhm_mean) for successful fits.
    """
    background = np.median(image_data)
    data_sub = image_data - background

    local_max = maximum_filter(data_sub, size=15) == data_sub
    peaks = np.where(local_max & (data_sub > 5 * np.std(data_sub)))

    if len(peaks[0]) == 0:
        raise ValueError("No bright sources detected.")

    star_data = []
    box_half_size = 15
    fitter = fitting.LevMarLSQFitter()

    # Sort peaks by brightness descending
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
                star_data.append((full_x, full_y, fwhm_mean))
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


def find_bright_star(star_data):
    if len(star_data) == 0:
        raise ValueError("No stars for zooming.")
    x, y, _ = star_data[0]  # brightest
    return y, x


def align_images(reference, target, ref_star_data, num_stars=20):
    """
    Aligns target to reference using matched centroids.
    Returns aligned target and list of matched centroid data for table.
    """
    targ_star_data = find_star_centroids_and_fwhms(target, num_stars)

    if len(ref_star_data) < 5 or len(targ_star_data) < 5:
        raise ValueError("Insufficient stars for alignment.")

    ref_pos = np.array([[x, y] for x, y, _ in ref_star_data])
    targ_pos = np.array([[x, y] for x, y, _ in targ_star_data])

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
            # Store for table: ref (x,y), original targ (x,y), aligned targ (x,y) = original - mean_delta
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

    aligned = shift(target, (aligned_shift_y, aligned_shift_x), order=5, mode='constant', cval=np.median(target))

    # Add aligned positions to table
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
        # Load images
        images_data, image_files = load_matching_fits_images(
            image_directory_path, prefix, suffix
        )

        # Select first two images
        ref_img = images_data[0]
        targ_img = images_data[1]
        file1 = image_files[0]
        file2 = image_files[1]

        print(f"\nUsing images:")
        print(f"  Reference (fixed): {os.path.basename(file1)}")
        print(f"  Target: {os.path.basename(file2)}")

        # Find stars in reference
        print("\nFinding stars in reference image...")
        ref_star_data = find_star_centroids_and_fwhms(ref_img, num_stars_for_alignment)
        fwhm_pixels = estimate_fwhm(ref_star_data)
        print(f"Estimated median FWHM: {fwhm_pixels:.2f} pixels")

        # Find bright star for zoom
        print("\nFinding a bright star for zoom...")
        center_y, center_x = find_bright_star(ref_star_data)
        print(f"Selected star at (x, y) = ({center_x:.1f}, {center_y:.1f})")
        zoom_half_size = int(5 * fwhm_pixels)

        # Compute unaligned difference
        unaligned_diff = ref_img - targ_img

        # Align using centroids
        print("\nAligning target image to reference using star centroids...")
        aligned_target, matched_centroids = align_images(ref_img, targ_img, ref_star_data, num_stars_for_alignment)

        # Compute aligned difference
        aligned_diff = ref_img - aligned_target

        # Save differences
        unaligned_output_path = os.path.join(cwd, "unaligned_difference.fits")
        aligned_output_path = os.path.join(cwd, "aligned_difference.fits")
        fits.PrimaryHDU(unaligned_diff).writeto(unaligned_output_path, overwrite=True)
        fits.PrimaryHDU(aligned_diff).writeto(aligned_output_path, overwrite=True)

        # Print statistics
        print("\nUnaligned difference:")
        print(f"  Mean: {np.mean(unaligned_diff):.4f}   Std: {np.std(unaligned_diff):.4f}")
        print("\nAligned difference:")
        print(f"  Mean: {np.mean(aligned_diff):.4f}   Std: {np.std(aligned_diff):.4f}")
        print(f"\nDifference images saved to:\n  {unaligned_output_path}\n  {aligned_output_path}")

        # Print table of centroids
        print("\n" + "=" * 80)
        print("CENTROID TABLE FOR MATCHED STARS")
        print("=" * 80)
        df = pd.DataFrame(matched_centroids)
        df = df[['ref_x', 'ref_y', 'targ_x', 'targ_y', 'aligned_x', 'aligned_y']]
        df.columns = ['Ref X', 'Ref Y', 'Targ X', 'Targ Y', 'Aligned Targ X', 'Aligned Targ Y']
        df = df.round(3)
        print(df.to_string(index=False))
        print("=" * 80)

        # Display zoomed differences
        fig, axs = plt.subplots(1, 2, figsize=(20, 8), sharey=True)

        im0 = axs[0].imshow(unaligned_diff, cmap='RdBu', origin='lower',
                            vmin=-np.std(unaligned_diff) * 3, vmax=np.std(unaligned_diff) * 3)
        axs[0].set_title(f'Unaligned Difference\n{os.path.basename(file1)} − {os.path.basename(file2)}\n'
                         f'FWHM ≈ {fwhm_pixels:.2f} px | Zoom on star at ({center_x:.1f}, {center_y:.1f})')
        axs[0].set_xlabel('X pixel')
        axs[0].set_ylabel('Y pixel')
        axs[0].set_xlim(center_x - zoom_half_size, center_x + zoom_half_size)
        axs[0].set_ylim(center_y - zoom_half_size, center_y + zoom_half_size)
        fig.colorbar(im0, ax=axs[0], label='Difference (counts)', shrink=0.8)

        im1 = axs[1].imshow(aligned_diff, cmap='RdBu', origin='lower',
                            vmin=-np.std(aligned_diff) * 3, vmax=np.std(aligned_diff) * 3)
        axs[1].set_title(f'Aligned Difference\n{os.path.basename(file1)} − {os.path.basename(file2)}\n'
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
