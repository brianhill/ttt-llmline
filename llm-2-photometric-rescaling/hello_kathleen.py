import os
import glob
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import maximum_filter, shift
from astropy.modeling import models, fitting
from scipy.spatial import KDTree
import pandas as pd

# Plate-solving & WCS imports
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS, FITSFixedWarning
from astroquery.astrometry_net import AstrometryNet

# Suppress harmless WCS warning and enable non-blocking plots
import warnings
warnings.simplefilter("ignore", FITSFixedWarning)
plt.ion()

# ────────────────────────────────────────────────────────────────
# CONFIGURATION
# ────────────────────────────────────────────────────────────────
cwd = os.getcwd()
image_directory_path = "/Volumes/Astronomy Data/2025 TTT Targets/ttt-00167/bin3x3"
prefix = "TTT1_QHY411-1_2025-04-26-23-0"
suffix = ".fits"

target_alignment_accuracy = 0.1          # pixels
num_stars_for_alignment = 20
num_images_for_template = 7
saturation_threshold = 60000.0
duplicate_distance_threshold = 3.0
aperture_radius = 5.0
aperture_max_threshold = 10000.0
edge_margin = 200


###############################################################################
# IMAGE LOADING BLOCK
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
        with fits.open(filepath, uint=True) as hdul:
            data = hdul[0].data
            if data is None:
                raise ValueError(f"No image data in primary HDU of {filepath}")
            float_data = np.array(data, dtype=np.float32)
            float_data[float_data < 0] = 32768.0
            background = np.median(float_data)
            print(f"  Background (median) for {os.path.basename(filepath)}: {background:.1f} counts")
            float_data -= background
            image_arrays.append(float_data)

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

    existing_centroids = []
    y_grid, x_grid = np.indices(image_data.shape)
    ny, nx = image_data.shape

    for i in range(min(len(sorted_peaks_y), num_stars * 5)):
        y = sorted_peaks_y[i]
        x = sorted_peaks_x[i]

        y0, y1 = max(0, y - box_half_size), min(ny, y + box_half_size + 1)
        x0, x1 = max(0, x - box_half_size), min(nx, x + box_half_size + 1)
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

                dist_sq = (x_grid - full_x)**2 + (y_grid - full_y)**2
                aperture_mask = dist_sq <= aperture_radius**2
                if np.max(image_data[aperture_mask]) > aperture_max_threshold:
                    continue

                if (full_x < edge_margin or full_x > nx - edge_margin or
                    full_y < edge_margin or full_y > ny - edge_margin):
                    continue

                too_close = any(
                    np.sqrt((full_x - ex_x)**2 + (full_y - ex_y)**2) < duplicate_distance_threshold
                    for ex_x, ex_y in existing_centroids
                )
                if too_close:
                    continue

                star_data.append((full_x, full_y, fwhm_mean, peak_pixel))
                existing_centroids.append((full_x, full_y))
        except Exception:
            continue

        if len(star_data) >= num_stars:
            break

    if len(star_data) < num_stars:
        print(f"Warning: Only found {len(star_data)} good stars (requested {num_stars}).")

    return star_data


def estimate_fwhm(star_data):
    if len(star_data) == 0:
        raise ValueError("No star data for FWHM estimation.")
    return np.median([s[2] for s in star_data])


def find_suitable_bright_star(star_data, image_shape):
    if len(star_data) < 2:
        raise ValueError("Fewer than 2 stars found.")
    sorted_stars = sorted(star_data, key=lambda s: s[3], reverse=True)
    full_x, full_y, _, peak = sorted_stars[1]
    return full_y, full_x, peak


def extract_local_patch(image, center_y, center_x, radius=5):
    y0 = int(np.floor(center_y - radius))
    y1 = int(np.floor(center_y + radius + 1))
    x0 = int(np.floor(center_x - radius))
    x1 = int(np.floor(center_x + radius + 1))
    ny, nx = image.shape
    return image[max(0, y0):min(ny, y1), max(0, x0):min(nx, x1)]


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
    for pos_template in template_pos:
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

    aligned = shift(target, (aligned_shift_y, aligned_shift_x), order=5,
                    mode='constant', cval=np.median(target))

    for entry in matched_data:
        entry['aligned_x'] = entry['targ_x'] + aligned_shift_x
        entry['aligned_y'] = entry['targ_y'] + aligned_shift_y

    print(f"  Matched {len(deltas)} stars")
    print(f"  Applied shift: Δy = {aligned_shift_y:+.3f} px, Δx = {aligned_shift_x:+.3f} px")
    print(f"  (Target alignment accuracy: ±{target_alignment_accuracy:.2f} pixel)")

    return aligned, matched_data


# ────────────────────────────────────────────────────────────────
# MAIN EXECUTION
# ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        images_data, image_files = load_matching_fits_images(
            image_directory_path, prefix, suffix
        )

        print(f"\nTotal number of images in the sequence: {len(images_data)}")

        targ_img = images_data[-1]
        file2 = image_files[-1]
        print(f"\nTarget (last in sequence): {os.path.basename(file2)}")

        # Build fresh median template
        print(f"\nCreating median template from the first {num_images_for_template} images...")
        first_images = images_data[:num_images_for_template]
        template_base = first_images[0]
        template_star_data = find_star_centroids_and_fwhms(template_base, num_stars_for_alignment)

        aligned_first_images = [template_base]
        for img in first_images[1:]:
            aligned_img, _ = align_images(template_base, img, template_star_data, num_stars_for_alignment)
            aligned_first_images.append(aligned_img)

        template_img = np.median(aligned_first_images, axis=0)
        print("Median template created.")

        template_final_background = np.median(template_img)
        print(f"  Final background level: {template_final_background:.1f} counts (should be near 0)")

        # ────────────────────────────────────────────────────────────────
        # Plate solve the freshly built median template
        # ────────────────────────────────────────────────────────────────
        print("\nPlate-solving the newly built median template (ICRS/J2000)...")

        center_guess = SkyCoord(
            ra="12h30m30s",
            dec="+12d21m05s",
            frame='icrs',
            equinox='J2000'
        )

        plate_scale_arcsec_per_pix = 0.44
        approx_fov_deg = max(template_img.shape) * (plate_scale_arcsec_per_pix / 3600.0) * 1.15

        ast = AstrometryNet()
        ast.api_key = 'paurzuggyfqztyup'

        solved_header = None

        try:
            temp_path = os.path.join(cwd, "temp_median_for_solve.fits")
            fits.PrimaryHDU(template_img).writeto(temp_path, overwrite=True)

            solved_header = ast.solve_from_image(
                temp_path,
                center_ra=center_guess.ra.deg,
                center_dec=center_guess.dec.deg,
                radius=approx_fov_deg / 2.0,
                scale_lower=plate_scale_arcsec_per_pix * 0.65,
                scale_upper=plate_scale_arcsec_per_pix * 1.35,
                scale_units='arcsecperpix',
                solve_timeout=300,
                verbose=True
            )

            print("Plate solving succeeded!")

        except Exception as e:
            print(f"Plate solving failed: {e}")
            print("Continuing without WCS.")

        # ────────────────────────────────────────────────────────────────
        # Display central region of template (non-blocking)
        # ────────────────────────────────────────────────────────────────
        ny, nx = template_img.shape
        half = 500
        center_y_t, center_x_t = ny // 2, nx // 2
        template_central = template_img[center_y_t - half:center_y_t + half, center_x_t - half:center_x_t + half]

        fig1 = plt.figure(figsize=(10, 10))
        plt.imshow(template_central, cmap='gray', origin='lower',
                   vmin=np.percentile(template_central, 1), vmax=np.percentile(template_central, 99))
        plt.title(f'Central 1000×1000 of Median Template\nCenter: ({center_x_t:.1f}, {center_y_t:.1f})')
        plt.xlabel('X pixel')
        plt.ylabel('Y pixel')
        plt.colorbar(label='Counts', shrink=0.8)
        plt.tight_layout()
        plt.show(block=False)

        # (Add the other display blocks with plt.show(block=False) as needed)

        fwhm_pixels = estimate_fwhm(template_star_data)
        print(f"Estimated median FWHM: {fwhm_pixels:.2f} pixels")

        print("\nSelecting bright star for zoom...")
        center_y, center_x, template_star_peak = find_suitable_bright_star(template_star_data, template_img.shape)
        print(f"Selected star at (x, y) = ({center_x:.2f}, {center_y:.2f})")
        print(f"Net peak value: {template_star_peak:.1f} counts")

        unaligned_diff = template_img - targ_img

        print("\nAligning target image to template...")
        aligned_target, matched_centroids = align_images(template_img, targ_img, template_star_data,
                                                         num_stars_for_alignment)

        aligned_diff = template_img - aligned_target

        # ────────────────────────────────────────────────────────────────
        # Aperture photometry & photometric scaling (this block defines scaled_aligned_target)
        # ────────────────────────────────────────────────────────────────
        print("\nComputing aperture photometry for alignment stars...")
        photometry_data = []
        y_grid, x_grid = np.indices(template_img.shape)

        for i, (full_x, full_y, fwhm, peak) in enumerate(template_star_data):
            dist_sq = (x_grid - full_x)**2 + (y_grid - full_y)**2
            mask = dist_sq <= aperture_radius**2
            template_sum = np.sum(template_img[mask])
            science_sum = np.sum(aligned_target[mask])
            photometry_data.append({
                'Star': i + 1,
                'X': round(full_x, 2),
                'Y': round(full_y, 2),
                'Template Net Counts': round(template_sum),
                'Science Net Counts': round(science_sum)
            })

        df_phot = pd.DataFrame(photometry_data)
        print("\nAPERTURE PHOTOMETRY TABLE")
        print("="*80)
        print(df_phot.to_string(index=False))
        print("="*80)

        avg_template = np.mean(df_phot['Template Net Counts'])
        avg_science = np.mean(df_phot['Science Net Counts'])
        scale_ratio = avg_science / avg_template if avg_template != 0 else 1.0
        print(f"\nPhotometric scale ratio (science/template): {scale_ratio:.4f}")

        # This line defines the missing variable
        scaled_aligned_target = aligned_target / scale_ratio

        # Recompute photometry after scaling
        print("\nRecomputing photometry on rescaled science image...")
        for row in photometry_data:
            full_x = row['X']
            full_y = row['Y']
            mask = ((x_grid - full_x)**2 + (y_grid - full_y)**2) <= aperture_radius**2
            row['Rescaled Science Net Counts'] = round(np.sum(scaled_aligned_target[mask]))

        print("\nRESCALED PHOTOMETRY TABLE")
        print("="*100)
        print(pd.DataFrame(photometry_data).to_string(index=False))
        print("="*100)

        # Now use scaled_aligned_target safely
        scaled_aligned_diff = template_img - scaled_aligned_target

        local_diff_patch = extract_local_patch(scaled_aligned_diff, center_y, center_x, radius=5)
        local_diff_min = np.min(local_diff_patch)
        local_diff_max = np.max(local_diff_patch)

        # ────────────────────────────────────────────────────────────────
        # Save outputs with WCS propagation
        # ────────────────────────────────────────────────────────────────
        template_output_path = os.path.join(cwd, "median_template.fits")
        if solved_header is not None:
            fits.PrimaryHDU(template_img, header=solved_header).writeto(template_output_path, overwrite=True)
        else:
            fits.PrimaryHDU(template_img).writeto(template_output_path, overwrite=True)

        scaled_diff_path = os.path.join(cwd, "scaled_aligned_template_last_difference.fits")
        if solved_header is not None:
            fits.PrimaryHDU(scaled_aligned_diff, header=solved_header).writeto(scaled_diff_path, overwrite=True)
        else:
            fits.PrimaryHDU(scaled_aligned_diff).writeto(scaled_diff_path, overwrite=True)

        fits.PrimaryHDU(unaligned_diff).writeto(os.path.join(cwd, "unaligned_template_last_difference.fits"), overwrite=True)
        fits.PrimaryHDU(aligned_diff).writeto(os.path.join(cwd, "aligned_template_last_difference.fits"), overwrite=True)

        np.save(os.path.join(cwd, "median_template.npy"), template_img)
        np.save(os.path.join(cwd, "rescaled_aligned_science.npy"), scaled_aligned_target)

        print(f"\nSaved:")
        print(f"  Median template (with WCS if solved): {template_output_path}")
        print(f"  Scaled aligned difference (with WCS if solved): {scaled_diff_path}")

        # Final non-blocking plot
        fig_final, axs = plt.subplots(1, 2, figsize=(20, 8), sharey=True)
        rel_extent = [-15.5, 15.5, -15.5, 15.5]

        axs[0].imshow(unaligned_diff, cmap='RdBu', origin='lower',
                      vmin=-np.std(unaligned_diff)*3, vmax=np.std(unaligned_diff)*3,
                      extent=rel_extent)
        axs[0].set_title('Unaligned Difference')
        axs[0].set_xlabel('ΔX (pixels)')
        axs[0].set_ylabel('ΔY (pixels)')

        axs[1].imshow(scaled_aligned_diff, cmap='RdBu', origin='lower',
                      vmin=-np.std(scaled_aligned_diff)*3, vmax=np.std(scaled_aligned_diff)*3,
                      extent=rel_extent)
        axs[1].set_title('Scaled Aligned Difference')
        axs[1].set_xlabel('ΔX (pixels)')

        plt.suptitle("Difference Images (zoomed)")
        plt.tight_layout()
        plt.show(block=False)

        print("\nScript finished. All figures are non-blocking.")

    except Exception as e:
        print(f"Error: {e}")
