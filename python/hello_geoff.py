import os
import glob
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import maximum_filter
from astropy.modeling import models, fitting

# Your original configuration (kept exactly as provided)
cwd = os.getcwd()
image_directory_path = "/Volumes/Astronomy Data/2025 TTT Targets/ttt-00167/bin3x3"
prefix = "TTT1_QHY411-1_2025-04-26-23-0"
suffix = ".fits"


# Renamed function to avoid any shadowing
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


def estimate_fwhm(image_data, num_stars=20):
    """
    Simple FWHM estimation using bright, isolated sources.
    Finds bright peaks, fits 2D Gaussian around each, and returns median FWHM in pixels.
    """
    # Rough background subtraction using median
    background = np.median(image_data)
    data_sub = image_data - background

    # Find local maxima (potential stars)
    local_max = maximum_filter(data_sub, size=15) == data_sub
    peaks = np.where(local_max & (data_sub > 5 * np.std(data_sub)))

    if len(peaks[0]) == 0:
        raise ValueError("No bright sources detected for FWHM estimation.")

    fwhm_values = []
    box_half_size = 15  # pixels around each peak

    fitter = fitting.LevMarLSQFitter()

    for y, x in zip(peaks[0], peaks[1]):
        # Extract small cutout around the star
        y0, y1 = max(0, y - box_half_size), min(image_data.shape[0], y + box_half_size + 1)
        x0, x1 = max(0, x - box_half_size), min(image_data.shape[1], x + box_half_size + 1)
        cutout = data_sub[y0:y1, x0:x1]

        if cutout.size < 100:
            continue  # too close to edge

        yy, xx = np.mgrid[:cutout.shape[0], :cutout.shape[1]]
        y_center = y - y0
        x_center = x - x0

        # Initial guess
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
                fwhm_x = g_fit.x_stddev.value * 2.355  # sigma to FWHM
                fwhm_y = g_fit.y_stddev.value * 2.355
                fwhm_mean = np.mean([fwhm_x, fwhm_y])
                fwhm_values.append(fwhm_mean)
        except Exception:  # Catch fitting failures gracefully
            continue

        if len(fwhm_values) >= num_stars:
            break

    if len(fwhm_values) == 0:
        raise ValueError("Failed to fit Gaussian to any sources.")

    median_fwhm = np.median(fwhm_values)
    return median_fwhm


# Main execution
if __name__ == "__main__":
    try:
        # Load images
        images_data, image_files = load_matching_fits_images(
            image_directory_path, prefix, suffix
        )

        # Select first two images
        img1 = images_data[0]
        img2 = images_data[1]
        file1 = image_files[0]
        file2 = image_files[1]

        print(f"\nUsing images:")
        print(f"  Image 1: {os.path.basename(file1)}")
        print(f"  Image 2: {os.path.basename(file2)}")

        # Check compatibility
        if img1.shape != img2.shape:
            raise ValueError(f"Image shapes mismatch: {img1.shape} vs {img2.shape}")

        # Compute difference
        diff_image = img1 - img2

        # Save difference image
        output_path = os.path.join(cwd, "difference_image.fits")
        fits.PrimaryHDU(diff_image).writeto(output_path, overwrite=True)

        # Basic statistics
        print("\nDifference computation complete.")
        print(f"Mean:     {np.mean(diff_image):.4f}")
        print(f"Std dev:  {np.std(diff_image):.4f}")
        print(f"Min/Max:  {np.min(diff_image):.4f} / {np.max(diff_image):.4f}")
        print(f"\nDifference image saved to:\n  {output_path}")

        # Estimate FWHM on the first image
        print("\nEstimating FWHM on first image...")
        fwhm_pixels = estimate_fwhm(img1)
        print(f"Estimated median FWHM: {fwhm_pixels:.2f} pixels")

        # Display the difference image
        plt.figure(figsize=(10, 8))
        plt.imshow(diff_image, cmap='RdBu', origin='lower',
                   vmin=-np.std(diff_image) * 3, vmax=np.std(diff_image) * 3)
        plt.colorbar(label='Difference (counts)')
        plt.title(f'Difference Image\n{os.path.basename(file1)} − {os.path.basename(file2)}\n'
                  f'FWHM (first image) ≈ {fwhm_pixels:.2f} pixels')
        plt.xlabel('X pixel')
        plt.ylabel('Y pixel')
        plt.tight_layout()
        plt.show()

    except FileNotFoundError as e:
        print(f"\nFile error: {e}")
    except ValueError as e:
        print(f"\nProcessing error: {e}")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
