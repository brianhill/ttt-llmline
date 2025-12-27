import os
import glob
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt

# Your original configuration (kept exactly as provided)
cwd = os.getcwd()
image_directory_path = "/Volumes/Astronomy Data/2025 TTT Targets/ttt-00167/bin3x3"
prefix = "TTT1_QHY411-1_2025-04-26-23-0"
suffix = ".fits"


# Renamed function to avoid any potential shadowing/conflict
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


# Main execution
if __name__ == "__main__":
    try:
        # Load images using the renamed function
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

        # Compute difference (image1 - image2)
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

        # Display the difference image using matplotlib
        plt.figure(figsize=(10, 8))
        # Use a diverging colormap to highlight positive/negative differences
        plt.imshow(diff_image, cmap='RdBu', origin='lower', vmin=-np.std(diff_image) * 3, vmax=np.std(diff_image) * 3)
        plt.colorbar(label='Difference (counts)')
        plt.title(f'Difference Image\n{os.path.basename(file1)} − {os.path.basename(file2)}')
        plt.xlabel('X pixel')
        plt.ylabel('Y pixel')
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"\nError: {str(e)}")
