from astropy.io import fits
import os
import numpy as np
import matplotlib.pyplot as plt

# Define the directory path (peer to the current working directory)
dir_path = os.path.join('..', 'llm-1-alignment')

# File paths
diff_file = os.path.join(dir_path, 'aligned_template_last_difference.fits')
median_file = os.path.join(dir_path, 'median_template.fits')

# Read the FITS files
with fits.open(diff_file) as hdul_diff:
    diff_data = hdul_diff[0].data  # Assuming primary HDU contains the data
    diff_header = hdul_diff[0].header

with fits.open(median_file) as hdul_median:
    median_data = hdul_median[0].data  # Assuming primary HDU contains the data
    median_header = hdul_median[0].header

# Search the difference image for pixels exceeding 100
threshold = 100.0

# Find indices where pixel value > threshold
y_indices, x_indices = np.where(diff_data > threshold)

# Get the corresponding pixel values
values = diff_data[y_indices, x_indices]

print(f"Found {len(values)} pixels with value > {threshold}:\n")
print("     X       Y       Value")
print("-----------------------------")

for x, y, val in zip(x_indices, y_indices, values):
    print(f"{x:7d} {y:7d} {val:12.3f}")

if len(values) == 0:
    print("No pixels exceed the threshold.")

# Find the brightest pixel in the entire difference image
max_value = np.nanmax(diff_data)  # Use nanmax to ignore any NaNs
if np.isnan(max_value):
    print("Difference image contains only NaNs.")
else:
    # Get the coordinates of the brightest pixel (row, col) = (y, x)
    cy, cx = np.unravel_index(np.nanargmax(diff_data), diff_data.shape)

    print(f"\nBrightest pixel in difference image: value = {max_value:.3f} at (x={cx}, y={cy})")

    # Define patch size: 30x30 pixels
    half_size = 15

    # Compute patch boundaries, clamping to image edges
    y_start = max(0, cy - half_size)
    y_end = min(diff_data.shape[0], cy + half_size)
    x_start = max(0, cx - half_size)
    x_end = min(diff_data.shape[1], cx + half_size)

    # Extract patches
    diff_patch = diff_data[y_start:y_end, x_start:x_end]
    median_patch = median_data[y_start:y_end, x_start:x_end]

    # Create coordinate arrays centered on zero
    patch_height, patch_width = diff_patch.shape
    x_coords = np.arange(x_start, x_end) - cx
    y_coords = np.arange(y_start, y_end) - cy

    # Display the patches side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Median template patch
    im0 = axes[0].imshow(median_patch, cmap='gray', origin='lower', extent=[x_coords[0]-0.5, x_coords[-1]+0.5, y_coords[0]-0.5, y_coords[-1]+0.5])
    axes[0].set_title('Median Template Patch\n' f'(centered on x={cx}, y={cy})')
    axes[0].set_xlabel('ΔX (pixels)')
    axes[0].set_ylabel('ΔY (pixels)')
    axes[0].set_xlim(-15, 15)
    axes[0].set_ylim(-15, 15)
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # Difference patch
    im1 = axes[1].imshow(diff_patch, cmap='gray', origin='lower', extent=[x_coords[0]-0.5, x_coords[-1]+0.5, y_coords[0]-0.5, y_coords[-1]+0.5])
    axes[1].set_title(f'Difference Patch\n(centered on x={cx}, y={cy}, peak = {max_value:.1f})')
    axes[1].set_xlabel('ΔX (pixels)')
    axes[1].set_ylabel('ΔY (pixels)')
    axes[1].set_xlim(-15, 15)
    axes[1].set_ylim(-15, 15)
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # Mark the center pixel with a red cross
    axes[0].plot(0, 0, 'r+', markersize=12, markeredgewidth=2)
    axes[1].plot(0, 0, 'r+', markersize=12, markeredgewidth=2)

    plt.tight_layout()
    plt.show()
