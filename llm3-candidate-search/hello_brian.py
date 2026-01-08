import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter

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

print("\nRescaled science image statistics:")
print(f"  Min: {np.min(rescaled_science_img):.1f}, Max: {np.max(rescaled_science_img):.1f}, Mean: {np.mean(rescaled_science_img):.1f}")

# Compute original difference: rescaled science - template
original_difference = rescaled_science_img - template_img

# Apply 5×5 boxcar smoothing to the difference image
print("\nApplying 5×5 boxcar smoothing to the difference image...")
smoothed_difference = uniform_filter(original_difference, size=5)

print("Smoothed difference statistics:")
print(f"  Min: {np.min(smoothed_difference):.1f}, Max: {np.max(smoothed_difference):.1f}, Mean: {np.mean(smoothed_difference):.1f}, Std: {np.std(smoothed_difference):.1f}")

# Absolute value version of the smoothed difference
abs_smoothed_difference = np.abs(smoothed_difference)
abs_median = np.median(abs_smoothed_difference)
print(f"\nAbsolute smoothed difference median: {abs_median:.1f} counts")

# Display central 2000×2000 regions in a 2×2 grid
half_large = 1000
ny, nx = template_img.shape
center_y_large, center_x_large = ny // 2, nx // 2

template_central_large = template_img[center_y_large - half_large:center_y_large + half_large, center_x_large - half_large:center_x_large + half_large]
science_central_large = rescaled_science_img[center_y_large - half_large:center_y_large + half_large, center_x_large - half_large:center_x_large + half_large]
smoothed_diff_central_large = smoothed_difference[center_y_large - half_large:center_y_large + half_large, center_x_large - half_large:center_x_large + half_large]
abs_smoothed_diff_central_large = abs_smoothed_difference[center_y_large - half_large:center_y_large + half_large, center_x_large - half_large:center_x_large + half_large]

# 2×2 grid layout for large central regions
fig_large, axs_large = plt.subplots(2, 2, figsize=(24, 24))

# Template (top-left)
im0 = axs_large[0, 0].imshow(template_central_large, cmap='gray', origin='lower',
                             vmin=np.percentile(template_central_large, 1), vmax=np.percentile(template_central_large, 99))
axs_large[0, 0].set_title('Central 2000×2000 of Median Template')
axs_large[0, 0].set_xlabel('X pixel')
axs_large[0, 0].set_ylabel('Y pixel')
fig_large.colorbar(im0, ax=axs_large[0, 0], shrink=0.8)

# Rescaled science (top-right)
im1 = axs_large[0, 1].imshow(science_central_large, cmap='gray', origin='lower',
                             vmin=np.percentile(science_central_large, 1), vmax=np.percentile(science_central_large, 99))
axs_large[0, 1].set_title('Central 2000×2000 of Rescaled Aligned Science Image')
axs_large[0, 1].set_xlabel('X pixel')
axs_large[0, 1].set_ylabel('Y pixel')
fig_large.colorbar(im1, ax=axs_large[0, 1], shrink=0.8)

# Smoothed difference (bottom-left)
vmax = np.std(smoothed_diff_central_large) * 3
im2 = axs_large[1, 0].imshow(smoothed_diff_central_large, cmap='RdBu', origin='lower',
                             vmin=-vmax, vmax=vmax)
axs_large[1, 0].set_title('Central 2000×2000 Smoothed Difference\n(5×5 Boxcar Average of Rescaled Science − Template)')
axs_large[1, 0].set_xlabel('X pixel')
axs_large[1, 0].set_ylabel('Y pixel')
fig_large.colorbar(im2, ax=axs_large[1, 0], label='Difference (counts)', shrink=0.8)

# Absolute smoothed difference (bottom-right)
vmax_abs = np.percentile(abs_smoothed_diff_central_large, 99.5)
im3 = axs_large[1, 1].imshow(abs_smoothed_diff_central_large, cmap='viridis', origin='lower',
                             vmin=0, vmax=vmax_abs)
axs_large[1, 1].set_title(f'Central 2000×2000 Absolute Smoothed Difference\nMedian: {abs_median:.1f} counts')
axs_large[1, 1].set_xlabel('X pixel')
axs_large[1, 1].set_ylabel('Y pixel')
fig_large.colorbar(im3, ax=axs_large[1, 1], label='Absolute Difference (counts)', shrink=0.8)

plt.tight_layout()
plt.show()

# NEW: 100×100 zooms of the two difference images centered on (x=1699, y=2154)
zoom_center_x = 1699
zoom_center_y = 2154
zoom_half = 50  # For 100×100 region

# Extract zoomed regions (handle edges if necessary)
smoothed_diff_zoom = smoothed_difference[zoom_center_y - zoom_half:zoom_center_y + zoom_half,
                                        zoom_center_x - zoom_half:zoom_center_x + zoom_half]
abs_smoothed_diff_zoom = abs_smoothed_difference[zoom_center_y - zoom_half:zoom_center_y + zoom_half,
                                                 zoom_center_x - zoom_half:zoom_center_x + zoom_half]

# 1×2 figure for zoomed differences
fig_zoom, axs_zoom = plt.subplots(1, 2, figsize=(24, 12))

# Smoothed difference zoom
vmax_zoom = np.std(smoothed_diff_zoom) * 3
im_zoom0 = axs_zoom[0].imshow(smoothed_diff_zoom, cmap='RdBu', origin='lower',
                              vmin=-vmax_zoom, vmax=vmax_zoom)
axs_zoom[0].set_title(f'100×100 Zoom of Smoothed Difference\nCentered at ({zoom_center_x}, {zoom_center_y})')
axs_zoom[0].set_xlabel('X pixel')
axs_zoom[0].set_ylabel('Y pixel')
fig_zoom.colorbar(im_zoom0, ax=axs_zoom[0], label='Difference (counts)', shrink=0.8)

# Absolute smoothed difference zoom
vmax_abs_zoom = np.percentile(abs_smoothed_diff_zoom, 99.5)
im_zoom1 = axs_zoom[1].imshow(abs_smoothed_diff_zoom, cmap='viridis', origin='lower',
                              vmin=0, vmax=vmax_abs_zoom)
axs_zoom[1].set_title(f'100×100 Zoom of Absolute Smoothed Difference\nCentered at ({zoom_center_x}, {zoom_center_y})\nMedian (whole image): {abs_median:.1f} counts')
axs_zoom[1].set_xlabel('X pixel')
axs_zoom[1].set_ylabel('Y pixel')
fig_zoom.colorbar(im_zoom1, ax=axs_zoom[1], label='Absolute Difference (counts)', shrink=0.8)

plt.tight_layout()
plt.show()

# Histogram of the absolute smoothed difference values (central region) with logarithmic y-axis
print("\nDisplaying histogram of the absolute smoothed difference values (central 2000×2000 region)...")
plt.figure(figsize=(12, 8))
hist_values, bins, _ = plt.hist(abs_smoothed_diff_central_large.flatten(), bins=200, color='skyblue', alpha=0.7, edgecolor='black')
plt.yscale('log')  # Logarithmic vertical axis
plt.axvline(abs_median, color='red', linestyle='--', linewidth=2, label=f'Median: {abs_median:.2f}')
plt.axvline(np.mean(abs_smoothed_diff_central_large), color='orange', linestyle=':', linewidth=2, label=f'Mean: {np.mean(abs_smoothed_diff_central_large):.2f}')
plt.title('Histogram of Absolute Smoothed Difference Values\n(Central 2000×2000, 5×5 Boxcar Smoothed, Logarithmic Y-Axis)')
plt.xlabel('Absolute Difference (counts)')
plt.ylabel('Number of pixels (log scale)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f"Absolute smoothed difference median (whole image): {abs_median:.1f} counts")
