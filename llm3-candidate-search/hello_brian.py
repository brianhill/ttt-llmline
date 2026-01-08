import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
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

# Compute difference: rescaled science - template
difference = rescaled_science_img - template_img

print("\nDifference (rescaled science − template) statistics:")
print(f"  Min: {np.min(difference):.1f}, Max: {np.max(difference):.1f}, Mean: {np.mean(difference):.1f}, Std: {np.std(difference):.1f}")

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

# Initial candidates: pixels exceeding threshold
print("\nFinding initial candidate pixels exceeding threshold...")
y_indices, x_indices = np.where(abs_smoothed_sum > threshold)
candidates = list(zip(x_indices, y_indices))
values = abs_smoothed_sum[y_indices, x_indices].tolist()

print(f"Found {len(candidates)} initial candidate pixels")

# Print first 20 initial candidates
if len(candidates) > 0:
    print("First 20 initial candidates (x, y) with values:")
    for i in range(min(20, len(candidates))):
        x, y = candidates[i]
        val = values[i]
        print(f"  {i+1:2d}: ({x}, {y}) value = {val:.1f}")
else:
    print("No initial candidates found.")

# Edge culling: eliminate candidates within 50 pixels of image edges
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

# Adjacency culling: keep only the brightest within 5 pixels
print("\nAdjacency culling: keeping brightest within 5 pixels...")
culled_candidates = []

# Convert values to NumPy array for safe sorting
values_array = np.array(values)
sorted_indices = np.argsort(-values_array)
sorted_candidates = [candidates[i] for i in sorted_indices]

for i, (x, y) in enumerate(sorted_candidates):
    too_close = False
    for cx, cy in culled_candidates:
        dist = np.sqrt((x - cx)**2 + (y - cy)**2)
        if dist <= 5.0:
            too_close = True
            break
    if not too_close:
        culled_candidates.append((x, y))

candidates = culled_candidates

print(f"After adjacency culling: {len(candidates)} unique bright candidates remain")

# 3×3 boxcar sum-smoothed template
print("\nCreating 3×3 boxcar sum-smoothed template image...")
smoothed_template = uniform_filter(template_img, size=3) * 9

# Secondary culling using Poisson noise estimate
poisson_multiplier = 10
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
if len(candidates) > 0:
    print("First 20 final candidates (x, y) with values:")
    for i in range(min(20, len(candidates))):
        x, y = candidates[i]
        val = abs_smoothed_sum[y, x]
        print(f"  {i+1:2d}: ({x}, {y}) value = {val:.1f}")
else:
    print("No candidates remain after secondary culling.")

# Display FULL rescaled science image with red circles (diameter 20 px) around final candidates
print("\nDisplaying FULL rescaled science image with red circles (diameter 20 px) on all final candidates...")
fig_science_full, ax_science_full = plt.subplots(1, 1, figsize=(20, 20))
im_science_full = ax_science_full.imshow(rescaled_science_img, cmap='gray', origin='lower',
                                         vmin=np.percentile(rescaled_science_img, 1),
                                         vmax=np.percentile(rescaled_science_img, 99))

for x, y in candidates:
    circ = Circle((x, y), radius=10, fill=False, color='red', linewidth=2)
    ax_science_full.add_patch(circ)

ax_science_full.set_title(f'Full Rescaled Aligned Science Image\nRed circles (diameter 20 px) on all {len(candidates)} final candidates')
ax_science_full.set_xlabel('X pixel')
ax_science_full.set_ylabel('Y pixel')
fig_science_full.colorbar(im_science_full, ax=ax_science_full, shrink=0.8, label='Counts')
plt.tight_layout()
plt.show()

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
fig_large, axs_large = plt.subplots(2, 2, figsize=(24, 24))

# Template (top-left)
im0 = axs_large[0, 0].imshow(template_central_large, cmap='gray', origin='lower',
                             vmin=np.percentile(template_central_large, 1),
                             vmax=np.percentile(template_central_large, 99))
axs_large[0, 0].set_title('Central 2000×2000 of Median Template')
axs_large[0, 0].set_xlabel('X pixel')
axs_large[0, 0].set_ylabel('Y pixel')
fig_large.colorbar(im0, ax=axs_large[0, 0], shrink=0.8)

# Rescaled science (top-right)
im1 = axs_large[0, 1].imshow(science_central_large, cmap='gray', origin='lower',
                             vmin=np.percentile(science_central_large, 1),
                             vmax=np.percentile(science_central_large, 99))
axs_large[0, 1].set_title('Central 2000×2000 of Rescaled Aligned Science Image')
axs_large[0, 1].set_xlabel('X pixel')
axs_large[0, 1].set_ylabel('Y pixel')
fig_large.colorbar(im1, ax=axs_large[0, 1], shrink=0.8)

# Difference (bottom-left)
vmax = np.std(diff_central_large) * 3
im2 = axs_large[1, 0].imshow(diff_central_large, cmap='RdBu', origin='lower',
                             vmin=-vmax, vmax=vmax)
axs_large[1, 0].set_title('Central 2000×2000 Difference\n(Rescaled Science − Template)')
axs_large[1, 0].set_xlabel('X pixel')
axs_large[1, 0].set_ylabel('Y pixel')
fig_large.colorbar(im2, ax=axs_large[1, 0], label='Difference (counts)', shrink=0.8)

# Absolute difference (bottom-right)
vmax_abs = np.percentile(abs_diff_central_large, 99.5)
im3 = axs_large[1, 1].imshow(abs_diff_central_large, cmap='viridis', origin='lower',
                             vmin=0, vmax=vmax_abs)
axs_large[1, 1].set_title(f'Central 2000×2000 Absolute Difference\nMedian: {abs_median:.1f} counts')
axs_large[1, 1].set_xlabel('X pixel')
axs_large[1, 1].set_ylabel('Y pixel')
fig_large.colorbar(im3, ax=axs_large[1, 1], label='Absolute Difference (counts)', shrink=0.8)

plt.tight_layout()
plt.show()

# 100×100 zooms of the two difference images centered on (x=1699, y=2154)
zoom_center_x = 1699
zoom_center_y = 2154
zoom_half = 50  # For 100×100 region

# Extract zoomed regions
diff_zoom = difference[zoom_center_y - zoom_half:zoom_center_y + zoom_half,
                       zoom_center_x - zoom_half:zoom_center_x + zoom_half]
abs_diff_zoom = abs_difference[zoom_center_y - zoom_half:zoom_center_y + zoom_half,
                               zoom_center_x - zoom_half:zoom_center_x + zoom_half]

# 1×2 figure for zoomed differences
fig_zoom, axs_zoom = plt.subplots(1, 2, figsize=(24, 12))

# Difference zoom
vmax_zoom = np.std(diff_zoom) * 3
im_zoom0 = axs_zoom[0].imshow(diff_zoom, cmap='RdBu', origin='lower',
                              vmin=-vmax_zoom, vmax=vmax_zoom)
axs_zoom[0].set_title(f'100×100 Zoom of Difference\nCentered at ({zoom_center_x}, {zoom_center_y})')
axs_zoom[0].set_xlabel('X pixel')
axs_zoom[0].set_ylabel('Y pixel')
fig_zoom.colorbar(im_zoom0, ax=axs_zoom[0], label='Difference (counts)', shrink=0.8)

# Absolute difference zoom
vmax_abs_zoom = np.percentile(abs_diff_zoom, 99.5)
im_zoom1 = axs_zoom[1].imshow(abs_diff_zoom, cmap='viridis', origin='lower',
                              vmin=0, vmax=vmax_abs_zoom)
axs_zoom[1].set_title(f'100×100 Zoom of Absolute Difference\nCentered at ({zoom_center_x}, {zoom_center_y})\nMedian (whole image): {abs_median:.1f} counts')
axs_zoom[1].set_xlabel('X pixel')
axs_zoom[1].set_ylabel('Y pixel')
fig_zoom.colorbar(im_zoom1, ax=axs_zoom[1], label='Absolute Difference (counts)', shrink=0.8)

plt.tight_layout()
plt.show()

# Histogram of the absolute difference values (central region) with logarithmic y-axis
print("\nDisplaying histogram of the absolute difference values (central 2000×2000 region)...")
plt.figure(figsize=(12, 8))
hist_values, bins, _ = plt.hist(abs_diff_central_large.flatten(), bins=200, color='skyblue', alpha=0.7, edgecolor='black')
plt.yscale('log')  # Logarithmic vertical axis
plt.axvline(abs_median, color='red', linestyle='--', linewidth=2, label=f'Median: {abs_median:.2f}')
plt.axvline(np.mean(abs_diff_central_large), color='orange', linestyle=':', linewidth=2, label=f'Mean: {np.mean(abs_diff_central_large):.2f}')
plt.title('Histogram of Absolute Difference Values\n(Central 2000×2000, Logarithmic Y-Axis)')
plt.xlabel('Absolute Difference (counts)')
plt.ylabel('Number of pixels (log scale)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f"Absolute difference median (whole image): {abs_median:.1f} counts")
