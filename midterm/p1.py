#!/usr/bin/env python3

import cv2
import numpy as np
import matplotlib.pyplot as plt
path_to_image = "PA120272.JPG"

# Read image
original_ig = cv2.imread(path_to_image)

# # Changing image from BGR to LAB 
# l_ab_ig = cv2.cvtColor(original_ig, cv2.COLOR_BGR2LAB)

# # Get the l channel
# l, a, b = cv2.split(l_ab_ig)

# # On l channel apply histogram equalisation
# l_hist_eq = cv2.equalizeHist(l)

# # Merge modified L channel with A and B
# merged_l_ab_ig = cv2.merge((l_hist_eq, a, b))

# # Change LAB to BGR 
# final_ig = cv2.cvtColor(merged_l_ab_ig, cv2.COLOR_LAB2BGR)

# # Display images
# print ("Original Image")
# plt.imshow( original_ig)
# print()
# print("Equalized Image")
# plt.imshow(final_ig)





print (original_ig.shape)
# Perform gamma correction using a for loop
gamma_ig = original_ig.copy()
for width in range(480):
    for height in range(640):
        for channels in range(3):
            gamma_ig[width, height, channels] = 255*((original_ig[width, height, channels] / 255) ** (1 / 2.2))

# Convert the resulting array back to the unsigned integer 8-bit format (uint8)
#gamma_ig = gamma_ig.astype(np.uint8)

# Display the original and gamma-corrected images
cv2_imshow( original_ig)
cv2_imshow( gamma_ig)
