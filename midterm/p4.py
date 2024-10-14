#!/usr/bin/env python3

import cv2
import numpy as np
import matplotlib.pyplot as plt
# write code for problem-3 here ...






    

# Get image
original_train_track_ig = cv2.imread('train_track.jpg')
train_track_ig = original_train_track_ig.copy()

# Create Gray image
gray_train_track_ig = cv2.cvtColor(train_track_ig, cv2.COLOR_BGR2GRAY)

# Gaussian blur
g_blur_train_track_ig = cv2.GaussianBlur(gray_train_track_ig, (19,19), 0)
# Canny edge detector
edg_train_track_ig = cv2.Canny(g_blur_train_track_ig, 49, 149)
# get lines using hough lines
val = np.pi/180
hough_lns_train_track_ig = cv2.HoughLinesP(edg_train_track_ig, 1, val, 79, 99, 19)


hough_lns_all = []
max = np.inf
for ln in hough_lns_train_track_ig:
    for x_a, y_a, x_b, y_b in ln:
        slp = (y_b - y_a) / (x_b - x_a) if (x_b - x_a) != 0 else max
        hough_lns_all.append((x_a, y_a, x_b, y_b, slp))

hough_lns_all.sort(key=lambda line: line[4])

# Add line to image that has highest slope
Mx_a, My_a, Mx_b, My_b, M_slp = hough_lns_all[-1]

# Extend line with gray pixel values (195, 195, 195)
px_distance = 390
Mx_a_extended = int(Mx_a - px_distance)
My_a_extended = int(My_a - M_slp * px_distance)
Mx_b_extended = int(Mx_b + px_distance)
My_b_extended = int(My_b + M_slp * px_distance)

cv2.line(train_track_ig, (Mx_a_extended, My_a_extended), (Mx_b_extended, My_b_extended), (195,195,195), 10)

# Add line to image that has lowest slope
mx_a, my_a, mx_b, my_b, m_slp = hough_lns_all[0]

# Extend line with gray pixel values (195, 195, 195)
mx_a_extended = int(mx_a - px_distance)
my_a_extended = int(my_a - m_slp *px_distance)
mx_b_extended = int(mx_b + px_distance)
my_b_extended = int(my_b + m_slp * px_distance)

cv2.line(train_track_ig, (mx_a_extended, my_a_extended), (mx_b_extended, my_b_extended), (195,195,195), 10)

# Generate Start and End points for perspective transform
val1 = 1480
val2 = 1920
corner_a=((val1 - my_b)/ m_slp +mx_a )
corner_b = ((val1 - My_b)/ M_slp +Mx_a )
corner_c = (( val2- my_b)/ m_slp +mx_a )
corner_d = (( val2- My_b)/ M_slp +Mx_a )

perspective_start = np.float32([[int(corner_a), val1], [int(corner_b), val1], [int(corner_c), val2], [int(corner_d), val2] ])
perspective_end = np.float32([[int(corner_c), val1], [int(corner_b), val1], [int(corner_c), val2], [int(corner_b), val2] ])

# Perspective transform
homography_p = cv2.getPerspectiveTransform(perspective_start, perspective_end)
width = train_track_ig.shape[1]
height = train_track_ig.shape[0]
view_from_top = cv2.warpPerspective(train_track_ig,homography_p, (width,height))

identified_px = []

r_start = 1890
r_end = 1300
step = -1

# Check for grey pixels in the image 

# In left half of image
for height in range(r_start, r_end, step): 
    s_start = 1499
    s_end = -1
    for width in range(s_start, s_end, step): 
        # See if pixel is grey 
        if all(view_from_top[height, width] == [195,195,195]): 
            identified_px.append(width)
            break
    if identified_px: 
        break

# In right half of image
for height in range(r_start, r_end, step):
    s_start = 1600
    s_end = view_from_top.shape[1]
    for width in range(s_start, s_end):
    # See if pixel is grey 
        if all(view_from_top[height, width] == [195,195,195]): 
            identified_px.append(width)
            break
    if len(identified_px) == 2: 
        break

mean_d = None
x_a, x_b = identified_px
mean_d = abs(x_b - x_a)


plt.figure()
plt.title("Original image")
plt.imshow(cv2.cvtColor(original_train_track_ig, cv2.COLOR_BGR2RGB))

plt.figure()
plt.title("Perspective transformed image")
plt.imshow(cv2.cvtColor(view_from_top, cv2.COLOR_BGR2RGB)) 

print("Mean distance between the train tracks for every row: " +  str(mean_d) + " px")



cv2.waitKey(0)
cv2.destroyAllWindows()