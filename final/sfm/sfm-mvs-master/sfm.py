# sfm - code
# Abhishek Avhad, Piyush Goenka, Ishan Kharat


from tqdm import tqdm
import open3d
import numpy
from scipy.optimize import least_squares
import matplotlib.pyplot
import copy
import os
import cv2

# camera parameters- intrinsic
v1 = [2393.9522001, -3.41060512997e-13, 932.38217699987]
v2 = [0, 2398.118540301132, 628.26499499879]
v3 = [0, 0, 1]
mat_K = numpy.array([v1, v2, v3])

# downsample images to reduce computation load
compute_scale = 2
c = compute_scale*1.0
mat_K[0, 0] /= c
mat_K[1, 1] /= c
mat_K[0, 2] /= c
mat_K[1, 2] /= c

# directory path for loading and saving images
directory_cwd = os.getcwd()

# path of images folder
path_of_image = '/home/robotics/courses/perception/final/sfm/sfm-mvs-master/LUsphinx'

# bundle adjustment flag
adj_bundle = False

# function to downscale images 
def down_scaling_image(input_image, factor_down_scaling):
    factor_down_scaling = factor_down_scaling/2
    factor_down_scaling = int (factor_down_scaling)
    iteration = 2-1
    while not (iteration > factor_down_scaling):
        input_image = cv2.pyrDown(input_image)
        iteration += 1
    return input_image
	
# triangulation function - input are images (2) and reprojection matrices
def perform_tri_angltion(mat_P_a, mat_P_b, set_points_a, set_points_b, matrice_K, flag_do_again):
    if flag_do_again:
        pt_a = set_points_a
        pt_b = set_points_b
    else:
        pt_a = numpy.transpose(set_points_a)
        pt_b = numpy.transpose(set_points_b)

    generated_pt_cld = cv2.triangulatePoints(mat_P_a, mat_P_b, pt_a, pt_b)
    generated_pt_cld = generated_pt_cld / generated_pt_cld[3]

    return pt_a, pt_b, generated_pt_cld


# Function for Perspective-n-Point
def perspective_n_point(mat_X, vec_p, mat_K, val_d, vec_p0, starting):
    
    if starting == 1:
        mat_X = mat_X[:, 0, :]
        vec_p = vec_p.T
        vec_p0 = vec_p0.T

    flag = cv2.SOLVEPNP_ITERATIVE
    _, rot_vectors, trans_vec, list_of_in_lirs = cv2.solvePnPRansac(mat_X, vec_p, mat_K, val_d, flag )
    
    mat_Re, not_used = cv2.Rodrigues(rot_vectors)

    if list_of_in_lirs is not None:
        vec_p0 = vec_p0[list_of_in_lirs[:,0]]
        mat_X = mat_X[list_of_in_lirs[:,0]]
        vec_p = vec_p[list_of_in_lirs[:,0]]

    return mat_Re, trans_vec, vec_p, mat_X, vec_p0

# Re-projection error computation
def calculate_error_re_prjection(homogeneous_trans, point_vals, rot_plus_trans, mat_K, flag_homo):
    
    complete_err = 0
    dimension = 3
    trans_vec = rot_plus_trans[:dimension, dimension]
    rot_vec = rot_plus_trans[:dimension, :dimension]

    rodrigues_val, not_used = cv2.Rodrigues(rot_vec)
    if flag_homo == 1:
        homogeneous_trans = cv2.convertPointsFromHomogeneous(homogeneous_trans.T)

    coeff_val = None
    projection_points, not_used = cv2.projectPoints(homogeneous_trans, rodrigues_val, trans_vec, mat_K, distCoeffs=coeff_val)
    projection_points = projection_points[:, 0, :]
    projection_points = numpy.float32(projection_points)
    point_vals = numpy.float32(point_vals)

    if not flag_homo == 1:
        norm_flag = cv2.NORM_L2
        complete_err = cv2.norm(projection_points, point_vals, norm_flag)
    else:
        norm_flag = cv2.NORM_L2
        complete_err = cv2.norm(projection_points, point_vals.T, norm_flag)

    point_vals = point_vals.T
    tot_error = complete_err / len(projection_points)

    return tot_error, homogeneous_trans, projection_points


# Reprojection error calculation with bundle adjustment
def optimezed_re_prjection_err(input_val):
	
    rot_plus_trans = input_val[0:12].reshape((3,4))
    mat_K = input_val[12:21].reshape((3,3))
    other_part = len(input_val[21:])
    scl = 0.4
    other_part = int(other_part * scl)
    lngth = len(input_val[21 + other_part:])
    mat_X = input_val[21 + other_part:].reshape((int( lngth/3), 3))
    mat_R = rot_plus_trans[:3, :3]
    vec_p = input_val[21:21 + other_part].reshape((2, int(other_part/2)))
    vec_t = rot_plus_trans[:3, 3]

    vec_p = vec_p.T
    number_of_points = len(vec_p)
    rodrigues_val, not_used = cv2.Rodrigues(mat_R)
    gotten_err = []

    flag = None
    points_projection, not_used = cv2.projectPoints(mat_X, rodrigues_val, vec_t, mat_K, distCoeffs = flag)
    points_projection = points_projection[:, 0, :]

    for iteration in range(number_of_points):
        point_in_image = vec_p[iteration]
        point_re_prjected = points_projection[iteration]
        error_val = (point_in_image - point_re_prjected)**2
        gotten_err.append(error_val)

    array_of_error = numpy.array(gotten_err).ravel()/number_of_points

    sum_err = numpy.sum(array_of_error)
    print(sum_err)

    return array_of_error


def adjustment_bundling(three_d_pts, t_2, latest_rot_plus_trans, mat_K, rodri_er):

    # variables for optimization
    optimal_v = numpy.hstack((latest_rot_plus_trans.ravel(), mat_K.ravel()))
    optimal_v = numpy.hstack((optimal_v, t_2.ravel()))
    optimal_v = numpy.hstack((optimal_v, three_d_pts.ravel()))

    er = numpy.sum(optimezed_re_prjection_err(optimal_v))
    re_evaluated_vals = least_squares(fun = optimezed_re_prjection_err, x0 = optimal_v, gtol = rodri_er)

    re_evaluated_vals = re_evaluated_vals.x
    rot_plus_trans = re_evaluated_vals[0:12].reshape((3,4))
    mat_K = re_evaluated_vals[12:21].reshape((3,3))
    other_vals = len(re_evaluated_vals[21:])
    factor = 0.4
    other_vals = int(other_vals * factor)
    vec_p = re_evaluated_vals[21:21 + other_vals].reshape((2, int(other_vals/2)))
    length = int(len(re_evaluated_vals[21 + other_vals:])/3)
    mat_X = re_evaluated_vals[21 + other_vals:].reshape((length, 3))
    vec_p = vec_p.T

    return mat_X, vec_p, rot_plus_trans
	
# draw points
def points_display(ip_img, ip_points, is_reprojection):
    if is_reprojection == False:
        rang = (0, 255, 0)
        ip_img = cv2.drawKeypoints(ip_img, ip_points, ip_img, color=rang, flags=0)
    else:
        for pt in ip_points:
            rang = (0, 0, 255)
            ip_img = cv2.circle(ip_img, tuple(pt), 2, rang, -1)
    return ip_img

# convert to ply
def convert_to_pointcloud_ply(directory_path, p_cld, colour_scheme, flag_to_dense):

    ftr = 200
    processed_pts = p_cld.reshape(-1, 3) * ftr
    processed_clrs = colour_scheme.reshape(-1, 3)
    print(processed_clrs.shape, processed_pts.shape)
    vrtices = numpy.hstack([processed_pts, processed_clrs])

    # point cloud - clean data
    get_average = numpy.mean(vrtices[:, :3], axis=0)
    tp = vrtices[:, :3] - get_average
    dstnce = numpy.sqrt(tp[:, 0] ** 2 + tp[:, 1] ** 2 + tp[:, 2] ** 2)
    add_val = 300
    ix = numpy.where(dstnce < numpy.mean(dstnce) + add_val)
    vrtices = vrtices[ix]

    header_data_pointcloud = '''ply
		format ascii 1.0
		element vertex %(vert_num)d
		property float x
		property float y
		property float z
		property uchar blue
		property uchar green
		property uchar red
		end_header
		'''
    
    if not flag_to_dense:
        with open(directory_path + '/Point_Cloud/sparse.ply', 'w') as opened_file:
            opened_file.write(header_data_pointcloud % dict(vert_num=len(vrtices)))
            numpy.savetxt(opened_file, vrtices, '%f %f %f %d %d %d')
    else:
        with open(directory_path + '/Point_Cloud/dense.ply', 'w') as opened_file:
            opened_file.write(header_data_pointcloud % dict(vert_num=len(vrtices)))
            numpy.savetxt(opened_file, vrtices, '%f %f %f %d %d %d')
            
# Registration of camera pose
def get_ori_cam(file_pth, input_msh, rot_plus_trans, iter):
    homo_transform = numpy.zeros((4, 4))
    homo_transform[:3, ] = rot_plus_trans
    vctr = [0, 0, 0, 1]
    homo_transform[3, :] = numpy.array(vctr)
    updated_msh = copy.deepcopy(input_msh).transform(homo_transform)
    open3d.io.write_triangle_mesh(file_pth + "/Point_Cloud/camerapose" + str(iter) + '.ply', updated_msh)
    return

# get the common points 
def get_pts_cmmn(p_a, p_b, p_c):

    i_a = []
    i_b = []
    for iterator in range(p_a.shape[0]):
        got_pt = numpy.where(p_b == p_a[iterator, :])
        if got_pt[0].size == 0:
            pass
        else:
            i_a.append(iterator)
            i_b.append(got_pt[0][0])

    # non-common array
    t_a1 = numpy.ma.array(p_b, mask=False)
    t_a1.mask[i_b] = True
    t_a1 = t_a1.compressed()
    vl = 2
    t_a1 = t_a1.reshape(int(t_a1.shape[0] / vl), vl)

    t_a2 = numpy.ma.array(p_c, mask=False)
    t_a2.mask[i_b] = True
    t_a2 = t_a2.compressed()
    t_a2 = t_a2.reshape(int(t_a2.shape[0] / vl), vl)
    print("Shape of new array ", t_a1.shape, t_a2.shape)
    return numpy.array(i_a), numpy.array(i_b), t_a1, t_a2

# feature detection and matching
def detect_feturs(input_image_a, input_image_b):
    cv_flag = cv2.COLOR_BGR2GRAY
    i_a_gr = cv2.cvtColor(input_image_a, cv_flag)
    i_b_gr = cv2.cvtColor(input_image_b, cv_flag)

    object_s_i_f_t = cv2.xfeatures2d.SIFT_create()
    keypoint_a, description_a = object_s_i_f_t.detectAndCompute(i_a_gr, None)    
    
    keypoint_b, description_b = object_s_i_f_t.detectAndCompute(i_b_gr, None)

    brute_force_matcher = cv2.BFMatcher()
    keypoint_matching_result = brute_force_matcher.knnMatch(description_a, description_b, k=2)

    selected_points = []
    for a, b in keypoint_matching_result:
        if a.distance < 0.7 * b.distance:
            selected_points.append(a)

    points_a = numpy.float32([keypoint_a[le.queryIdx].pt for le in selected_points])
    points_b = numpy.float32([keypoint_b[le.trainIdx].pt for le in selected_points])

    return points_a, points_b


window_normal = cv2.WINDOW_NORMAL
cv2.namedWindow('sfm', window_normal)

array_of_poses = mat_K.ravel()

row1=[1, 0, 0, 0]
row2=[0, 1, 0, 0]
row3=[0, 0, 1, 0]
homogeneous_mtx = numpy.array([row1,row2,row3])
homogeneous_mtx_1 = numpy.empty((3, 4))

mtx_P_1 = numpy.matmul(mat_K, homogeneous_mtx)
reference_P_mtx = mtx_P_1
mtx_P_2 = numpy.empty((3, 4))

total_mtx_X = numpy.zeros((1, 3))
total_colrs = numpy.zeros((1, 3))

directory_list = os.listdir(path_of_image)
list_of_images = sorted(directory_list)
array_igs = []
for ip_image in list_of_images:
    if '.jpg' in ip_image.lower() or '.png' in ip_image.lower():
        array_igs = array_igs + [ip_image]
created_mesh = open3d.geometry.TriangleMesh.create_coordinate_frame()
iteration_variable = 0

flag_to_make_denser = False  

image_a = down_scaling_image(cv2.imread(path_of_image + '/' + array_igs[iteration_variable]), compute_scale)
image_b = down_scaling_image(cv2.imread(path_of_image + '/' + array_igs[iteration_variable + 1]), compute_scale)

points_a, points_b = detect_feturs(image_a, image_b)

# Computing essential mtx
mtx_Essential, amount_of_masking = cv2.findEssentialMat(points_a, points_b, mat_K, method=cv2.RANSAC, prob=0.999, threshold=0.4, mask=None)
points_a = points_a[amount_of_masking.ravel() == 1]
points_b = points_b[amount_of_masking.ravel() == 1]

# pose of second img w.r.t first img
_, rot_vec, trans_vec, amount_of_masking = cv2.recoverPose(mtx_Essential, points_a, points_b, mat_K)  # computing pose
points_a = points_a[amount_of_masking.ravel() > 0]
points_b = points_b[amount_of_masking.ravel() > 0]
homogeneous_mtx_1[:3, :3] = numpy.matmul(rot_vec, homogeneous_mtx[:3, :3])
homogeneous_mtx_1[:3, 3] = homogeneous_mtx[:3, 3] + numpy.matmul(homogeneous_mtx[:3, :3], trans_vec.ravel())

mtx_P_2 = numpy.matmul(mat_K, homogeneous_mtx_1)

# Triangulation performed for first pair of images. These poses are taken as reference, used later for incremental sfm
points_a, points_b, three_d_pts = perform_tri_angltion(mtx_P_1, mtx_P_2, points_a, points_b, mat_K, flag_do_again=False)

# Backtracking  3-D points on image then computing re-projection error 
repro_error, three_d_pts, repro_pts = calculate_error_re_prjection(three_d_pts, points_b, homogeneous_mtx_1, mat_K, flag_homo = 1)
print("||||| Reprojection Error: ", repro_error)
rotation_mtx_pnp, translation_mtx_pnp, points_b, three_d_pts, points_0_trans = perspective_n_point(three_d_pts, points_b, mat_K, numpy.zeros((5, 1), dtype=numpy.float32), points_a, starting=1)

rot_vec = numpy.eye(3)
my_vec = [[0], [0], [0]]
trans_vec = numpy.array(my_vec, dtype=numpy.float32)

# amount of images taken
total_images = len(array_igs) - 2 

array_of_poses = numpy.hstack((array_of_poses, mtx_P_1.ravel()))
array_of_poses = numpy.hstack((array_of_poses, mtx_P_2.ravel()))

threshold_total_g = 0.5

for iteration_variable in tqdm(range(total_images)):
    # get new image to the pipeline and find matches with pair of images
    scaled_down_ig = down_scaling_image(cv2.imread(path_of_image + '/' + array_igs[iteration_variable + 2]), compute_scale)


    points_got_1, points_got_2 = detect_feturs(image_b, scaled_down_ig)
    if iteration_variable != 0:
        points_a, points_b, three_d_pts = perform_tri_angltion(mtx_P_1, mtx_P_2, points_a, points_b, mat_K, flag_do_again = False)
        points_b = points_b.T
        three_d_pts = cv2.convertPointsFromHomogeneous(three_d_pts.T)
        three_d_pts = three_d_pts[:, 0, :]
    

    # find points 1 and points 2 index match
    index_one, index_two, one_tp, two_tp = get_pts_cmmn(points_b, points_got_1, points_got_2)
    common_p2 = points_got_2[index_two]
    common_p = points_got_1[index_two]
    common_p0 = points_a.T[index_one]
        
    # PnP 
    rotation_mtx_pnp, translation_mtx_pnp, common_p2, three_d_pts, common_p = perspective_n_point(three_d_pts[index_one], common_p2, mat_K, numpy.zeros((5, 1), dtype=numpy.float32), common_p, starting = 0)
    
    # get projection mtx of new img
    reprojection_homo_transform_mtx = numpy.hstack((rotation_mtx_pnp, translation_mtx_pnp))
    updated_P_mtx = numpy.matmul(mat_K, reprojection_homo_transform_mtx)

    repro_error, three_d_pts, _ = calculate_error_re_prjection(three_d_pts, common_p2, reprojection_homo_transform_mtx, mat_K, flag_homo = 0)
   
    
    one_tp, two_tp, three_d_pts = perform_tri_angltion(mtx_P_2, updated_P_mtx, one_tp, two_tp, mat_K, flag_do_again = False)
    repro_error, three_d_pts, _ = calculate_error_re_prjection(three_d_pts, two_tp, reprojection_homo_transform_mtx, mat_K, flag_homo = 1)
    print("Re-projection error: ", repro_error)
    # store each img pose
    array_of_poses = numpy.hstack((array_of_poses, updated_P_mtx.ravel()))


    if adj_bundle:
        print("Bundle adjustment...")
        three_d_pts, two_tp, reprojection_homo_transform_mtx = adjustment_bundling(three_d_pts, two_tp, reprojection_homo_transform_mtx, mat_K, threshold_total_g)
        updated_P_mtx = numpy.matmul(mat_K, reprojection_homo_transform_mtx)
        repro_error, three_d_pts, _ = calculate_error_re_prjection(three_d_pts, two_tp, reprojection_homo_transform_mtx, mat_K, flag_homo = 0)
        print("Mini-mized Error: ",repro_error)
        total_mtx_X = numpy.vstack((total_mtx_X, three_d_pts))
        registered_points_1 = numpy.array(two_tp, dtype=numpy.int32)
        gotten_colrs = numpy.array([scaled_down_ig[ppts[1], ppts[0]] for ppts in registered_points_1])
        total_colrs = numpy.vstack((total_colrs, gotten_colrs))
    else:
        total_mtx_X = numpy.vstack((total_mtx_X, three_d_pts[:, 0, :]))
        registered_points_1 = numpy.array(two_tp, dtype=numpy.int32)
        gotten_colrs = numpy.array([scaled_down_ig[ppts[1], ppts[0]] for ppts in registered_points_1.T])
        total_colrs = numpy.vstack((total_colrs, gotten_colrs)) 
 


    homogeneous_mtx = numpy.copy(homogeneous_mtx_1)
    mtx_P_1 = numpy.copy(mtx_P_2)
    matplotlib.pyplot.scatter(iteration_variable, repro_error)
    matplotlib.pyplot.pause(0.05)

    image_a = numpy.copy(image_b)
    image_b = numpy.copy(scaled_down_ig)
    points_a = numpy.copy(points_got_1)
    points_b = numpy.copy(points_got_2)
    #P1 = numpy.copy(P2)
    mtx_P_2 = numpy.copy(updated_P_mtx)
    cv2.imshow('image', scaled_down_ig)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

matplotlib.pyplot.show()
cv2.destroyAllWindows()

# pointcloud is registered. Is is saved via open3d in .ply form
print("Computing Point Cloud data...")
print(total_mtx_X.shape, total_colrs.shape)
convert_to_pointcloud_ply(directory_cwd, total_mtx_X, total_colrs, flag_to_make_denser)
print("Operation Complete!")
# save all images pro-jection matrices
numpy.savetxt('pose.csv', array_of_poses, delimiter = '\n')

