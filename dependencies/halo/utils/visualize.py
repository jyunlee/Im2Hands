import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
# from torchvision.utils import save_image
# import im2mesh.common as common


# def visualize_data(data, data_type, out_file):
#     r''' Visualizes the data with regard to its type.

#     Args:
#         data (tensor): batch of data
#         data_type (string): data type (img, voxels or pointcloud)
#         out_file (string): output file
#     '''
#     if data_type == 'trans_matrix':
#         visualize_transmatrix(data, out_file=out_file)

#     elif data_type == 'img':
#         if data.dim() == 3:
#             data = data.unsqueeze(0)
#         save_image(data, out_file, nrow=4)
#     elif data_type == 'voxels':
#         visualize_voxels(data, out_file=out_file)
#     elif data_type == 'pointcloud':
#         visualize_pointcloud(data, out_file=out_file)
#     elif data_type is None or data_type == 'idx':
#         pass
#     else:
#         raise ValueError('Invalid data_type "%s"' % data_type)


def set_ax_limits(ax, lim=100.0): # , lim=10.0): # 0.1
    ax.set_xlim3d(-lim, lim)
    ax.set_ylim3d(-lim, lim)
    ax.set_zlim3d(-lim, lim)


def plot_skeleton_single_view(joints, joint_order='biomech', object_points=None,
                              color='r', ax=None, show=True):
    if ax is None:
        print('new fig')
        print(joints)
        fig = plt.figure()
        ax = fig.gca(projection=Axes3D.name)
    # set_ax_limits(ax)

    # Skeleton definition
    mano_joint_parent = np.array([0, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19])
    biomech_joint_parent = np.array([0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
    halo_joint_parent = np.array([0, 0, 1, 2, 0, 4, 5, 0, 7, 8, 0, 10, 11, 0, 13, 14, 15, 3, 6, 12, 9])
    if joint_order == 'mano':
        joint_parent = mano_joint_parent
    elif joint_order == 'biomech':
        joint_parent = biomech_joint_parent
    elif joint_order == 'halo':
        joint_parent = halo_joint_parent

    if object_points is not None:
        ax.scatter(object_points[:, 0], object_points[:, 1], object_points[:, 2], alpha=0.1, c='r')
    ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], alpha=0.8, c='b')

    b_start_loc = joints[joint_parent]  # joints[biomech_joint_parent]
    b_end_loc = joints
    for b in range(21):
        if b == 1:
            cur_color = 'g'
        else:
            cur_color = color
        ax.plot([b_start_loc[b, 0], b_end_loc[b, 0]],
                [b_start_loc[b, 1], b_end_loc[b, 1]],
                [b_start_loc[b, 2], b_end_loc[b, 2]], color=cur_color)

    if show:
        print('show')
        fig.show()
        # plt.close(fig)


def get_joint_parent(joint_order):
    if joint_order == 'mano':
        return np.array([0, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19])
    if joint_order == 'biomech':
        return np.array([0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
    if joint_order == 'halo':
        return np.array([0, 0, 1, 2, 0, 4, 5, 0, 7, 8, 0, 10, 11, 0, 13, 14, 15, 3, 6, 12, 9])


def plot_skeleton(ax, joints, joint_parent, color, object_points=None):
    if object_points is not None:
        ax.scatter(object_points[:, 0], object_points[:, 1], object_points[:, 2], alpha=0.1, c='r')
    ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], alpha=0.8, c='b')

    b_start_loc = joints[joint_parent]  # joints[biomech_joint_parent]
    b_end_loc = joints
    n_joint = 21
    if len(joints) == 16:
        n_joint = 16
    # import pdb; pdb.set_trace()
    for b in range(n_joint):
        ax.plot([b_start_loc[b, 0], b_end_loc[b, 0]],
                [b_start_loc[b, 1], b_end_loc[b, 1]],
                [b_start_loc[b, 2], b_end_loc[b, 2]], color=color)


def visualise_skeleton(joints, object_points=None, joint_order='biomech', out_file=None,
                       color='g', title=None, show=False):
    # Skeleton definition
    mano_joint_parent = np.array([0, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19])
    biomech_joint_parent = np.array([0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
    halo_joint_parent = np.array([0, 0, 1, 2, 0, 4, 5, 0, 7, 8, 0, 10, 11, 0, 13, 14, 15, 3, 6, 12, 9])
    if joint_order == 'mano':
        joint_parent = mano_joint_parent
    elif joint_order == 'biomech':
        joint_parent = biomech_joint_parent
    elif joint_order == 'halo':
        joint_parent = halo_joint_parent
    # Create plot
    fig = plt.figure(figsize=(13.5, 9))
    if title is not None:
        fig.suptitle(title)
    # ax = fig.add_subplot(111, projection='3d')
    # ax = fig.gca(projection=Axes3D.name)
    # set_ax_limits(ax)

    for i in range(6):
        ax = fig.add_subplot(2, 3, i + 1, projection='3d')
        set_ax_limits(ax)
        ax.view_init(elev=10., azim=i * 60)
        plot_skeleton(ax, joints, joint_parent, color, object_points=object_points)

    if out_file is not None:
        plt.savefig(out_file)
    if show:
        plt.show()
    # if show or (out_file is not None):
    plt.close(fig)


def display_mano_hand(hand_info, mano_faces=None, ax=None, alpha=0.2, show=True):
    """
    Displays hand batch_idx in batch of hand_info, hand_info as returned by
    generate_random_hand
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    verts, joints = hand_info['verts'], hand_info['joints']
    rest_joints = hand_info['rest_joints']
    # verts_joints_assoc = hand_info['verts_assoc']

    # import pdb; pdb.set_trace()
    visualize_bone = 13
    # rest_verts = hand_info['rest_verts']
    if mano_faces is None:
        ax.scatter(verts[:, 0], verts[:, 1], verts[:, 2], alpha=0.1)
    else:
        mesh = Poly3DCollection(verts[mano_faces], alpha=alpha)
        face_color = (141 / 255, 184 / 255, 226 / 255)
        edge_color = (50 / 255, 50 / 255, 50 / 255)
        mesh.set_edgecolor(edge_color)
        mesh.set_facecolor(face_color)
        ax.add_collection3d(mesh)
    # print("Joints", joints)
    print("joint shape", joints.shape)
    ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], color='r')

    # ax.scatter(joints[:16, 0], joints[:16, 1], joints[:16, 2], color='r')
    ax.scatter(rest_joints[:, 0], rest_joints[:, 1], rest_joints[:, 2], color='g')
    # ax.scatter(rest_joints[:4, 0], rest_joints[:4, 1], rest_joints[:4, 2], color='g')
    # ax.scatter(rest_joints[4:, 0], rest_joints[4:, 1], rest_joints[4:, 2], color='b')

    # visualize only some part
    # seleceted = verts_joints_assoc[:-1] == visualize_bone
    # ax.scatter(verts[seleceted, 0], verts[seleceted, 1], verts[seleceted, 2], color='black', alpha=0.5)

    # cam_equal_aspect_3d(ax, verts.numpy())
    cam_equal_aspect_3d(ax, verts)
    # cam_equal_aspect_3d(ax, rest_joints.numpy())
    if show:
        plt.show()


def cam_equal_aspect_3d(ax, verts, flip_x=False):
    """
    Centers view on cuboid containing hand and flips y and z axis
    and fixes azimuth
    """
    extents = np.stack([verts.min(0), verts.max(0)], axis=1)
    sz = extents[:, 1] - extents[:, 0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize / 2
    if flip_x:
        ax.set_xlim(centers[0] + r, centers[0] - r)
    else:
        ax.set_xlim(centers[0] - r, centers[0] + r)
    # Invert y and z axis
    ax.set_ylim(centers[1] + r, centers[1] - r)
    ax.set_zlim(centers[2] + r, centers[2] - r)


# def visualize_voxels(voxels, out_file=None, show=False):
#     r''' Visualizes voxel data.

#     Args:
#         voxels (tensor): voxel data
#         out_file (string): output file
#         show (bool): whether the plot should be shown
#     '''
#     # Use numpy
#     voxels = np.asarray(voxels)
#     # Create plot
#     fig = plt.figure()
#     ax = fig.gca(projection=Axes3D.name)
#     voxels = voxels.transpose(2, 0, 1)
#     ax.voxels(voxels, edgecolor='k')
#     ax.set_xlabel('Z')
#     ax.set_ylabel('X')
#     ax.set_zlabel('Y')
#     ax.view_init(elev=30, azim=45)
#     if out_file is not None:
#         plt.savefig(out_file)
#     if show:
#         plt.show()
#     plt.close(fig)


# def visualize_transmatrix(voxels, out_file=None, show=False):
#     r''' Visualizes voxel data.

#     Args:
#         voxels (tensor): voxel data
#         out_file (string): output file
#         show (bool): whether the plot should be shown
#     '''
#     # Use numpy
#     voxels = np.asarray(voxels)
#     # Create plot
#     fig = plt.figure()
#     ax = fig.gca(projection=Axes3D.name)
#     voxels = voxels.transpose(2, 0, 1)
#     ax.voxels(voxels, edgecolor='k')
#     ax.set_xlabel('Z')
#     ax.set_ylabel('X')
#     ax.set_zlabel('Y')
#     ax.view_init(elev=30, azim=45)
#     if out_file is not None:
#         plt.savefig(out_file)
#     if show:
#         plt.show()
#     plt.close(fig)


# def visualize_pointcloud(points, normals=None,
#                          out_file=None, show=False):
#     r''' Visualizes point cloud data.

#     Args:
#         points (tensor): point data
#         normals (tensor): normal data (if existing)
#         out_file (string): output file
#         show (bool): whether the plot should be shown
#     '''
#     # Use numpy
#     points = np.asarray(points)
#     # Create plot
#     fig = plt.figure()
#     ax = fig.gca(projection=Axes3D.name)
#     ax.scatter(points[:, 2], points[:, 0], points[:, 1])
#     if normals is not None:
#         ax.quiver(
#             points[:, 2], points[:, 0], points[:, 1],
#             normals[:, 2], normals[:, 0], normals[:, 1],
#             length=0.1, color='k'
#         )
#     ax.set_xlabel('Z')
#     ax.set_ylabel('X')
#     ax.set_zlabel('Y')
#     ax.set_xlim(-0.5, 0.5)
#     ax.set_ylim(-0.5, 0.5)
#     ax.set_zlim(-0.5, 0.5)
#     ax.view_init(elev=30, azim=45)
#     if out_file is not None:
#         plt.savefig(out_file)
#     if show:
#         plt.show()
#     plt.close(fig)


# def visualise_projection(
#         self, points, world_mat, camera_mat, img, output_file='out.png'):
#     r''' Visualizes the transformation and projection to image plane.

#         The first points of the batch are transformed and projected to the
#         respective image. After performing the relevant transformations, the
#         visualization is saved in the provided output_file path.

#     Arguments:
#         points (tensor): batch of point cloud points
#         world_mat (tensor): batch of matrices to rotate pc to camera-based
#                 coordinates
#         camera_mat (tensor): batch of camera matrices to project to 2D image
#                 plane
#         img (tensor): tensor of batch GT image files
#         output_file (string): where the output should be saved
#     '''
#     points_transformed = common.transform_points(points, world_mat)
#     points_img = common.project_to_camera(points_transformed, camera_mat)
#     pimg2 = points_img[0].detach().cpu().numpy()
#     image = img[0].cpu().numpy()
#     plt.imshow(image.transpose(1, 2, 0))
#     plt.plot(
#         (pimg2[:, 0] + 1)*image.shape[1]/2,
#         (pimg2[:, 1] + 1) * image.shape[2]/2, 'x')
#     plt.savefig(output_file)