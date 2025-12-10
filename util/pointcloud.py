import numpy as np
import open3d as o3d

def backproject(depth, K, mask):
    ys, xs = np.where(mask > 0)
    zs = depth[ys, xs]

    x = (xs - K[0,2]) * zs / K[0,0]
    y = (ys - K[1,2]) * zs / K[1,1]

    pts = np.vstack((x, y, zs)).T
    return pts

def transform_points(pts, T):
    pts_h = np.hstack((pts, np.ones((pts.shape[0], 1))))
    return (T @ pts_h.T).T[:, :3]

def dbscan_filter(pts):
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    labels = np.array(pc.cluster_dbscan(eps=0.03, min_points=20))
    if len(labels) == 0:
        return pts
    largest = np.argmax(np.bincount(labels[labels >= 0]))
    return pts[labels == largest]
