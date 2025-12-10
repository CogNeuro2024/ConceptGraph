import numpy as np
from scipy.spatial import ConvexHull

def bbox_3d(pts):
    """
       Compute an axis-aligned bounding box from a point cloud.

       Parameters
       ----------
       pts : np.ndarray
           Point cloud (N x 3).

       Returns
       -------
       dict
           Dictionary with:
               - "min": np.ndarray of shape (3,)
               - "max": np.ndarray of shape (3,)
       """

    hull = ConvexHull(pts)
    min_pt = pts.min(axis=0)
    max_pt = pts.max(axis=0)
    return {"min": min_pt, "max": max_pt}

def iou_3d(boxA, boxB):
    """
       Compute 3D Intersection-over-Union (IoU) between two bounding boxes.

       Parameters
       ----------
       boxA : dict
           Output of bbox_3d for object A.
       boxB : dict
           Output of bbox_3d for object B.

       Returns
       -------
       float
           IoU value in [0, 1].

       Notes
       -----
       Used to determine likely adjacency for Scene Graph edges.
       """

    inter_min = np.maximum(boxA["min"], boxB["min"])
    inter_max = np.minimum(boxA["max"], boxB["max"])
    inter_vol = np.prod(np.maximum(inter_max - inter_min, 0))

    volA = np.prod(boxA["max"] - boxA["min"])
    volB = np.prod(boxB["max"] - boxB["min"])

    return inter_vol / (volA + volB - inter_vol + 1e-6)

def build_edges(objects, captioner):
    """
        Construct the semantic 3D scene graph.

        Parameters
        ----------
        objects : list[dict]
            Object nodes augmented with 3D geometry and captions.
        captioner : InterVLCaptioner
            Captioner used to infer inter-object relations.

        Returns
        -------
        list[dict]
            Edge list, where each entry contains:
                - "i": object index
                - "j": object index
                - "relation": string description

        Notes
        -----
        Steps:
            1. Compute 3D bounding boxes for all objects
            2. Compute pairwise IoU similarities
            3. Build a minimum spanning tree (ensures connectivity)
            4. For each MST edge, query InterVL for relationship reasoning

        Implements Scene Graph Generation (Section II-B).
        """
    # build MST based on IoU weights
    bboxes = [bbox_3d(obj["points"]) for obj in objects]

    weights = {}
    for i in range(len(objects)):
        for j in range(i+1, len(objects)):
            weights[(i,j)] = iou_3d(bboxes[i], bboxes[j])

    # MST = connect by highest IoU
    sorted_edges = sorted(weights.items(), key=lambda x: -x[1])
    parent = list(range(len(objects)))

    def find(x):
        while parent[x] != x:
            x = parent[x]
        return x

    E = []
    for (i,j), _ in sorted_edges:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj
            rel = captioner.relation(objects[i]["caption"], objects[j]["caption"])
            E.append({"i": i, "j": j, "relation": rel})

    return E
