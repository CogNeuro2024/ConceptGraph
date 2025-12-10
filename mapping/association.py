import numpy as np
from sklearn.neighbors import NearestNeighbors

def geometric_similarity(A, B, thresh=0.025):
    """
        Compute geometric similarity between two point clouds.

        Parameters
        ----------
        A : np.ndarray
            First point cloud (N x 3).
        B : np.ndarray
            Second point cloud (M x 3).
        thresh : float
            Maximum Euclidean distance to consider points as overlapping.

        Returns
        -------
        float
            Ratio of points in A that have a neighbor in B within `thresh`.

        Notes
        -----
        Implements ϕ_geo from ConceptGraphs Section II-A.
        """

    if len(A) == 0 or len(B) == 0:
        return 0
    nbrs = NearestNeighbors(n_neighbors=1).fit(B)
    dists, _ = nbrs.kneighbors(A)
    ratio = np.mean(dists < thresh)
    return ratio

def semantic_similarity(fA, fB):
    """
       Compute semantic similarity between two feature vectors.

       Parameters
       ----------
       fA : np.ndarray | torch.Tensor
           CLIP embedding of object A.
       fB : np.ndarray | torch.Tensor
           CLIP embedding of object B.

       Returns
       -------
       float
           Normalized cosine similarity in [0, 1].

       Notes
       -----
       Implements ϕ_sem from the paper.
       """

    cos = (fA @ fB.T).item()
    return (cos + 1) / 2

def match_detections(detections, objects, delta_sim=1.1):
    """
       Associate new object detections to existing 3D object nodes.

       Parameters
       ----------
       detections : list[dict]
           Each detection contains:
               - points : np.ndarray
               - feat : np.ndarray
               - views : list of view images
       objects : list[dict]
           Existing object nodes in the map.
       delta_sim : float
           Minimum similarity required to match a detection to an existing object.

       Returns
       -------
       dict
           Mapping { detection_id : object_id or None }.

       Notes
       -----
       - If no match exceeds delta_sim, a new object must be created.
       - Corresponds to "Greedy Association" in Section II-A.
       """
    associations = {}
    for i, det in enumerate(detections):
        best, best_sim = None, 0
        for j, obj in enumerate(objects):
            geo = geometric_similarity(det["points"], obj["points"])
            sem = semantic_similarity(det["feat"], obj["feat"])
            sim = geo + sem
            if sim > best_sim:
                best, best_sim = j, sim
        associations[i] = best if best_sim >= delta_sim else None
    return associations
