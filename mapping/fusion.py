import numpy as np
from utils.pointcloud import dbscan_filter

def fuse(objects, detections, assoc):
    """
        Fuse new detections into the global 3D object map.

        Parameters
        ----------
        objects : list[dict]
            Current map containing fused object nodes.
        detections : list[dict]
            Newly detected objects from the current RGB-D frame.
        assoc : dict
            Output of `match_detections`, mapping detections to object indices.

        Returns
        -------
        list[dict]
            Updated list of fused object nodes.

        Notes
        -----
        This function performs:
            - Semantic feature averaging
            - Point cloud merging and denoising
            - View accumulation for captioning

        Implements the "Object Fusion" part of Section II-A.
        """
    for det_id, obj_id in assoc.items():
        det = detections[det_id]
        if obj_id is None:
            objects.append({
                "points": det["points"],
                "feat": det["feat"],
                "count": 1,
                "views": det["views"],
            })
        else:
            obj = objects[obj_id]
            obj["feat"] = (obj["count"] * obj["feat"] + det["feat"]) / (obj["count"] + 1)
            obj["count"] += 1
            obj["points"] = np.vstack([obj["points"], det["points"]])
            obj["points"] = dbscan_filter(obj["points"])
            obj["views"] += det["views"]
    return objects
