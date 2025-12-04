from segmentation.sam_segment import SAMSegmenter
from features.clip_encoder import CLIPEncoder
from mapping.association import match_detections
from mapping.fusion import fuse
from captioning.intervl_captioner import InterVLCaptioner
from graph.graph_builder import build_edges
from utils.pointcloud import backproject, transform_points, dbscan_filter

import cv2
import numpy as np

def process_frame(rgb, depth, K, T, sam, clip):
    masks = sam.segment(rgb)
    detections = []
    for mask in masks:
        feat = clip.encode_crop(rgb, mask)
        pts = backproject(depth, K, mask)
        pts = transform_points(pts, T)
        pts = dbscan_filter(pts)
        detections.append({
            "points": pts,
            "feat": feat.cpu().numpy(),
            "views": [rgb],   # store view for later captioning
        })
    return detections


def run_conceptgraphs(frames):
    sam = SAMSegmenter()
    clip = CLIPEncoder()
    captioner = InterVLCaptioner()

    objects = []

    for rgb, depth, K, T in frames:
        dets = process_frame(rgb, depth, K, T, sam, clip)
        assoc = match_detections(dets, objects)
        objects = fuse(objects, dets, assoc)

    # caption nodes
    for obj in objects:
        view_imgs = obj["views"][:10]
        caps = [captioner.caption_view(v) for v in view_imgs]
        obj["caption"] = captioner.summarize(caps)

    # build graph
    edges = build_edges(objects, captioner)

    return {"objects": objects, "edges": edges}
