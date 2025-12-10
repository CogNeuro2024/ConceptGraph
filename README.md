# ConceptGraph
Create a concept graph 


# class diagram

                            ┌──────────────────────────┐
                            │      SAMSegmenter        │
                            ├──────────────────────────┤
                            │ - model: SAM             │
                            │ - predictor: SamPredictor│
                            ├──────────────────────────┤
                            │ + segment(rgb): masks    │
                            └──────────────────────────┘
                                      │
                                      ▼
 ┌──────────────────────────┐      uses       ┌────────────────────────────┐
 │      CLIPEncoder         │<───────────────│      RGB + SAM Masks        │
 ├──────────────────────────┤                 └────────────────────────────┘
 │ - model: CLIP            │
 │ - preprocess             │
 ├──────────────────────────┤
 │ + encode_crop(): feat    │
 └──────────────────────────┘
                                      │
                                      ▼
                       ┌──────────────────────────┐
                       │   Association (functions)│
                       ├──────────────────────────┤
                       │ + geometric_similarity() │
                       │ + semantic_similarity()  │
                       │ + match_detections()     │
                       └──────────────────────────┘
                                      │
                                      ▼
                       ┌──────────────────────────┐
                       │         Fusion           │
                       ├──────────────────────────┤
                       │ + fuse(objects, dets)    │
                       │   merges+updates objects │
                       └──────────────────────────┘
                                      │
                                      ▼
                   ┌──────────────────────────────────┐
                   │       InterVLCaptioner           │
                   ├──────────────────────────────────┤
                   │ - model: InterVL                 │
                   │ - processor                      │
                   ├──────────────────────────────────┤
                   │ + caption_view(image): str       │
                   │ + summarize(caps): str           │
                   │ + relation(capA, capB): str      │
                   └──────────────────────────────────┘
                                      │
                                      ▼
                    ┌────────────────────────────────┐
                    │      GraphBuilder (MST)        │
                    ├────────────────────────────────┤
                    │ + bbox_3d(points)              │
                    │ + iou_3d(boxA, boxB)           │
                    │ + build_edges(): edges         │
                    └────────────────────────────────┘
                                      │
                                      ▼
                    ┌────────────────────────────────┐
                    │       SceneGraph Output        │
                    ├────────────────────────────────┤
                    │ Nodes: objects (feat, points,  │
                    │         caption, bbox ...)     │
                    │ Edges: relations               │
                    └────────────────────────────────┘
