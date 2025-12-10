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
