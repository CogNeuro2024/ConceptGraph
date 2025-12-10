from segment_anything import sam_model_registry, SamPredictor


class SAMSegmenter:
    """
        SAMSegmenter
        -------------
        Performs class-agnostic instance segmentation using the Segment Anything Model (SAM).

        This module takes a raw RGB image and produces a set of binary masks,
        each corresponding to a potential object instance. SAM does not rely on
        predefined categories, making it ideal for open-vocabulary mapping.

        Attributes
        ----------
        model : torch.nn.Module
            The SAM model architecture loaded with pretrained weights.
        predictor : SamPredictor
            High-level prediction interface for SAM, providing mask extraction functionality.
    """

    def __init__(self, sam_checkpoint="sam_vit_h_4b8939.pth"):
        """
                Initialize the SAMSegmenter by loading SAM weights and preparing the predictor.

                Parameters
                ----------
                sam_checkpoint : str
                    Path to the SAM model checkpoint (.pth file).
        """
        self.model = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
        self.model.to("cuda")
        self.predictor = SamPredictor(self.model)

    def segment(self, rgb):
        """
               Generate object masks from an RGB image.

               Parameters
               ----------
               rgb : np.ndarray
                   RGB image of shape (H, W, 3) in uint8 format.

               Returns
               -------
               list[np.ndarray]
                   List of binary masks, each of shape (H, W), where 1 = object, 0 = background.

               Notes
               -----
               These masks serve as input to:
                   - CLIPEncoder for feature extraction
                   - 3D point cloud backprojection
        """
        self.predictor.set_image(rgb)
        masks, _, _ = self.predictor.predict(
            point_coords=None,
            point_labels=None,
            multimask_output=True
        )
        return masks
