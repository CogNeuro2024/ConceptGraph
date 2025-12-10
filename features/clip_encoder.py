import torch
import open_clip
from torchvision import transforms
from PIL import Image




class CLIPEncoder:
    """
       CLIPEncoder
       ------------
       Extracts semantic embedding vectors for object regions using CLIP.

       This encoder provides semantic features that allow ConceptGraphs to:
           - Match objects across multiple views
           - Reason about similarity between detections
           - Provide semantic grounding for captioning and downstream tasks

       Attributes
       ----------
       model : torch.nn.Module
           Pretrained CLIP image encoder.
       preprocess : torchvision.transforms.Compose
           Preprocessing pipeline for feeding images into CLIP.
       tokenizer : Callable
           Tokenizer for optional text conditioning (not used here).
       """
    def __init__(self, model_name="ViT-B-32"):
        """
               Load the CLIP model and preprocessing functions.

               Parameters
               ----------
               model_name : str
                   Name of CLIP variant to use.
               """
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained="laion2b_s34b_b79k")
        self.model.to("cuda")
        self.tokenizer = open_clip.get_tokenizer(model_name)

    def encode_crop(self, rgb, mask):
        """
                Compute the CLIP embedding of a masked object region.

                Parameters
                ----------
                rgb : np.ndarray
                    The full RGB image.
                mask : np.ndarray
                    Binary mask indicating which pixels belong to the object.

                Returns
                -------
                torch.Tensor
                    A normalized CLIP feature vector with unit norm.

                Notes
                -----
                This feature vector is used during:
                    - Object association
                    - Node fusion
                    - Semantic similarity scoring
                """

        crop = rgb * mask[..., None]
        img = Image.fromarray(crop.astype("uint8"))
        img = self.preprocess(img).unsqueeze(0).to("cuda")
        with torch.no_grad():
            feat = self.model.encode_image(img)
        return feat / feat.norm()
