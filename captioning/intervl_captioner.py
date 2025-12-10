import torch
from internvl.model.internvl_chat import InterVLChatModel, InterVLChatProcessor

class InterVLCaptioner:
    """
       InterVLCaptioner
       -----------------
       Generates object descriptions and inter-object relationship labels
       using the InterVL multimodal large language model.

       This module replaces:
           - LLaVA (multi-view vision-language captioning)
           - GPT-4 (caption summarization and relation reasoning)

       Attributes
       ----------
       model : InterVLChatModel
           The loaded InterVL vision-language transformer.
       proc : InterVLChatProcessor
           Preprocessor for converting images and prompts into model inputs.
       """

    def __init__(self, model_path="OpenGVLab/InternVL2-26B"):
        """
               Load the InterVL model and preprocessing pipeline.

               Parameters
               ----------
               model_path : str
                   HuggingFace model identifier for an InterVL variant.
               """
        self.model = InterVLChatModel.from_pretrained(model_path).to("cuda")
        self.proc = InterVLChatProcessor.from_pretrained(model_path)

    def caption_view(self, image):
        """
             Produce a natural-language caption describing the central object.

             Parameters
             ----------
             image : np.ndarray | PIL.Image
                 Input image crop containing the object.

             Returns
             -------
             str
                 Caption describing the object.

             Notes
             -----
             Called once per "best view" per object node.
             """

        msg = [{"role": "user", "content": "Describe the central object."}]
        inputs = self.proc(messages=msg, images=[image], return_tensors="pt").to("cuda")
        out = self.model.generate(**inputs, max_length=80)
        return self.proc.batch_decode(out, skip_special_tokens=True)[0]

    def summarize(self, captions):
        """
               Summarize multiple view-level captions into a single coherent description.

               Parameters
               ----------
               captions : list[str]
                   Up to 10 captions from InterVL describing the object.

               Returns
               -------
               str
                   Final, coherent, refined caption.

               Notes
               -----
               Implements the caption refinement step from ConceptGraphs Appendix A2.
               """
        text = "Summarize these object captions into one object description:\n" + "\n".join(captions)
        msg = [{"role": "user", "content": text}]
        inputs = self.proc(messages=msg, return_tensors="pt").to("cuda")
        out = self.model.generate(**inputs, max_length=100)
        return self.proc.batch_decode(out, skip_special_tokens=True)[0]

    def relation(self, capA, capB):
        """
               Infer semantic or spatial relationship between two objects.

               Parameters
               ----------
               capA : str
                   Caption of object A.
               capB : str
                   Caption of object B.

               Returns
               -------
               str
                   Relationship descriptor, e.g.:
                       - "A is on B"
                       - "B is inside A"
                       - "A is next to B"

               Notes
               -----
               Implements Section II-B (Scene Graph Generation).
               """

        text = f"""
        Object A: {capA}
        Object B: {capB}
        Describe the spatial relationship between them (e.g. A on B).
        """
        msg = [{"role": "user", "content": text}]
        inputs = self.proc(messages=msg, return_tensors="pt").to("cuda")
        out = self.model.generate(**inputs, max_length=50)
        return self.proc.batch_decode(out, skip_special_tokens=True)[0]
