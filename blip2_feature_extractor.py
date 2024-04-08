from PIL import Image
import torch
from lavis.models import load_model_and_preprocess
import numpy as np

class BLIP2FeatureExtractor:
    def __init__(self, device='cpu'):
        self.device = device
        self.model, self.vis_processors, self.txt_processors = load_model_and_preprocess(
            name="blip2_feature_extractor",
            model_type="coco",
            is_eval=True,
            device=device
        )

    def extract_image_features(self, image):
        # Ensure image is a PIL Image
        if not isinstance(image, Image.Image):
            raise ValueError("Input image must be a PIL.Image.Image instance")
        
        # Process the image using the provided visual processor
        try:
            image_processed = self.vis_processors["eval"](image).unsqueeze(0).to(self.device)
            image_emb = self.model.extract_features({"image": image_processed}, mode="image").image_embeds[:,0,:]
            image_emb /= image_emb.norm(dim=-1, keepdim=True)
            return image_emb.cpu().numpy()
        except Exception as e:
            # Add error handling here
            print(f"Error processing image: {e}")
            return None

    def extract_text_features(self, text):
        text_input = self.txt_processors["eval"](text)
        text_emb = self.model.extract_features({"text_input": [text_input]}, mode="text").text_embeds[:,0,:]
        return text_emb.cpu().numpy()

    def extract_multimodal_features(self, text, image):
        image_processed = self.vis_processors["eval"](image).unsqueeze(0).to(self.device)
        text_input = self.txt_processors["eval"](text)
        sample = {"image": image_processed, "text_input": [text_input]}
        multimodal_emb = self.model.extract_features(sample).multimodal_embeds[:,0,:]
        return multimodal_emb.cpu().numpy()

    