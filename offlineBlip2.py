from lavis.models import load_model_and_preprocess
from PIL import Image
from pathlib import Path
import torch
import numpy as np

import json
image_map = None
with open('./static/mappa.json', 'r') as f:
    image_map = json.load(f)
print(image_map)
# Assuming 'device' is set appropriately (e.g., 'cuda' for GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model, vis_processors, txt_processors = load_model_and_preprocess(
    name="blip2_feature_extractor", model_type="coco", is_eval=True, device=device
)

# Assuming 'texts' and 'images' are lists of text descriptions and PIL images respectively
# texts = [...]  # Your text descriptions here
# images = [Image.open(img_path) for img_path in sorted(Path("./staticTemp/img").glob("*.jpg"))]  # Load images

# for image in images:
#     # Image processing
#     image_processed = vis_processors["eval"](image).unsqueeze(0).to(device)
    
#     # Text processing (assuming direct usage, replace 'translate_caption' as necessary)
#     #text_input = txt_processors["eval"](text).to(device)
    
#     # Feature extraction
#     sample = {"image": image_processed}
#     image_emb = model.extract_features(sample, mode="image").image_embeds[:, 0, :]  # Image features
#     #text_emb = model.extract_features(sample, mode="text").text_embeds[:, 0, :]  # Text features
#     #multimodal_emb = model.extract_features(sample).multimodal_embeds[:, 0, :]  # Multimodal features

#     # Save features (adjust path and saving mechanism as needed)
#     img_feature_path = Path("./staticTemp/feature") / "image_features.npy"
#     #text_feature_path = Path("./static/feature") / "text_features.npy"
#     #multimodal_feature_path = Path("./staticTemp/feature") / "multimodal_features.npy"
#     np.save(img_feature_path, image_emb.cpu().numpy())
#     #np.save(text_feature_path, text_emb.cpu().numpy())
#     #np.save(multimodal_feature_path, multimodal_emb.cpu().numpy())

save_path = './static/images_embedding.pt'
img_dir = Path("./static/img")
img_paths = [p for p in img_dir.iterdir() if p.suffix.lower() in ['.jpg', '.jpeg', '.png']]
img_paths = sorted(img_paths)
images_embedding = None

# backup
# for img_path in sorted(Path("./staticTemp/img").glob("*.jpg")):
#         print(img_path)  # e.g., ./static/img/xxx.jpg
#         raw_image = Image.open(img_path).convert("RGB")
#         image_processed = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
#         sample = {"image": image_processed}
#         image_emb = model.extract_features(sample, mode="image").image_embeds[:, 0, :]  # Image features
#         img_feature_path = Path("./staticTemp/feature") / (img_path.stem + ".npy")  # e.g., ./static/feature/xxx.npy
#         np.save(img_feature_path, image_emb.cpu().numpy())


for img_path in img_paths:
    print(img_path)  # Print the image path for verification
    raw_image = Image.open(img_path).convert("RGB")
    image_processed = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
    sample = {"image": image_processed}
    image_emb = model.extract_features(sample, mode="image").image_embeds[:, 0, :]  # Image features
    img_feature_path = Path("./static/feature") / (img_path.stem + ".npy")
    #np.save(img_feature_path, image_emb.squeeze().cpu().numpy())

        # stack tensor
    if images_embedding is None:
        images_embedding = image_emb
    else:
       images_embedding = torch.cat((images_embedding, image_emb),0)


torch.save(images_embedding, save_path)



feature_dir = Path("./static/feature")
print(f"Image Features Shape: {images_embedding.size()}")
#comment file
# feature_tensors = [torch.from_numpy(np.load(feature_path)).squeeze(0) for feature_path in feature_dir.glob("*.npy")]
# all_features_tensor = torch.stack(feature_tensors)
# print('all_features_Image_tensor.size(): ', all_features_tensor.size())
loaded_images_embedding = torch.load(save_path)
print(f'loaded_images_embedding {loaded_images_embedding.size()}')

# Example to print sizes, adjust as needed for your logic
#print(f"Text Features Shape: {text_emb.size()}")
#print(f"Multimodal Features Shape: {multimodal_emb.size()}")
