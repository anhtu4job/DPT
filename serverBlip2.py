import numpy as np
from datetime import datetime
from flask import Flask, request, render_template
from PIL import Image
from pathlib import Path
from blip2_feature_extractor import BLIP2FeatureExtractor
from lavis.models import load_model_and_preprocess

import torch
# Assuming 'device' is set appropriately (e.g., 'cuda' for GPU)


app = Flask(__name__)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, vis_processors, txt_processors = load_model_and_preprocess(
    name="blip2_feature_extractor", model_type="coco", is_eval=True, device=device
)

# Load your dataset features and paths
features = []
img_paths = []
feature_tensors = []

feature_dir = Path("./static/feature")
img_dir = Path("./static/img")

img_dir = Path("./static/img")
img_paths = []

# List of image file extensions to include
extensions = ['*.jpg', '*.jpeg', '*.png']

# Iterate over each extension and add matching paths to img_paths
for extension in extensions:
    img_paths.extend(img_dir.glob(extension))

# If you need img_paths to be a list of Path objects (and it currently is),
# this is already the case. If you need strings instead, you can convert them:
img_paths = [str(path) for path in img_paths]

# Print paths to verify
for path in img_paths:
    print(path)

feature_dir = Path("./static/feature")
# feature_tensors = [torch.from_numpy(np.load(feature_path)).squeeze(0) for feature_path in feature_dir.glob("*.npy")]
# all_features_tensor = torch.stack(feature_tensors)
# all_features_tensor /= all_features_tensor.norm(dim=-1, keepdim=True)
save_path = './static/images_embedding.pt'
all_features_tensor = torch.load(save_path)
all_features_tensor /= all_features_tensor.norm(dim=-1, keepdim=True)


def search_similar_product(image_target, text_target,len_dataset = 68, number_retrieval=2, search_modality="II", dataset_embeddings=None):
        
        retrieval_map = {}

        if image_target is None:
            image_processed = torch.rand(1, 3, 224, 224).to(device)
                
        else:
            image_processed = vis_processors["eval"](image_target).unsqueeze(0).to(device)

        sample = {"image": image_processed}
        image_emb = model.extract_features(sample, mode="image").image_embeds[0,0,:] # size (768)

        # normalize
        image_emb /= image_emb.norm(dim=-1, keepdim=True)

        # transform to numpy tensor
        image_emb = image_emb.cpu().detach().numpy()


        similarity_vector = dataset_embeddings.cpu().detach().numpy() @ image_emb.T

        #Sort Similarity
        index_sorted = np.argsort(similarity_vector)

        for i in range(1, number_retrieval+1):
            print("Extracted: ", i)

            # index of the db product
            #idx = index_sorted[len(img_paths)-i]
            idx = index_sorted[len_dataset-i]
            similarity_value = similarity_vector[idx]

            print("idx: ", idx)
            print("similarity value: ", similarity_value)
            image_found = img_paths[idx]
            print("Image found ",i," ",image_found,similarity_value)
            #fill the extracted infos in the map
            retrieval_map[i] = (image_found, similarity_value)
        
        return retrieval_map

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['query_img']
        img = Image.open(file.stream).convert("RGB")  # Ensure image is in the correct format

        uploaded_img_path = "static/uploaded/" + datetime.now().isoformat().replace(":", ".") + "_" + file.filename
        img.save(uploaded_img_path)  # Save query image

        # Example text target, modify as needed
        text_target = "example text description"
        # Call the search function, modify parameters as needed
        results = search_similar_product(img, text_target, number_retrieval=10, search_modality="II", dataset_embeddings=all_features_tensor)

        # Example way to process results, modify according to your `search_similar_product` implementation
        scores = [(value[1], value[0]) for key, value in results.items()]

        return render_template('index.html', query_path=uploaded_img_path, scores=scores)
    else:
        return render_template('index.html')

if __name__=="__main__":
    app.run("0.0.0.0")