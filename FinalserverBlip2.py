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

save_path_img = './static/images_embedding.pt'
save_path_text = './static/texts_embedding.pt'
save_path_multimodal = './static/multimodal_embedding.pt'
#Save file texts
texts_file_path = './static/texts.txt'
images_name_path = './static/imageName.txt'


# Initialize an empty list to hold the loaded texts
images = []
texts = []
names_image = []

# Load the texts from the file, each text on a new line
with open(texts_file_path, 'r', encoding='utf-8') as file:
    for line in file:
        # Strip the newline character and append to the list
        texts.append(line.rstrip('\n'))

with open(images_name_path, 'r', encoding='utf-8') as file:
    for line in file:
        # Strip the newline character and append to the list
        names_image.append(line.rstrip('\n'))


# # Assuming you want to see the loaded data
# print(len(loaded_texts))
# setting device on GPU if available, else CPU


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, vis_processors, txt_processors = load_model_and_preprocess(
    name="blip2_feature_extractor", model_type="coco", is_eval=True, device=device
)

#Load image to images
img_dir = Path("./static/img")
img_paths = [p for p in img_dir.iterdir() if p.suffix.lower() in ['.jpg', '.jpeg', '.png']]
for img_path in img_paths:
    image = Image.open(img_path).convert("RGB")
    images.append(image)



images_embedding = torch.load(save_path_img)
texts_embedding = torch.load(save_path_text)
multimodal_embedding = torch.load(save_path_multimodal)

images_embedding /= images_embedding.norm(dim=-1, keepdim=True)
texts_embedding /= texts_embedding.norm(dim=-1, keepdim=True)
multimodal_embedding /= multimodal_embedding.norm(dim=-1, keepdim=True)

def search_similar_product(image_target, text_target, number_retrieval, search_modality, len_dataset = 68, is_coco_finetuned = False):

    retrieval_map = {}

    # preprocess image. check if the image is None. In this case build a tensor with shape (1, 3, 224, 224)
    if image_target is None:
        if is_coco_finetuned:
            image_processed = torch.rand(1, 3, 364, 364).to(device)
        else:
            image_processed = torch.rand(1, 3, 224, 224).to(device)
    else:
      image_processed = vis_processors["eval"](image_target).unsqueeze(0).to(device)

    # preprocess text
    text_translated = text_target
    text_input = txt_processors["eval"](text_translated)

    # build sample
    sample = {"image": image_processed, "text_input": [text_input]}

    # find features of image and text target
    multimodal_emb = model.extract_features(sample).multimodal_embeds[0,0,:] # size (768)
    image_emb = model.extract_features(sample, mode="image").image_embeds[0,0,:] # size (768)
    text_emb = model.extract_features(sample, mode="text").text_embeds[0,0,:] # size (768)

    # normalize
    image_emb /= image_emb.norm(dim=-1, keepdim=True)
    text_emb /= text_emb.norm(dim=-1, keepdim=True)
    multimodal_emb /= multimodal_emb.norm(dim=-1, keepdim=True)

    # transform to numpy tensor
    image_emb = image_emb.cpu().detach().numpy()
    text_emb = text_emb.cpu().detach().numpy()
    multimodal_emb = multimodal_emb.cpu().detach().numpy()

    if search_modality == "II":
      similarity_vector = images_embedding.cpu().detach().numpy() @ image_emb.T
    elif search_modality == "TI":
      similarity_vector = images_embedding.cpu().detach().numpy() @ text_emb.T
    elif search_modality == "MI":
      similarity_vector = images_embedding.cpu().detach().numpy() @ multimodal_emb.T

    elif search_modality == "IT":
      similarity_vector = texts_embedding.cpu().detach().numpy() @ image_emb.T
    elif search_modality == "TT":
      similarity_vector = texts_embedding.cpu().detach().numpy() @ text_emb.T
    elif search_modality == "MT":
      similarity_vector = texts_embedding.cpu().detach().numpy() @ multimodal_emb.T

    elif search_modality == "IM":
      similarity_vector = multimodal_embedding.cpu().detach().numpy() @ image_emb.T
    elif search_modality == "TM":
      similarity_vector = multimodal_embedding.cpu().detach().numpy() @ text_emb.T
    elif search_modality == "MM":
      similarity_vector = multimodal_embedding.cpu().detach().numpy() @ multimodal_emb.T

    index_sorted = np.argsort(similarity_vector)

    for i in range(1, number_retrieval+1):

        print("Extracted: ", i)

        # index of the db product
        idx = index_sorted[len_dataset-i]
        similarity_value = similarity_vector[idx]

        print("idx: ", idx)
        print("similarity value: ", similarity_value)

        # extract name of the image
        name_image_found = names_image[idx]
        print("name_image_found: ", name_image_found)
        print("-----------------------")

        # extract text of the image
        text_image_found = texts[idx]
        translated_text_image_found = texts[idx]

        # extract image
        image_found = img_paths[idx]
        image_test = "static/uploaded/" + name_image_found;
        #fill the extracted infos in the map
        retrieval_map[i] = (image_found, name_image_found, text_image_found, translated_text_image_found, similarity_value)

    return retrieval_map


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['query_img']
        text_input = request.form['query_text']
        search_modality = request.form['search_modality']
        img = ""
        uploaded_img_path = ""
        if file and file.filename != '':
            img = Image.open(file.stream).convert("RGB")
            uploaded_img_path = "static/uploaded/" + datetime.now().isoformat().replace(":", ".") + "_" + file.filename
            img.save(uploaded_img_path)  # Save query image

        
        

        # Example text target, modify as needed
        text_target = text_input
        number_retrieval = 10
        # Call the search function, modify parameters as needed
        #results = search_similar_product(img, text_target, number_retrieval=10, search_modality="II", dataset_embeddings=all_features_tensor)
        results = search_similar_product(image_target=img, text_target=text_target,number_retrieval=number_retrieval,search_modality=search_modality)
        # Example way to process results, modify according to your `search_similar_product` implementation
        scores = [(value[4], value[0]) for key, value in results.items()]

        return render_template('index.html', query_path=uploaded_img_path, scores=scores)
    else:
        return render_template('index.html')

if __name__=="__main__":
    app.run("0.0.0.0")