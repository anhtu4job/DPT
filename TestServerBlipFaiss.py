import faiss
import numpy as np
from datetime import datetime
from flask import Flask, request, render_template
from PIL import Image
from pathlib import Path
from blip2_feature_extractor import BLIP2FeatureExtractor
from lavis.models import load_model_and_preprocess
import torch

app = Flask(__name__)
print("Start App")
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

print("Load Text")

# Load the texts from the file, each text on a new line
with open(texts_file_path, 'r', encoding='utf-8') as file:
    for line in file:
        # Strip the newline character and append to the list
        texts.append(line.rstrip('\n'))

print("Load Image")

with open(images_name_path, 'r', encoding='utf-8') as file:
    for line in file:
        # Strip the newline character and append to the list
        names_image.append(line.rstrip('\n'))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, vis_processors, txt_processors = load_model_and_preprocess(
    name="blip2_feature_extractor", model_type="coco", is_eval=True, device=device
)
print("Load list Images")

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



def create_faiss_index(embeddings_np):
    dimension = embeddings_np.shape[1]  # assuming embeddings_np is a numpy array of your embeddings
    index = faiss.IndexFlatL2(dimension)  # Using L2 distance for similarity
    index.add(embeddings_np)  # Add embeddings to the index
    return index

# Convert PyTorch tensors to numpy arrays for FAISS
images_embedding_np = images_embedding.cpu().detach().numpy()
texts_embedding_np = texts_embedding.cpu().detach().numpy()
multimodal_embedding_np = multimodal_embedding.cpu().detach().numpy()

# Create FAISS indexes for each embedding type
images_index = create_faiss_index(images_embedding_np)
texts_index = create_faiss_index(texts_embedding_np)
multimodal_index = create_faiss_index(multimodal_embedding_np)

def search_similar_product(image_target, text_target, number_retrieval, search_modality, len_dataset=68, is_coco_finetuned=False):

    print("Start image_processed")
    # if image_target is None:
    #     if is_coco_finetuned:
    #         image_processed = torch.rand(1, 3, 364, 364).to(device)
    #     else:
    #         image_processed = torch.rand(1, 3, 224, 224).to(device)
    # else:
    #   image_processed = vis_processors["eval"](image_target).unsqueeze(0).to(device)
    image_processed = None
    if search_modality == "II":
        image_processed = vis_processors["eval"](image_target).unsqueeze(0).to(device)

    # preprocess text
    text_translated = text_target
    text_input = txt_processors["eval"](text_translated)

    # build sample
    sample = {"image": image_processed, "text_input": [text_input]}

    print("Extract image")
    multimodal_emb = None
    image_emb = None
    text_emb = None

    if search_modality == "II":
        image_emb = model.extract_features(sample, mode="image").image_embeds[0,0,:] # size (768)
        image_emb /= image_emb.norm(dim=-1, keepdim=True)

    elif search_modality == "TI":
        text_emb = model.extract_features(sample, mode="text").text_embeds[0,0,:] # size (768)
        text_emb /= text_emb.norm(dim=-1, keepdim=True)





    # find features of image and text target
    # multimodal_emb = model.extract_features(sample).multimodal_embeds[0,0,:] # size (768)
    # image_emb = model.extract_features(sample, mode="image").image_embeds[0,0,:] # size (768)
    # text_emb = model.extract_features(sample, mode="text").text_embeds[0,0,:] # size (768)

    print("normalize image")

    # normalize
    # image_emb /= image_emb.norm(dim=-1, keepdim=True)
    # text_emb /= text_emb.norm(dim=-1, keepdim=True)
    #multimodal_emb /= multimodal_emb.norm(dim=-1, keepdim=True)

    print("Choose the correct FAISS")

    # Choose the correct FAISS index and embedding vector based on search_modality
    if search_modality in ["II", "TI", "MI"]:
        faiss_index = images_index
        query_embedding = image_emb if search_modality == "II" else text_emb if search_modality == "TI" else multimodal_emb
    elif search_modality in ["IT", "TT", "MT"]:
        faiss_index = texts_index
        query_embedding = image_emb if search_modality == "IT" else text_emb if search_modality == "TT" else multimodal_emb
    elif search_modality in ["IM", "TM", "MM"]:
        faiss_index = multimodal_index
        query_embedding = image_emb if search_modality == "IM" else text_emb if search_modality == "TM" else multimodal_emb

    # Convert query embedding to FAISS-compatible format (numpy array, reshape if necessary)
    query_embedding_np = np.expand_dims(query_embedding, axis=0)

    print("Search the FAISS index")
    # Search the FAISS index
    D, I = faiss_index.search(query_embedding_np, number_retrieval)  # D: distances, I: indices

    print("add retrieval_map")

    retrieval_map = {}
    for i, idx in enumerate(I[0]):
        similarity_value = D[0][i]
        name_image_found = names_image[idx]
        text_image_found = texts[idx]
        translated_text_image_found = texts[idx]
        image_found = img_paths[idx]
        image_test = "static/uploaded/" + name_image_found

        # Fill the extracted infos in the map
        retrieval_map[i + 1] = (image_found, name_image_found, text_image_found, translated_text_image_found, text_image_found)
    return retrieval_map


# test_img = images[0];
# test_text = "Pizza";
# test_number_retrieval = 10;
# test_search_modality = "II";

# print("Start Search Images")

# results = search_similar_product(image_target=test_img, text_target=test_text,number_retrieval=test_number_retrieval,search_modality=test_search_modality)
# print(results);
# print("end Search Images")

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['query_img']
        text_input = request.form['query_text']
        search_modality = request.form['search_modality']
        number_retrieval = request.form['retrieval_number']
        img = ""
        uploaded_img_path = ""

        if file and file.filename != '' and search_modality == "II":
            img = Image.open(file.stream).convert("RGB")
            uploaded_img_path = "static/uploaded/" + datetime.now().isoformat().replace(":", ".") + "_" + file.filename
            img.save(uploaded_img_path)  # Save query image
        # Example text target, modify as needed
        text_target = text_input
        # Call the search function, modify parameters as needed
        #results = search_similar_product(img, text_target, number_retrieval=10, search_modality="II", dataset_embeddings=all_features_tensor)
        results = search_similar_product(image_target=img, text_target=text_target,number_retrieval=int(number_retrieval),search_modality=search_modality)
        # Example way to process results, modify according to your `search_similar_product` implementation
        print("End Result")
        scores = [(value[4], value[0]) for key, value in results.items()]

        return render_template('index.html', query_path=uploaded_img_path, scores=scores)
    else:
        return render_template('index.html')

if __name__=="__main__":
    app.run("0.0.0.0")