import json
from lavis.models import load_model_and_preprocess
from PIL import Image
from pathlib import Path
import torch
import numpy as np
import os

image_map = None
with open('./static/FinalData.json', 'r') as f:
    image_map = json.load(f)
#print(image_map)

images = []
texts = []
names_image = []

save_path = './static/images_embedding.pt'
img_dir = Path("./static/img")
img_paths = [p for p in img_dir.iterdir() if p.suffix.lower() in ['.jpg', '.jpeg', '.png']]



for img_path in img_paths:
    file_name = os.path.basename(img_path)
    print(file_name)
    print(image_map[file_name])
    image = Image.open(img_path).convert("RGB")
    text = image_map[file_name]
    images.append(image)
    texts.append(text)
    names_image.append(file_name)

#Save file texts
texts_file_path = './static/texts.txt'
images_name_path = './static/imageName.txt'

# Save texts to the file, each text on a new line
with open(texts_file_path, 'w', encoding='utf-8') as file:
    for text in texts:
        file.write(f"{text}\n")

with open(images_name_path, 'w', encoding='utf-8') as file:
    for text in names_image:
        file.write(f"{text}\n")



# # Initialize an empty list to hold the loaded texts
# loaded_texts = []

# # Load the texts from the file, each text on a new line
# with open(texts_file_path, 'r', encoding='utf-8') as file:
#     for line in file:
#         # Strip the newline character and append to the list
#         loaded_texts.append(line.rstrip('\n'))

# # Assuming you want to see the loaded data
# print(len(loaded_texts))
# setting device on GPU if available, else CPU
        
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

#Additional Info when using cuda
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))  
   
from lavis.models import load_model_and_preprocess
model, vis_processors, txt_processors = load_model_and_preprocess(name="blip2_feature_extractor", model_type="coco", is_eval=True, device=device)
    

images_embedding = None
texts_embedding = None
multimodal_embedding = None

# iter over the list of text and images

for text, image in zip(texts,images):
    image_processed = vis_processors["eval"](image).unsqueeze(0).to(device)
    text_translated = text
    text_input = txt_processors["eval"](text_translated)
    sample = {"image": image_processed, "text_input": [text_input]}

    multimodal_emb = model.extract_features(sample).multimodal_embeds[:,0,:] # size (1, 768)
    image_emb = model.extract_features(sample, mode="image").image_embeds[:,0,:] # size (1, 768)
    text_emb = model.extract_features(sample, mode="text").text_embeds[:,0,:] # size (1, 768)

    # stack tensor
    if images_embedding is None:
        images_embedding = image_emb
    else:
       images_embedding = torch.cat((images_embedding, image_emb),0)

    if texts_embedding is None:
        texts_embedding = text_emb
    else:
       texts_embedding = torch.cat((texts_embedding, text_emb),0)

    if multimodal_embedding is None:
        multimodal_embedding = multimodal_emb
    else:
       multimodal_embedding = torch.cat((multimodal_embedding, multimodal_emb),0)

print('images_embedding.size(): ', images_embedding.size())
print('texts_embedding.size(): ', texts_embedding.size())
print('multimodal_embedding.size(): ', multimodal_embedding.size())

save_path_img = './static/images_embedding.pt'
save_path_text = './static/texts_embedding.pt'
save_path_multimodal = './static/multimodal_embedding.pt'

torch.save(images_embedding, save_path_img)
torch.save(texts_embedding, save_path_text)
torch.save(multimodal_embedding, save_path_multimodal)

loaded_images_embedding = torch.load(save_path_img)
print(f'loaded_images_embedding {loaded_images_embedding.size()}')
loaded_texts_embedding = torch.load(save_path_text)
print(f'loaded_texts_embedding {loaded_texts_embedding.size()}')
loaded_multimodal_embedding = torch.load(save_path_multimodal)
print(f'loaded_multimodal_embedding {loaded_multimodal_embedding.size()}')