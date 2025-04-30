# ==========================
# 📦 라이브러리 Import
# ==========================

import torch
import torch.nn.functional as F
from transformers import (
    CLIPProcessor, CLIPModel,
    BlipProcessor, BlipForConditionalGeneration,
    ViltProcessor, ViltForQuestionAnswering
)
from PIL import Image
import json
import os
from tqdm import tqdm

# ==========================
# ⚙️ 모델 로딩
# ==========================

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model.eval()

caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
caption_model.eval()

vqa_model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
vqa_processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
vqa_model.eval()

# ==========================
# 📂 데이터 경로 설정
# ==========================

COCO_IMAGE_DIR = "./val2017"
COCO_ANN_PATH = "annotations_trainval2017/annotations/captions_val2017.json"

# 데이터 다운로드
# https://cocodataset.org/#download
# 2017 Val images [5K/1GB]
# 2017 Train/Val annotations [241MB]

# ==========================
# 📚 캡션 로딩
# ==========================

with open(COCO_ANN_PATH, 'r') as f:
    coco_data = json.load(f)

from collections import defaultdict
id_to_captions = defaultdict(list)
for ann in coco_data['annotations']:
    id_to_captions[ann['image_id']].append(ann['caption'])

id_to_filename = {img['id']: img['file_name'] for img in coco_data['images']}

# ==========================
# 🔍 Image-Text Retrieval (CLIP)
# ==========================

def encode_image_text(image, text):
    inputs = clip_processor(text=[text], images=image, return_tensors="pt", padding=True)
    outputs = clip_model(**inputs)
    return outputs.image_embeds, outputs.text_embeds

def cosine_similarity(img_embeds, text_embeds):
    img_embeds = F.normalize(img_embeds, dim=-1)
    text_embeds = F.normalize(text_embeds, dim=-1)
    return torch.matmul(img_embeds, text_embeds.T)

def compute_recall(similarity_matrix, k):
    correct = 0
    for i in range(similarity_matrix.size(0)):
        values, indices = similarity_matrix[i].topk(k)
        if i in indices:
            correct += 1
    return correct / similarity_matrix.size(0)

# ▶ 유효한 이미지 ID만 필터링
valid_ids = []
for img_id in id_to_captions.keys():
    if img_id in id_to_filename:
        file_path = os.path.join(COCO_IMAGE_DIR, id_to_filename[img_id])
        if os.path.exists(file_path):
            valid_ids.append(img_id)

selected_ids = valid_ids[:100]
image_embeds_list = []
text_embeds_list = []

for img_id in tqdm(selected_ids, desc="Encoding images and captions"):
    file_path = os.path.join(COCO_IMAGE_DIR, id_to_filename[img_id])
    image = Image.open(file_path).convert("RGB")
    caption = id_to_captions[img_id][0]
    img_embed, txt_embed = encode_image_text(image, caption)
    image_embeds_list.append(img_embed)
    text_embeds_list.append(txt_embed)

if len(image_embeds_list) == 0 or len(text_embeds_list) == 0:
    print("❗ No valid image-caption pairs found. Check val2017 image folder or selected IDs.")
    exit()

image_embeds_all = torch.cat(image_embeds_list, dim=0)
text_embeds_all = torch.cat(text_embeds_list, dim=0)

sim_matrix = cosine_similarity(image_embeds_all, text_embeds_all)
recall1 = compute_recall(sim_matrix, 1)
recall5 = compute_recall(sim_matrix, 5)
recall10 = compute_recall(sim_matrix, 10)

print(f"\n[Image-Text Retrieval Results - COCO]")
print(f"Recall@1: {recall1:.4f}")
print(f"Recall@5: {recall5:.4f}")
print(f"Recall@10: {recall10:.4f}")

# ==========================
# 📝 Image Captioning (BLIP) + 이미지 show()
# ==========================

def generate_caption(image):
    inputs = caption_processor(image, return_tensors="pt")
    out = caption_model.generate(**inputs)
    caption = caption_processor.decode(out[0], skip_special_tokens=True)
    return caption

print("\n[Image Captioning Results - COCO]")
for img_id in selected_ids[:5]:
    file_name = id_to_filename[img_id]
    file_path = os.path.join(COCO_IMAGE_DIR, file_name)
    image = Image.open(file_path).convert("RGB")
    gen_caption = generate_caption(image)
    print(f"Image: {file_name} | Generated Caption: {gen_caption}")
    image.show(title=f"{file_name}")

# ==========================
# # ❓ Visual Question Answering (VQA) [데이터셋 링크 이상으로 보류]
# # ==========================

# vqa_data = load_dataset("HuggingFaceM4/VQAv2", split="validation", trust_remote_code=True)

# print("\n[VQA Results - COCO + HuggingFace vqa]")
# for sample in vqa_data.select(range(5)):
#     image_id = sample["image_id"]
#     question = sample["question"]
#     answer = sample["answer"]

#     file_name = f"{image_id:012d}.jpg"
#     file_path = os.path.join(COCO_IMAGE_DIR, file_name)
#     if not os.path.exists(file_path):
#         print(f"Image not found: {file_name}")
#         continue

#     image = Image.open(file_path).convert("RGB")
#     encoding = vqa_processor(image, question, return_tensors="pt")
#     output = vqa_model(**encoding)
#     pred_id = output.logits.argmax(-1).item()
#     pred_answer = vqa_model.config.id2label[pred_id]

#     print(f"Q: {question}\nPredicted: {pred_answer} | Ground Truth: {answer}\n")

# ==========================
# ❓ Custom VQA 실험
# ==========================

print("\n[Custom VQA Results - ViLT]")
custom_vqa_samples = [
    {"file": "000000179765.jpg", "question": "What is the vehicle in this image?"},
    {"file": "000000190236.jpg", "question": "What shape is shown in the picture?"},
    {"file": "000000331352.jpg", "question": "Is this a bathroom?"},
    {"file": "000000517069.jpg", "question": "Where is the woman sitting?"},
    {"file": "000000182417.jpg", "question": "Is there food on the table?"}
]

for sample in custom_vqa_samples:
    file_path = os.path.join(COCO_IMAGE_DIR, sample["file"])
    if not os.path.exists(file_path):
        print(f"Image not found: {sample['file']}")
        continue
    image = Image.open(file_path).convert("RGB")
    inputs = vqa_processor(image, sample["question"], return_tensors="pt")
    outputs = vqa_model(**inputs)
    pred_id = outputs.logits.argmax(-1).item()
    pred_answer = vqa_model.config.id2label[pred_id]
    print(f"Image: {sample['file']}\nQ: {sample['question']}\nA: {pred_answer}\n")