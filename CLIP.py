# ==========================
# 📦 라이브러리 Import
# ==========================

import torch
import torch.nn.functional as F
from transformers import (
    CLIPProcessor, CLIPModel
)
from PIL import Image
import json
import os
from tqdm import tqdm
from evaluate import load as load_metric


# ==========================
# ⚙️ 모델 로딩
# ==========================

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model.eval()

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
text_list = []

for img_id in tqdm(selected_ids, desc="Encoding images and captions"):
    file_path = os.path.join(COCO_IMAGE_DIR, id_to_filename[img_id])
    image = Image.open(file_path).convert("RGB")
    caption = id_to_captions[img_id][0]
    img_embed, txt_embed = encode_image_text(image, caption)
    image_embeds_list.append(img_embed)
    text_embeds_list.append(txt_embed)
    text_list.append(caption)

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
# 📝 간접 방식 Image Captioning 평가 (BLEU)
# ==========================

print("\n[Indirect Captioning Evaluation - BLEU Metric]")
metric = load_metric("bleu")
predictions = []
references = []

for i, img_id in enumerate(selected_ids[:20]):
    pred_caption = text_list[sim_matrix[i].argmax().item()]
    ref_captions = id_to_captions[img_id][:5]
    predictions.append(pred_caption)
    references.append(ref_captions)

score = metric.compute(predictions=predictions, references=references)
print(f"BLEU: {score['bleu']:.4f}")

# ==========================
# 🧠 간접 방식 VQA 평가 예시
# ==========================

print("\n[Indirect VQA Evaluation - Matching Accuracy]")
custom_vqa = [
    {"image": "000000179765.jpg", "question": "What is the vehicle?", "choices": ["motorcycle", "bicycle", "car"]},
    {"image": "000000331352.jpg", "question": "Is this a bathroom?", "choices": ["yes", "no"]},
    {"image": "000000517069.jpg", "question": "Where is the person sitting?", "choices": ["on a bench", "on a sofa", "on the ground"]},
]

correct = 0
for sample in custom_vqa:
    image_path = os.path.join(COCO_IMAGE_DIR, sample["image"])
    image = Image.open(image_path).convert("RGB")
    img_inputs = clip_processor(images=image, return_tensors="pt")
    ques_inputs = clip_processor(text=[sample["question"]], return_tensors="pt")

    img_embed = clip_model.get_image_features(**img_inputs)
    question_embed = clip_model.get_text_features(**ques_inputs)

    choice_embeds = []
    for ans in sample["choices"]:
        ans_embed = clip_model.get_text_features(**clip_processor(text=[ans], return_tensors="pt"))
        choice_embeds.append(ans_embed)

    choice_embeds_all = torch.cat(choice_embeds, dim=0)
    question_embed = F.normalize(question_embed, dim=-1)
    choice_embeds_all = F.normalize(choice_embeds_all, dim=-1)

    sim = torch.matmul(question_embed, choice_embeds_all.T)
    pred_idx = torch.argmax(sim, dim=-1).item()
    pred_answer = sample["choices"][pred_idx]

    print(f"Image: {sample['image']}\nQ: {sample['question']}\nA: {pred_answer}\n")

    if "answer" in sample and pred_answer == sample["answer"]:
        correct += 1

if any("answer" in s for s in custom_vqa):
    acc = correct / sum("answer" in s for s in custom_vqa)
    print(f"Accuracy: {acc:.2%}")
else:
    print("(Ground truth 정답이 포함되어 있지 않아 Accuracy 계산 생략)")