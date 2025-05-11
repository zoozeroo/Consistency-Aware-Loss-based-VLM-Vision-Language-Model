import json
import re
from collections import defaultdict
import spacy
import inflect
import os

# 필수 라이브러리 준비
nlp = spacy.load("en_core_web_sm")  # 사전 설치 필요
p = inflect.engine()

# 경로 설정 (환경에 맞게 수정)
captions_path = "annotations/captions_train2017.json"
instances_path = "annotations/instances_train2017.json"
output_path = "coco_token_bbox_matched.json"

# Load JSON 파일
with open(captions_path, "r") as f:
    captions_data = json.load(f)
with open(instances_path, "r") as f:
    instances_data = json.load(f)

# 카테고리 매핑
category_name_to_id = {cat["name"]: cat["id"] for cat in instances_data["categories"]}
category_id_to_name = {cat["id"]: cat["name"] for cat in instances_data["categories"]}
category_names = set(category_name_to_id.keys())

# 이미지 ID → caption 리스트
image_id_to_captions = defaultdict(list)
for ann in captions_data["annotations"]:
    image_id_to_captions[ann["image_id"]].append(ann["caption"])

# 이미지 ID → bbox들 (category_id 포함)
image_id_to_bboxes = defaultdict(list)
for ann in instances_data["annotations"]:
    image_id_to_bboxes[ann["image_id"]].append({
        "category_id": ann["category_id"],
        "bbox": ann["bbox"]
    })

# spaCy + inflect로 명사만 추출 + 단수화
def extract_normalized_nouns(captions):
    found_tokens = set()
    for cap in captions:
        doc = nlp(cap)
        for token in doc:
            if token.pos_ == "NOUN":
                word = token.text.lower()
                singular = p.singular_noun(word) if p.singular_noun(word) else word
                found_tokens.add(singular)
    return found_tokens

# 최종 결과 생성
final_results = []
matched_count = 0
unmatched_count = 0

for image_id, captions in image_id_to_captions.items():
    found_tokens = extract_normalized_nouns(captions)

    matched_entries = []
    for bbox_entry in image_id_to_bboxes.get(image_id, []):
        category_id = bbox_entry["category_id"]
        label = category_id_to_name[category_id].lower()
        label_singular = p.singular_noun(label) if p.singular_noun(label) else label
        if label_singular in found_tokens:
            matched_entries.append({
                "token": label_singular,
                "bbox": bbox_entry["bbox"],
                "label": label
            })

    if matched_entries:
        matched_count += 1
    else:
        unmatched_count += 1

    final_results.append({
        "image_id": image_id,
        "captions": captions,
        "matches": matched_entries
    })

# 결과 저장
with open(output_path, "w") as f:
    json.dump(final_results, f, indent=2)

# 요약 출력
total = matched_count + unmatched_count
print("총 이미지 수:", total)
print(f"매칭된 이미지 수: {matched_count} ({matched_count/total:.2%})")
print(f"매칭 안된 이미지 수: {unmatched_count} ({unmatched_count/total:.2%})")
print(f"결과 저장 위치: {output_path}")
