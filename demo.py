import argparse
import os
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
SCORE_THRESHOLD = 0.75

os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT / ".cache"))

import cv2
import numpy as np
from PIL import Image
from pycocotools import mask as coco_mask
from scipy.sparse.csgraph import connected_components
import torch
from torch.nn import functional as F
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor


def get_masks(output, size):
    logits = output.masks_queries_logits[0].float()
    scores = output.class_queries_logits[0].float().softmax(dim=-1)[:, 0]
    foreground = logits > 0
    pixels = foreground.sum(dim=(1, 2))
    scores = scores * (logits.sigmoid() * foreground).sum(dim=(1, 2)) / pixels.clamp_min(1)

    masks = []
    for logit in logits[scores >= SCORE_THRESHOLD]:
        resized = F.interpolate(
            logit[None, None], size=size, mode="bilinear", align_corners=False
        )[0, 0]
        mask = coco_mask.encode(np.asfortranarray((resized > 0).cpu().numpy(), dtype=np.uint8))
        if coco_mask.area(mask):
            masks.append(mask)
    return masks


def merge_masks(masks):
    if not masks:
        return []

    overlap = coco_mask.iou(masks, masks, [0] * len(masks)) > 0
    count, labels = connected_components(overlap, directed=False)
    groups = [[] for _ in range(count)]
    for mask, label in zip(masks, labels):
        groups[label].append(mask)
    return [coco_mask.merge(group) for group in groups]


def main():
    parser = argparse.ArgumentParser(description="Mask2Former demo")
    parser.add_argument("image", type=Path, help="Input image")
    parser.add_argument("--model", type=Path, default=PROJECT / "best", help="Model directory")
    parser.add_argument("--output", type=Path, help="Output PNG")
    args = parser.parse_args()

    source = args.image.resolve()
    output = (args.output or PROJECT / "demo_outputs" / f"{source.stem}_contour.png").resolve()
    with Image.open(source) as original:
        image = original.convert("RGB")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = Mask2FormerImageProcessor.from_pretrained(args.model, local_files_only=True)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        args.model, local_files_only=True
    ).to(device).eval()
    inputs = processor(images=image, return_tensors="pt").to(device)
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    with torch.inference_mode(), torch.autocast(
        device.type, dtype=torch.bfloat16, enabled=use_bf16
    ):
        result = model(**inputs)

    masks = merge_masks(get_masks(result, (image.height, image.width)))
    overlay = np.array(image)
    for mask in masks:
        contours, _ = cv2.findContours(
            coco_mask.decode(mask), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(overlay, contours, -1, (255, 0, 0), thickness=3)

    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overlay).save(output)
    print(f"Instances: {len(masks)}")
    print(f"Saved to {output}")


if __name__ == "__main__":
    main()
