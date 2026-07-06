#!/usr/bin/env python
import gzip
import io
import json
import os
import tarfile
import textwrap

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from ocl.cli import train
from omegaconf import OmegaConf
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer, CLIPTextModelWithProjection


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RUN_DIR = os.path.join(
    REPO_ROOT,
    "outputs/projects/prompting/vg/"
    "prompt_vg_small14_dinov2_mapping_lang_point_pred_sep_clip/"
    "2026-07-03_18-43-50_clip_layer2_tuned_5k_bs128_amp",
)
CONFIG_PATH = os.environ.get("CTRLO_CONFIG_PATH", os.path.join(RUN_DIR, "config/config.yaml"))
CHECKPOINT_PATH = os.environ.get(
    "CTRLO_CHECKPOINT_PATH", os.path.join(RUN_DIR, "checkpoints/epoch=1-step=5000.ckpt")
)
VAL_SHARD = os.environ.get(
    "CTRLO_VG_VAL_SHARD",
    os.path.join(REPO_ROOT, "scripts/datasets/outputs/vg_disjoint_clip/val/shard-000000.tar"),
)
OUTPUT_IMAGE = os.environ.get(
    "CTRLO_OUTPUT_IMAGE",
    os.path.join(REPO_ROOT, "_001n_my/_008n_clip_layer2_5k_qual/vg_val_clip_5k.png"),
)
CLIP_MODEL = os.environ.get("CTRLO_CLIP_MODEL", "openai/clip-vit-base-patch32")
N_SAMPLES = int(os.environ.get("CTRLO_QUAL_N_SAMPLES", "4"))
NUM_SLOTS = int(os.environ.get("CTRLO_NUM_SLOTS", "7"))


def load_val_samples(shard_path, n_samples):
    samples = []
    with tarfile.open(shard_path) as tar:
        keys = sorted({name.split(".")[0] for name in tar.getnames()})
        for key in keys[:n_samples]:
            image = np.load(io.BytesIO(gzip.decompress(tar.extractfile(f"{key}.image.npy.gz").read())))
            names = json.loads(tar.extractfile(f"{key}.name.json").read())
            mask = np.load(
                io.BytesIO(gzip.decompress(tar.extractfile(f"{key}.instance_mask.npy.gz").read()))
            )
            centroids = np.load(io.BytesIO(tar.extractfile(f"{key}.bbox_centroids.npy").read()))
            name_embedding = np.load(
                io.BytesIO(gzip.decompress(tar.extractfile(f"{key}.name_embedding.npy.gz").read()))
            ).astype("float32")
            samples.append(
                {
                    "key": key,
                    "image": image,
                    "names": names,
                    "mask": mask,
                    "centroids": centroids,
                    "name_embedding": name_embedding,
                }
            )
    return samples


class ClipTextEncoder:
    def __init__(self, model_name, device):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = CLIPTextModelWithProjection.from_pretrained(model_name).to(device)
        self.model.eval()

    @torch.no_grad()
    def encode(self, prompts):
        inputs = self.tokenizer(prompts, padding=True, truncation=True, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        embeddings = self.model(**inputs).text_embeds.float()
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        valid_mask = torch.tensor(
            [prompt != "other" for prompt in prompts],
            dtype=embeddings.dtype,
            device=embeddings.device,
        ).unsqueeze(-1)
        return embeddings * valid_mask


def pad_list(values, fill, length):
    values = list(values[:length])
    return values + [fill] * (length - len(values))


def colorize_label_mask(mask):
    palette = np.array(
        [
            [0, 0, 0],
            [230, 57, 70],
            [42, 157, 143],
            [69, 123, 157],
            [233, 196, 106],
            [155, 93, 229],
            [244, 162, 97],
            [46, 204, 113],
            [231, 111, 81],
            [91, 192, 190],
            [247, 37, 133],
        ],
        dtype=np.float32,
    )
    colored = palette[np.asarray(mask, dtype=np.int64) % len(palette)] / 255.0
    return colored


def make_overlay(image, mask, color):
    image_float = image.astype(np.float32) / 255.0
    color_arr = np.zeros_like(image_float)
    for channel, value in enumerate(color):
        color_arr[:, :, channel] = mask * value
    alpha = 0.45
    return np.clip(image_float * (1.0 - alpha) + color_arr * alpha, 0.0, 1.0)


@torch.no_grad()
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(os.path.dirname(OUTPUT_IMAGE), exist_ok=True)

    config = OmegaConf.load(CONFIG_PATH)
    model = train.build_model_from_config(config, CHECKPOINT_PATH).to(device)
    model.eval()

    image_transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((224, 224), interpolation=torchvision.transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    samples = load_val_samples(VAL_SHARD, N_SAMPLES)
    images = torch.stack([image_transform(sample["image"]) for sample in samples]).to(device)

    padded_prompts = [pad_list(sample["names"], "other", NUM_SLOTS) for sample in samples]
    name_embeddings = torch.tensor(
        [
            pad_list(sample["name_embedding"].tolist(), [0.0] * 512, NUM_SLOTS)
            for sample in samples
        ],
        dtype=torch.float32,
        device=device,
    )
    contrastive_loss_mask = torch.tensor(
        [[1 if prompt != "other" else 0 for prompt in prompts] for prompts in padded_prompts],
        dtype=torch.long,
        device=device,
    )
    bbox_centroids = torch.tensor(
        [pad_list(sample["centroids"].tolist(), [-1, -1], NUM_SLOTS) for sample in samples],
        dtype=torch.float32,
        device=device,
    )
    instance_bbox = torch.full((len(samples), NUM_SLOTS, 4), -1.0, dtype=torch.float32, device=device)

    inputs = {
        "image": images,
        "bbox_centroids": bbox_centroids,
        "contrastive_loss_mask": contrastive_loss_mask,
        "name_embedding": name_embeddings,
        "instance_bbox": instance_bbox,
        "batch_size": len(samples),
    }
    outputs = model(inputs)
    masks_as_image = outputs["object_decoder"].masks_as_image.detach().cpu()

    colors = [
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (1.0, 1.0, 0.0),
        (1.0, 0.0, 1.0),
        (0.0, 1.0, 1.0),
        (0.75, 0.75, 0.75),
    ]

    fig, axes = plt.subplots(len(samples), NUM_SLOTS + 2, figsize=(3.0 * (NUM_SLOTS + 2), 3.2 * len(samples)))
    if len(samples) == 1:
        axes = axes[None, :]

    for row, sample in enumerate(samples):
        image = sample["image"]
        axes[row, 0].imshow(image)
        axes[row, 0].set_title(f"id {sample['key']}\\nimage", fontsize=9)
        axes[row, 0].axis("off")

        gt_overlay = 0.55 * image.astype(np.float32) / 255.0 + 0.45 * colorize_label_mask(sample["mask"])
        axes[row, 1].imshow(np.clip(gt_overlay, 0.0, 1.0))
        axes[row, 1].set_title("GT mask", fontsize=9)
        axes[row, 1].axis("off")

        pred_masks = masks_as_image[row].view(NUM_SLOTS, 1, 224, 224).squeeze(1).numpy()
        image_224 = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)
        for slot_idx in range(NUM_SLOTS):
            overlay = make_overlay(image_224, pred_masks[slot_idx], colors[slot_idx % len(colors)])
            title = "" if padded_prompts[row][slot_idx] == "other" else textwrap.fill(padded_prompts[row][slot_idx], 18)
            axes[row, slot_idx + 2].imshow(overlay)
            axes[row, slot_idx + 2].set_title(title, fontsize=8)
            axes[row, slot_idx + 2].axis("off")

    plt.tight_layout()
    fig.savefig(OUTPUT_IMAGE, dpi=180)
    plt.close(fig)
    print(f"Saved qualitative VG validation visualization to {OUTPUT_IMAGE}")


if __name__ == "__main__":
    main()
