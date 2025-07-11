import os
import json
import torch
import numpy as np
import torchvision.transforms as T
import torchvision.models as models
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from tqdm import tqdm

# --- CONFIG --- #
DATASET_PATH = "detections"
VIEWS = ["broadcast", "tacticam"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- Load Pretrained Model (ResNet50) for Embedding --- #
def load_model():
    model = models.resnet50(pretrained=True)
    model = torch.nn.Sequential(*list(model.children())[:-1])  # Remove FC layer
    model.eval().to(device)
    return model


# --- Preprocess image for ResNet --- #
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# --- Extract embeddings for each crop --- #
def extract_embeddings(view, model):
    crop_dir = os.path.join(DATASET_PATH, view, "crops")
    meta_path = os.path.join(DATASET_PATH, view, "tracking_metadata.json")
    
    with open(meta_path, 'r') as f:
        metadata = json.load(f)

    track_embeddings = defaultdict(list)

    for key, info in tqdm(metadata.items(), desc=f"Processing {view} crops"):
        frame = info['frame']
        track_id = str(info['track_id'])
        crop_file = f"{frame}_{track_id}.jpg"
        crop_path = os.path.join(crop_dir, crop_file)

        if not os.path.exists(crop_path):
            continue

        try:
            image = Image.open(crop_path).convert("RGB")
        except:
            continue

        input_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            embedding = model(input_tensor).squeeze().cpu().numpy()

        track_embeddings[track_id].append(embedding)

    # Average embeddings per track
    avg_embeddings = {
        tid: np.mean(np.stack(embs), axis=0)
        for tid, embs in track_embeddings.items() if len(embs) > 0
    }

    return avg_embeddings


# --- Match tacticam tracks to broadcast tracks --- #
def match_tracks(embeddings_tacticam, embeddings_broadcast, threshold=0.5):
    matched = {}
    used_broadcast_ids = set()

    for tid_tac, emb_tac in embeddings_tacticam.items():
        best_id = None
        best_score = -1

        for tid_brd, emb_brd in embeddings_broadcast.items():
            if tid_brd in used_broadcast_ids:
                continue

            score = cosine_similarity([emb_tac], [emb_brd])[0][0]
            if score > best_score:
                best_score = score
                best_id = tid_brd

        if best_score >= threshold:
            matched[tid_tac] = best_id
            used_broadcast_ids.add(best_id)
        else:
            matched[tid_tac] = None  # No good match

    return matched


# --- Save player_id mapping --- #
def save_mapping(mapping, output_file="player_mapping.json"):
    with open(output_file, 'w') as f:
        json.dump({"tacticam_to_broadcast": mapping}, f, indent=2)
    print(f"Player mapping saved to {output_file}")


# --- Main Execution --- #
def main():
    print("Loading model...")
    model = load_model()

    print("Extracting embeddings...")
    emb_brd = extract_embeddings("broadcast", model)
    emb_tac = extract_embeddings("tacticam", model)

    print("Matching players...")
    mapping = match_tracks(emb_tac, emb_brd, threshold=0.6)

    save_mapping(mapping)


if __name__ == "__main__":
    main()
