import os
import hashlib
import numpy as np
import torch

try:
    import clip
except ImportError:
    clip = None

CLASS_PROMPTS = {
    "smoke": [
        "a person smoking a cigarette",
        "someone holding and smoking tobacco",
        "smoke coming from a cigarette or pipe",
    ],
    "blood": [
        "blood on the ground, bloody scene, gore",
        "a person bleeding from an injury",
        "graphic medical scene with blood",
    ],
    "violent": [
        "people fighting and hitting each other",
        "physical altercation and violence",
        "riot or street fight scene",
    ],
    "abusive": [
        "person making aggressive gestures",
        "verbal harassment and threatening behavior",
        "someone using abusive language or gestures",
    ],
    "sexy": [
        "sexually suggestive content and exposure",
        "inappropriate revealing clothing or acts",
        "explicit adult content",
    ],
    "money": [
        "displaying large amounts of cash suspiciously",
        "gambling or scam related content",
        "fraudulent money scheme promotion",
    ],
    "policy": [
        "politically sensitive content and symbols",
        "unauthorized political commentary",
        "political slander or misinformation",
    ],
}


class CLIPFeatureExtractor:
    def __init__(self, model_name="ViT-B/16", device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval()
        if self.device.type == "cuda":
            self.model = self.model.float()

    def extract_visual_features(self, frames, batch_size=32):
        features = self._batch_extract(frames, batch_size)
        return features

    def extract_text_features(self, text_prompts):
        tokens = clip.tokenize(text_prompts).to(self.device)
        with torch.no_grad():
            # 本地 CLIP.encode_text 需双参数: text=token_embedding, token=原始 tokens
            token_embedding = self.model.encode_token(tokens)
            text_features = self.model.encode_text(token_embedding, tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features.cpu().numpy()

    def compute_similarity(self, visual_feats, text_feats):
        visual_feats = visual_feats / np.linalg.norm(visual_feats, axis=-1, keepdims=True)
        text_feats = text_feats / np.linalg.norm(text_feats, axis=-1, keepdims=True)
        similarity = visual_feats @ text_feats.T
        return similarity

    def _batch_extract(self, frames, batch_size):
        from PIL import Image

        all_features = []
        num_frames = frames.shape[0]
        for start in range(0, num_frames, batch_size):
            end = min(start + batch_size, num_frames)
            batch_frames = frames[start:end]
            batch_tensors = []
            for frame in batch_frames:
                image = Image.fromarray(frame)
                tensor = self.preprocess(image)
                batch_tensors.append(tensor)
            batch_tensor = torch.stack(batch_tensors).to(self.device)
            with torch.no_grad():
                batch_features = self.model.encode_image(batch_tensor)
                batch_features = batch_features / batch_features.norm(dim=-1, keepdim=True)
            all_features.append(batch_features.cpu().numpy())
        return np.concatenate(all_features, axis=0)

    def _cache_features(self, features, cache_path):
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.save(cache_path, features)

    def _load_cache(self, cache_path, source_hash):
        if not os.path.exists(cache_path):
            return None
        try:
            data = np.load(cache_path, allow_pickle=False)
            hash_path = cache_path + ".hash"
            if os.path.exists(hash_path):
                with open(hash_path, "r") as f:
                    cached_hash = f.read().strip()
                if cached_hash != source_hash:
                    return None
            else:
                return None
            return data
        except Exception:
            return None

    def _compute_file_hash(self, file_path):
        hasher = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
