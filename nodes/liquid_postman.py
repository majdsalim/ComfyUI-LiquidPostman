import base64
import io
import os
import re

import numpy as np
import requests
import torch
from PIL import Image

GEMINI_API_BASE = "https://generativelanguage.googleapis.com"

DEFAULT_SYSTEM_PROMPT = (
    "You are an expert image-generation engine. You must ALWAYS produce an image.\n"
    "Interpret all user input—regardless of format, intent, or abstraction—as literal "
    "visual directives for image composition.\n"
    "If a prompt is conversational or lacks specific visual details, you must creatively "
    "invent a concrete visual scenario that depicts the concept.\n"
    "Prioritize generating the visual representation above any text, formatting, or "
    "conversational requests."
)


class LiquidPostmanGeminiImage:
    CATEGORY = "google/gemini"
    FUNCTION = "generate"
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "text")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": ""}),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "model": ("STRING", {"default": "gemini-2.0-flash-preview-image-generation"}),
                "seed": ("INT", {"default": 42, "min": -1, "max": 2**31 - 1}),
                "aspect_ratio": (["1:1", "2:3", "3:2", "3:4", "4:3", "9:16", "16:9", "21:9"],),
                "resolution": (["1K", "2K", "4K"],),
                "response_modalities": (["IMAGE", "IMAGE+TEXT"],),
                "system_prompt": ("STRING", {"default": DEFAULT_SYSTEM_PROMPT, "multiline": True}),
                "search_grounding": ("BOOLEAN", {"default": False}),
                "files": ("STRING", {"default": "", "multiline": True}),
                "debug": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "images": ("IMAGE",),
            },
        }

    # ------------------------------------------------------------------ #
    #  Main entry point
    # ------------------------------------------------------------------ #
    def generate(
        self,
        api_key: str,
        prompt: str,
        model: str,
        seed: int,
        aspect_ratio: str,
        resolution: str,
        response_modalities: str,
        system_prompt: str,
        search_grounding: bool,
        files: str,
        debug: bool,
        images: torch.Tensor | None = None,
    ):
        if not api_key.strip():
            raise ValueError("[Liquid Postman] API key is required. Please provide your Gemini API key.")

        parts = self._build_parts(prompt, images, files, api_key, debug)
        body = self._build_request_body(
            parts, model, seed, aspect_ratio, resolution,
            response_modalities, system_prompt, search_grounding,
        )

        if debug:
            import json
            safe = json.loads(json.dumps(body))
            # Truncate base64 blobs for readability
            for part in safe.get("contents", [{}])[0].get("parts", []):
                if "inlineData" in part:
                    part["inlineData"]["data"] = part["inlineData"]["data"][:60] + "..."
            print(f"[Liquid Postman] Request body:\n{json.dumps(safe, indent=2)}")

        url = f"{GEMINI_API_BASE}/v1beta/models/{model}:generateContent"
        headers = {
            "x-goog-api-key": api_key,
            "Content-Type": "application/json",
        }

        resp = requests.post(url, headers=headers, json=body, timeout=120)
        if resp.status_code != 200:
            msg = resp.text[:500]
            raise RuntimeError(
                f"[Liquid Postman] Gemini API returned {resp.status_code}: {msg}"
            )

        data = resp.json()
        if debug:
            print(f"[Liquid Postman] Response keys: {list(data.keys())}")

        out_images, out_text = self._parse_response(data)

        if not out_images:
            raise RuntimeError(
                "[Liquid Postman] Gemini returned no images. "
                "Try adjusting your prompt or changing response_modalities."
            )

        tensor = self._pil_images_to_tensor(out_images)
        return (tensor, out_text)

    # ------------------------------------------------------------------ #
    #  Build parts array
    # ------------------------------------------------------------------ #
    def _build_parts(self, prompt, images, files_str, api_key, debug):
        parts = []

        # 1) Text prompt
        if prompt.strip():
            parts.append({"text": prompt})

        # 2) Input images → inline base64 PNG
        if images is not None:
            for i in range(images.shape[0]):
                img_np = (images[i].cpu().numpy() * 255).astype(np.uint8)
                pil_img = Image.fromarray(img_np)
                buf = io.BytesIO()
                pil_img.save(buf, format="PNG")
                b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
                parts.append({
                    "inlineData": {
                        "mimeType": "image/png",
                        "data": b64,
                    }
                })
                if debug:
                    print(f"[Liquid Postman] Attached input image {i} ({pil_img.width}x{pil_img.height})")

        # 3) File uploads (PDF / DOCX)
        if files_str.strip():
            paths = re.split(r"[,\n]+", files_str.strip())
            for raw_path in paths:
                p = raw_path.strip()
                if not p:
                    continue
                if not os.path.isfile(p):
                    raise FileNotFoundError(
                        f"[Liquid Postman] File not found: {p}"
                    )
                ext = os.path.splitext(p)[1].lower()
                if ext == ".pdf":
                    mime = "application/pdf"
                elif ext == ".docx":
                    mime = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                else:
                    raise ValueError(
                        f"[Liquid Postman] Unsupported file type '{ext}'. Only .pdf and .docx are allowed."
                    )
                file_uri = self._upload_file(p, mime, api_key, debug)
                parts.append({
                    "fileData": {
                        "fileUri": file_uri,
                        "mimeType": mime,
                    }
                })

        return parts

    # ------------------------------------------------------------------ #
    #  Gemini Files API upload
    # ------------------------------------------------------------------ #
    def _upload_file(self, filepath, mime_type, api_key, debug):
        filename = os.path.basename(filepath)
        file_size = os.path.getsize(filepath)

        if debug:
            print(f"[Liquid Postman] Uploading {filename} ({file_size} bytes, {mime_type})")

        # Step 1: Initiate resumable upload
        init_url = f"{GEMINI_API_BASE}/upload/v1beta/files"
        init_headers = {
            "x-goog-api-key": api_key,
            "X-Goog-Upload-Protocol": "resumable",
            "X-Goog-Upload-Command": "start",
            "X-Goog-Upload-Header-Content-Length": str(file_size),
            "X-Goog-Upload-Header-Content-Type": mime_type,
            "Content-Type": "application/json",
        }
        init_body = {"file": {"display_name": filename}}
        init_resp = requests.post(init_url, headers=init_headers, json=init_body, timeout=60)
        if init_resp.status_code != 200:
            raise RuntimeError(
                f"[Liquid Postman] File upload init failed ({init_resp.status_code}): {init_resp.text[:300]}"
            )

        upload_url = init_resp.headers.get("X-Goog-Upload-URL")
        if not upload_url:
            raise RuntimeError("[Liquid Postman] File upload init did not return an upload URL.")

        # Step 2: Upload the bytes
        with open(filepath, "rb") as f:
            file_bytes = f.read()

        upload_headers = {
            "x-goog-api-key": api_key,
            "X-Goog-Upload-Offset": "0",
            "X-Goog-Upload-Command": "upload, finalize",
            "Content-Length": str(file_size),
        }
        upload_resp = requests.post(upload_url, headers=upload_headers, data=file_bytes, timeout=120)
        if upload_resp.status_code != 200:
            raise RuntimeError(
                f"[Liquid Postman] File upload failed ({upload_resp.status_code}): {upload_resp.text[:300]}"
            )

        file_info = upload_resp.json().get("file", {})
        file_uri = file_info.get("uri", "")
        if not file_uri:
            raise RuntimeError("[Liquid Postman] File upload succeeded but no URI returned.")

        if debug:
            print(f"[Liquid Postman] Uploaded {filename} → {file_uri}")

        return file_uri

    # ------------------------------------------------------------------ #
    #  Build Gemini request body
    # ------------------------------------------------------------------ #
    def _build_request_body(
        self, parts, model, seed, aspect_ratio, resolution,
        response_modalities, system_prompt, search_grounding,
    ):
        modalities = ["TEXT", "IMAGE"] if response_modalities == "IMAGE+TEXT" else ["IMAGE"]

        generation_config = {
            "responseModalities": modalities,
            "imageConfig": {
                "aspectRatio": aspect_ratio,
                "imageSize": resolution,
            },
        }
        if seed >= 0:
            generation_config["seed"] = seed

        body = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": generation_config,
        }

        if system_prompt.strip():
            body["system_instruction"] = {
                "parts": [{"text": system_prompt}]
            }

        if search_grounding:
            body["tools"] = [{"google_search": {}}]

        return body

    # ------------------------------------------------------------------ #
    #  Parse Gemini response
    # ------------------------------------------------------------------ #
    def _parse_response(self, data):
        images = []
        text_parts = []

        for candidate in data.get("candidates", []):
            for part in candidate.get("content", {}).get("parts", []):
                if "text" in part:
                    text_parts.append(part["text"])

                inline = part.get("inlineData") or part.get("inline_data")
                if inline and "data" in inline:
                    img_bytes = base64.b64decode(inline["data"])
                    pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                    images.append(pil_img)

        return images, "\n".join(text_parts)

    # ------------------------------------------------------------------ #
    #  PIL images → ComfyUI tensor
    # ------------------------------------------------------------------ #
    def _pil_images_to_tensor(self, pil_images):
        tensors = []
        for img in pil_images:
            arr = np.array(img).astype(np.float32) / 255.0
            tensors.append(torch.from_numpy(arr))
        return torch.stack(tensors)
