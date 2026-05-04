"""
Image generation tool - Generate images from text prompts.

This tool prefers the model profile configured in default_model_image_generation.
If no dedicated profile is configured, it falls back to the current bot's create_img
implementation when available.
"""

import base64
import os
import uuid
from typing import Any, Dict, Optional, Tuple

import requests

from agent.tools.base_tool import BaseTool, ToolResult
from common.log import logger
from config import conf

DEFAULT_IMAGE_GENERATION_PROFILE_KEY = "default_model_image_generation"
DEFAULT_TIMEOUT = 90


class ImageGenerate(BaseTool):
    """Generate images from text prompts"""

    name: str = "image_generate"
    description: str = (
        "Generate images from a text prompt, or edit an input image with a prompt. "
        "Use text-to-image when only `prompt` is provided. Use image-to-image when `image` is also provided. "
        "The `image` input can be a URL, a local file path, or a data URL in the format "
        "data:image/<type>;base64,<base64_data>. The output will be saved to the path in `output_path` "
        "when provided, otherwise it is stored under the workspace tmp/output directory with an auto-generated name."
    )

    params: dict = {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": "Text prompt describing the image to generate or how to edit the input image",
            },
            "size": {
                "type": "string",
                "description": "Optional image size (e.g. 1024x1024, 2K)",
            },
            "response_format": {
                "type": "string",
                "description": "Optional response format: url or b64_json (default: url)",
            },
            "image": {
                "type": "string",
                "description": "Optional input image for image-to-image. Supports URL, local file path, or data URL (data:image/<type>;base64,<base64_data>)",
            },
            "output_format": {
                "type": "string",
                "description": "Optional output image format for image-to-image, such as png or jpeg (default: png)",
            },
            "watermark": {
                "type": "boolean",
                "description": "Whether to add a watermark in image-to-image mode (default: false)",
            },
            "output_path": {
                "type": "string",
                "description": "Optional output file path, relative to the agent workspace or absolute. If omitted, the image is saved under workspace tmp/output with an auto-generated name.",
            },
            "save_local": {
                "type": "boolean",
                "description": "Whether to save the generated image locally and return a file_to_send payload (default: true)",
            },
        },
        "required": ["prompt"],
    }

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        self.cwd = self.config.get("cwd", os.getcwd())

    def execute(self, args: Dict[str, Any]) -> ToolResult:
        prompt = (args.get("prompt") or "").strip()
        if not prompt:
            return ToolResult.fail("Error: 'prompt' parameter is required")

        size = (args.get("size") or "").strip()
        response_format = (args.get("response_format") or "url").strip().lower()
        if response_format not in ("url", "b64_json"):
            response_format = "url"
        image = (args.get("image") or "").strip()
        output_format = (args.get("output_format") or "png").strip().lower() or "png"
        watermark = bool(args.get("watermark", False))
        save_local = args.get("save_local", True)
        output_path = (args.get("output_path") or "").strip()

        profile, profile_error = self._resolve_preferred_profile()
        if profile_error:
            return ToolResult.fail(profile_error)

        image_input = None
        if image:
            image_input, image_error = self._normalize_image_input(image)
            if image_error:
                return ToolResult.fail(image_error)

        try:
            if profile:
                result = self._generate_via_profile(
                    profile=profile,
                    prompt=prompt,
                    size=size,
                    response_format=response_format,
                    image=image_input,
                    output_format=output_format,
                    watermark=watermark,
                )
            else:
                if image_input:
                    return ToolResult.fail(
                        "Error: image-to-image requires an image generation profile; current bot fallback only supports text-to-image"
                    )
                result = self._generate_via_bot_fallback(prompt=prompt)
        except requests.Timeout:
            return ToolResult.fail(f"Error: Image generation timed out after {DEFAULT_TIMEOUT}s")
        except requests.ConnectionError:
            return ToolResult.fail("Error: Failed to connect to image generation API")
        except Exception as e:
            logger.error(f"[ImageGenerate] Unexpected error: {e}", exc_info=True)
            return ToolResult.fail(f"Error: Image generation failed - {str(e)}")

        if not result.get("ok"):
            return ToolResult.fail(result.get("error") or "Error: Image generation failed")

        image_url = result.get("image_url")
        image_b64 = result.get("image_b64")
        model = result.get("model")
        provider = result.get("provider")

        if save_local:
            local_path = None
            if image_url:
                local_path = self._download_image(image_url, output_path)
            elif image_b64:
                local_path = self._save_b64_image(image_b64, output_path)

            if local_path:
                file_size = os.path.getsize(local_path)
                file_name = os.path.basename(local_path)
                mime_type = self._mime_type_from_path(local_path)
                return ToolResult.success({
                    "type": "file_to_send",
                    "file_type": "image",
                    "path": local_path,
                    "file_name": file_name,
                    "mime_type": mime_type,
                    "size": file_size,
                    "size_formatted": self._format_size(file_size),
                    "message": "已生成图片，正在发送",
                    "model": model,
                    "provider": provider,
                    "image_url": image_url,
                })

        return ToolResult.success({
            "model": model,
            "provider": provider,
            "image_url": image_url,
            "image_b64": image_b64,
        })

    def _resolve_preferred_profile(self) -> Tuple[Optional[dict], Optional[str]]:
        profile_id = (conf().get(DEFAULT_IMAGE_GENERATION_PROFILE_KEY, "") or "").strip()
        if not profile_id:
            return None, None

        custom_models = conf().get("custom_models", []) or []
        if not isinstance(custom_models, list):
            custom_models = []
        profile = next((m for m in custom_models if m.get("id") == profile_id), None)
        if not profile:
            return None, (
                f"Error: '{DEFAULT_IMAGE_GENERATION_PROFILE_KEY}' is set to '{profile_id}', "
                "but this model profile does not exist."
            )

        model_id = (profile.get("model") or "").strip()
        api_key = (profile.get("api_key") or "").strip()
        api_base = (profile.get("api_base") or "").strip().rstrip("/")
        profile_name = profile.get("name") or profile_id

        if not model_id:
            return None, f"Error: Image generation profile '{profile_name}' is missing model field"
        if not api_key:
            return None, f"Error: Image generation profile '{profile_name}' is missing api_key field"
        if not api_base:
            return None, f"Error: Image generation profile '{profile_name}' is missing api_base field"

        return {
            "id": profile_id,
            "name": profile_name,
            "model": model_id,
            "api_key": api_key,
            "api_base": api_base,
        }, None

    def _generate_via_profile(
        self,
        profile: dict,
        prompt: str,
        size: str,
        response_format: str,
        image: Optional[str] = None,
        output_format: str = "png",
        watermark: bool = False,
    ) -> dict:
        payload = {
            "model": profile["model"],
            "prompt": prompt,
            "response_format": response_format,
        }
        if size:
            payload["size"] = size
        if image:
            payload["image"] = image
            # payload["output_format"] = output_format or "png"
            payload["watermark"] = watermark

        headers = {
            "Authorization": f"Bearer {profile['api_key']}",
            "Content-Type": "application/json",
        }

        resp = requests.post(
            f"{profile['api_base']}/images/generations",
            headers=headers,
            json=payload,
            timeout=DEFAULT_TIMEOUT,
        )

        if resp.status_code != 200:
            return {
                "ok": False,
                "error": f"HTTP {resp.status_code}: {resp.text[:300]}",
            }

        data = resp.json()
        items = data.get("data") or []
        if not items:
            return {"ok": False, "error": "Image API returned empty data"}

        first = items[0] if isinstance(items[0], dict) else {}
        image_url = first.get("url")
        image_b64 = first.get("b64_json")
        if not image_url and not image_b64:
            return {"ok": False, "error": "Image API returned neither url nor b64_json"}

        return {
            "ok": True,
            "model": profile["model"],
            "provider": f"profile:{profile['name']}",
            "image_url": image_url,
            "image_b64": image_b64,
        }

    def _generate_via_bot_fallback(self, prompt: str) -> dict:
        bot = getattr(getattr(self, "model", None), "bot", None)
        if not bot or not hasattr(bot, "create_img"):
            return {
                "ok": False,
                "error": (
                    "No image generation profile configured and current model bot "
                    "does not support create_img."
                ),
            }

        ok, output = bot.create_img(prompt)
        if not ok:
            return {"ok": False, "error": str(output)}

        return {
            "ok": True,
            "model": conf().get("text_to_image", ""),
            "provider": "bot_fallback",
            "image_url": output,
            "image_b64": None,
        }

    def _download_image(self, url: str, output_path: str = "") -> Optional[str]:
        try:
            resp = requests.get(url, timeout=DEFAULT_TIMEOUT)
            resp.raise_for_status()
        except Exception as e:
            logger.warning(f"[ImageGenerate] Failed to download image URL: {e}")
            return None

        content_type = (resp.headers.get("Content-Type") or "image/png").split(";")[0].strip().lower()
        ext_map = {
            "image/jpeg": ".jpg",
            "image/png": ".png",
            "image/webp": ".webp",
            "image/gif": ".gif",
            "image/bmp": ".bmp",
        }
        ext = ext_map.get(content_type, ".png")

        # Resolve output path
        if output_path:
            # User-specified path (relative or absolute)
            if os.path.isabs(output_path):
                file_path = output_path
            else:
                file_path = os.path.abspath(os.path.join(self.cwd, output_path))
        else:
            # Default: workspace_root/tmp/output/image-<uuid>.<ext>
            out_dir = os.path.join(self.cwd, "tmp", "output")
            file_path = os.path.join(out_dir, f"image-{uuid.uuid4().hex}{ext}")

        # Create output directory if needed
        out_dir = os.path.dirname(file_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        with open(file_path, "wb") as f:
            f.write(resp.content)
        return file_path

    def _save_b64_image(self, image_b64: str, output_path: str = "") -> Optional[str]:
        try:
            raw = base64.b64decode(image_b64)
        except Exception as e:
            logger.warning(f"[ImageGenerate] Invalid b64 image: {e}")
            return None

        # Resolve output path
        if output_path:
            # User-specified path (relative or absolute)
            if os.path.isabs(output_path):
                file_path = output_path
            else:
                file_path = os.path.abspath(os.path.join(self.cwd, output_path))
        else:
            # Default: workspace_root/tmp/output/image-<uuid>.png
            out_dir = os.path.join(self.cwd, "tmp", "output")
            file_path = os.path.join(out_dir, f"image-{uuid.uuid4().hex}.png")

        # Create output directory if needed
        out_dir = os.path.dirname(file_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        with open(file_path, "wb") as f:
            f.write(raw)
        return file_path

    @staticmethod
    def _mime_type_from_path(path: str) -> str:
        ext = os.path.splitext(path)[1].lower()
        mime_map = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".webp": "image/webp",
            ".gif": "image/gif",
            ".bmp": "image/bmp",
            ".tiff": "image/tiff",
            ".tif": "image/tiff",
        }
        return mime_map.get(ext, "application/octet-stream")

    def _normalize_image_input(self, image: str) -> Tuple[Optional[str], Optional[str]]:
        image = image.strip()
        if not image:
            return None, None

        if image.startswith("data:image/"):
            if ";base64," not in image:
                return None, "Error: image data URL must use the format data:image/<type>;base64,<base64_data>"
            return image, None

        if image.startswith(("http://", "https://")):
            return image, None

        resolved_path = image
        if not os.path.isabs(resolved_path):
            resolved_path = os.path.abspath(os.path.join(self.cwd, resolved_path))

        if not os.path.exists(resolved_path):
            return None, f"Error: image file not found: {image}"
        if not os.path.isfile(resolved_path):
            return None, f"Error: image path is not a file: {image}"

        ext = os.path.splitext(resolved_path)[1].lower()
        mime_map = {
            ".jpg": "jpeg",
            ".jpeg": "jpeg",
            ".png": "png",
            ".webp": "webp",
            ".gif": "gif",
            ".bmp": "bmp",
            ".tiff": "tiff",
            ".tif": "tiff",
        }
        image_type = mime_map.get(ext)
        if not image_type:
            return None, f"Error: unsupported input image format: {ext or '(no extension)'}"

        try:
            with open(resolved_path, "rb") as f:
                raw = f.read()
            encoded = base64.b64encode(raw).decode("utf-8")
            return f"data:image/{image_type};base64,{encoded}", None
        except Exception as e:
            logger.warning(f"[ImageGenerate] Failed to encode local image: {e}")
            return None, f"Error: failed to read input image: {e}"

    @staticmethod
    def _format_size(size_bytes: int) -> str:
        size = float(size_bytes)
        for unit in ["B", "KB", "MB", "GB"]:
            if size < 1024.0:
                return f"{size:.1f}{unit}"
            size /= 1024.0
        return f"{size:.1f}TB"
