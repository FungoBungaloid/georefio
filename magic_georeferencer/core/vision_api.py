"""
Vision API client for geographic bounding box estimation.

Supports OpenAI and Anthropic vision-capable models for analyzing
images and estimating their geographic location.
"""

import base64
import json
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Tuple, Callable
import io

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


class APIProvider(Enum):
    """Supported API providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


@dataclass
class VisionModel:
    """Vision model configuration."""
    provider: APIProvider
    model_id: str
    display_name: str
    cost_per_image_low: float  # Estimated cost in USD (low end)
    cost_per_image_high: float  # Estimated cost in USD (high end)

    @property
    def avg_cost_per_image(self) -> float:
        """Average estimated cost per image."""
        return (self.cost_per_image_low + self.cost_per_image_high) / 2


# Available vision models
VISION_MODELS = {
    "gpt-5-2025-08-07": VisionModel(
        provider=APIProvider.OPENAI,
        model_id="gpt-5-2025-08-07",
        display_name="GPT-5",
        cost_per_image_low=0.01,
        cost_per_image_high=0.03
    ),
    "gpt-5-nano-2025-08-07": VisionModel(
        provider=APIProvider.OPENAI,
        model_id="gpt-5-nano-2025-08-07",
        display_name="GPT-5 Nano",
        cost_per_image_low=0.001,
        cost_per_image_high=0.005
    ),
    "claude-sonnet-4-5-20250929": VisionModel(
        provider=APIProvider.ANTHROPIC,
        model_id="claude-sonnet-4-5-20250929",
        display_name="Claude Sonnet 4.5",
        cost_per_image_low=0.01,
        cost_per_image_high=0.03
    ),
    "claude-haiku-4-5-20251001": VisionModel(
        provider=APIProvider.ANTHROPIC,
        model_id="claude-haiku-4-5-20251001",
        display_name="Claude Haiku 4.5",
        cost_per_image_low=0.002,
        cost_per_image_high=0.008
    ),
}

# Models grouped by provider for UI
MODELS_BY_PROVIDER = {
    APIProvider.OPENAI: ["gpt-4.1-2025-04-14", "gpt-4.1-nano-2025-04-14"],
    APIProvider.ANTHROPIC: ["claude-sonnet-4-5-20250514", "claude-haiku-4-5-20250514"],
}


# Available basemap sources (must match keys in tile_sources.json)
BASEMAP_OPTIONS = {
    "osm_standard": "OpenStreetMap Standard - best for road maps, city maps, labeled features",
    "esri_world_imagery": "ESRI World Imagery - best for aerial photos, satellite imagery, natural features",
    "osm_humanitarian": "OpenStreetMap Humanitarian - best for high contrast, simplified features, developing regions"
}

DEFAULT_BASEMAP = "osm_standard"


@dataclass
class BoundingBoxEstimate:
    """Result of geographic bounding box estimation."""
    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float
    reasoning: str
    recommended_basemap: str = DEFAULT_BASEMAP  # Recommended basemap source key
    confidence: Optional[str] = None  # Optional confidence level from model

    @property
    def center_lon(self) -> float:
        """Center longitude."""
        return (self.min_lon + self.max_lon) / 2

    @property
    def center_lat(self) -> float:
        """Center latitude."""
        return (self.min_lat + self.max_lat) / 2

    @property
    def width_degrees(self) -> float:
        """Width in degrees."""
        return abs(self.max_lon - self.min_lon)

    @property
    def height_degrees(self) -> float:
        """Height in degrees."""
        return abs(self.max_lat - self.min_lat)

    def to_extent_meters(self, at_latitude: Optional[float] = None) -> float:
        """
        Approximate extent in meters (using the larger of width/height).
        Uses center latitude if not specified.
        """
        import math
        lat = at_latitude if at_latitude is not None else self.center_lat

        # Approximate meters per degree at given latitude
        meters_per_degree_lat = 111320  # ~constant
        meters_per_degree_lon = 111320 * math.cos(math.radians(lat))

        width_m = self.width_degrees * meters_per_degree_lon
        height_m = self.height_degrees * meters_per_degree_lat

        return max(width_m, height_m)

    def is_valid(self) -> Tuple[bool, str]:
        """Check if the bounding box is geographically valid."""
        if not (-180 <= self.min_lon <= 180):
            return False, f"Invalid min_lon: {self.min_lon}"
        if not (-180 <= self.max_lon <= 180):
            return False, f"Invalid max_lon: {self.max_lon}"
        if not (-90 <= self.min_lat <= 90):
            return False, f"Invalid min_lat: {self.min_lat}"
        if not (-90 <= self.max_lat <= 90):
            return False, f"Invalid max_lat: {self.max_lat}"
        if self.min_lon >= self.max_lon:
            return False, f"min_lon ({self.min_lon}) >= max_lon ({self.max_lon})"
        if self.min_lat >= self.max_lat:
            return False, f"min_lat ({self.min_lat}) >= max_lat ({self.max_lat})"
        return True, "Valid"


# System prompt for geographic analysis
VISION_SYSTEM_PROMPT = (
    "You are a geographic analysis expert specializing in precise location identification. "
    "Your task is to analyze images (maps, aerial photos, sketches, historical documents) "
    "and determine the EXACT geographic bounding box they depict.\n\n"
    "CRITICAL: You must identify the SPECIFIC area shown in the image, not a general region. "
    "If the image shows a neighborhood, identify THAT neighborhood. If it shows a specific "
    "intersection or landmark, identify THAT exact location.\n\n"
    "Analyze the image systematically for geographic identifiers:\n"
    "1. TEXT LABELS: Read ALL visible text - street names, place names, building labels, "
    "   district names, postal codes. These are your most reliable identifiers.\n"
    "2. WATER FEATURES: Rivers, lakes, coastlines, harbors - match their exact shapes.\n"
    "3. ROAD NETWORKS: Highway interchanges, distinctive intersections, road patterns.\n"
    "4. LANDMARKS: Recognizable buildings, monuments, bridges, stadiums.\n"
    "5. BOUNDARIES: Administrative borders, park boundaries, coastlines.\n"
    "6. SCALE INDICATORS: Use any scale bars to estimate the area's extent.\n\n"
    "BOUNDING BOX REQUIREMENTS:\n"
    "- The bounding box must tightly encompass ONLY the area shown in the image.\n"
    "- Add a small margin (~5-10%) to ensure the edges are captured, but no more.\n"
    "- Do NOT return a bounding box for a larger region (city, country) when the image "
    "  shows a smaller area (neighborhood, district).\n"
    "- If the image shows a 2km x 2km area, your bounding box should be approximately that size.\n\n"
    "IMPORTANT: You MUST respond with ONLY valid JSON in this exact format, "
    "with no additional text before or after:\n"
    '{"min_lon": <number>, "min_lat": <number>, "max_lon": <number>, '
    '"max_lat": <number>, "recommended_basemap": "<basemap_key>", '
    '"reasoning": "<explanation>"}\n\n'
    "Basemap options:\n"
    '- "osm_standard": OpenStreetMap - for maps, street plans, urban areas with roads/buildings\n'
    '- "esri_world_imagery": Satellite imagery - for aerial photos, natural features, rural areas\n'
    '- "osm_humanitarian": High-contrast OSM - for historical/faded maps needing clear features\n\n'
    "Coordinate format:\n"
    "- min_lon/max_lon: West/East bounds (-180 to 180)\n"
    "- min_lat/max_lat: South/North bounds (-90 to 90)\n\n"
    "Reasoning: Briefly explain the key features you identified that determined the location "
    "(e.g., 'Identified intersection of Main St and 5th Ave in downtown Portland based on "
    "visible street labels and distinctive river bend')."
)


class VisionAPIClient:
    """Client for vision API calls to estimate geographic bounding boxes."""

    MAX_IMAGE_SIZE = 2048  # Max dimension for resizing before API call

    def __init__(self, provider: APIProvider, api_key: str, model_key: str):
        """
        Initialize the vision API client.

        Args:
            provider: API provider (OpenAI or Anthropic)
            api_key: API key for the provider
            model_key: Key from VISION_MODELS dict
        """
        self.provider = provider
        self.api_key = api_key
        self.model_key = model_key

        if model_key not in VISION_MODELS:
            raise ValueError(f"Unknown model: {model_key}")

        self.model = VISION_MODELS[model_key]

        if self.model.provider != provider:
            raise ValueError(f"Model {model_key} is not available for provider {provider}")

    def preprocess_image(self, image_path: Path) -> Tuple[bytes, str]:
        """
        Preprocess image for API call: resize if needed and convert to base64.

        Args:
            image_path: Path to the image file

        Returns:
            Tuple of (base64_encoded_bytes, media_type)
        """
        if not HAS_PIL:
            raise ImportError("PIL/Pillow is required for image preprocessing")

        with Image.open(image_path) as img:
            # Convert to RGB if necessary (handles RGBA, grayscale, etc.)
            if img.mode not in ('RGB', 'L'):
                img = img.convert('RGB')

            # Resize if larger than max size
            width, height = img.size
            if width > self.MAX_IMAGE_SIZE or height > self.MAX_IMAGE_SIZE:
                # Calculate new size maintaining aspect ratio
                if width > height:
                    new_width = self.MAX_IMAGE_SIZE
                    new_height = int(height * (self.MAX_IMAGE_SIZE / width))
                else:
                    new_height = self.MAX_IMAGE_SIZE
                    new_width = int(width * (self.MAX_IMAGE_SIZE / height))

                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

            # Save to bytes
            buffer = io.BytesIO()

            # Determine format based on original file
            ext = image_path.suffix.lower()
            if ext in ('.jpg', '.jpeg'):
                img_format = 'JPEG'
                media_type = 'image/jpeg'
            elif ext == '.png':
                img_format = 'PNG'
                media_type = 'image/png'
            elif ext == '.gif':
                img_format = 'GIF'
                media_type = 'image/gif'
            elif ext == '.webp':
                img_format = 'WEBP'
                media_type = 'image/webp'
            else:
                # Default to JPEG for unknown formats
                img_format = 'JPEG'
                media_type = 'image/jpeg'

            img.save(buffer, format=img_format, quality=85)
            buffer.seek(0)

            return base64.standard_b64encode(buffer.read()).decode('utf-8'), media_type

    def estimate_bounding_box(
        self,
        image_path: Path,
        progress_callback: Optional[Callable[[str], None]] = None
    ) -> BoundingBoxEstimate:
        """
        Estimate the geographic bounding box of an image.

        Args:
            image_path: Path to the image file
            progress_callback: Optional callback for progress updates

        Returns:
            BoundingBoxEstimate with coordinates and reasoning

        Raises:
            Exception on API errors or invalid responses
        """
        if progress_callback:
            progress_callback(f"Preprocessing image: {image_path.name}")

        # Preprocess image
        image_base64, media_type = self.preprocess_image(image_path)

        if progress_callback:
            progress_callback(f"Sending to {self.model.display_name}...")

        # Make API call based on provider
        if self.provider == APIProvider.OPENAI:
            response_text = self._call_openai(image_base64, media_type)
        else:
            response_text = self._call_anthropic(image_base64, media_type)

        if progress_callback:
            progress_callback("Parsing response...")

        # Parse JSON response
        return self._parse_response(response_text)

    def _call_openai(self, image_base64: str, media_type: str) -> str:
        """Make OpenAI API call."""
        import urllib.request
        import urllib.error

        url = "https://api.openai.com/v1/chat/completions"

        payload = {
            "model": self.model.model_id,
            "messages": [
                {
                    "role": "system",
                    "content": VISION_SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{media_type};base64,{image_base64}",
                                "detail": "high"
                            }
                        },
                        {
                            "type": "text",
                            "text": "Analyze this image and estimate its geographic bounding box. Respond with only the JSON object."
                        }
                    ]
                }
            ],
            "max_tokens": 500,
            "temperature": 0.1  # Low temperature for more consistent output
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        data = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(url, data=data, headers=headers, method='POST')

        try:
            with urllib.request.urlopen(req, timeout=60) as response:
                result = json.loads(response.read().decode('utf-8'))
                return result['choices'][0]['message']['content']
        except urllib.error.HTTPError as e:
            error_body = e.read().decode('utf-8') if e.fp else ""
            raise Exception(f"OpenAI API error {e.code}: {error_body}")
        except urllib.error.URLError as e:
            raise Exception(f"Network error: {e.reason}")

    def _call_anthropic(self, image_base64: str, media_type: str) -> str:
        """Make Anthropic API call."""
        import urllib.request
        import urllib.error

        url = "https://api.anthropic.com/v1/messages"

        payload = {
            "model": self.model.model_id,
            "max_tokens": 500,
            "system": VISION_SYSTEM_PROMPT,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": image_base64
                            }
                        },
                        {
                            "type": "text",
                            "text": "Analyze this image and estimate its geographic bounding box. Respond with only the JSON object."
                        }
                    ]
                }
            ]
        }

        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01"
        }

        data = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(url, data=data, headers=headers, method='POST')

        try:
            with urllib.request.urlopen(req, timeout=60) as response:
                result = json.loads(response.read().decode('utf-8'))
                # Anthropic returns content as a list
                for block in result.get('content', []):
                    if block.get('type') == 'text':
                        return block.get('text', '')
                raise Exception("No text content in Anthropic response")
        except urllib.error.HTTPError as e:
            error_body = e.read().decode('utf-8') if e.fp else ""
            raise Exception(f"Anthropic API error {e.code}: {error_body}")
        except urllib.error.URLError as e:
            raise Exception(f"Network error: {e.reason}")

    def _parse_response(self, response_text: str) -> BoundingBoxEstimate:
        """Parse the JSON response from the vision model."""
        # Try to extract JSON from the response
        # Models sometimes include extra text before/after JSON
        response_text = response_text.strip()

        # Try direct JSON parse first
        try:
            data = json.loads(response_text)
        except json.JSONDecodeError:
            # Try to find JSON object in the response
            json_match = re.search(r'\{[^{}]*"min_lon"[^{}]*\}', response_text, re.DOTALL)
            if json_match:
                try:
                    data = json.loads(json_match.group())
                except json.JSONDecodeError:
                    raise ValueError(f"Could not parse JSON from response: {response_text[:500]}")
            else:
                raise ValueError(f"No valid JSON found in response: {response_text[:500]}")

        # Validate required fields
        required_fields = ['min_lon', 'min_lat', 'max_lon', 'max_lat']
        for field in required_fields:
            if field not in data:
                raise ValueError(f"Missing required field: {field}")
            try:
                data[field] = float(data[field])
            except (TypeError, ValueError):
                raise ValueError(f"Invalid value for {field}: {data[field]}")

        # Parse recommended basemap (with fallback to default)
        recommended_basemap = data.get('recommended_basemap', DEFAULT_BASEMAP)
        if recommended_basemap not in BASEMAP_OPTIONS:
            # If model returned invalid basemap, use default
            recommended_basemap = DEFAULT_BASEMAP

        return BoundingBoxEstimate(
            min_lon=data['min_lon'],
            min_lat=data['min_lat'],
            max_lon=data['max_lon'],
            max_lat=data['max_lat'],
            reasoning=data.get('reasoning', 'No reasoning provided'),
            recommended_basemap=recommended_basemap,
            confidence=data.get('confidence')
        )


def estimate_batch_cost(
    image_count: int,
    model_key: str
) -> Tuple[float, float]:
    """
    Estimate the cost range for processing a batch of images.

    Args:
        image_count: Number of images to process
        model_key: Key from VISION_MODELS dict

    Returns:
        Tuple of (low_estimate, high_estimate) in USD
    """
    if model_key not in VISION_MODELS:
        raise ValueError(f"Unknown model: {model_key}")

    model = VISION_MODELS[model_key]
    low = image_count * model.cost_per_image_low
    high = image_count * model.cost_per_image_high

    return low, high


def get_models_for_provider(provider: APIProvider) -> list:
    """Get list of model keys available for a provider."""
    return MODELS_BY_PROVIDER.get(provider, [])


def get_provider_display_name(provider: APIProvider) -> str:
    """Get display name for a provider."""
    return {
        APIProvider.OPENAI: "OpenAI",
        APIProvider.ANTHROPIC: "Anthropic"
    }.get(provider, str(provider))
