import base64
import io
import logging
import os
from PIL import Image
from typing import Optional

try:
    import torch
    import torch.nn as nn
    import torchvision.transforms as transforms
    from torchvision.models import ResNet50_Weights

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    transforms = None
    ResNet50_Weights = None

try:
    from torchreid.utils import FeatureExtractor

    TORCHREID_AVAILABLE = True
except ImportError:
    TORCHREID_AVAILABLE = False
    FeatureExtractor = None

try:
    import numpy as np
except ImportError:
    np = None

# Models supported via torchreid
TORCHREID_MODELS = {
    "osnet_x1_0",
    "osnet_x0_75",
    "osnet_x0_5",
    "osnet_x0_25",
    "osnet_ibn_x1_0",
    "osnet_ain_x1_0",
}

# Default directory where bundled weights are stored inside the image.
# Can be overridden by setting the TORCH_HOME environment variable.
_DEFAULT_WEIGHTS_DIR = "/app/weights"

_LOGGER = logging.getLogger(__name__)


class ReIDModel:
    """Person re-identification model wrapper.

    Supports torchreid models (e.g. OSNet variants) for dedicated person
    re-identification, with an automatic fallback to a generic ResNet50
    feature extractor when torchreid is not installed or when the requested
    model is ``resnet50``.
    """

    def __init__(self, device: Optional[str] = None, model_name: str = "osnet_x1_0"):
        """Initialize the ReID model.

        Args:
            device: Device to run the model on ("cuda" or "cpu").
                Auto-detected if None.
            model_name: Name of the model to load.  Supported torchreid
                models include ``osnet_x1_0``, ``osnet_x0_75``,
                ``osnet_x0_5``, ``osnet_x0_25``, ``osnet_ibn_x1_0``, and
                ``osnet_ain_x1_0``.  Set to ``resnet50`` to force the
                generic ImageNet ResNet50 fallback.

        Raises:
            RuntimeError: If torch is not installed
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError(
                "torch is not installed. To use the ReID model, install PyTorch:\n"
                "  pip install torch torchvision\n"
                "Or install all dependencies:\n"
                "  pip install -r requirements.txt"
            )

        # Auto-detect device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)
        self.model_name = model_name
        self.model = None
        self._use_torchreid = False
        self._extractor = None
        self._load_model()

        # Preprocessing transforms (used only for the ResNet50 fallback)
        if not self._use_torchreid:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((256, 128)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

    def _load_model(self):
        """Load the re-ID model.

        Tries to load the requested model via torchreid first.  Falls back
        to a generic ResNet50 feature extractor when torchreid is not
        available, when the model name is ``resnet50``, or when the torchreid
        model fails to load.
        """
        if self.model_name in TORCHREID_MODELS and TORCHREID_AVAILABLE:
            try:
                self._load_torchreid_model()
                return
            except Exception as e:
                _LOGGER.error(
                    f"Failed to load torchreid model '{self.model_name}': {e}. "
                    "Falling back to ResNet50. Re-ID accuracy will be reduced. "
                    "Check that weights are accessible and torchreid is correctly installed."
                )
        else:
            if self.model_name in TORCHREID_MODELS and not TORCHREID_AVAILABLE:
                _LOGGER.warning(
                    f"torchreid is not installed; cannot load {self.model_name}. "
                    "Falling back to ResNet50. Install torchreid for better re-ID accuracy: pip install torchreid"
                )
        self._load_resnet50_fallback()

    def _load_torchreid_model(self):
        """Load a torchreid person re-identification model.

        Prefers bundled weights from TORCH_HOME or /app/weights so that no
        network access is required at runtime.
        """
        # Point torchreid's cache at the bundled weights directory so it finds
        # weights without hitting the network.
        weights_dir = os.environ.setdefault("TORCH_HOME", _DEFAULT_WEIGHTS_DIR)

        _LOGGER.info(
            f"Loading torchreid model: {self.model_name} (weights dir: {weights_dir})..."
        )
        self._extractor = FeatureExtractor(
            model_name=self.model_name,
            model_path="",
            device=str(self.device),
        )
        self._use_torchreid = True
        _LOGGER.info(
            f"Successfully loaded {self.model_name} via torchreid on device: {self.device}"
        )

    def _load_resnet50_fallback(self):
        """Load the pre-trained ResNet50 feature extractor."""
        from torchvision.models import resnet50

        _LOGGER.info("Loading ResNet50 feature extractor...")
        resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        # Remove the final classification layer to get 2048-dim features
        self.model = nn.Sequential(*list(resnet.children())[:-1])
        self.model = self.model.to(self.device)
        self.model.eval()
        _LOGGER.info(f"Successfully loaded ResNet50 on device: {self.device}")

    def extract_embedding(self, base64_image: str) -> np.ndarray:
        """Extract re-identification embedding from a base64-encoded image.

        Args:
            base64_image: Base64-encoded image string

        Returns:
            Feature vector as numpy array (shape: (embedding_dim,))
        """
        try:
            # Decode base64 image
            image_data = base64.b64decode(base64_image)
            image = Image.open(io.BytesIO(image_data)).convert("RGB")

            if self._use_torchreid:
                return self._extract_torchreid(image)

            # ResNet50 fallback path
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                embedding = self.model(image_tensor)

                if len(embedding.shape) > 2:
                    embedding = embedding.view(embedding.size(0), -1)

                embedding = embedding.cpu().numpy()[0]
                embedding = embedding / (
                    np.linalg.norm(embedding) + 1e-8
                )  # L2 normalization

            return embedding

        except Exception as e:
            _LOGGER.error(f"Error extracting embedding from image: {e}")
            raise

    def extract_embedding_from_file(self, image_path: str) -> np.ndarray:
        """Extract re-identification embedding from an image file.

        Args:
            image_path: Path to image file

        Returns:
            Feature vector as numpy array (shape: (embedding_dim,))
        """
        try:
            if self._use_torchreid:
                features = self._extractor([image_path])
                embedding = features.cpu().numpy()[0]
                embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
                return embedding

            image = Image.open(image_path).convert("RGB")
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                embedding = self.model(image_tensor)

                if len(embedding.shape) > 2:
                    embedding = embedding.view(embedding.size(0), -1)

                embedding = embedding.cpu().numpy()[0]
                embedding = embedding / (np.linalg.norm(embedding) + 1e-8)

            return embedding

        except Exception as e:
            _LOGGER.error(f"Error extracting embedding from file {image_path}: {e}")
            raise

    def _extract_torchreid(self, image: Image.Image) -> np.ndarray:
        """Extract embedding using the torchreid FeatureExtractor.

        Args:
            image: PIL Image in RGB mode.

        Returns:
            L2-normalised feature vector as a numpy array.
        """
        img_array = np.array(image)
        features = self._extractor([img_array])
        embedding = features.cpu().numpy()[0]
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
        return embedding
