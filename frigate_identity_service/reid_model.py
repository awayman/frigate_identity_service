import base64
import io
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
    import numpy as np
except ImportError:
    np = None


class ReIDModel:
    """Person re-identification model wrapper."""

    def __init__(self, device: Optional[str] = None):
        """Initialize the ReID model.

        Args:
            device: Device to run the model on ("cuda" or "cpu"). Auto-detected if None.

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
        self.model = None
        self._load_model()

        # Preprocessing transforms
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
        """Load the pre-trained ResNet50 feature extractor."""
        from torchvision.models import resnet50

        print("Loading ResNet50 feature extractor...")
        resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        # Remove the final classification layer to get 2048-dim features
        self.model = nn.Sequential(*list(resnet.children())[:-1])
        self.model = self.model.to(self.device)
        self.model.eval()
        print(f"Successfully loaded ResNet50 on device: {self.device}")

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

            # Preprocess
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)

            # Extract features
            with torch.no_grad():
                embedding = self.model(image_tensor)

                # Flatten if needed
                if len(embedding.shape) > 2:
                    embedding = embedding.view(embedding.size(0), -1)

                # Convert to numpy and normalize
                embedding = embedding.cpu().numpy()[0]
                embedding = embedding / (
                    np.linalg.norm(embedding) + 1e-8
                )  # L2 normalization

            return embedding

        except Exception as e:
            print(f"Error extracting embedding from image: {e}")
            raise

    def extract_embedding_from_file(self, image_path: str) -> np.ndarray:
        """Extract re-identification embedding from an image file.

        Args:
            image_path: Path to image file

        Returns:
            Feature vector as numpy array (shape: (embedding_dim,))
        """
        try:
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
            print(f"Error extracting embedding from file {image_path}: {e}")
            raise
