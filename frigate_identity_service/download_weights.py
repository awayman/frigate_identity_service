"""Download OSNet weights for bundling into the Docker image."""

import os
import sys


def download_osnet_weights(
    model_name: str = "osnet_x1_0", output_dir: str = "/app/weights"
):
    """Download torchreid model weights to a local directory."""
    os.makedirs(output_dir, exist_ok=True)
    try:
        from torchreid.utils import FeatureExtractor

        print(f"Downloading {model_name} weights to {output_dir} ...")
        # torchreid caches weights in TORCH_HOME; we set it to output_dir
        os.environ["TORCH_HOME"] = output_dir
        FeatureExtractor(model_name=model_name, model_path="", device="cpu")
        print(f"Weights for {model_name} downloaded successfully.")
    except Exception as e:
        print(f"ERROR: Failed to download weights: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else "osnet_x1_0"
    out = sys.argv[2] if len(sys.argv) > 2 else "/app/weights"
    download_osnet_weights(model, out)
