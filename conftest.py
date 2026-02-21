import os
import sys

# Add the source package directory to sys.path so that tests can import
# modules like embedding_store, matcher, identity_service, etc.
sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), "frigate_identity_service"),
)
