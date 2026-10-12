# Face-detection fixture

`astronaut.png` is the public-domain NASA photograph of astronaut Eileen Collins
distributed by scikit-image. It is used only to validate face detection and
tracking, not to recognize or impersonate this person. NASA does not endorse
this project.

- Source: https://github.com/scikit-image/scikit-image/blob/49a98547e909697f6ee89ec70691c7cad7eeb823/skimage/data/astronaut.png
- Attribution and public-domain statement: the `astronaut()` documentation in
  `skimage/data/_fetchers.py` at the same commit.
- SHA-256: `88431cd9653ccd539741b555fb0a46b61558b301d4110412b5bc28b5e3ea6cb5`.
- This hash was verified against the upstream `_registry.py` before use.

Integration tests create their videos in temporary directories and never send
the fixture or generated videos to external services.
