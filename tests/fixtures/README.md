# Face-analysis fixtures

`astronaut.png` is the public-domain NASA photograph of astronaut Eileen Collins
distributed by scikit-image. It is used to validate detection, tracking and
recognition between generated test clips. It is not used to impersonate anyone.
NASA does not endorse this project.

- Source: https://github.com/scikit-image/scikit-image/blob/49a98547e909697f6ee89ec70691c7cad7eeb823/skimage/data/astronaut.png
- Attribution and public-domain statement: the `astronaut()` documentation in
  `skimage/data/_fetchers.py` at the same commit.
- SHA-256: `88431cd9653ccd539741b555fb0a46b61558b301d4110412b5bc28b5e3ea6cb5`.
- This hash was verified against the upstream `_registry.py` before use.

`grace_hopper.jpg` is the public-domain official US Navy portrait of Rear
Admiral Grace Hopper distributed as Matplotlib sample data. It is a second
test identity for rejecting false matches; the US Navy does not endorse this project.

- Source: https://github.com/matplotlib/matplotlib/blob/1392cbe3c79cdb93f9282747841d648770f60249/lib/matplotlib/mpl-data/sample_data/grace_hopper.jpg
- Upstream commit: `1392cbe3c79cdb93f9282747841d648770f60249` (v3.10.8).
- Git blob: `478720d6694a56b8962630e6dde8a230fc937049`.
- SHA-256: `a8ca6d734765703b09728ab47fe59f473d93ae3967fc24c7c0288c3c7adb7130`.
- Photograph attribution: US Navy, official portrait, public domain as a work
  of the US federal government; see https://commons.wikimedia.org/wiki/File:Grace_Hopper.jpg.

Integration tests create their videos in temporary directories and never send
the fixture or generated videos to external services.

`virtual_adult.png` is a newly generated fictional adult portrait, created with
the image generation tool for this project's virtual-identity tests. No source
portrait or real person's name was provided to the generator. It is a test
asset, not a virtual identity assigned to a user. Generation provenance cannot
prove that the image resembles no real person.

- Prompt: one fictional adult woman aged 35–40, medium warm brown skin, subtle
  eyeliner, no eyewear, ordinary facial traits, frontal neutral studio portrait.
- Generated on 2026-10-12; original file kept in the workspace generation directory.
- Size: 2,082,644 bytes; dimensions: 1254×1254 RGB PNG.
- SHA-256: `6c075392f5b27ba0f612f8e3de39400a89eb22a7727686e170490f05cf31a1c4`.
- Its actual pixels are checked with YuNet/SFace for usability and separation
  from the existing public-domain source fixtures. Tests of the cloud HTTP
  adapter mock network responses; the local command adapter runs real subprocesses.
