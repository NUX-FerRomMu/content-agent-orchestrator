# Bundled YuNet detector

`face_detection_yunet_2023mar.onnx` comes from OpenCV Zoo, commit
`47534e27c9851bb1128ccc0102f1145e27f23f98`, under
`models/face_detection_yunet/`. Its MIT license is included in `LICENSE`.

- Size: 232589 bytes.
- SHA-256: `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4`.
- The expected hash and size were verified against the Git LFS pointer in that
  pinned upstream commit before using the downloaded model.
- Source: https://github.com/opencv/opencv_zoo/tree/47534e27c9851bb1128ccc0102f1145e27f23f98/models/face_detection_yunet

The model is shipped with this source checkout so analysis never needs to
download weights or send input media to a network service. Detection checks
this checksum before loading the model, and the model is an explicit pipeline
input so its content participates in cache fingerprints.
