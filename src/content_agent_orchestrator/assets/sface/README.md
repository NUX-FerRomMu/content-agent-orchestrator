# Pinned SFace recognition model

- Upstream: OpenCV Zoo `models/face_recognition_sface/face_recognition_sface_2021dec.onnx`.
- Commit: `47534e27c9851bb1128ccc0102f1145e27f23f98`.
- Source: https://github.com/opencv/opencv_zoo/blob/47534e27c9851bb1128ccc0102f1145e27f23f98/models/face_recognition_sface/face_recognition_sface_2021dec.onnx
- SHA-256: `0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79`.
- Size: 38,696,353 bytes. Both size and SHA were verified against the upstream
  Git LFS pointer before using the actual model.
- License: Apache 2.0, reproduced in `LICENSE`; upstream README explicitly
  licenses all files in this directory under Apache 2.0.
- Contributors: Yaoyao Zhong (SFace), Chengrui Wang (ONNX conversion).

This is the original float32 MobileFaceNet model trained with SFace loss,
not the int8 variants. OpenCV `FaceRecognizerSF` aligns five YuNet landmarks,
then produces a 128-dimensional feature, normalized by this application.
The runtime verifies the model checksum and uses CPU inference. No downloads
or inference services are invoked during user execution.
