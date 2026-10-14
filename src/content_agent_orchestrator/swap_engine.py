"""Local InSwapper/ArcFace ONNX adapter; no downloads or image services."""

import cv2
import numpy as np

from .contracts import ExecutionError
from .virtual_face import png_image, validate_portrait

ARCFACE_POINTS = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float64,
)


def alignment(points, size):
    """Least-squares similarity transform, without a random RANSAC fit."""
    source = np.asarray(points, dtype=np.float64)
    target = ARCFACE_POINTS.copy()
    if size == 128:
        target[:, 0] += 8
    elif size != 112:
        raise ExecutionError("Unsupported face alignment size")
    if source.shape != (5, 2) or not np.isfinite(source).all():
        raise ExecutionError("Invalid swap landmarks")
    centered = source - source.mean(axis=0)
    variance = (centered**2).sum() / 5
    if variance < 1e-8:
        raise ExecutionError("Degenerate face alignment")
    u, singular, vt = np.linalg.svd((target - target.mean(axis=0)).T @ centered / 5)
    signs = np.array([1, np.linalg.det(u @ vt)])
    rotation = u @ np.diag(signs) @ vt
    scale = float(singular @ signs) / variance
    linear = scale * rotation
    return np.column_stack(
        (linear, target.mean(axis=0) - linear @ source.mean(axis=0))
    ).astype(np.float32)


def _session(model, device):
    try:
        import onnx
        import onnxruntime as ort
    except ImportError:
        raise ExecutionError(
            "Install the optional swap or swap-cuda uv group"
        ) from None
    ort.disable_telemetry_events()
    if model.stat().st_size > 2 * 1024**3:
        raise ExecutionError("Swap model exceeds the 2 GiB limit")
    graph = onnx.load(str(model), load_external_data=False)
    messages = [graph]
    external = False
    while messages:
        message = messages.pop()
        if isinstance(message, onnx.TensorProto) and (
            message.external_data or message.data_location == onnx.TensorProto.EXTERNAL
        ):
            external = True
            break
        for field, value in message.ListFields():
            if field.message_type is not None:
                messages.extend(value if field.is_repeated else [value])
    if external:
        raise ExecutionError("Swap models must contain their weights internally")
    providers = ["CPUExecutionProvider"]
    if device == "cuda":
        if "CUDAExecutionProvider" not in ort.get_available_providers():
            raise ExecutionError(
                "CUDA ONNX provider unavailable; use swap-cuda with a compatible GPU"
            )
        providers.insert(0, "CUDAExecutionProvider")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    options.log_severity_level = 3
    try:
        session = ort.InferenceSession(
            graph.SerializeToString(), sess_options=options, providers=providers
        )
    except Exception as exc:
        raise ExecutionError("Cannot initialize the local ONNX model") from exc
    if device == "cuda" and session.get_providers()[0] != "CUDAExecutionProvider":
        raise ExecutionError("Requested CUDA provider failed to initialize")
    return session, graph


class InSwapper:
    @staticmethod
    def _infer(session, bindings):
        try:
            return session.run(None, bindings)[0]
        except Exception as exc:
            raise ExecutionError("Local ONNX inference failed") from exc

    def __init__(self, swap_model, arcface_model, *, device="cpu"):
        self.swap, graph = _session(swap_model, device)
        self.arcface, _ = _session(arcface_model, device)
        swap_inputs = self.swap.get_inputs()
        arc_inputs = self.arcface.get_inputs()
        if (
            len(swap_inputs) != 2
            or swap_inputs[0].shape != [1, 3, 128, 128]
            or swap_inputs[1].shape != [1, 512]
            or len(arc_inputs) != 1
            or arc_inputs[0].shape
            not in ([1, 3, 112, 112], ["None", 3, 112, 112], [None, 3, 112, 112])
            or any(item.type != "tensor(float)" for item in [*swap_inputs, *arc_inputs])
            or not graph.graph.initializer
            or list(graph.graph.initializer[-1].dims) != [512, 512]
        ):
            raise ExecutionError(
                "Expected original InSwapper 128 and ArcFace 112 model signatures"
            )
        from onnx.numpy_helper import to_array

        self.emap = to_array(graph.graph.initializer[-1]).astype(np.float32)
        if not np.isfinite(self.emap).all():
            raise ExecutionError("Invalid InSwapper embedding map")
        self.swap_names = [item.name for item in swap_inputs]
        self.arc_name = arc_inputs[0].name
        self.latents = {}

    def prepare_target(self, name, content):
        image = png_image(content)
        face = validate_portrait(content)["face"]
        aligned = cv2.warpAffine(image, alignment(face["landmarks"], 112), (112, 112))
        blob = cv2.dnn.blobFromImage(
            aligned, 1 / 127.5, (112, 112), (127.5, 127.5, 127.5), swapRB=True
        )
        features = np.asarray(
            self._infer(self.arcface, {self.arc_name: blob}), dtype=np.float32
        ).reshape(1, -1)
        if (
            features.shape != (1, 512)
            or not np.isfinite(features).all()
            or np.linalg.norm(features) < 1e-8
        ):
            raise ExecutionError("ArcFace returned an invalid target embedding")
        latent = (features / np.linalg.norm(features)) @ self.emap
        norm = np.linalg.norm(latent)
        if not np.isfinite(latent).all() or norm < 1e-8:
            raise ExecutionError("Invalid mapped target embedding")
        self.latents[name] = (latent / norm).astype(np.float32)

    def replace(self, image, face, target):
        transform = alignment(face["landmarks"], 128)
        crop = cv2.warpAffine(
            image, transform, (128, 128), borderMode=cv2.BORDER_REPLICATE
        )
        blob = cv2.dnn.blobFromImage(crop, 1 / 255, (128, 128), swapRB=True)
        prediction = np.asarray(
            self._infer(
                self.swap,
                dict(zip(self.swap_names, (blob, self.latents[target]), strict=True)),
            )
        )
        if prediction.shape != (1, 3, 128, 128) or not np.isfinite(prediction).all():
            raise ExecutionError("Invalid swap prediction")
        replacement = (
            (np.clip(prediction[0].transpose(1, 2, 0)[:, :, ::-1], 0, 1) * 255)
            .round()
            .astype(np.uint8)
        )
        inverse = cv2.invertAffineTransform(transform)
        height, width = image.shape[:2]
        mask = np.zeros((128, 128), np.float32)
        # Keep feathering at the crop perimeter, away from identity-bearing pixels.
        mask[4:-4, 4:-4] = 1
        mask = cv2.GaussianBlur(mask, (9, 9), 0)
        restored = cv2.warpAffine(replacement, inverse, (width, height))
        alpha = cv2.warpAffine(mask, inverse, (width, height))[:, :, None]
        return (
            np.clip(alpha * restored + (1 - alpha) * image, 0, 255)
            .round()
            .astype(np.uint8)
        )
