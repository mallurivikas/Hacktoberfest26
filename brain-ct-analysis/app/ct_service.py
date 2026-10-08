from functools import lru_cache
from io import BytesIO
from typing import Literal

import numpy as np
from huggingface_hub import hf_hub_download
from PIL import Image, UnidentifiedImageError

from .schemas import BrainCTAnalysis, RankedLabel

MODEL_REPOSITORY = "kimsungil/brain-ich-ensemble"
MODEL_FILENAME = "ich_resnet18.pt"
MODEL_ID = "brain-ct-resnet18-ich"
LABELS = ("epidural", "intraparenchymal", "intraventricular", "subarachnoid", "subdural", "any")
MODEL_THRESHOLD = 0.5


class ImageValidationError(ValueError):
    pass


def _checkpoint_path() -> str:
    return hf_hub_download(repo_id=MODEL_REPOSITORY, filename=MODEL_FILENAME)


@lru_cache(maxsize=1)
def _model_and_device():
    import torch
    from torch import nn
    from torchvision import models

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(LABELS))
    checkpoint = torch.load(_checkpoint_path(), map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("model"), dict):
        raise RuntimeError("The CT checkpoint has an unexpected weights-only format.")
    checkpoint_labels = tuple(checkpoint.get("subtypes", ()))
    if checkpoint_labels != LABELS:
        raise RuntimeError("The CT checkpoint labels do not match the supported API contract.")
    state = {key.removeprefix("module."): value for key, value in checkpoint["model"].items()}
    model.load_state_dict(state, strict=True)
    return model.to(device).eval(), device


def _value(value: object, fallback: float) -> float:
    if value is None:
        return fallback
    if isinstance(value, (list, tuple)):
        value = value[0] if value else fallback
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _window_hu(pixels: np.ndarray, center: float, width: float) -> np.ndarray:
    lower, upper = center - width / 2, center + width / 2
    clipped = np.clip(pixels, lower, upper)
    return ((clipped - lower) * (255.0 / (upper - lower))).astype(np.uint8)


def _prepare_dicom(payload: bytes) -> Image.Image:
    try:
        import pydicom

        dataset = pydicom.dcmread(BytesIO(payload), force=False)
        if str(getattr(dataset, "Modality", "")).upper() != "CT":
            raise ImageValidationError("The uploaded DICOM is not labelled as a CT image.")
        pixels = dataset.pixel_array
    except ImageValidationError:
        raise
    except Exception as exc:
        raise ImageValidationError("The uploaded DICOM file cannot be read as a CT image.") from exc
    if pixels.ndim != 2:
        raise ImageValidationError("Upload one 2D CT DICOM slice; CT volumes and multi-frame DICOM files are not supported by this model.")
    if pixels.shape[0] < 32 or pixels.shape[1] < 32:
        raise ImageValidationError("Upload a CT image at least 32 pixels wide and high.")
    hu = pixels.astype(np.float32)
    hu = hu * _value(getattr(dataset, "RescaleSlope", None), 1.0) + _value(getattr(dataset, "RescaleIntercept", None), 0.0)
    # The source model is documented for brain/subdural window inputs. The third
    # channel repeats the brain window so the 3-channel ResNet contract is preserved.
    brain = _window_hu(hu, 40.0, 80.0)
    subdural = _window_hu(hu, 80.0, 200.0)
    return Image.fromarray(np.stack((brain, subdural, brain), axis=-1), mode="RGB")


def _prepare_raster(payload: bytes) -> Image.Image:
    try:
        with Image.open(BytesIO(payload)) as candidate:
            candidate.verify()
        with Image.open(BytesIO(payload)) as source:
            image = source.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ImageValidationError("The uploaded file is not a readable JPG, PNG, or WEBP image.") from exc
    if min(image.size) < 32:
        raise ImageValidationError("Upload a CT image at least 32 pixels wide and high.")
    return image


def _prepare(payload: bytes, is_dicom: bool):
    from torchvision import transforms

    image = _prepare_dicom(payload) if is_dicom else _prepare_raster(payload)
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return preprocess(image).unsqueeze(0)


def analyze_brain_ct(payload: bytes, is_dicom: bool) -> BrainCTAnalysis:
    import torch

    tensor = _prepare(payload, is_dicom)
    model, device = _model_and_device()
    with torch.no_grad():
        # Labels are not mutually exclusive: "any" may co-exist with one or more subtypes.
        probabilities = torch.sigmoid(model(tensor.to(device))[0]).detach().cpu().tolist()
    scores = {label: round(float(score), 4) for label, score in zip(LABELS, probabilities)}
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    threshold_labels = [label for label in LABELS if scores[label] >= MODEL_THRESHOLD]
    input_type: Literal["brain_ct_2d_image", "brain_ct_dicom_slice"] = "brain_ct_dicom_slice" if is_dicom else "brain_ct_2d_image"
    return BrainCTAnalysis(
        request_id="",
        model=MODEL_ID,
        device=device,
        input_type=input_type,
        highest_scoring_label=ranked[0][0],
        any_hemorrhage_score=scores["any"],
        class_scores=scores,
        ranked_labels=[RankedLabel(label=label, score=score) for label, score in ranked],
        labels_at_or_above_model_threshold=threshold_labels,
        integrated_summary="No tabular-model health summary was supplied with this CT image.",
    )
