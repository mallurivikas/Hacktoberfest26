# Brain CT Analysis API

Research-only local API for **single axial brain-CT slice** screening. It loads a public Apache-2.0 ResNet-18 checkpoint from [`kimsungil/brain-ich-ensemble`](https://huggingface.co/kimsungil/brain-ich-ensemble) on first use, then caches it locally. This service deliberately uses the compact `ich_resnet18.pt` checkpoint (about 45 MB), not the repository's much larger three-model ensemble.

The model returns independent screening scores for `epidural`, `intraparenchymal`, `intraventricular`, `subarachnoid`, `subdural`, and `any` intracranial haemorrhage. It is a research/demo model, not clinically validated. It must not be used for diagnosis, triage, treatment, or patient-care decisions.

## Supported input

- One de-identified axial brain CT slice as JPG, PNG, or WEBP.
- One de-identified single-frame DICOM CT slice (`.dcm` or `.dicom`). DICOM data are transformed into brain and subdural windows before inference.
- CT series/volumes and multi-frame DICOMs are rejected because this is a 2D slice model. DICOM uploads must identify their modality as CT. The service does not determine body region from a raster image, so callers must submit only an axial brain CT slice.

## Run

```powershell
cd brain-ct-analysis
C:\Siddu\H\gemma-food-smoke-test\.venv\Scripts\python.exe -m pip install -r requirements.txt
C:\Siddu\H\gemma-food-smoke-test\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8004
```

Open `http://127.0.0.1:8004` for the upload UI or `http://127.0.0.1:8004/docs` for the API contract.

## Endpoint

`POST /api/v1/medical/ct/analyze` accepts multipart fields:

- `file`: the CT slice, up to 24 MB.
- `health_model_summary` (optional): JSON from the tabular risk-model branch. It is carried as review context but never changes CT scores.

The response includes independent sigmoid scores (not mutually exclusive probabilities), the `any` score, all scores ranked, and any labels at the checkpoint's documented 0.5 threshold.
