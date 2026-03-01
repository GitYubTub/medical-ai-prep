## Section 7: Data Formats in Medical Imaging

### Why Formats Matter
- Medical images are not JPG/PNG
- Metadata is critical:
  - Patient info
  - Scanner settings
  - Orientation

### Main Formats
- DICOM
  - Hospital standard
  - Contains:
    - Image
    - Metadata
  - Used in clinical settings
    
- NIfTI
  - Common in research
  - Supports 3D & 4D
  - Cleaner for ML pipelines
    
- Loading Tools
  - pydicom to DICOM
  - nibabel to NIfTI
  - Convert to NumPy / PyTorch tensors

### Typical Pipeline
```nginx
DICOM / NIfTI
- NumPy
- PyTorch Tensor
- Model
```

### Key Takeaway
Understanding formats = controlling the data, not guessing.
 
