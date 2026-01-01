# Deep Learning with PyTorch for Medical Image Analysis  
### Student Notes & Foundations

These notes document my understanding of core concepts needed for **medical image analysis using deep learning**, based on coursework and independent study.  
Written from a **learner’s perspective**, focusing on intuition and practical relevance.

---

## Section 2: Crash Course — NumPy

### What is NumPy?
- Python library for **numerical computing**
- Optimized for fast operations on arrays
- Foundation for **PyTorch tensors**

### Why NumPy Matters for Medical Imaging
- Medical images are stored as **arrays of pixel/voxel values**
- CT/MRI scans = large 2D or 3D matrices
- Used for preprocessing before deep learning

### Core Concepts
- Arrays: `np.array()`, `np.zeros()`, `np.ones()`
- Shape & dimensions: `.shape`, `.ndim`
- Indexing & slicing: `image[20:100, 30:120]`
  
