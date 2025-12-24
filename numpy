Section 2: NumPy
What NumPy Is

Python library for fast numerical computing

Core object: ndarray (n-dimensional array)

Foundation for PyTorch tensors

Why It Matters for Medical Imaging

Medical images = matrices (2D X-ray, 3D CT/MRI)

NumPy handles:

Pixel values

Slices

Preprocessing math

Key Concepts

Array creation:

np.array(), np.zeros(), np.ones()

Shape & dimensions:

.shape, .ndim

Indexing & slicing:

img[10:50, 20:80]

Vectorized operations (FAST):

img / 255

Broadcasting:

Apply operations across arrays automatically

Important Takeaway

NumPy lets you manipulate medical images as data, not pictures.
