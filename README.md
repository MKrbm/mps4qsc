# mps4qsc

A small collection of experiments and utilities for working with matrix product states (MPS) in quantum state classification tasks. The `qmpsqsc` package provides helpers for building and manipulating MPS tensors, while the notebooks folder captures interactive experiments.

## Repository layout
- `qmpsqsc/`: Python package with MPS construction, optimization, and supporting utilities.
- `notebooks/`: Jupyter notebooks that explore different experiments (e.g., variational autoencoders, classifiers, and optimization routines).
- `qmpsqsc/test/`: Unit tests covering the core tensor utilities and optimizers.

## `mps_add_svm` notebook
The `notebooks/mps_add_svm.ipynb` experiment builds GHZ and phase-flipped MPS states, augments them with single-qubit error variants, and combines them with `add_mpstates` to form training examples. It then uses MPS-based probability predictions (`mps_binary_predict`) as inputs to a simple linear SVM trained with a polynomial feature map (`poly2_features`) and hinge loss.

### Running the notebook
1. Install dependencies (Python 3, PyTorch, and Jupyter are required):
   ```bash
   pip install torch jupyter matplotlib
   ```
2. Launch the notebook with the repository on the Python path:
   ```bash
   PYTHONPATH=. jupyter notebook notebooks/mps_add_svm.ipynb
   ```
3. Run all cells to reproduce the SVM experiment or adapt the code for other MPS-derived datasets.

## Development
If you add new experiments that produce model checkpoints or other large artifacts, ensure they are ignored via `.gitignore` before committing.
