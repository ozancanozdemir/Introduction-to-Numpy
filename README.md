# Introduction to NumPy – STAT 112 Recitation 6

This repository contains the **lecture / recitation notebook**  
for **STAT 112 – Introduction to Programming (Recitation 6)**  
at **Middle East Technical University (METU), Department of Statistics**.

The material focuses on a **hands-on introduction to NumPy** in Python, with many small exercises, examples, and explanations aimed at beginners.

Prepared by **Ozancan Ozdemir**  
📧 **ozancan@metu.edu.tr**

---

## Sources Used

This document was created using and inspired by the following resources:

- **CS231n – Convolutional Neural Networks for Visual Recognition, Stanford University**
- **Official NumPy documentation**
- **Jay Alammar’s blog posts and visual explanations**
- **Google Colaboratory (Colab) documentation and guides**

---

## Environment and Tools

The notebook assumes basic familiarity with:

- **Python 3** (Python 3.7+; examples mention Colab using Python 3.7 / 3.9)
- **Google Colab** as the main environment:
  - Running notebooks in the browser
  - Saving notebooks
  - Connecting with Google Drive / GitHub

Useful Colab references covered in the notes:

- Overview of Colaboratory  
- Guide to Markdown  
- Importing libraries and installing dependencies  
- Saving and loading notebooks in GitHub  
- Interactive forms & widgets

---

## Topics Covered

Below is an overview of the main sections in the notebook and what each of them teaches.

### 1. Introduction to NumPy

- What is **NumPy** and why it is fundamental for:
  - Scientific computing
  - Data science & machine learning
- Relationship to other libraries:
  - Used by **Pandas, SciPy, Matplotlib, scikit-learn, scikit-image**, and more.
- Import convention:
  ```python
  import numpy as np
````

**Key idea:** NumPy provides the `ndarray` object, a powerful **n-dimensional array** that enables fast, vectorized operations.

---

### 2. Creating and Inspecting Arrays

* Creating 1D and 2D arrays with `np.array()`:

  ```python
  a = np.array([1, 2, 3])
  b = np.array([[1, 2], [3, 4]])
  ```
* Basic attributes:

  * `ndim` – number of dimensions (rank)
  * `shape` – size along each dimension
  * `size` – total number of elements
  * `dtype` – data type of elements

**Goal:** Understand how NumPy arrays are structured and how they differ from plain Python lists.

---

### 3. Creating Special Arrays

Functions for quickly creating arrays with specific patterns:

* `np.zeros((m,n))` – all zeros
* `np.ones((m,n))` – all ones
* `np.full((m,n), value)` – constant array
* `np.eye(n)` – identity matrix
* `np.random.random((m,n))` – random values in [0, 1)

Also introduces **exercise-style problems**, for example:

* Create a 0-vector except for a specific index = 1
* Create a 3×3 identity matrix
* Create random 3×3×3 arrays

**Goal:** Get comfortable initializing arrays for later computations.

---

### 4. Number Sequences: `arange` and `linspace`

Two ways to generate regular numeric sequences:

* `np.arange(start, stop, step)` – like Python’s `range`, but returns an array
* `np.linspace(start, stop, num)` – generates `num` equally spaced points between start and stop (inclusive)

Examples:

```python
f = np.arange(10, 50, 5)
g = np.linspace(0., 1., num=5)
```

**Goal:** Quickly create numerical grids and ranges for experiments and plotting.

---

### 5. Basic Operators and Vectorization

* Elementwise operations:

  * `+`, `-`, `*`, `/` vs. `np.add`, `np.subtract`, `np.multiply`, `np.divide`
* Unary operations:

  * `np.sqrt`, `np.exp`, `np.round`, `np.ceil`, `np.floor`
* Demonstration of **performance benefits** of vectorized operations vs. Python `for` loops.

**Key concept:** Use NumPy’s vectorized functions instead of explicit loops to gain speed and cleaner code.

---

### 6. Matrix Multiplication and Dot Product

* Clarifies that `*` is **elementwise multiplication**, not matrix multiplication.
* Introduces:

  * `np.dot(a, b)` or `a.dot(b)` or `a @ b`
* Inner product of vectors, matrix–vector multiplication, and matrix–matrix multiplication.

**Example exercise:** Multiply a 5×3 matrix by a 3×2 matrix where all entries are 1.

**Goal:** Understand how to perform proper linear algebra operations in NumPy.

---

### 7. Mathematical & Statistical Functions

Common aggregation functions on arrays:

* `np.min`, `np.max`, `np.mean`, `np.median`, `np.quantile`, `np.std`, `np.sum`, `np.cumsum`
* Demonstrates:

  * Global aggregation (over all elements)
  * Aggregation along axes (`axis=0` for columns, `axis=1` for rows)
* Intro to:

  * `np.cov` – covariance matrix
  * `np.corrcoef` – correlation matrix

Also includes a small example to compute **five-number summary** using quantiles.

**Goal:** Use NumPy as a basic stats engine for arrays.

---

### 8. Sorting and Shuffling

* `np.sort(a)` – returns a sorted copy
* `np.sort(a)[::-1]` – reverse for descending order
* `np.random.shuffle(a)` – shuffles array *in place*

**Goal:** Reorder your data deterministically or randomly.

---

### 9. Indexing and Slicing

Covers:

* 1D indexing:

  * `a[i]`, `a[start:stop]`, negative indices (`a[-1]`, `a[-2]`, …)
* 2D indexing:

  * `a[row, col]`
  * Row access: `a[1, :]`, `a[1:2, :]`, `a[[1], :]`
  * Column access: `a[:, 1]`, `a[:, 1:2]`
* Subarrays via slicing, e.g.:

  ```python
  b = a[:2, 1:3]
  ```

Explains shape differences between rank-1 and rank-2 views.

**Goal:** Master array access patterns, which is crucial for all later work in NumPy.

---

### 10. Integer & Boolean Indexing

* **Integer array indexing**:

  ```python
  a[[0, 1, 2], [0, 1, 0]]
  ```
* **Boolean indexing**:

  ```python
  bool_idx = (a > 2)
  a[bool_idx]
  a[a > 2]
  ```
* Practical example:

  * Filtering values greater than a threshold
  * Detecting outliers using **1.5 × IQR rule** (quantiles + boolean mask)

**Goal:** Learn how to select arbitrary elements based on conditions.

---

### 11. `take` and `put`

* `np.take(a, indices, axis=...)` – select elements along a given axis
* `np.put(a, indices, values)` – replace specified positions in a 1D view

**Goal:** Understand more explicit index-based manipulation, especially in multi-dimensional arrays.

---

### 12. Reshaping, Transposing, and Squeezing

Key operations:

* `.T` – transpose:

  ```python
  x.T
  ```
* `reshape` – change shape while keeping the number of elements:

  ```python
  y = w.reshape(-1,)      # 2D → 1D
  y.reshape((-1, 1))      # 1D → column vector
  ```
* `squeeze` – remove dimensions of size 1:

  ```python
  z = w.squeeze()
  ```

**Goal:** Prepare data shapes for functions and models that expect specific input dimensions.

---

### 13. Concatenating Arrays

Using `np.concatenate`:

* Concatenate along rows (`axis=0`) or columns (`axis=1`)
* Examples:

  * Stack extra rows below an array
  * Add new columns to an existing matrix
  * Use transpose `.T` to match shapes before concatenation

**Goal:** Combine multiple data pieces into a single array for analysis or modeling.

---

### 14. Random Numbers and Seeding

* `np.random.random(n)` – random floats in [0, 1)
* `np.random.randint(low, high, size)` – random integers
* `np.random.seed(some_number)` – make random results reproducible

**Goal:** Generate reproducible random data for experiments and assignments.

---

## Exercises

Throughout the notebook, there are **short exercises** such as:

* Creating specific arrays (zeros/ones/identity/random)
* Constructing sequences with `arange` and `linspace`
* Implementing matrix products using `@`
* Calculating five-number summaries and outliers
* Practicing slicing, boolean indexing, and reshaping
* Concatenating arrays of compatible shapes

These exercises are designed to **build intuition** for NumPy by doing, not only by reading.

---

## How to Use This Material

1. Open the notebook in **Google Colab** or locally in **Jupyter Notebook**.
2. Run each code cell, read the explanations, and try modifying the examples.
3. Solve the exercises *without* looking at the answers first.
4. Use this notebook as a **reference** for future courses that rely on NumPy (such as statistics, machine learning, and data analysis).

This recitation is meant to give you a **solid foundation in NumPy**, so that you can focus on concepts in later courses rather than on basic Python array mechanics.
