# Animal Classification using SIFT and K-Means

A computer vision project for animal image classification using **Canny Edge Detection, SIFT feature extraction, and K-Means clustering**.

## Dataset

The project uses an animal image dataset containing multiple animal categories.

## Method

The basic pipeline is:

```text
Input Image
    ↓
Canny Edge Detection
    ↓
SIFT Feature Extraction
    ↓
K-Means Clustering
    ↓
Image Classification
```

K-Means is used with **17 clusters** to create visual feature representations from the extracted SIFT features.

## Tools

- Python
- OpenCV
- NumPy
- Scikit-learn
- Matplotlib

## Project

The complete implementation is available in:

`animal_classification_sift_kmeans.ipynb`

The notebook includes image preprocessing, Canny edge detection, SIFT feature extraction, K-Means clustering, and classification results.

## Note

This project was created for learning and experimentation with traditional computer vision techniques.
