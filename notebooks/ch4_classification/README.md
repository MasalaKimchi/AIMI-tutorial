# Chapter 4: Image Classification with Machine Learning

This chapter introduces a minimal end-to-end pipeline for training a classifier on medical images. We use a subset of the publicly available COVID-19 chest X‑ray dataset. The notebook demonstrates how to:

- Programmatically download labeled images and organize them into class folders.
- Preprocess images by resizing and flattening to feed into a traditional machine learning model.
- Train a Support Vector Machine to distinguish COVID‑19 cases from other findings.
- Evaluate performance using `classification_report`.

While the dataset here is intentionally tiny for demonstration, the same workflow can be scaled to larger collections and more sophisticated models.
