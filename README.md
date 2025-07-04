# AIMI-Tutorial: AI in Medical Imaging

A comprehensive tutorial series for learning Artificial Intelligence in Medical Imaging.

## Chapters

### Chapter 2: Understanding Medical Images: Formats & Visualization

This chapter introduces the fundamental concepts of medical image formats and their visualization. Topics covered include:
- DICOM format and metadata
- Medical image visualization techniques
- Window/level adjustments
- Working with image series
- Basic image processing for medical images

The chapter includes hands-on exercises and examples using real medical imaging data.

## Setup

1. Clone this repository:
```bash
git clone https://github.com/<your-username>/AIMI-tutorial.git
cd AIMI-tutorial
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Launch Jupyter Notebook:
```bash
jupyter notebook
```

4. Navigate to the desired chapter in the `notebooks` directory.

## Data

The tutorial uses publicly available medical imaging datasets. Chapter 2 relies on the [example-dicom-structural](https://github.com/datalad/example-dicom-structural) dataset, which provides anonymized T1-weighted brain scans.

### Cloning the dataset

1. Install [DataLad](https://www.datalad.org/) if it is not already available:
   ```bash
   pip install datalad
   ```
2. Clone the dataset:
   ```bash
   datalad clone https://github.com/datalad/example-dicom-structural.git path/to/data/example-dicom-structural
   ```
   Replace `path/to/data` with your preferred location.

We acknowledge the maintainers of the `example-dicom-structural` project for providing this sample DICOM dataset. The data are distributed under the Open Data Commons Public Domain Dedication & License; see the dataset repository for details.

## License

This tutorial is released under the MIT License. See the LICENSE file for details.