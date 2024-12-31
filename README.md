# Environment Setup

This repository contains instructions for setting up the environment for this project.

## Prerequisites

- Anaconda or Miniconda installed on your system [Anaconda](https://www.anaconda.com/products/distribution)
- Access to terminal or command prompt 'ctrl' + 'R', then entre 'cmd' and press 'ENTER'.

## Installation Steps

### 1. Create Environment

Create a new conda environment with Python 3.8:

```bash
conda create -n env_name python=3.8 -y
```

Note : This project was implemented using Python 3.8.20

```bash
conda create -n env_name python=3.8.20 -y
```

### 2. Activate Environment

```bash
conda activate env_name
```

### 3. Update Package Managers

Update pip to the latest version:

```bash
python -m pip install --upgrade pip
```

Update setuptools and wheel:

```bash
pip install --upgrade setuptools wheel
```

### 4. Install Dependencies

Install required packages:

```bash
pip install Flask Flask-SQLAlchemy face-recognition ultralytics supervision imutils

pip install face-recognition
```

## Dependencies List

- Flask
- Flask-SQLAlchemy
- face-recognition
- ultralytics
- supervision
- imutils
