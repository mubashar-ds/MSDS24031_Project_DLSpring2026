# Datasets

This project uses preprocessed datasets for training and evaluation. Due to GitHub's file size limitations, the datasets are **not included** in this repository.

## Download

### Original Datasets

* **FaceForensics++ (C23)**

  * https://www.kaggle.com/datasets/xdxd003/ff-c23

* **Celeb-DF v2**

  * https://www.kaggle.com/datasets/reubensuju/celeb-df-v2

### Preprocessed Datasets

* **FaceForensics++ Extracted Frames**

  * https://drive.google.com/drive/folders/1nEx405KvXlWR_66kDl4stOHM4AneR5BC?usp=sharing

* **Celeb-DF v2 Extracted Frames**

  * https://drive.google.com/drive/folders/1XIi0S0D1TQD8Mh5Iayqwt6q8beO91X76?usp=sharing

* **FaceForensics++ Cropped Faces**

  * https://drive.google.com/drive/folders/10zkmyOsvJhGsGd1VowyZh38Mpc2YQzv_?usp=sharing

* **Celeb-DF v2 Cropped Faces**

  * https://drive.google.com/drive/folders/1qZA18HPYUZzfR9nY1Sr4kj_u51rXlKv6?usp=sharing

* **JPEG Compressed Datasets (Q90, Q70, Q50, Q30, Q10)**

  * https://drive.google.com/file/d/1umN7kCqt1EqJsMzsQ0cASNeD9qIlzOm5/view?usp=sharing

---

## Directory Structure

After downloading and extracting the datasets, organize them as follows:

```text
datasets/
├── ffpp_frames/
├── ffpp_cropped_faces/
├── celebv2_frames/
├── celebv2_cropped_faces/
└── compressed_ffpp_celebv2/
```

---

## Dataset Usage

This project uses the datasets for the following purposes:

### FaceForensics++ (FF++)

* Frame extraction
* Face cropping
* Model training
* Validation
* In-domain evaluation

### Celeb-DF v2

* Frame extraction
* Face cropping
* Cross-dataset evaluation

### JPEG Compressed Datasets

Compression levels include:

* Quality 90
* Quality 70
* Quality 50
* Quality 30
* Quality 10

These datasets are used to evaluate the robustness of deepfake detection models under varying JPEG compression levels.

---

## Notes

* Place the downloaded folders directly inside the `datasets/` directory before running any notebooks or scripts.
* The preprocessing scripts for frame extraction, face cropping, and JPEG compression are available in the `data_preparation/` directory.
