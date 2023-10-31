# Data Folder for PCB Defect Detection Project

This folder contains various datasets that are crucial for the PCB Defect Detection project. Below is an outline of the primary datasets and their details.

---

## Table of Contents

1. [Datasets](#datasets)
    - [PCB Defects Dataset (Kaggle)](#pcb-defects-dataset-kaggle)
    - [DeepPCB (tangsali5201)](#deeppcb-tangsali5201)
2. [Annotation Summary](#annotation-summary)

---

## Datasets

### PCB Defects Dataset (Kaggle)

- **Authors**: Huang, Weibo, and Peng Wei
- **Provider**: Open Lab on Human Robot Interaction, Peking University 
- **Date**: 2019
- **Description**: This dataset offers various types of artificially created PCB defects using Adobe Photoshop. It focuses on six major categories of defects: missing hole, mouse bite, open circuit, short, spur, and spurious copper.
- **Access**: [Kaggle PCB Defects Dataset](https://www.kaggle.com/datasets/akhatova/pcb-defects/data)
- **Accessed**: 12.10.2023 

### DeepPCB (tangsali5201)

- **Authors**: tangsanli5201 (Github Name)
- **Provider**: GitHub repository by tangsali5201
- **Date**: 19. December 2018
- **Description**: The DeepPCB dataset comprises 1,500 image pairs. Each pair consists of a defect-free template image and a corresponding tested image. Annotations for six common types of PCB defects are provided: open, short, mousebite, spur, pin hole, and spurious copper.
- **Access**: [DeepPCB GitHub Repository](https://github.com/tangsanli5201/DeepPCB)
- **Accessed**: 29.10.2023 

---

## Annotation Summary

The `annotation_summary.csv` file serves to consolidate all relevant information for the images from each dataset into a single file. This includes columns specifying the absolute paths to the images.

### How to Update

To update the `annotation_summary.csv`, call the function `get_dataframe(create_annotation_summary=True)` with the `create_annotation_summary` parameter set to `True` of the corresponding dataset.

---
