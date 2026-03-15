# Tri-MARF: 3D Point Cloud Visual Language Model Augmentation System

This repository provides the core modules for Tri-MARF, a cutting-edge 3D Point Cloud Visual Language Model Augmentation System, as supplementary material for the NeurIPS paper.

## Core Components

Tri-MARF comprises three innovative components:

### 1. Point Cloud Gating Mechanism (Point_Gating.py)

The PointCloudFilter class enhances visual language model (VLM) outputs by aligning them with 3D point cloud features:

- Extracts features via a pre-trained 3D encoder
- Filters irrelevant VLM responses using adaptive similarity thresholds
- Integrates with deduplication for refined outputs
- Prioritizes key object categories for precise augmentation



### 2. BERT deduplication mechanism (Bert_Deduplication.py)

This module contains multiple classes for response deduplication and aggregation:
- `BertDeduplicator`: Cluster and merge similar responses using BERT model embedding
- `ClipWeightedDeduplicator`: Extend the basic deduplication, combining the CLIP model to consider image-text similarity
- `ResponseAggregator`: Aggregate and sort normalized responses
- `MABResponseAggregator`: Use the multi-armed bandit (MAB) algorithm to optimize response selection
- `UserFeedbackCollector`: Collect user feedback to improve response sorting

### 3. VLM preliminary annotation (VLM_preliminary_annotation.py)

The `VLMPreliminaryAnnotation` class implements the function of preliminary annotation of 3D point cloud views using the visual language model:
- Construct multiple rounds of prompts to obtain high-quality descriptions
- Process multiple image paths and generate candidate descriptions
- Calculate and store the probability score of each candidate
- Save the results in JSON format

### 4.System Requirements

Note: This repository includes only core modules and is not standalone. To build the full system, ensure:

**3D Processing Framework**:

- Configure a 3D point cloud processing library (e.g., Open3D or PointNet)
- Set up necessary environment variables

**Dependencies**:

- PyTorch
- Transformers
- CLIP
- scikit-learn
- Open3D or similar

**Pre-trained Models**:

- Download BERT, CLIP, and 3D encoder models

**API Access**:

- Configure VLM API keys for annotation tasks