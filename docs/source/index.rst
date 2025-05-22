.. SuperDARN documentation master file, created by
   sphinx-quickstart on Tue May 13 09:46:44 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

SuperDARN Codebase Overview
===========================

The SuperDARN codebase provides a proof-of-concept framework for processing radar data and training machine learning models for weather forecasting. It is designed to handle large-scale datasets stored in MinIO buckets and includes tools for efficient data processing, as well as machine learning pipelines for forecasting tasks.

Key Components
--------------

1. **Data Processing with MinIO**:
   - The `MinioToDisk` module provides functionality for downloading radar data from MinIO buckets to local storage. It supports multithreaded downloads to ensure high performance and preserves the folder structure of the bucket locally.
   - The `generateFitToConv` module processes FITACF radar data files and associates them with corresponding CONVMAP files. This enables the creation of structured datasets for downstream tasks.

2. **WeatherLearn Models**:
   - The `weatherlearn` package includes tools for training machine learning models on radar data. It features the Pangu-Weather model, a 3D transformer-based architecture optimized for spatiotemporal weather forecasting.
   - The `PTL` subpackage provides PyTorch Lightning-based modules for data handling, model training, and evaluation. It includes utilities for hyperparameter optimization and integration with HPC environments using SLURM.

3. **Data Handling**:
   - The `DataModule` module defines classes for loading and preprocessing radar data. It supports both flat and grid-based data representations and includes functionality for caching datasets and saving/loading preprocessed data.
   - The `DatasetFromMinioBucket` class enables seamless integration with MinIO, allowing radar data to be loaded directly into PyTorch workflows.

4. **Forecasting Models**:
   - The `model` module implements the Pangu-Weather model, which uses earth-specific attention mechanisms and transformer blocks to process spatiotemporal data. It supports multi-step forecasting and includes utilities for patch embedding, window partitioning, and attention masking.

5. **Utilities**:
   - The `utils` module provides helper functions and classes for tensor operations, such as up-sampling, down-sampling, cropping, and patch recovery. These utilities are essential for preparing data for the Pangu-Weather model.

Proof-of-Concept Goals
-----------------------

The codebase demonstrates the following capabilities:

1. **Efficient Data Processing**:
   - By leveraging MinIO for distributed object storage, the framework enables rapid access to radar data. The `MinioToDisk` and `generateFitToConv` modules streamline the process of preparing datasets for machine learning tasks.

2. **Forecasting with Machine Learning**:
   - The `weatherlearn` package provides a pipeline for training forecasting models on radar data. The Pangu-Weather model serves as a proof-of-concept for using transformer-based architectures in weather prediction.

3. **Scalability and HPC Integration**:
   - The framework is designed to scale across HPC environments, with support for SLURM job scheduling and distributed training. This ensures that the tools can handle large datasets and complex computations efficiently.

4. **Modularity and Extensibility**:
   - The modular design of the codebase allows for easy integration of new data sources, preprocessing methods, and machine learning models. This makes it adaptable to future research and development needs.

Conclusion
----------

The SuperDARN codebase provides a robust foundation for processing radar data and training machine learning models for weather forecasting. By combining efficient data handling with state-of-the-art machine learning techniques, it offers a scalable and extensible solution for tackling complex forecasting challenges.

.. toctree::
   :maxdepth: 6
   :caption: Contents:
   
   chorddht
   weatherlearn
  