.. SuperDARN documentation master file, created by
   sphinx-quickstart on Tue May 13 09:46:44 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

SuperDARN Future Work
===========================

The planned future work for the SuperDARN codebase focuses on enhancing the existing framework and expanding its capabilities. The following areas are identified for further development using CUDA and parrallelization:

1. **Optimizing Data Processing**:
   - Implementing more efficient data processing algorithms to reduce the time taken for downloading and preprocessing radar data.
   - Exploring advanced caching mechanisms to minimize redundant data transfers and improve access times.
   - Improving Data structures away from linked lists, to enable parrallelization

2. **Enhancing Machine Learning Models**:
   - Investigating new architectures and techniques for improving the performance of the Pangu-Weather model.
   - Experimenting with different hyperparameter optimization strategies to fine-tune model performance.
   - Integrating additional machine learning frameworks and libraries to provide users with more options for model training and evaluation.


Requirements
-----------------

For the future work, the available resources are: 

- 2 years x 2.0 FTE (PDRA/RSE)
- Hardware stack:
   - Workstation(s)
      - Local GPU
      - Local Storage 
   - HPC 
      - GPU nodes
      - Storage (MinIO)
- Software stack:
   - Existing RST for testing
   - Existing codebase for reference
   - Existing data for testing

Minimum Viable Requirements
------------------------------------------------

- The minimum viable requirements for the future work include:
   - A working code base for testing and validation.
   - A GPU (per person) for development and testing. 
   - The existing store of files
   
Under this, a significant amount of time will be spent developing plans and tooling for building tests, comparing outputs across all stored data. 


General Requirements
------------------------------------------------

If more storage is available, the following general requirements can be considered:
   - A more extensive set of radar data for testing and validation.
   - Additional hardware resources for parallel processing and distributed training.
   - Faster access to Testing, means more code changes, and therefore more dev time, thus reducing project risk

If a larger GPU is available, the following general requirements can be considered:
   - A larger GPU for training larger models and handling more complex datasets.
   - Significantly reduces code change time, and therefore reduces project risk
   - Multiple GPUs enables tools like local LLMs to be used to generate code, and therefore reduces project risk

If a storage Server is available, the following general requirements can be considered:
   - A storage server for hosting MinIO buckets and providing distributed access to radar data.
   - Data sharing between multiple users and teams can enable collaboration and knowledge sharing, and shared tooling
   - The project can better pivot towards future Machine learning applications. 

