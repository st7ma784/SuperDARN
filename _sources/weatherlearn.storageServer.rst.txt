weatherlearn.storageServer package
==================================

This package provides functionality for managing storage operations, which is hosted in docker compose.
This spins up a MinIO server for object storage, allowing for efficient data management and retrieval.

Minio was chosed for its compatibility with AWS S3, making it easy to integrate with existing workflows and tools.
It works exceedingly well with random access data, and offers redundancy and fault tolerance.
(this was ideal given the hardware stack available at the time of writing).

Feel free to modify the docker compose file to suit your needs.

Modules for how to interact with the storage server
-------------------------------------------------------

- **DataModule**:
  Handles data loading and preprocessing, including downloading and uploading data to MinIO.

- **MinioToDisk**:
  Handles downloading data from MinIO buckets to local storage.

- **DiskToMinio**:
  Handles uploading data from local storage to MinIO buckets.

- **utils**:
  Contains utility functions for storage operations.

Module contents
---------------

.. automodule:: weatherlearn.storageServer
   :members:
   :show-inheritance:
   :undoc-members:
