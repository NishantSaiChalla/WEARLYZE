"""
S3 data loading utilities for fashion detection system.

This module provides efficient S3 integration with caching, batch downloads,
and parallel loading capabilities.
"""

import os
import io
import json
import hashlib
import shutil
import logging
from typing import List, Dict, Optional, Union, Any, Tuple
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import pickle
from functools import lru_cache
from datetime import datetime, timedelta

import numpy as np
from PIL import Image
import torch

try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
    HAS_BOTO3 = True
except ImportError:
    HAS_BOTO3 = False
    boto3 = None

logger = logging.getLogger(__name__)


class S3DataLoader:
    """
    S3 data loader with caching and efficient batch downloading.
    """
    
    def __init__(
        self,
        bucket_name: str,
        cache_dir: Optional[Union[str, Path]] = None,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        region_name: str = 'us-east-1',
        max_workers: int = 8,
        cache_ttl_days: int = 30,
        verify_ssl: bool = True
    ):
        """
        Initialize S3 data loader.
        
        Args:
            bucket_name: S3 bucket name
            cache_dir: Local cache directory (default: ~/.cache/fashion_detection)
            aws_access_key_id: AWS access key (uses default credentials if None)
            aws_secret_access_key: AWS secret key (uses default credentials if None)
            region_name: AWS region
            max_workers: Maximum number of parallel download workers
            cache_ttl_days: Cache time-to-live in days
            verify_ssl: Whether to verify SSL certificates
        """
        if not HAS_BOTO3:
            raise ImportError("boto3 is required for S3 functionality. Install with: pip install boto3")
        
        self.bucket_name = bucket_name
        self.region_name = region_name
        self.max_workers = max_workers
        self.cache_ttl_days = cache_ttl_days
        self.verify_ssl = verify_ssl
        
        # Setup cache directory
        if cache_dir is None:
            cache_dir = Path.home() / '.cache' / 'fashion_detection' / bucket_name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize S3 client
        self._init_s3_client(aws_access_key_id, aws_secret_access_key)
        
        # Cache metadata
        self.cache_metadata_file = self.cache_dir / 'cache_metadata.json'
        self.cache_metadata = self._load_cache_metadata()
        
        # In-memory cache for frequently accessed items
        self._memory_cache = {}
        self._memory_cache_size = 100  # Max items in memory cache
    
    def _init_s3_client(
        self,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None
    ):
        """Initialize S3 client with credentials."""
        try:
            if aws_access_key_id and aws_secret_access_key:
                self.s3_client = boto3.client(
                    's3',
                    aws_access_key_id=aws_access_key_id,
                    aws_secret_access_key=aws_secret_access_key,
                    region_name=self.region_name,
                    verify=self.verify_ssl
                )
            else:
                # Use default credentials (IAM role, ~/.aws/credentials, etc.)
                self.s3_client = boto3.client(
                    's3',
                    region_name=self.region_name,
                    verify=self.verify_ssl
                )
            
            # Test connection
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            logger.info(f"Successfully connected to S3 bucket: {self.bucket_name}")
            
        except NoCredentialsError:
            raise ValueError("No AWS credentials found. Please provide credentials or configure AWS CLI.")
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == '404':
                raise ValueError(f"Bucket '{self.bucket_name}' not found")
            else:
                raise ValueError(f"Error connecting to S3: {e}")
    
    def _load_cache_metadata(self) -> Dict[str, Any]:
        """Load cache metadata from disk."""
        if self.cache_metadata_file.exists():
            try:
                with open(self.cache_metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Error loading cache metadata: {e}")
        return {}
    
    def _save_cache_metadata(self):
        """Save cache metadata to disk."""
        try:
            with open(self.cache_metadata_file, 'w') as f:
                json.dump(self.cache_metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving cache metadata: {e}")
    
    def _get_cache_path(self, s3_key: str) -> Path:
        """Get local cache path for an S3 key."""
        # Create a safe filename from S3 key
        safe_key = s3_key.replace('/', '_')
        return self.cache_dir / safe_key
    
    def _is_cached(self, s3_key: str) -> bool:
        """Check if an S3 object is cached and valid."""
        cache_path = self._get_cache_path(s3_key)
        
        if not cache_path.exists():
            return False
        
        # Check cache metadata
        if s3_key in self.cache_metadata:
            cached_time = datetime.fromisoformat(self.cache_metadata[s3_key]['cached_at'])
            ttl = timedelta(days=self.cache_ttl_days)
            
            if datetime.now() - cached_time > ttl:
                # Cache expired
                cache_path.unlink()
                del self.cache_metadata[s3_key]
                self._save_cache_metadata()
                return False
            
            # Verify file integrity with etag
            cached_etag = self.cache_metadata[s3_key].get('etag')
            if cached_etag:
                try:
                    response = self.s3_client.head_object(Bucket=self.bucket_name, Key=s3_key)
                    current_etag = response.get('ETag', '').strip('"')
                    if current_etag != cached_etag:
                        # File changed on S3
                        cache_path.unlink()
                        del self.cache_metadata[s3_key]
                        self._save_cache_metadata()
                        return False
                except Exception:
                    pass
        
        return True
    
    def _download_from_s3(self, s3_key: str, cache_path: Path) -> Path:
        """Download a file from S3 to local cache."""
        try:
            # Ensure parent directory exists
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Download with progress callback
            def download_callback(bytes_transferred):
                # Could add progress tracking here if needed
                pass
            
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=s3_key)
            
            # Save to cache
            with open(cache_path, 'wb') as f:
                f.write(response['Body'].read())
            
            # Update cache metadata
            self.cache_metadata[s3_key] = {
                'cached_at': datetime.now().isoformat(),
                'etag': response.get('ETag', '').strip('"'),
                'size': response.get('ContentLength', 0)
            }
            self._save_cache_metadata()
            
            logger.debug(f"Downloaded {s3_key} to cache")
            return cache_path
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'NoSuchKey':
                raise FileNotFoundError(f"S3 key not found: {s3_key}")
            else:
                raise RuntimeError(f"Error downloading from S3: {e}")
    
    def load_file(self, s3_key: str, force_download: bool = False) -> Path:
        """
        Load a file from S3, using cache if available.
        
        Args:
            s3_key: S3 object key
            force_download: Force download even if cached
            
        Returns:
            Path to local file
        """
        cache_path = self._get_cache_path(s3_key)
        
        if not force_download and self._is_cached(s3_key):
            logger.debug(f"Using cached file for {s3_key}")
            return cache_path
        
        return self._download_from_s3(s3_key, cache_path)
    
    def load_image(self, s3_key: str, force_download: bool = False) -> Image.Image:
        """
        Load an image from S3.
        
        Args:
            s3_key: S3 object key for the image
            force_download: Force download even if cached
            
        Returns:
            PIL Image object
        """
        # Check memory cache first
        if s3_key in self._memory_cache and not force_download:
            return self._memory_cache[s3_key]
        
        try:
            # Check if we should download to cache or load directly
            if self._is_cached(s3_key) and not force_download:
                cache_path = self._get_cache_path(s3_key)
                image = Image.open(cache_path).convert('RGB')
            else:
                # For images, we can load directly from S3 without saving to disk
                response = self.s3_client.get_object(Bucket=self.bucket_name, Key=s3_key)
                image_data = response['Body'].read()
                image = Image.open(io.BytesIO(image_data)).convert('RGB')
                
                # Optionally save to cache
                cache_path = self._get_cache_path(s3_key)
                image.save(cache_path)
                
                # Update cache metadata
                self.cache_metadata[s3_key] = {
                    'cached_at': datetime.now().isoformat(),
                    'etag': response.get('ETag', '').strip('"'),
                    'size': len(image_data)
                }
                self._save_cache_metadata()
            
            # Add to memory cache
            if len(self._memory_cache) >= self._memory_cache_size:
                # Remove oldest item
                self._memory_cache.pop(next(iter(self._memory_cache)))
            self._memory_cache[s3_key] = image
            
            return image
            
        except Exception as e:
            logger.error(f"Error loading image from S3: {e}")
            raise
    
    def load_json(self, s3_key: str, force_download: bool = False) -> Dict[str, Any]:
        """
        Load a JSON file from S3.
        
        Args:
            s3_key: S3 object key for the JSON file
            force_download: Force download even if cached
            
        Returns:
            Parsed JSON data
        """
        file_path = self.load_file(s3_key, force_download)
        
        with open(file_path, 'r') as f:
            return json.load(f)
    
    def load_numpy(self, s3_key: str, force_download: bool = False) -> np.ndarray:
        """
        Load a numpy array from S3.
        
        Args:
            s3_key: S3 object key for the numpy file
            force_download: Force download even if cached
            
        Returns:
            Numpy array
        """
        file_path = self.load_file(s3_key, force_download)
        return np.load(file_path)
    
    def load_torch(self, s3_key: str, force_download: bool = False) -> Any:
        """
        Load a PyTorch tensor or model from S3.
        
        Args:
            s3_key: S3 object key for the PyTorch file
            force_download: Force download even if cached
            
        Returns:
            PyTorch object
        """
        file_path = self.load_file(s3_key, force_download)
        return torch.load(file_path, map_location='cpu')
    
    def batch_download(
        self,
        s3_keys: List[str],
        force_download: bool = False,
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Path]:
        """
        Download multiple files from S3 in parallel.
        
        Args:
            s3_keys: List of S3 object keys
            force_download: Force download even if cached
            progress_callback: Callback function for progress updates
            
        Returns:
            Dictionary mapping S3 keys to local paths
        """
        results = {}
        to_download = []
        
        # Check what needs to be downloaded
        for s3_key in s3_keys:
            if force_download or not self._is_cached(s3_key):
                to_download.append(s3_key)
            else:
                results[s3_key] = self._get_cache_path(s3_key)
        
        if not to_download:
            return results
        
        # Download in parallel
        completed = 0
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_key = {
                executor.submit(self.load_file, key, True): key
                for key in to_download
            }
            
            for future in as_completed(future_to_key):
                s3_key = future_to_key[future]
                try:
                    local_path = future.result()
                    results[s3_key] = local_path
                    completed += 1
                    
                    if progress_callback:
                        progress_callback(completed, len(to_download))
                        
                except Exception as e:
                    logger.error(f"Error downloading {s3_key}: {e}")
                    results[s3_key] = None
        
        return results
    
    def list_objects(
        self,
        prefix: str = '',
        delimiter: str = '',
        max_keys: int = 1000
    ) -> List[Dict[str, Any]]:
        """
        List objects in S3 bucket with given prefix.
        
        Args:
            prefix: S3 key prefix to filter objects
            delimiter: Delimiter for grouping keys
            max_keys: Maximum number of keys to return
            
        Returns:
            List of object metadata dictionaries
        """
        objects = []
        
        try:
            paginator = self.s3_client.get_paginator('list_objects_v2')
            page_iterator = paginator.paginate(
                Bucket=self.bucket_name,
                Prefix=prefix,
                Delimiter=delimiter,
                PaginationConfig={'MaxItems': max_keys}
            )
            
            for page in page_iterator:
                if 'Contents' in page:
                    for obj in page['Contents']:
                        objects.append({
                            'key': obj['Key'],
                            'size': obj['Size'],
                            'last_modified': obj['LastModified'].isoformat(),
                            'etag': obj['ETag'].strip('"')
                        })
            
            return objects
            
        except Exception as e:
            logger.error(f"Error listing objects: {e}")
            raise
    
    def clear_cache(self, older_than_days: Optional[int] = None):
        """
        Clear local cache.
        
        Args:
            older_than_days: Only clear files older than this many days
        """
        if older_than_days is None:
            # Clear entire cache
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.cache_metadata = {}
            self._save_cache_metadata()
            logger.info("Cleared entire cache")
        else:
            # Clear old files
            cutoff_date = datetime.now() - timedelta(days=older_than_days)
            keys_to_remove = []
            
            for s3_key, metadata in self.cache_metadata.items():
                cached_at = datetime.fromisoformat(metadata['cached_at'])
                if cached_at < cutoff_date:
                    cache_path = self._get_cache_path(s3_key)
                    if cache_path.exists():
                        cache_path.unlink()
                    keys_to_remove.append(s3_key)
            
            for key in keys_to_remove:
                del self.cache_metadata[key]
            
            self._save_cache_metadata()
            logger.info(f"Removed {len(keys_to_remove)} cached files older than {older_than_days} days")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about the cache."""
        total_size = 0
        file_count = 0
        
        for cache_file in self.cache_dir.rglob('*'):
            if cache_file.is_file() and cache_file != self.cache_metadata_file:
                total_size += cache_file.stat().st_size
                file_count += 1
        
        return {
            'cache_dir': str(self.cache_dir),
            'total_files': file_count,
            'total_size_mb': total_size / (1024 * 1024),
            'metadata_entries': len(self.cache_metadata),
            'ttl_days': self.cache_ttl_days
        }


class S3DatasetWrapper:
    """
    Wrapper class to make any dataset work with S3 storage.
    """
    
    def __init__(
        self,
        dataset: Any,
        s3_loader: S3DataLoader,
        preload_batch_size: int = 32
    ):
        """
        Initialize S3 dataset wrapper.
        
        Args:
            dataset: Base dataset that expects S3 paths
            s3_loader: S3DataLoader instance
            preload_batch_size: Number of files to preload in parallel
        """
        self.dataset = dataset
        self.s3_loader = s3_loader
        self.preload_batch_size = preload_batch_size
        
        # Enable S3 mode in wrapped dataset
        if hasattr(dataset, 'use_s3'):
            dataset.use_s3 = True
    
    def __len__(self) -> int:
        """Return length of wrapped dataset."""
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Any:
        """Get item from wrapped dataset with S3 loading."""
        # Get sample from base dataset
        sample = self.dataset[idx]
        
        # The base dataset should return S3 keys instead of loading images
        # This wrapper handles the actual S3 loading
        if 'image_s3_key' in sample:
            sample['image'] = self.s3_loader.load_image(sample['image_s3_key'])
            del sample['image_s3_key']
        
        return sample
    
    def preload_batch(self, indices: List[int]):
        """
        Preload a batch of samples in parallel.
        
        Args:
            indices: List of sample indices to preload
        """
        s3_keys = []
        
        for idx in indices:
            sample = self.dataset.samples[idx]
            if 'image_path' in sample:
                s3_keys.append(sample['image_path'])
        
        if s3_keys:
            self.s3_loader.batch_download(s3_keys)