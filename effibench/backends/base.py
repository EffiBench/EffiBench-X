"""
Base backend module providing unified access to execution backends.
"""

import logging
from typing import Dict, Literal, Optional, Type, Union

from fastapi import FastAPI

from effibench.backends.backend_utils import (
    BaseExecutionManager,
    create_fastapi_app
)
from effibench.utils import setup_logger

# Re-exported from the specific backend modules to avoid cyclic imports
from effibench.backends.docker import DockerBackend
from effibench.backends.local import LocalBackend


# Global registry of backend instances
_backends: Dict[str, tuple[BaseExecutionManager, FastAPI]] = {}


def get_backend(
    backend_type: Literal["docker", "local"] = "docker",
    num_workers: Optional[int] = None,
    setup_logging: bool = True,
    skip_setup: bool = False
) -> tuple[Union[DockerBackend, LocalBackend], FastAPI]:
    """
    Get or create a backend of the specified type.
    
    Args:
        backend_type: Type of backend to use ("docker" or "local")
        num_workers: Number of worker threads for the backend
        setup_logging: Whether to set up logging
        skip_setup: Whether to skip language environment setup
        
    Returns:
        Tuple of (backend_manager, fastapi_app)
    """
    if setup_logging:
        setup_logger()
    
    # Check cache first
    cache_key = f"{backend_type}_{num_workers}"
    if cache_key in _backends:
        return _backends[cache_key]
    
    # Create new backend
    if backend_type == "docker":
        backend_class: Type[BaseExecutionManager] = DockerBackend
    elif backend_type == "local":
        backend_class = LocalBackend
    else:
        raise ValueError(f"Unknown backend type: {backend_type}")
    
    # Initialize the backend manager
    manager = backend_class(num_workers=num_workers, skip_setup=skip_setup)
    
    # Create FastAPI app
    app = create_fastapi_app(manager)
    
    # Cache the result
    _backends[cache_key] = (manager, app)
    
    logging.info(f"Created {backend_type} backend with {manager.num_workers} workers")
    return manager, app