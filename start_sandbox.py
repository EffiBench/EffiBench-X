#!/usr/bin/env python
"""
Start Sandbox - Unified CLI to start either the backend.

Usage:
    python start_sandbox.py --type [docker|local] [--host HOST] [--port PORT] [--workers NUM] [--skip-setup]
"""

import argparse
import logging
import signal
import uvicorn

from effibench.utils import setup_logger
from effibench.backends import get_backend


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Start the EffiBench-X backend server")
    parser.add_argument(
        "--type", 
        choices=["docker", "local"], 
        default="docker",
        help="Backend type: docker (containerized) or local (process-based)"
    )
    parser.add_argument(
        "--host", 
        default="0.0.0.0", 
        help="Bind socket to this host"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=8000, 
        help="Bind socket to this port"
    )
    parser.add_argument(
        "--workers", 
        type=int, 
        default=None, 
        help=f"Number of worker threads (default: num_physical_cores - 1)"
    )
    parser.add_argument(
        "--skip-setup",
        action="store_true",
        help="Skip backend setup and initialization"
    )
    return parser.parse_args()


def main():
    """Run the backend server with the specified configuration."""
    # Setup logging
    setup_logger()
    
    # Parse command line arguments
    args = parse_args()
    
    # Get backend type and configuration
    backend_type = args.type
    num_workers = args.workers
    host = args.host
    port = args.port
    skip_setup = args.skip_setup
    
    # Get backend and app
    manager, app = get_backend(
        backend_type=backend_type,
        num_workers=num_workers,
        setup_logging=False,  # Already set up logging
        skip_setup=skip_setup
    )
    
    # Log configuration
    logging.info(f"Using {backend_type} backend")
    if skip_setup:
        logging.info("Setup skipped (--skip-setup flag)")
    logging.info(f"Starting server with {manager.num_workers} workers")
    logging.info(f"Server will listen on {host}:{port}")
    
    # Start the server
    module_name = f"effibench.backends.{backend_type}"
    app_var = "app"
    
    # Create a temporary module with the app
    import sys
    import types
    temp_module = types.ModuleType("temp_app_module")
    temp_module.app = app
    sys.modules["temp_app_module"] = temp_module
    
    # Setup signal handlers for graceful shutdown
    original_sigint_handler = signal.getsignal(signal.SIGINT)
    original_sigterm_handler = signal.getsignal(signal.SIGTERM)
    
    shutdown_initiated = False
    
    def shutdown_handler(signum, frame):
        nonlocal shutdown_initiated
        if shutdown_initiated:
            # If shutdown already initiated and another signal is received,
            # restore original handler and let it process the signal
            signal.signal(signal.SIGINT, original_sigint_handler)
            signal.signal(signal.SIGTERM, original_sigterm_handler)
            return
            
        shutdown_initiated = True
        logging.info("Shutdown initiated. Cleaning up resources...")
        manager.stop_workers()
        logging.info("Worker threads stopped. Exiting...")
        sys.exit(0)
    
    # Install custom signal handlers
    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)
    
    try:
        # Start the server
        uvicorn.run(
            "temp_app_module:app",
            host=host,
            port=port,
            reload=False
        )
    finally:
        # Ensure resources are cleaned up even if uvicorn exits without SIGINT
        if not shutdown_initiated:
            manager.stop_workers()


if __name__ == "__main__":
    main()