"""Base session functionality for LLM Sandbox."""

import io
import os
import logging
import tarfile
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class ExecutionResult:
    """Result of code execution in sandbox."""

    exit_code: int
    output: str
    error: str
    execution_time: float
    resource_usage: dict


class ConsoleOutput:
    """Console output from code execution."""

    def __init__(self, text: str, exit_code: int = 0):
        self._text = text
        self._exit_code = exit_code
        self._runtime = None
        self._memory = None
        self._integral = None

    @property
    def text(self) -> str:
        return self._text

    @property
    def exit_code(self) -> int:
        return self._exit_code
        
    @property
    def integral(self) -> float | None:
        return self._integral
        
    @property
    def runtime(self) -> float | None:
        return self._runtime
        
    @runtime.setter
    def runtime(self, value: float):
        self._runtime = value
        
    @property
    def memory(self) -> float | None:
        return self._memory
        
    @memory.setter
    def memory(self, value: float):
        self._memory = value
        
    @integral.setter
    def integral(self, value: float):
        self._integral = value

    def __repr__(self):
        parts = [f"text={self.text}", f"exit_code={self.exit_code}"]
        if self._runtime is not None:
            parts.append(f"runtime={self.runtime}")
        if self._memory is not None:
            parts.append(f"memory={self.memory}")
        if self._integral is not None:
            parts.append(f"integral={self.integral}")
        return f"ConsoleOutput({', '.join(parts)})"

    def __str__(self):
        return self.text


class Session(ABC):
    """Abstract base class for sandbox sessions."""

    def __init__(
        self,
        lang: str,
        verbose: bool = True,
        strict_security: bool = True,
        runtime_configs: dict | None = None,
        logger: logging.Logger | None = None,
    ):
        self.lang = lang
        self.verbose = verbose
        self.runtime_configs = runtime_configs
        self.strict_security = strict_security
        self.logger = logger or logging.getLogger(__name__)
        self._dir_cache: set[str] = {"", "/"}

    def _log(self, message: str, level: str = "info"):
        """Log message if verbose is enabled."""
        if self.verbose:
            getattr(self.logger, level)(message)

    @abstractmethod
    def open(self):
        """Open the sandbox session."""
        raise NotImplementedError

    @abstractmethod
    def close(self):
        """Close the sandbox session."""
        raise NotImplementedError

    @abstractmethod
    def copy_to_runtime(self, src: str, dest: str):
        """Copy file to sandbox runtime."""
        raise NotImplementedError

    @abstractmethod
    def copy_from_runtime(self, src: str, dest: str):
        """Copy file from sandbox runtime."""
        raise NotImplementedError

    @abstractmethod
    def execute_command(self, command: str) -> ConsoleOutput:
        """Execute command in sandbox."""
        raise NotImplementedError

    def __enter__(self):
        """Context manager entry."""
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        
    def _ensure_dir(self, directory: str) -> bool:
        """Ensure directory exists in container, with caching to avoid repeated operations.
        
        Args:
            directory: Container path to ensure exists
            
        Returns:
            True if directory was created, False if it already existed
        """
        if not directory or directory in self._dir_cache:
            return False
            
        # Create directory and cache it
        self.execute_command(f"mkdir -p {directory}")
        
        # Cache this directory immediately
        self._dir_cache.add(directory)
        
        # Also cache parent directories with exact path format
        # Handle absolute paths correctly
        is_absolute = directory.startswith("/")
        path_parts = directory.split("/")
        current_path = "/" if is_absolute else ""
        
        for i, part in enumerate(path_parts):
            if not part:  # Skip empty parts (like after leading slash)
                continue
                
            if current_path == "/" or current_path == "":
                current_path = f"{current_path}{part}"
            else:
                current_path = f"{current_path}/{part}"
                
            self._dir_cache.add(current_path)
            
        return True
        
    def _put_bytes(self, dest_path: str, data: bytes) -> None:
        """Send bytes directly to container as a file without using a local temp file.
        
        Args:
            dest_path: Destination path in container
            data: File content as bytes
        """
        directory = os.path.dirname(dest_path) or "/"
        
        # Ensure destination directory exists
        created = self._ensure_dir(directory)
        if created and self.verbose:
            self._log(f"Created directory {directory}")
        
        # Create in-memory tar with file data
        tarstream = io.BytesIO()
        with tarfile.open(fileobj=tarstream, mode="w") as tar:
            info = tarfile.TarInfo(name=os.path.basename(dest_path))
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        
        tarstream.seek(0)
        self._put_archive(directory, tarstream)
        
    def _put_archive(self, dest_dir: str, tarstream: io.BytesIO) -> None:
        """Abstract method to put a tar archive into the container.
        Implemented by concrete subclasses (Docker, Podman, K8s).
        
        Args:
            dest_dir: Destination directory in container
            tarstream: In-memory tar stream
        """
        raise NotImplementedError
