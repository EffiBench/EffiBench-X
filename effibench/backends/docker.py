"""
Docker-based backend for code execution.
"""

import logging
from typing import Optional

from llm_sandbox import SandboxSession
from effibench.utils import EFFIBENCH_REGISTRY
from effibench.backends.backend_utils import (
    BaseExecutionManager,
    get_cpu_topology, 
    get_num_physical_cores,
)


class DockerBackend(BaseExecutionManager):
    """Execution manager that uses Docker-based SandboxSession for code execution."""
    
    def __init__(self, num_workers: Optional[int] = None, skip_setup: bool = False):
        super().__init__(
            session_class=SandboxSession,
            num_workers=num_workers,
            skip_setup=skip_setup,
        )

    def _create_initial_session(self, lang: str, config: dict) -> None:
        """Initialize a Docker session for the given language."""
        with SandboxSession(
            lang=config["llm_sandbox_lang"],
            verbose=True,
            stream=False,
            keep_template=True,
            commit_container=True,
            libraries=config.get("packages", []),
            runtime_configs={"ulimits": [{"name": "nofile", "soft": 65536, "hard": 65536}]},
        ) as session:
            pass
    
    def get_session(self, worker_id: int, lang: str) -> SandboxSession:
        """Get an existing session or create a new one with CPU pinning."""
        with self.sessions_lock:
            key = (worker_id, lang)
            if key in self.sessions:
                return self.sessions[key]
        
        # Pin to CPU core
        cpu_topology = get_cpu_topology()
        pcore = list(cpu_topology.keys())[worker_id % get_num_physical_cores()]
        cpuset = ",".join(map(str, cpu_topology.get(pcore, [pcore])))
        # logging.info(f"[Worker {worker_id}] Pinning to CPU core {pcore} with cpuset {cpuset}")
        
        # Configure container with CPU pinning and increased file limits
        container_configs = {
            "cpuset_cpus": cpuset,
            "ulimits": [{"name": "nofile", "soft": 65536, "hard": 65536}]
        }
        
        session = SandboxSession(
            lang=lang,
            verbose=False,
            stream=False,
            keep_template=True,
            commit_container=True if worker_id == 0 else False,
            runtime_configs=container_configs,
        )
        session.open(skip_setup=True)
        
        with self.sessions_lock:
            self.sessions[key] = session
        return session