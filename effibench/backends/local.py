"""
Local process-based backend for code execution.
"""

import threading
from typing import Optional

from effibench.utils import EFFIBENCH_REGISTRY
from effibench.backends.local_sandbox import LocalSession
from effibench.backends.backend_utils import (
    BaseExecutionManager,
    get_cpu_topology, 
    get_num_physical_cores,
)


class LocalBackend(BaseExecutionManager):
    """Execution manager that uses LocalSession for code execution."""
    
    def __init__(self, num_workers: Optional[int] = None, skip_setup: bool = False):
        super().__init__(
            session_class=LocalSession,
            num_workers=num_workers,
            skip_setup=skip_setup,
        )

    def _create_initial_session(self, lang: str, config: dict) -> None:
        """Initialize a local session for the given language."""
        with LocalSession(
            lang=config["llm_sandbox_lang"],
            verbose=True,
            libraries=config.get("packages", None) if lang != "javascript" else None, # do not install packages for javascript during _create_initial_session
        ) as session:
            pass
    
    def get_session(self, worker_id: int, lang: str) -> LocalSession:
        """Get an existing session or create a new one with CPU pinning."""
        with self.sessions_lock:
            key = (worker_id, lang)
            if key in self.sessions:
                return self.sessions[key]
        
        # Pin to CPU core
        cpu_topology = get_cpu_topology()
        pcore = list(cpu_topology.keys())[worker_id % get_num_physical_cores()]
        cpuset = ",".join(map(str, cpu_topology.get(pcore, [pcore])))
        
        session = LocalSession(
            lang=lang,
            verbose=False,
            cpuset=cpuset,
            nice_level=-19,
            libraries=EFFIBENCH_REGISTRY["javascript"]["packages"] if lang == "javascript" else None,
        )
        session.open()
        
        with self.sessions_lock:
            self.sessions[key] = session
        return session