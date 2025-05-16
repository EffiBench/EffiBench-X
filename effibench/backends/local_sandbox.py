import os
import re
import shutil
import logging
import subprocess
import platform
import tempfile
from typing import Optional
from pathlib import Path
from functools import cache

from llm_sandbox.base import Session, ConsoleOutput
from llm_sandbox.const import SupportedLanguage, SupportedLanguageValues
from llm_sandbox.utils import (
    get_libraries_installation_command,
    get_code_file_extension,
    get_code_execution_command,
    ProfilerScript,
    ProfilerCpp,
)


@cache
def is_macos() -> bool:
    """Check if the current platform is macOS."""
    return platform.system() == "Darwin"


@cache
def find_go_binary_directory(binary_name: str) -> Optional[str]:
    """Find a Go binary in common installation locations."""
    # Check if it's in PATH first
    if shutil.which(binary_name):
        return binary_name
    
    # Check common Go binary locations
    possible_paths = [
        Path.home() / "go" / "bin" / binary_name,
        Path.home() / ".go" / "bin" / binary_name,
        Path.home() / ".gvm" / "pkgsets" / "go1.23" / "global" / "bin" / binary_name,
        Path.home() / ".gvm" / "gos" / "go1.23" / "bin" / binary_name,
    ]
    
    # Add GOPATH/bin if GOPATH is set
    if "GOPATH" in os.environ:
        possible_paths.append(Path(os.environ["GOPATH"]) / "bin" / binary_name)
    
    for path in possible_paths:
        if path.exists() and os.access(path, os.X_OK):
            return str(path.parent)
    
    return None


def graceful_kill(process: subprocess.Popen):
    process.terminate()
    try:
        process.wait(timeout=1)
    except subprocess.TimeoutExpired:
        process.kill()


def get_zsh_env():
    cmd = ["/bin/zsh", "-i", "-c", "printenv"]
    output = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, start_new_session=True)
    return dict(line.split("=", 1) for line in output.decode().splitlines() if "=" in line)


def find_java_public_class_name(code: str) -> str | None:
    """Find the name of the first public class in Java code."""
    public_class_matches = list(re.finditer(r'public\s+class\s+(\w+)', code))
    
    if not public_class_matches:
        return None
    
    return public_class_matches[0].group(1)


class LocalSession(Session):
    """Session that executes code locally with resource constraints."""

    def __init__(
        self,
        lang: str = SupportedLanguage.PYTHON,
        cpuset: Optional[str] = None,
        nice_level: Optional[int] = -19,
        verbose: bool = False,
        libraries: Optional[list[str]] = None,
        **kwargs,
    ):
        """
        Create a new local execution session.
        
        Args:
            cpuset: CPU cores to pin this session to (comma-separated list for taskset)
                   Set to None on macOS (will be ignored)
            nice_level: Process priority (-20 to 19, lower is higher priority)
                       Set to None on macOS (will be ignored)
            verbose: Enable verbose logging
        """
        # Initialize with a temporary language (will be overridden in run())
        super().__init__(lang, verbose)
        
        if lang not in SupportedLanguageValues:
            raise ValueError(
                f"Language {lang} is not supported. Must be one of {SupportedLanguageValues}"
            )
        
        self.lang: str = lang
        self.cpuset = cpuset
        self.nice_level = nice_level
        self.init_libraries = libraries
        self.installed_libraries = set()
        self.env = None
        self.session_dir = None
        self.current_process = None
        self._file_content_cache: str | None = None

    @property
    def is_open(self) -> bool:
        """Return whether the session is open based on session_dir existence."""
        return self.session_dir is not None

    def open(self, skip_setup=False):
        """Open the local session by creating necessary directories."""
        if self.is_open:
            self._log("Session already open", level="warning")
            return
        
        # Create a temporary directory for this session
        self.session_dir = Path(tempfile.mkdtemp())
        self._log(f"Created session directory: {self.session_dir}")
        
        # Should always setup
        self.setup()

    def setup(self, libraries: Optional[list] = None):
        """
        Set up the environment for code execution.
        This function sets up the required directories and tools for code execution.
        Creates all possible profilers upfront for later use.
        
        :param libraries: List of libraries to install
        """
        if not self.is_open:
            raise RuntimeError("Session is not open. Please call open() method before setting up.")

        # Setup environment variables
        self.env = get_zsh_env()
        logging.info(f"Environment variables: {self.env}")

        # Set up Go environment if needed
        if self.lang == SupportedLanguage.GO:
            self.execute_command("go mod init example", workdir=self.session_dir)
            self.execute_command("go mod tidy", workdir=self.session_dir)
            self.install_libraries(["golang.org/x/tools/cmd/goimports@latest"])
            go_bin_dir = find_go_binary_directory("goimports")
            if go_bin_dir:
                self.env["PATH"] = os.pathsep.join([go_bin_dir, self.env.get("PATH", "")])
            else:
                self._log("Warning: goimports binary not found", level="warning")
        
        # Set up all profiler tools upfront
        # Bash script profiler
        profiler_script_path = self.session_dir / "profiler.sh"
        self.create_file(ProfilerScript, profiler_script_path)
        profiler_script_path.chmod(0o755)
        
        # C++ profiler
        profiler_cpp_path = self.session_dir / "profiler.cpp"
        profiler_binary_path = self.session_dir / "profiler"
        self.create_file(ProfilerCpp, profiler_cpp_path)
        
        self._log("Compiling profiler binary")
        cmd = f"g++ -std=c++11 -O2 {profiler_cpp_path} -o {profiler_binary_path}"
        try:
            subprocess.run(cmd, shell=True, check=True)
            self._log("Profiler compilation successful")
        except subprocess.CalledProcessError as e:
            self._log(f"Failed to compile profiler: {e}", level="error")
        
        # Install libraries if provided
        libraries = list(set((self.init_libraries or []) + (libraries or [])))
        self.install_libraries(libraries)

    def install_libraries(self, libraries: Optional[list[str]]):
        if not self.is_open:
            raise RuntimeError("Session is not open. Please call open() method before installing libraries.")
        
        if libraries is None:
            return
        
        for library in libraries:
            if library not in self.installed_libraries:
                command = get_libraries_installation_command(self.lang, library)
                output = self.execute_command(command, workdir=self.session_dir)
                if output.exit_code != 0:
                    raise RuntimeError(f"Failed to install [{self.lang}] library [{library}] with command [{command}]: {output.text}")
                self.installed_libraries.add(library)

    def close(self):
        """Clean up the session, terminate any running processes, and remove temp directory."""
        if self.current_process:
            graceful_kill(self.current_process)
        self.current_process = None
        
        if self.session_dir and self.session_dir.exists():
            try:
                shutil.rmtree(self.session_dir)
                self._log(f"Cleaned up session directory: {self.session_dir}")
            except Exception as e:
                self._log(f"Error cleaning session directory: {e}", level="error")
        self.session_dir = None


    def run(self, code: str, libraries: Optional[list] = None, stdin: Optional[str] = None,
            time_limit: Optional[float] = None, memory_limit: Optional[float] = None,
            return_statistics: bool = True) -> ConsoleOutput:
        """
        Run code in the sandbox environment
        :param code: Code to run
        :param libraries: List of libraries to install
        :param stdin: Standard input to pass to the program
        :param time_limit: Maximum execution time in seconds (None or 0 = no time limit)
        :param memory_limit: Memory limit in MB (None or 0 = no memory limit)
        :return: Console output
        """
        if not self.is_open:
            raise RuntimeError("Session is not open. Please call open() method before running code.")

        # Install required libraries
        self.install_libraries(libraries)

        # Determine destination path for code
        if self.lang == SupportedLanguage.GO:
            code_file = self.session_dir / "main.go"
        elif self.lang == SupportedLanguage.JAVA:
            code_file = self.session_dir / f"{find_java_public_class_name(code) or 'Main'}.java"
        else:
            code_file = self.session_dir / f"code.{get_code_file_extension(self.lang)}"
        
        # Only write the code file if it has changed
        skip_create_code_file = True
        if self._file_content_cache != code:
            self._file_content_cache = code
            skip_create_code_file = False
            self.create_file(code, code_file)

        # Write stdin to a file if provided
        stdin_file_path = self.session_dir / "stdin.txt"
        if stdin is not None:
            if stdin == '':
                stdin = '\n'
            self.create_file(stdin, stdin_file_path)

        # Get execution commands
        commands = get_code_execution_command(
            self.lang,
            code_file,
            time_limit=time_limit,
            memory_limit=memory_limit,
            has_stdin=stdin is not None,
            stdin_file_path=stdin_file_path,
            profiler_script_path=self.session_dir / "profiler.sh",
            profiler_binary_path=self.session_dir / "profiler",
            profiler_log_path=self.session_dir / "profiler.log",
        )

        # Execute the commands
        if skip_create_code_file:
            commands = commands[-1:]
        for command in commands:
            output = self.execute_command(command, workdir=self.session_dir)
            if output.exit_code != 0:
                return output
        return output
    
    def copy_from_runtime(self, src: str, dest: str):
        if not self.is_open:
            raise RuntimeError("Session is not open. Please call open() method before copying files.")
        
        self.execute_command(f"cp -r {src} {dest}")

    def copy_to_runtime(self, src: str, dest: str):
        if not self.is_open:
            raise RuntimeError("Session is not open. Please call open() method before copying files.")
        
        self.execute_command(f"cp -r {src} {dest}")
    
    def create_file(self, content: str, dest: str | Path):
        """Create a file with the given content."""
        if not self.is_open:
            raise RuntimeError("Session is not open. Please call open() method before creating files.")
        
        dest = Path(dest)
        if not dest.is_absolute():
            dest = self.session_dir / dest
        
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content)

    def cat_profile(self) -> Optional[str]:
        """Get the content of the profiler log file."""
        profiler_log_path = self.session_dir / "profiler.log"
        if not profiler_log_path.exists():
            return None
        return profiler_log_path.read_text() or None
    
    def execute_command(self, command: Optional[str] = None, workdir: Optional[str | Path] = None, time_limit: Optional[float] = None) -> ConsoleOutput:
        if not command:
            raise ValueError("Command cannot be empty")

        if not self.is_open:
            raise RuntimeError("Session is not open. Please call open() method before executing commands.")

        self._log(f"Executing command: {command}")

        if self.cpuset and not is_macos():
            command = f"taskset -c {self.cpuset} {command}"
        # if self.nice_level is not None and not is_macos():
            # command = f"nice -n {self.nice_level} {command}"

        if workdir is None:
            workdir = self.session_dir
        
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(workdir),
            text=True,
            env=self.env,
            start_new_session=True
        )
        self.current_process = process
        
        time_limit = max(0, time_limit or 0) or None
        try:
            stdout, stderr = process.communicate(timeout=time_limit)
            exit_code = process.returncode
        except subprocess.TimeoutExpired:
            graceful_kill(process)
            stdout, stderr = process.communicate()
            exit_code = 124
        
        self.current_process = None
        text = stdout + ("\n" + stderr if stderr else "")
        self._log(f"Command exited with code {exit_code}")
        self._log(f"Output: {text}")
        return ConsoleOutput(text=text, exit_code=exit_code)
