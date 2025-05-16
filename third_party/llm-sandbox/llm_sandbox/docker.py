import io
import os
import re
import docker
import tarfile
from pathlib import Path

from docker.models.images import Image
from docker.models.containers import Container
from docker.types import Mount
from llm_sandbox.utils import (
    image_exists,
    get_libraries_installation_command,
    get_code_file_extension,
    get_code_execution_command,
    WORKSPACE_DIR,
    StdinFilePath,
    ProfilerLogPath,
    ProfilerScript,
    ProfilerScriptPath,
    ProfilerCpp,
    ProfilerCppPath,
    ProfilerBinaryPath,
    StatisticsCpp,
    StatisticsCppPath,
    StatisticsBinaryPath,
)
from llm_sandbox.base import Session, ConsoleOutput
from llm_sandbox.const import (
    SupportedLanguage,
    SupportedLanguageValues,
    DefaultImage,
    NotSupportedLibraryInstallation,
)


def find_java_public_class_name(code: str) -> str | None:
    """Find the name of the first public class in Java code."""
    public_class_matches = list(re.finditer(r'public\s+class\s+(\w+)', code))
    
    if not public_class_matches:
        return None
    
    return public_class_matches[0].group(1)


class SandboxDockerSession(Session):
    def __init__(
        self,
        client: docker.DockerClient | None = None,
        image: str | None = None,
        dockerfile: str | None = None,
        lang: str = SupportedLanguage.PYTHON,
        keep_template: bool = False,
        commit_container: bool = True,
        verbose: bool = False,
        mounts: list[Mount] | None = None,
        stream: bool = True,
        runtime_configs: dict | None = None,
        libraries: list | None = None,
        **kwargs,
    ):
        """
        Create a new sandbox session
        :param client: Docker client, if not provided, a new client will be created based on local Docker context
        :param image: Docker image to use
        :param dockerfile: Path to the Dockerfile, if image is not provided
        :param lang: Language of the code
        :param keep_template: if True, the image and container will not be removed after the session ends
        :param commit_container: if True, the Docker container will be commited after the session ends
        :param verbose: if True, print messages
        :param mounts: List of mounts to be mounted to the container
        :param stream: if True, the output will be streamed (enabling this option prevents obtaining an exit code of run command)
        :param runtime_configs: Additional configurations for the container, i.e. resources limits (cpu_count, mem_limit), etc.
        :param libraries: List of libraries to install after container creation
        """
        super().__init__(lang, verbose)
        if image and dockerfile:
            raise ValueError("Only one of image or dockerfile should be provided")

        if lang not in SupportedLanguageValues:
            raise ValueError(
                f"Language {lang} is not supported. Must be one of {SupportedLanguageValues}"
            )

        if not image and not dockerfile:
            image = DefaultImage.__dict__[lang.upper()]

        self.lang: str = lang
        self.client: docker.DockerClient | None = None

        if not client:
            if self.verbose:
                print("Using local Docker context since client is not provided..")

            self.client = docker.from_env()
        else:
            self.client = client

        self.image: Image | str = image
        self.dockerfile: str | None = None
        self.container: Container | None = None
        self.path = None
        self.keep_template = keep_template
        self.commit_container = commit_container
        self.is_create_template: bool = False
        self.verbose = verbose
        self.mounts = mounts
        self.stream = stream
        self.runtime_configs = runtime_configs
        self.init_libraries = libraries
        self.installed_libraries = set()
        self.session_dir = Path(WORKSPACE_DIR)
        self._file_content_cache: str | None = None
        
        # Pre-cache common directories
        self._dir_cache.add(str(WORKSPACE_DIR))

    def open(self, skip_setup=False):
        warning_str = (
            "Since the `keep_template` flag is set to True the docker image will not be removed after the session ends "
            "and remains for future use."
        )
        if self.dockerfile:
            self.path = os.path.dirname(self.dockerfile)
            if self.verbose:
                f_str = f"Building docker image from {self.dockerfile}"
                f_str = f"{f_str}\n{warning_str}" if self.keep_template else f_str
                print(f_str)

            self.image, _ = self.client.images.build(
                path=self.path,
                dockerfile=os.path.basename(self.dockerfile),
                tag=f"sandbox-{self.lang.lower()}-{os.path.basename(self.path)}",
            )
            self.is_create_template = True

        if isinstance(self.image, str):
            if not image_exists(self.client, self.image):
                if self.verbose:
                    f_str = f"Pulling image {self.image}.."
                    f_str = f"{f_str}\n{warning_str}" if self.keep_template else f_str
                    print(f_str)

                self.image = self.client.images.pull(self.image)
                self.is_create_template = True
            else:
                self.image = self.client.images.get(self.image)
                if self.verbose:
                    print(f"Using image {self.image.tags[-1]}")

        self.container = self.client.containers.run(
            self.image,
            detach=True,
            tty=True,
            mounts=self.mounts,
            **self.runtime_configs if self.runtime_configs else {},
        )

        # Create a temporary directory for this session
        self.execute_command(f"mkdir -p {self.session_dir}", workdir="/")
        self._log(f"Created session directory: {self.session_dir}")

        if not skip_setup:
            self.setup()

    def setup(self, libraries: list | None = None):
        """
        Set up the environment for code execution.
        This function sets up the required directories and tools for code execution.
        Creates all possible profilers upfront for later use.
        
        :param libraries: List of libraries to install
        """
        if not self.container:
            raise RuntimeError("Session is not open. Please call open() method before setting up.")

        # Set up Go environment if needed
        if self.lang == SupportedLanguage.GO:
            self.execute_command("go mod init example", workdir=self.session_dir)
            self.execute_command("go mod tidy", workdir=self.session_dir)
            self.install_libraries(["golang.org/x/tools/cmd/goimports@latest"])

        # Install necessary build tools for profiler compilation
        self._ensure_build_tools_installed()

        # Set up all profiler tools upfront
        # Bash script profiler
        self.create_file(ProfilerScript, ProfilerScriptPath)
        self.execute_command(f"chmod +x {ProfilerScriptPath}")

        # C++ profiler
        self.create_file(ProfilerCpp, ProfilerCppPath)
        compile_result = self.execute_command(f"g++ -std=c++20 -O3 {ProfilerCppPath} -o {ProfilerBinaryPath}")
        if self.verbose:
            print(f"Profiler compilation result: {compile_result}")
            
        # Set up statistics calculator
        self.create_file(StatisticsCpp, StatisticsCppPath)
        statistics_compile_result = self.execute_command(f"g++ -std=c++20 -O3 {StatisticsCppPath} -o {StatisticsBinaryPath}")
        if self.verbose:
            print(f"Statistics compilation result: {statistics_compile_result}")

        # Install libraries if provided
        libraries = list(set((self.init_libraries or []) + (libraries or [])))
        self.install_libraries(libraries)

    def _ensure_build_tools_installed(self):
        """
        Ensure necessary build tools are installed for profiler compilation.
        Installs packages based on the container's OS and package manager.
        """
        # Check if we're in a Debian/Ubuntu-based container
        apt_get_exists = self.execute_command("which apt-get || echo 'not found'").text
        if apt_get_exists != 'not found':
            if self.verbose:
                print("Detected Debian/Ubuntu-based container. Installing build tools...")
            # Update package lists and install essential build tools
            self.execute_command("apt-get update -y")
            self.execute_command("apt-get install -y build-essential")
            return

        # Check if we're in an Alpine-based container
        apk_exists = self.execute_command("which apk || echo 'not found'").text
        if apk_exists != 'not found':
            if self.verbose:
                print("Detected Alpine-based container. Installing build tools...")
            self.execute_command("apk add --no-cache g++ make")
            return

        # Check if we're in a CentOS/RHEL/Fedora-based container
        yum_exists = self.execute_command("which yum || echo 'not found'").text
        if yum_exists != 'not found':
            if self.verbose:
                print("Detected CentOS/RHEL/Fedora-based container. Installing build tools...")
            self.execute_command("yum install -y gcc-c++ make")
            return

        # Check if we're in an Arch-based container
        pacman_exists = self.execute_command("which pacman || echo 'not found'").text
        if pacman_exists != 'not found':
            if self.verbose:
                print("Detected Arch-based container. Installing build tools...")
            self.execute_command("pacman -Sy --noconfirm base-devel")
            return

        if self.verbose:
            print("Warning: Could not determine container OS or package manager. Compilation may fail if build tools are not already installed.")

    def install_libraries(self, libraries: list | None = None):
        if not self.container:
            raise RuntimeError("Session is not open. Please call open() method before installing libraries.")
        
        if libraries is None:
            return
        
        for library in libraries:
            if library not in self.installed_libraries:
                if self.lang.upper() in NotSupportedLibraryInstallation:
                    raise ValueError(f"Library installation has not been supported for {self.lang} yet!")
                command = get_libraries_installation_command(self.lang, library)
                output = self.execute_command(command, workdir=self.session_dir)
                if output.exit_code != 0:
                    raise RuntimeError(f"Failed to install [{self.lang}] library [{library}] with command [{command}]: {output.text}")
                self.installed_libraries.add(library)

    def close(self):
        if self.container:
            if self.commit_container and isinstance(self.image, Image):
                self.container.commit(self.image.tags[-1])

            self.container.remove(force=True)
            self.container = None

        if self.is_create_template and not self.keep_template:
            # check if the image is used by any other container
            containers = self.client.containers.list(all=True)
            image_id = (
                self.image.id
                if isinstance(self.image, Image)
                else self.client.images.get(self.image).id
            )
            image_in_use = any(
                container.image.id == image_id for container in containers
            )

            if not image_in_use:
                if isinstance(self.image, str):
                    self.client.images.remove(self.image)
                elif isinstance(self.image, Image):
                    self.image.remove(force=True)
                else:
                    raise ValueError("Invalid image type")
            else:
                if self.verbose:
                    print(
                        f"Image {self.image.tags[-1]} is in use by other containers. Skipping removal.."
                    )

    def run(self, code: str, libraries: list | None = None, stdin: str | None = None,
            time_limit: float | None = None, memory_limit: float | None = None,
            return_statistics: bool = False, use_tty: bool = None) -> ConsoleOutput:
        """
        Run code in the sandbox environment
        :param code: Code to run
        :param libraries: List of libraries to install
        :param stdin: Standard input to pass to the program
        :param time_limit: Maximum execution time in seconds (None or 0 = no time limit)
        :param memory_limit: Memory limit in MB (None or 0 = no memory limit)
        :param return_statistics: If True, calculate and return execution statistics (runtime, memory, integral)
        :return: Console output
        """
        if not self.container:
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
        if stdin is not None:
            if stdin == '':
                stdin = '\n'
            self.create_file(stdin, StdinFilePath)

        # Get execution commands
        commands = get_code_execution_command(
            self.lang,
            code_file,
            time_limit=time_limit,
            memory_limit=memory_limit,
            has_stdin=stdin is not None,
            stdin_file_path=StdinFilePath,
            profiler_script_path=ProfilerScriptPath,
            profiler_binary_path=ProfilerBinaryPath,
            profiler_log_path=ProfilerLogPath,
        )

        # Set use_tty based on language if not explicitly provided
        if use_tty is None:
            use_tty = self.lang != SupportedLanguage.JAVASCRIPT
        
        # Execute the commands
        if skip_create_code_file:
            commands = commands[-1:]
        for command in commands:
            output = self.execute_command(command, workdir=self.session_dir, use_tty=use_tty)
            if output.exit_code != 0:
                return output
        
        # Calculate statistics if required or this is a successful execution
        if return_statistics and output.exit_code == 0:
            if self.verbose:
                print("Calculating execution statistics...")
            stats_cmd = f"{StatisticsBinaryPath} {ProfilerLogPath}"
            stats_output = self.execute_command(stats_cmd, workdir=self.session_dir)
            
            # Parse statistics output (format: "runtime_ns max_memory_kb integral")
            try:
                runtime_ns, max_memory_kb, integral = map(float, stats_output.text.strip().split())
                output.runtime = runtime_ns
                output.memory = max_memory_kb
                output.integral = integral
            except Exception as e:
                if self.verbose:
                    print(f"Failed to parse statistics: {e} - {stats_output.text}")
                output.runtime = 0.0
                output.memory = 0.0
                output.integral = 0.0
                
        return output
    
    def copy_from_runtime(self, src: str, dest: str):
        if not self.container:
            raise RuntimeError(
                "Session is not open. Please call open() method before copying files."
            )

        if self.verbose:
            print(f"Copying {self.container.short_id}:{src} to {dest}..")

        bits, stat = self.container.get_archive(src)
        if stat["size"] == 0:
            raise FileNotFoundError(f"File {src} not found in the container")

        tarstream = io.BytesIO(b"".join(bits))
        with tarfile.open(fileobj=tarstream, mode="r") as tar:
            tar.extractall(os.path.dirname(dest))

    def _put_archive(self, dest_dir: str, tarstream: io.BytesIO) -> None:
        """Put a tar archive into the container."""
        if not self.container:
            raise RuntimeError("Session is not open. Please call open() method before putting archives.")
        
        self.container.put_archive(dest_dir, tarstream)

    def copy_to_runtime(self, src: str, dest: str):
        if not self.container:
            raise RuntimeError("Session is not open. Please call open() method before copying files.")

        directory = os.path.dirname(dest)
        created = self._ensure_dir(directory)

        if self.verbose:
            if created:
                self._log(f"Creating directory {self.container.short_id}:{directory}")
            self._log(f"Copying {src} to {self.container.short_id}:{dest}..")

        tarstream = io.BytesIO()
        with tarfile.open(fileobj=tarstream, mode="w") as tar:
            tar.add(src, arcname=os.path.basename(dest))

        tarstream.seek(0)
        self._put_archive(os.path.dirname(dest) or "/", tarstream)

    def create_file(self, content: str, dest: str | Path):
        dest_str = str(dest)
        if self.verbose:
            self._log(f"Creating file {self.container.short_id if self.container else ''}:{dest_str}")
        
        self._put_bytes(dest_str, content.encode('utf-8'))

    def cat_profile(self):
        output = self.execute_command(f"cat {ProfilerLogPath}")
        if output.exit_code != 0:
            return None
        return output.text or None

    def execute_command(self, command: str | None, workdir: str | Path | None = None, use_tty: bool = True) -> ConsoleOutput:
        if not command:
            raise ValueError("Command cannot be empty")

        if not self.container:
            raise RuntimeError("Session is not open. Please call open() method before executing commands.")

        if self.verbose:
            print(f"Executing command: {command}")

        if workdir is None:
            workdir = self.session_dir

        exec_run_args = {"stream": self.stream, "tty": use_tty}
        if workdir:
            exec_run_args["workdir"] = str(workdir)
        exit_code, exec_log = self.container.exec_run(command, **exec_run_args)

        text = ""
        if self.verbose:
            print(f"Command exited with code {exit_code}")
            print("Output:", end=" ")
        if not self.stream:
            exec_log = [exec_log]

        for chunk in exec_log:
            if chunk:
                chunk_str = chunk.decode("utf-8")
                text += chunk_str
                if self.verbose:
                    print(chunk_str, end="")

        return ConsoleOutput(text=text, exit_code=exit_code)
