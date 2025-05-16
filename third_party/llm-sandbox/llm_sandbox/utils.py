import docker
import docker.errors
from typing import Optional
from llm_sandbox.const import SupportedLanguage
from pathlib import Path

from docker import DockerClient


def image_exists(client: DockerClient, image: str) -> bool:
    """
    Check if a Docker image exists
    :param client: Docker client
    :param image: Docker image
    :return: True if the image exists, False otherwise
    """
    try:
        client.images.get(image)
        return True
    except docker.errors.ImageNotFound:
        return False
    except Exception as e:
        raise e


def get_libraries_installation_command(lang: str, library: str) -> Optional[str]:
    """
    Get the command to install libraries for the given language
    :param lang: Programming language
    :param library: List of libraries
    :return: Installation command
    """
    if lang == SupportedLanguage.PYTHON:
        return f"pip install {library}"
    elif lang == SupportedLanguage.JAVA:
        return f"mvn install:install-file -Dfile={library}"
    elif lang == SupportedLanguage.JAVASCRIPT:
        return f"npm install {library}"
    elif lang == SupportedLanguage.CPP:
        return f"apt-get install {library}"
    elif lang == SupportedLanguage.GO:
        return f"go install {library}"
    elif lang == SupportedLanguage.RUBY:
        return f"gem install {library}"
    else:
        raise ValueError(f"Language {lang} is not supported")


def get_code_file_extension(lang: str) -> str:
    """
    Get the file extension for the given language
    :param lang: Programming language
    :return: File extension
    """
    if lang == SupportedLanguage.PYTHON:
        return "py"
    elif lang == SupportedLanguage.JAVA:
        return "java"
    elif lang == SupportedLanguage.JAVASCRIPT:
        return "js"
    elif lang == SupportedLanguage.CPP:
        return "cpp"
    elif lang == SupportedLanguage.GO:
        return "go"
    elif lang == SupportedLanguage.RUBY:
        return "rb"
    else:
        raise ValueError(f"Language {lang} is not supported")


WORKSPACE_DIR = Path("/workspace")
StdinFilePath = WORKSPACE_DIR / "stdin.txt"
ProfilerLogPath = WORKSPACE_DIR / "memory_profile.log"
ProfilerScript = (Path(__file__).parent / "profiler.sh").read_text()
ProfilerScriptPath = WORKSPACE_DIR / "profile.sh"

ProfilerCpp = (Path(__file__).parent / "profiler.cpp").read_text()
ProfilerCppPath = WORKSPACE_DIR / "profiler.cpp"
ProfilerBinaryPath = WORKSPACE_DIR / "profiler"

# Define statistics constants using file paths
StatisticsCpp = (Path(__file__).parent / "statistics.cpp").read_text()
StatisticsCppPath = WORKSPACE_DIR / "statistics.cpp"
StatisticsBinaryPath = WORKSPACE_DIR / "statistics"


def get_code_execution_command(
    lang: str,
    code_file: str | Path, 
    time_limit: Optional[float] = None,
    memory_limit: Optional[float] = None,
    sampling_interval: float = 0.0001,
    has_stdin: bool = False, 
    stdin_file_path: str | Path = StdinFilePath,
    profiler_script_path: str | Path = ProfilerScriptPath,
    profiler_binary_path: str | Path = ProfilerBinaryPath,
    profiler_log_path: str | Path = ProfilerLogPath,
) -> list[str]:
    """
    Return the execution command for the given language and code file.
    Args:
        lang: Language of the code
        code_file: Path to the code file
        time_limit: Time limit in seconds (None or 0 = no time limit)
        memory_limit: Memory limit in MB (None or 0 = no memory limit)
        sampling_interval: Sampling interval for profiler
        has_stdin: Whether to read from stdin
        stdin_file_path: Path to read stdin from
        profiler_script_path: Path to the profiler script
        profiler_binary_path: Path to the profiler binary
        profiler_log_path: Path to write profiler logs
    Returns:
        A list of commands to execute for the given language and code file
    """
    commands = None
    if lang == SupportedLanguage.PYTHON:
        commands = [f"python -u {code_file}"]
    elif lang == SupportedLanguage.JAVA:
        commands = [f"javac --enable-preview --release 21 {code_file}", f"java --enable-preview {Path(code_file).stem}"]
    elif lang == SupportedLanguage.JAVASCRIPT:
        commands = [f"node --harmony {code_file}"]
    elif lang == SupportedLanguage.CPP:
        commands = [f"g++ -std=c++20 -O2 -o a.out {code_file}", "./a.out"]
    elif lang == SupportedLanguage.GO:
        commands = [f"goimports -w {code_file}", f"go run {code_file}"]
    elif lang == SupportedLanguage.RUBY:
        commands = [f"ruby {code_file}"]
    else:
        raise ValueError(f"Language {lang} is not supported")
    

    memory_limit = max(0, memory_limit or 0)
    cmd = f"{profiler_binary_path} {sampling_interval} {profiler_log_path} {memory_limit} -- {commands[-1]}"
    
    time_limit = max(0, time_limit or 0)
    if time_limit > 0:
        cmd = f"timeout {time_limit} {cmd}"
    if has_stdin:
        cmd = f"bash -c '{cmd} < {stdin_file_path}'"
    
    commands[-1] = cmd
    return commands
