from enum import Enum
from dataclasses import dataclass


class SandboxBackend(str, Enum):
    DOCKER = "docker"
    KUBERNETES = "kubernetes"
    PODMAN = "podman"
    MICROMAMBA = "micromamba"


@dataclass
class SupportedLanguage:
    PYTHON = "python"
    JAVA = "java"
    JAVASCRIPT = "javascript"
    CPP = "cpp"
    GO = "go"
    RUBY = "ruby"


@dataclass
class DefaultImage:
    PYTHON = "python:3.11.11-bookworm"
    JAVA = "openjdk:21-jdk-bookworm"
    JAVASCRIPT = "node:22.14.0-bookworm"
    CPP = "gcc:14.2.0-bookworm"
    GO = "golang:1.23.7-bookworm"
    RUBY = "ruby:3.2.7-bookworm"


NotSupportedLibraryInstallation = ["JAVA"]
SupportedLanguageValues = [
    v for k, v in SupportedLanguage.__dict__.items() if not k.startswith("__")
]