"""Minimal fcntl compatibility shim for Windows local Metaflow usage.

Metaflow imports modules that depend on fcntl (POSIX-only), even for local runs.
This shim provides the small surface needed to avoid import failures on Windows.
"""

LOCK_SH = 1
LOCK_EX = 2
LOCK_NB = 4
LOCK_UN = 8

F_GETFD = 1
F_SETFD = 2
F_GETFL = 3
F_SETFL = 4
FD_CLOEXEC = 1


def flock(fd, operation):
    """No-op file lock for Windows local development."""
    return 0


def fcntl(fd, op, arg=0):
    """No-op fcntl call for Windows local development."""
    return 0
