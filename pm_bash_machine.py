"""
bash_machine.py

A synchronous, thread-safe, in-memory Bash machine for LLM agents.

Dependency:
    pip install just-bash==0.2.1

The shell and filesystem come from just-bash-py. This module adds:
- named users
- per-file N / R / RW access
- lazy text and binary files
- persistent cwd and exported variables per user
- host-side load and dump helpers
- one machine-wide RLock

The access policy lives in the filesystem view. Shell commands do not get
parsed or filtered for permissions. grep, cat, redirection, sed, rm, cp, and
other commands therefore hit the same permission boundary.

Lazy content:
- str and bytes values are stored by reference.
- Any object with content() -> str or bytes is accepted.
- content() runs only when the file is first read.
- The returned object is cached by reference.
- A shell write replaces the lazy value with an ordinary VFS file.

Access:
- Access.N  : no content read, no mutation
- Access.R  : read only
- Access.RW : read and write
- No explicit rule means Access.RW.

This is intentionally not a POSIX UID/GID implementation. It is a small
per-file access layer for agent workspaces.

The public API is fully synchronous. just-bash-py itself uses async filesystem
methods internally, so _UserFs implements that interface with async def methods.
Callers never use asyncio.

Target: just-bash-py 0.2.1.
"""

from __future__ import annotations

import base64
import copy
import os
import posixpath
import shlex
import threading
import uuid
from dataclasses import dataclass, replace
from enum import StrEnum, auto
from pathlib import Path
from typing import Any, Mapping

from just_bash import Bash
from just_bash.fs import InMemoryFs


class Access(StrEnum):
    N = auto()
    R = auto()
    RW = auto()


AccessSpec = Access | Mapping[str, Access]


@dataclass(slots=True)
class BashResult:
    stdout: str
    stderr: str
    exit_code: int

    @property
    def ok(self) -> bool:
        return self.exit_code == 0

    def check(self) -> "BashResult":
        if self.exit_code != 0:
            raise RuntimeError(
                f"Bash exited with code {self.exit_code}\n"
                f"stdout:\n{self.stdout}\n"
                f"stderr:\n{self.stderr}"
            )
        return self


_UNSET = object()


@dataclass(slots=True)
class _Content:
    source: Any
    binary: bool
    cached: Any = _UNSET

    def get(self) -> str | bytes:
        if self.cached is _UNSET:
            value = self.source
            if not isinstance(value, (str, bytes)):
                value = value.content()

            if self.binary:
                if not isinstance(value, bytes):
                    raise TypeError("binary content() must return bytes")
            elif not isinstance(value, str):
                raise TypeError("text content() must return str")

            self.cached = value

        return self.cached

    def known_size(self) -> int | None:
        if self.cached is _UNSET:
            if isinstance(self.source, bytes):
                return len(self.source)
            if isinstance(self.source, str):
                return len(self.source.encode("utf-8"))
            return None

        if isinstance(self.cached, bytes):
            return len(self.cached)
        return len(self.cached.encode("utf-8"))


@dataclass(slots=True)
class _User:
    bash: Bash
    cwd: str
    exports: str = ""


def _path(path: str) -> str:
    return posixpath.normpath(path)


def _under(path: str, root: str) -> bool:
    path = _path(path)
    root = _path(root)
    return path == root or path.startswith(root.rstrip("/") + "/")


def _permission_error(path: str) -> PermissionError:
    return PermissionError(13, "Permission denied", path)


class _UserFs:
    """
    A just-bash filesystem view for one user.

    The backing InMemoryFs is shared by all users.
    """

    def __init__(self, machine: "BashMachine", user: str):
        self._machine = machine
        self._user = user
        self._base = machine._fs

    def __getattr__(self, name: str):
        return getattr(self._base, name)

    def _need_read(self, path: str) -> None:
        if self._machine._access(path, self._user) is Access.N:
            raise _permission_error(path)

    def _need_write(self, path: str) -> None:
        if self._machine._access(path, self._user) is not Access.RW:
            raise _permission_error(path)

    def _need_write_tree(self, path: str) -> None:
        self._need_write(path)
        for protected_path in self._machine._acl:
            if _under(protected_path, path):
                self._need_write(protected_path)

    async def read_file(self, path: str, *args, **kwargs):
        path = _path(path)
        self._need_read(path)

        content = self._machine._content.get(path)
        if content is not None:
            return content.get()

        return await self._base.read_file(path, *args, **kwargs)

    async def write_file(self, path: str, content, *args, **kwargs):
        path = _path(path)
        self._need_write(path)
        self._machine._content.pop(path, None)
        return await self._base.write_file(path, content, *args, **kwargs)

    async def append_file(self, path: str, content, *args, **kwargs):
        path = _path(path)
        self._need_write(path)

        lazy = self._machine._content.pop(path, None)
        if lazy is not None:
            current = lazy.get()
            if type(current) is not type(content):
                raise TypeError(
                    f"cannot append {type(content).__name__} "
                    f"to {type(current).__name__}"
                )
            return await self._base.write_file(
                path, current + content, *args, **kwargs
            )

        return await self._base.append_file(path, content, *args, **kwargs)

    async def exists(self, path: str, *args, **kwargs):
        return await self._base.exists(path, *args, **kwargs)

    async def readdir(self, path: str, *args, **kwargs):
        return await self._base.readdir(path, *args, **kwargs)

    async def stat(self, path: str, *args, **kwargs):
        path = _path(path)
        result = await self._base.stat(path, *args, **kwargs)
        return self._machine._visible_stat(result, path, self._user)

    async def lstat(self, path: str, *args, **kwargs):
        path = _path(path)
        method = getattr(self._base, "lstat", self._base.stat)
        result = await method(path, *args, **kwargs)
        return self._machine._visible_stat(result, path, self._user)

    async def mkdir(self, path: str, *args, **kwargs):
        self._need_write(_path(path))
        return await self._base.mkdir(path, *args, **kwargs)

    async def rm(self, path: str, *args, **kwargs):
        path = _path(path)
        self._need_write_tree(path)
        result = await self._base.rm(path, *args, **kwargs)
        self._machine._remove_metadata(path)
        return result

    async def unlink(self, path: str, *args, **kwargs):
        path = _path(path)
        self._need_write(path)
        method = getattr(self._base, "unlink", self._base.rm)
        result = await method(path, *args, **kwargs)
        self._machine._remove_metadata(path)
        return result

    async def mv(self, source: str, destination: str, *args, **kwargs):
        source = _path(source)
        destination = _path(destination)

        self._need_write_tree(source)
        self._need_write_tree(destination)

        result = await self._base.mv(source, destination, *args, **kwargs)
        self._machine._move_metadata(source, destination)
        return result

    async def rename(self, source: str, destination: str, *args, **kwargs):
        source = _path(source)
        destination = _path(destination)

        self._need_write_tree(source)
        self._need_write_tree(destination)

        method = getattr(self._base, "rename", self._base.mv)
        result = await method(source, destination, *args, **kwargs)
        self._machine._move_metadata(source, destination)
        return result

    async def cp(self, source: str, destination: str, *args, **kwargs):
        source = _path(source)
        destination = _path(destination)

        self._need_read_tree(source)
        self._need_write_tree(destination)

        result = await self._base.cp(source, destination, *args, **kwargs)
        self._machine._copy_metadata(source, destination)
        return result

    async def copy(self, source: str, destination: str, *args, **kwargs):
        source = _path(source)
        destination = _path(destination)

        self._need_read_tree(source)
        self._need_write_tree(destination)

        method = getattr(self._base, "copy", self._base.cp)
        result = await method(source, destination, *args, **kwargs)
        self._machine._copy_metadata(source, destination)
        return result

    def _need_read_tree(self, path: str) -> None:
        self._need_read(path)
        for protected_path in self._machine._acl:
            if _under(protected_path, path):
                self._need_read(protected_path)

    async def chmod(self, path: str, *args, **kwargs):
        path = _path(path)
        self._need_write(path)
        return await self._base.chmod(path, *args, **kwargs)

    async def truncate(self, path: str, *args, **kwargs):
        path = _path(path)
        self._need_write(path)
        self._machine._content.pop(path, None)
        return await self._base.truncate(path, *args, **kwargs)

    async def utimes(self, path: str, *args, **kwargs):
        path = _path(path)
        self._need_write(path)
        return await self._base.utimes(path, *args, **kwargs)

    async def symlink(self, *args, **kwargs):
        # A symlink could otherwise alias a protected target under an RW path.
        raise PermissionError(13, "symlinks are disabled in ACL views")

    async def link(self, *args, **kwargs):
        # A hard link has the same ACL aliasing problem.
        raise PermissionError(13, "hard links are disabled in ACL views")


class BashMachine:
    """
    One long-lived fake machine shared by many normal Python threads.

    User "user" exists by default.
    """

    def __init__(self, *, cwd: str = "/home/user"):
        self._lock = threading.RLock()
        self._fs = InMemoryFs()
        self._acl: dict[str, AccessSpec] = {}
        self._content: dict[str, _Content] = {}
        self._users: dict[str, _User] = {}

        # This shell bypasses ACLs. Only BashMachine host methods use it.
        self._admin = Bash(fs=self._fs, cwd=cwd)

        # Do not depend on Result.check() existing in just-bash.
        result = self._admin.run(
            "mkdir -p /home/user /tmp /tools /shared /scratch"
        )
        if result.exit_code != 0:
            raise RuntimeError(result.stderr)

        self.add_user("user", cwd=cwd)

    def add_user(self, name: str, *, cwd: str = "/home/user") -> None:
        with self._lock:
            if name in self._users:
                raise ValueError(f"user already exists: {name}")

            view = _UserFs(self, name)
            bash = Bash(
                fs=view,
                cwd=cwd,
                env={
                    "USER": name,
                    "LOGNAME": name,
                    "HOME": "/home/user",
                },
            )
            self._users[name] = _User(bash=bash, cwd=cwd)

    def exec(self, user: str, command: str) -> BashResult:
        """
        Execute one blocking shell call as one virtual user.

        Calls from all threads and users serialize on one RLock.
        """
        with self._lock:
            shell = self._users[user]
            marker = f"__BM_STATE_{uuid.uuid4().hex}__"
            cwd_marker = marker + "_CWD"
            exports_marker = marker + "_EXPORTS"
            end_marker = marker + "_END"

            wrapped = (
                f"{shell.exports}\n"
                f"cd {shlex.quote(shell.cwd)} || exit $?\n"
                f"{command}\n"
                "__bm_status=$?\n"
                f"printf {shlex.quote(cwd_marker + chr(10))}\n"
                "pwd\n"
                f"printf {shlex.quote(exports_marker + chr(10))}\n"
                "export -p\n"
                f"printf {shlex.quote(end_marker + chr(10))}\n"
                'exit "$__bm_status"\n'
            )

            try:
                result = shell.bash.run(wrapped)
            except PermissionError as exc:
                return BashResult(
                    stdout="",
                    stderr=f"{exc}\n",
                    exit_code=1,
                )

            stdout = result.stdout
            start = cwd_marker + "\n"

            if start in stdout:
                visible, state = stdout.rsplit(start, 1)
                cwd, state = state.split("\n", 1)

                exports_start = exports_marker + "\n"
                _, state = state.split(exports_start, 1)
                exports, after = state.split(end_marker + "\n", 1)

                shell.cwd = cwd
                shell.exports = exports
                stdout = visible + after

            return BashResult(
                stdout=stdout.strip(),
                stderr=result.stderr,
                exit_code=result.exit_code,
            )

    def write_text(
        self,
        path: str,
        content: str | Any,
        *,
        access: AccessSpec = Access.RW,
    ) -> None:
        """Add text without copying a large provider-backed value."""
        self._write_virtual(path, content, binary=False, access=access)

    def write_binary(
        self,
        path: str,
        content: bytes | Any,
        *,
        access: AccessSpec = Access.RW,
    ) -> None:
        """Add binary data without copying a large provider-backed value."""
        self._write_virtual(path, content, binary=True, access=access)

    def set_access(self, path: str, access: AccessSpec) -> None:
        with self._lock:
            self._acl[_path(path)] = access

    def read_text(self, path: str) -> str:
        """Read a file as the host, bypassing user ACLs."""
        with self._lock:
            path = _path(path)
            lazy = self._content.get(path)
            if lazy is not None:
                value = lazy.get()
                if not isinstance(value, str):
                    raise TypeError(f"{path} is binary")
                return value

            result = self._admin.run(f"cat -- {shlex.quote(path)}")
            if result.exit_code != 0:
                raise RuntimeError(result.stderr)
            return result.stdout.strip()

    def read_binary(self, path: str) -> bytes:
        """Read a file as bytes, bypassing user ACLs."""
        with self._lock:
            path = _path(path)
            lazy = self._content.get(path)
            if lazy is not None:
                value = lazy.get()
                if isinstance(value, bytes):
                    return value
                return value.encode("utf-8")

            result = self._admin.run(
                f"base64 < {shlex.quote(path)} | tr -d '\\n'"
            )
            if result.exit_code != 0:
                raise RuntimeError(result.stderr)
            return base64.b64decode(result.stdout)

    def load_text(
        self,
        real_path: str | os.PathLike[str],
        virtual_path: str | None = None,
        *,
        access: AccessSpec = Access.RW,
        encoding: str = "utf-8",
    ) -> str:
        real = Path(real_path)
        virtual = virtual_path or f"/home/user/{real.name}"
        self.write_text(
            virtual,
            real.read_text(encoding=encoding),
            access=access,
        )
        return virtual

    def load_binary(
        self,
        real_path: str | os.PathLike[str],
        virtual_path: str | None = None,
        *,
        access: AccessSpec = Access.RW,
    ) -> str:
        real = Path(real_path)
        virtual = virtual_path or f"/home/user/{real.name}"
        self.write_binary(virtual, real.read_bytes(), access=access)
        return virtual

    def dump(self, virtual_path: str, real_path: str | os.PathLike[str]) -> Path:
        """Dump one virtual file exactly as bytes."""
        real = Path(real_path)
        real.parent.mkdir(parents=True, exist_ok=True)
        real.write_bytes(self.read_binary(virtual_path))
        return real

    def _write_virtual(
        self,
        path: str,
        content: Any,
        *,
        binary: bool,
        access: AccessSpec,
    ) -> None:
        with self._lock:
            path = _path(path)
            parent = posixpath.dirname(path) or "/"

            result = self._admin.run(
                f"mkdir -p -- {shlex.quote(parent)} && "
                f": > {shlex.quote(path)}"
            )
            if result.exit_code != 0:
                raise RuntimeError(result.stderr)

            self._content[path] = _Content(content, binary=binary)
            self._acl[path] = access

    def _access(self, path: str, user: str) -> Access:
        spec = self._acl.get(_path(path), Access.RW)

        if isinstance(spec, Access):
            return spec

        return spec.get(user, spec.get("default", Access.RW))

    def _remove_metadata(self, root: str) -> None:
        for path in list(self._content):
            if _under(path, root):
                del self._content[path]

        for path in list(self._acl):
            if _under(path, root):
                del self._acl[path]

    def _move_metadata(self, source: str, destination: str) -> None:
        self._remap_metadata(source, destination, copy_values=False)

    def _copy_metadata(self, source: str, destination: str) -> None:
        self._remap_metadata(source, destination, copy_values=True)

    def _remap_metadata(
        self,
        source: str,
        destination: str,
        *,
        copy_values: bool,
    ) -> None:
        source = _path(source)
        destination = _path(destination)

        content_items = [
            (path, value)
            for path, value in self._content.items()
            if _under(path, source)
        ]
        acl_items = [
            (path, value)
            for path, value in self._acl.items()
            if _under(path, source)
        ]

        if not copy_values:
            self._remove_metadata(source)

        for path, value in content_items:
            relative = posixpath.relpath(path, source)
            target = (
                destination
                if relative == "."
                else posixpath.join(destination, relative)
            )
            self._content[target] = value

        for path, value in acl_items:
            relative = posixpath.relpath(path, source)
            target = (
                destination
                if relative == "."
                else posixpath.join(destination, relative)
            )
            self._acl[target] = value

    def _visible_stat(self, stat: Any, path: str, user: str):
        access = self._access(path, user)

        mode = stat.mode
        if access is Access.N:
            mode &= ~0o666
        elif access is Access.R:
            mode &= ~0o222

        size = stat.size
        lazy = self._content.get(path)
        if lazy is not None:
            known_size = lazy.known_size()
            if known_size is not None:
                size = known_size

        if mode == stat.mode and size == stat.size:
            return stat

        if hasattr(stat, "_replace"):
            return stat._replace(mode=mode, size=size)

        try:
            return replace(stat, mode=mode, size=size)
        except TypeError:
            clone = copy.copy(stat)
            clone.mode = mode
            clone.size = size
            return clone


if __name__ == "__main__":
    class VirtualText:
        def content(self):
            return "lorem ipsum\n" * 10_000

    class VirtualBinary:
        def content(self):
            return b"\x89PNG\r\n\x1a\n" + b"\x00" * 100_000

    vm = BashMachine()
    vm.add_user("floppa")

    vm.write_text("/home/user/def.md", "hello")
    vm.write_text("/home/user/readme.md", "hello", access=Access.R)
    vm.write_text("/home/user/dontreadme.md", "hello", access=Access.N)
    vm.write_text(
        "/home/user/maybe_readme.md",
        "hello",
        access={"default": Access.RW, "floppa": Access.N},
    )

    vm.write_text("/home/user/huge_markdown.md", VirtualText())
    vm.write_binary("/home/user/huge_image.png", VirtualBinary())

    vm.exec("user", "mkdir -p project && cd project").check()
    vm.exec("user", "echo alpha > notes.txt").check()
    vm.exec("user", "echo beta >> notes.txt").check()

    print(vm.exec("user", "pwd").check().stdout, end="")
    print(vm.exec("user", "grep -Rni alpha .").check().stdout, end="")

    denied = vm.exec("floppa", "cat /home/user/maybe_readme.md")
    print("floppa exit:", denied.exit_code)
    print("floppa stderr:", denied.stderr, end="")
