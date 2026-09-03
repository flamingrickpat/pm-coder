"""
test_bash_machine.py

Tests for bash_machine.py.

Install:
    pip install pytest just-bash==0.2.1

Run:
    pytest -v test_bash_machine.py

The tests use the real just-bash shell and in-memory filesystem.
No mocks are used.
"""

from __future__ import annotations

import threading

import pytest

from pm_bash_machine import Access, BashMachine


class LazyText:
    def __init__(self, value: str):
        self.value = value
        self.calls = 0

    def content(self) -> str:
        self.calls += 1
        return self.value


class LazyBinary:
    def __init__(self, value: bytes):
        self.value = value
        self.calls = 0

    def content(self) -> bytes:
        self.calls += 1
        return self.value


@pytest.fixture
def vm():
    machine = BashMachine()
    machine.add_user("floppa")
    return machine


def test_default_file_is_read_write(vm):
    vm.write_text("/home/user/file.txt", "hello")

    assert vm.exec("floppa", "cat /home/user/file.txt").check().stdout == "hello"

    vm.exec("floppa", "echo changed > /home/user/file.txt").check()

    assert vm.read_text("/home/user/file.txt") == "changed"


def test_read_only_file_can_be_read(vm):
    vm.write_text(
        "/home/user/readme.md",
        "hello",
        access=Access.R,
    )

    result = vm.exec("floppa", "cat /home/user/readme.md").check()

    assert result.stdout == "hello"


def test_read_only_file_cannot_be_overwritten(vm):
    vm.write_text(
        "/home/user/readme.md",
        "hello",
        access=Access.R,
    )

    result = vm.exec(
        "floppa",
        "echo changed > /home/user/readme.md",
    )

    assert not result.ok
    assert vm.read_text("/home/user/readme.md") == "hello"


def test_no_access_file_cannot_be_read(vm):
    vm.write_text(
        "/home/user/secret.md",
        "secret",
        access=Access.N,
    )

    result = vm.exec("floppa", "cat /home/user/secret.md")

    assert not result.ok
    assert "Permission denied" in result.stderr


def test_no_access_file_cannot_be_overwritten(vm):
    vm.write_text(
        "/home/user/secret.md",
        "secret",
        access=Access.N,
    )

    result = vm.exec(
        "floppa",
        "echo stolen > /home/user/secret.md",
    )

    assert not result.ok
    assert vm.read_text("/home/user/secret.md") == "secret"


def test_no_access_file_name_is_still_visible(vm):
    vm.write_text(
        "/home/user/secret.md",
        "secret",
        access=Access.N,
    )

    result = vm.exec("floppa", "ls /home/user").check()

    assert "secret.md" in result.stdout


def test_per_user_override(vm):
    vm.write_text(
        "/home/user/maybe.md",
        "hello",
        access={
            "default": Access.RW,
            "floppa": Access.N,
        },
    )

    assert vm.exec("user", "cat /home/user/maybe.md").check().stdout == "hello"

    denied = vm.exec("floppa", "cat /home/user/maybe.md")
    assert not denied.ok


def test_missing_user_rule_uses_default(vm):
    vm.write_text(
        "/home/user/default.md",
        "hello",
        access={
            "default": Access.R,
            "someone_else": Access.N,
        },
    )

    assert vm.exec("floppa", "cat /home/user/default.md").check().stdout == "hello"

    result = vm.exec("floppa", "echo nope > /home/user/default.md")
    assert not result.ok


def test_set_access_changes_existing_file(vm):
    vm.write_text("/home/user/file.txt", "hello")

    vm.exec("floppa", "cat /home/user/file.txt").check()

    vm.set_access(
        "/home/user/file.txt",
        {"default": Access.RW, "floppa": Access.N},
    )

    assert not vm.exec("floppa", "cat /home/user/file.txt").ok
    assert vm.exec("user", "cat /home/user/file.txt").check().stdout == "hello"


def test_filesystem_is_shared_between_users(vm):
    vm.exec("user", "echo shared > /shared/file.txt").check()

    result = vm.exec("floppa", "cat /shared/file.txt").check()

    assert result.stdout == "shared"


def test_each_user_has_own_persistent_cwd(vm):
    vm.exec("user", "mkdir -p /home/user/a && cd /home/user/a").check()
    vm.exec("floppa", "mkdir -p /home/user/b && cd /home/user/b").check()

    assert vm.exec("user", "pwd").check().stdout.strip() == "/home/user/a"
    assert vm.exec("floppa", "pwd").check().stdout.strip() == "/home/user/b"


def test_exported_environment_persists_per_user(vm):
    vm.exec("user", "export COLOR=blue").check()
    vm.exec("floppa", "export COLOR=red").check()

    assert vm.exec("user", 'printf "%s" "$COLOR"').check().stdout == "blue"
    assert vm.exec("floppa", 'printf "%s" "$COLOR"').check().stdout == "red"


def test_shell_script_can_be_saved_and_sourced(vm):
    vm.write_text(
        "/tools/search.sh",
        """
search_all() {
    pattern="$1"
    root="${2:-.}"
    grep -Rni -- "$pattern" "$root"
}
""".lstrip(),
    )

    vm.write_text("/shared/a.txt", "alpha\nbeta\n")
    vm.write_text("/shared/b.txt", "gamma\nalpha\n")

    result = vm.exec(
        "floppa",
        "source /tools/search.sh; search_all alpha /shared",
    ).check()

    assert "/shared/a.txt" in result.stdout
    assert "/shared/b.txt" in result.stdout


def test_lazy_text_is_not_loaded_on_write(vm):
    source = LazyText("very large text")

    vm.write_text("/home/user/lazy.txt", source)

    assert source.calls == 0


def test_lazy_text_loads_once_on_first_read(vm):
    source = LazyText("very large text")
    vm.write_text("/home/user/lazy.txt", source)

    assert vm.exec("user", "cat /home/user/lazy.txt").check().stdout == "very large text"
    assert source.calls == 1

    assert vm.exec("user", "cat /home/user/lazy.txt").check().stdout == "very large text"
    assert source.calls == 1


def test_overwriting_lazy_text_does_not_materialize_old_value(vm):
    source = LazyText("old")
    vm.write_text("/home/user/lazy.txt", source)

    vm.exec("user", "printf new > /home/user/lazy.txt").check()

    assert source.calls == 0
    assert vm.read_text("/home/user/lazy.txt") == "new"


def test_lazy_binary_is_not_loaded_on_write(vm):
    source = LazyBinary(b"\x00\x01\x02")

    vm.write_binary("/home/user/data.bin", source)

    assert source.calls == 0


def test_lazy_binary_loads_once_and_preserves_bytes(vm):
    payload = bytes(range(256)) * 32
    source = LazyBinary(payload)

    vm.write_binary("/home/user/data.bin", source)

    assert vm.read_binary("/home/user/data.bin") == payload
    assert source.calls == 1

    assert vm.read_binary("/home/user/data.bin") == payload
    assert source.calls == 1


def test_lazy_binary_obeys_acl(vm):
    source = LazyBinary(b"\x89PNG\r\n\x1a\n")

    vm.write_binary(
        "/home/user/image.png",
        source,
        access={"default": Access.RW, "floppa": Access.N},
    )

    result = vm.exec("floppa", "cat /home/user/image.png")

    assert not result.ok
    assert source.calls == 0


def test_host_read_bypasses_acl(vm):
    vm.write_text(
        "/home/user/secret.txt",
        "secret",
        access=Access.N,
    )

    assert vm.read_text("/home/user/secret.txt") == "secret"


def test_load_and_dump_text(vm, tmp_path):
    source = tmp_path / "source.txt"
    source.write_text("hello äöü\n", encoding="utf-8")

    vm.load_text(
        source,
        "/shared/source.txt",
        access=Access.R,
    )

    assert vm.exec("user", "cat /shared/source.txt").check().stdout == "hello äöü"
    assert not vm.exec("user", "echo nope > /shared/source.txt").ok

    target = tmp_path / "dumped.txt"
    vm.dump("/shared/source.txt", target)

    assert target.read_text(encoding="utf-8") == "hello äöü\n"


def test_load_and_dump_binary(vm, tmp_path):
    payload = bytes(range(256)) * 8
    source = tmp_path / "source.bin"
    source.write_bytes(payload)

    vm.load_binary(
        source,
        "/shared/source.bin",
        access=Access.R,
    )

    target = tmp_path / "dumped.bin"
    vm.dump("/shared/source.bin", target)

    assert target.read_bytes() == payload


def test_read_only_file_cannot_be_removed(vm):
    vm.write_text(
        "/home/user/readme.md",
        "hello",
        access=Access.R,
    )

    result = vm.exec("floppa", "rm /home/user/readme.md")

    assert not result.ok
    assert vm.read_text("/home/user/readme.md") == "hello"


def test_no_access_source_cannot_be_copied(vm):
    vm.write_text(
        "/home/user/secret.txt",
        "secret",
        access={"default": Access.RW, "floppa": Access.N},
    )

    result = vm.exec(
        "floppa",
        "cp /home/user/secret.txt /scratch/stolen.txt",
    )

    assert not result.ok


def test_symlinks_are_disabled_for_user_shells(vm):
    vm.write_text("/shared/file.txt", "hello")

    result = vm.exec(
        "floppa",
        "ln -s /shared/file.txt /scratch/link.txt",
    )

    assert not result.ok


def test_many_threads_can_share_one_machine(vm):
    thread_count = 20
    vm.write_text("/shared/threaded.txt", "")

    errors = []

    def worker(number: int):
        try:
            vm.exec(
                "user",
                f"echo thread-{number} >> /shared/threaded.txt",
            ).check()
        except Exception as exc:
            errors.append(exc)

    threads = [
        threading.Thread(target=worker, args=(i,))
        for i in range(thread_count)
    ]

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    assert errors == []

    lines = vm.read_text("/shared/threaded.txt").splitlines()

    assert len(lines) == thread_count
    assert set(lines) == {
        f"thread-{i}"
        for i in range(thread_count)
    }


def test_check_raises_for_failed_shell_command(vm):
    result = vm.exec("user", "false")

    with pytest.raises(RuntimeError):
        result.check()


def test_unknown_user_is_programmer_error(vm):
    with pytest.raises(KeyError):
        vm.exec("does-not-exist", "echo hello")
