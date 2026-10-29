"""Agent file tools must use the same user and ACLs as the real virtual shell."""
import json

import pytest

from pm_bash_machine import Access, BashMachine
from pm_coder import _run_subagent, build_settings, make_file_tools, prompt_text


@pytest.fixture
def target(tmp_path):
    machine = BashMachine()
    machine.exec("user", "mkdir -p /work/scratch /work/ghost").check()
    machine.add_user("workflow", cwd="/work")
    machine.write_text("/work/ghost/memory.yaml", "content: accepted\n", access=Access.R)
    machine.write_text("/work/ghost/hidden.yaml", "private memory", access=Access.N)
    machine.write_text("/work/scratch/notes.txt", "first\nsecond\n")
    settings = build_settings(cwd=tmp_path, model="unused", context_window=96_000)
    tools = {tool.name: tool.function for tool in make_file_tools(settings, machine, user="workflow")}
    return machine, tools


def test_file_read_uses_the_same_scope_as_bash(target):
    machine, tools = target
    assert "content: accepted" in tools["read"]("ghost/memory.yaml")
    denied = json.loads(tools["read"]("ghost/hidden.yaml"))
    assert denied["success"] is False
    assert "private memory" not in denied["error"]
    assert machine.exec("workflow", "cat ghost/hidden.yaml").exit_code != 0


@pytest.mark.parametrize("operation", ["write", "ranged-write", "edit"])
def test_every_write_path_preserves_read_only_memory(target, operation):
    machine, tools = target
    if operation == "write":
        result = tools["write"]("ghost/memory.yaml", "changed\n")
    elif operation == "ranged-write":
        result = tools["write"]("ghost/memory.yaml", "changed", 1, 1)
    else:
        result = tools["edit"]("ghost/memory.yaml", "accepted", "changed")
    assert json.loads(result)["success"] is False
    assert machine.read_text("/work/ghost/memory.yaml") == "content: accepted\n"
    assert machine.exec("workflow", "echo changed > ghost/memory.yaml").exit_code != 0


def test_relative_paths_follow_the_workflow_users_current_directory(target):
    machine, tools = target
    machine.exec("workflow", "cd scratch").check()
    assert "second" in tools["read"]("notes.txt")
    assert tools["edit"]("notes.txt", "second", "SECOND").startswith("edited")
    machine.write_text("/work/scratch/new.txt", "old\r\n")
    assert "whole file" in tools["write"]("new.txt", "  exact\r\n\r\n")
    assert machine.read_binary("/work/scratch/new.txt") == b"  exact\r\n\r\n"
    assert not machine.is_file("/home/user/new.txt")


def test_user_write_preserves_bytes_and_existing_access(target):
    machine, _ = target
    machine.write_text("/work/scratch/scoped.txt", "old", access={"workflow": Access.RW, "user": Access.N})
    machine.write_text_as("workflow", "scratch/scoped.txt", "  exact\r\n\r\n")
    assert machine.read_binary_as("workflow", "scratch/scoped.txt") == b"  exact\r\n\r\n"
    assert machine.exec("user", "cat /work/scratch/scoped.txt").exit_code != 0


def test_user_specific_grants_do_not_use_default_user_permissions(tmp_path):
    machine = BashMachine()
    machine.add_user("reader")
    machine.write_text("/home/user/per-user.txt", "shared", access={"user": Access.RW, "reader": Access.R})
    settings = build_settings(cwd=tmp_path, model="unused", context_window=96_000)
    tools = {tool.name: tool.function for tool in make_file_tools(settings, machine, user="reader")}
    assert "shared" in tools["read"]("per-user.txt")
    assert json.loads(tools["write"]("per-user.txt", "changed"))["success"] is False


def test_virtual_prompt_paths_never_read_host_files(tmp_path):
    host_file = tmp_path / "host-secret.txt"
    host_file.write_text("host content must stay hidden", encoding="utf-8")
    machine = BashMachine()
    machine.write_text("/work/request.md", "virtual request\n", access=Access.R)
    machine.write_text("/work/hidden.md", "hidden request", access=Access.N)
    machine.add_user("workflow", cwd="/work")
    assert prompt_text("request.md", tmp_path, bash_machine=machine, user="workflow") == "virtual request\n"
    assert prompt_text(str(host_file), tmp_path, bash_machine=machine, user="workflow") == str(host_file)
    with pytest.raises(PermissionError):
        prompt_text("hidden.md", tmp_path, bash_machine=machine, user="workflow")


def test_denied_subagent_prompt_records_failure_without_starting_inference(tmp_path):
    machine = BashMachine()
    machine.add_user("workflow", cwd="/work")
    machine.write_text("/work/hidden.md", "hidden request", access=Access.N)
    settings = build_settings(cwd=tmp_path, model="unused", context_window=96_000)
    record = {}
    _run_subagent(settings, "", "hidden.md", record, machine, "workflow")
    assert record["status"].startswith("crashed: PermissionError:")
    assert "hidden request" not in record["status"]
    assert "output" not in record


def test_denied_lazy_content_is_never_evaluated(tmp_path):
    class Unreadable:
        def content(self):
            raise AssertionError("The hidden lazy file was evaluated")

    machine = BashMachine()
    machine.add_user("workflow", cwd="/work")
    machine.write_text("/work/hidden.txt", Unreadable(), access=Access.N)
    settings = build_settings(cwd=tmp_path, model="unused", context_window=96_000)
    tools = {tool.name: tool.function for tool in make_file_tools(settings, machine, user="workflow")}
    assert json.loads(tools["read"]("scratch/../hidden.txt"))["success"] is False
    with pytest.raises(PermissionError):
        machine.read_binary_as("workflow", "hidden.txt")
