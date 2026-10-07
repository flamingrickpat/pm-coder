"""Generate the optional decision client's API from its source and docstrings."""
import mkdocs_gen_files

with mkdocs_gen_files.open("reference/pm_decision.md", "w") as stream:
    stream.write("# pm_decision\n\n::: pm_decision\n    options:\n        inherited_members: [__init__]\n        show_if_no_docstring: true\n")
mkdocs_gen_files.set_edit_path("reference/pm_decision.md", "pm_decision.py")
with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as stream:
    stream.write("* [pm_decision](pm_decision.md)\n* [Native MCP transport](mcp_transport.md)\n")
with mkdocs_gen_files.open("reference/mcp_transport.md", "w") as stream:
    stream.write("# Native MCP transport\n\n::: pm_coder.load_timed_mcp_toolsets\n")
mkdocs_gen_files.set_edit_path("reference/mcp_transport.md", "pm_coder.py")
