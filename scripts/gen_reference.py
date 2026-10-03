"""Generate the optional decision client's API from its source and docstrings."""
import mkdocs_gen_files

with mkdocs_gen_files.open("reference/pm_decision.md", "w") as stream:
    stream.write("# pm_decision\n\n::: pm_decision\n")
mkdocs_gen_files.set_edit_path("reference/pm_decision.md", "pm_decision.py")
with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as stream:
    stream.write("* [pm_decision](pm_decision.md)\n")
