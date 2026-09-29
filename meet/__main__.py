"""`python -m meet`, which is what the `meet` command runs.

`update` is dispatched before anything else is imported. On Windows a loaded
module's files (numpy's DLLs, the `meet.exe` launcher) cannot be replaced while
this process runs, and the updater's job is to replace them.
"""

import sys

if __name__ == "__main__":
    if sys.argv[1:2] == ["update"]:
        from meet.update import main

        sys.exit(main(sys.argv[2:]))

    from meet.cli import app

    app(prog_name="meet")
