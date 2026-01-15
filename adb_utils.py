import subprocess

def adb_shell(cmd: str) -> str:
    """
    Run an adb shell command and return stdout.
    """
    return subprocess.check_output(
        ["adb", "shell", cmd],
        stderr=subprocess.DEVNULL,
        text=True
    ).strip()
