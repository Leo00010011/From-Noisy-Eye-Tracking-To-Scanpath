import os
import shutil
from pathlib import Path

def move_logs():
    # Define the source (current directory) and destination
    source_dir = Path.cwd()
    target_dir = source_dir / "logs"

    # Create the 'logs' folder if it doesn't exist
    target_dir.mkdir(exist_ok=True)

    # Find all .log files in the current directory
    log_files = list(source_dir.glob("*.log"))

    if not log_files:
        print("No .log files found.")
        return

    # Move each file
    for file_path in log_files:
        try:
            shutil.move(str(file_path), target_dir / file_path.name)
            print(f"Moved: {file_path.name}")
        except Exception as e:
            print(f"Error moving {file_path.name}: {e}")

    print(f"\nSuccessfully moved {len(log_files)} file(s) to {target_dir}")

if __name__ == "__main__":
    move_logs()