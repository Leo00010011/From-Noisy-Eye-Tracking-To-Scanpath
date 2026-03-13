import os
import re

def rename_logs_by_date():
    # Regular expression to find the date string in the file content
    # Matches "Starting debug at: " followed by the rest of the line
    date_pattern = re.compile(r"Starting debug at:\s*(.*)")
    
    # Regular expression to match the filename format: something_something_number.log
    file_pattern = re.compile(r"^(.+_.+_)\d+\.log$")

    # Iterate through all files in the current directory
    for filename in os.listdir('.'):
        match = file_pattern.match(filename)
        
        if match:
            prefix = match.group(1)
            
            try:
                with open(filename, 'r') as f:
                    content = f.read()
                    date_match = date_pattern.search(content)
                    
                    if date_match:
                        # Clean the date string to be filename-friendly
                        # Removes characters like :, /, or spaces if necessary
                        raw_date = date_match.group(1).strip()
                        clean_date = re.sub(r'[\\/*?:"<>| ]', '_', raw_date)
                        
                        new_name = f"{prefix}{clean_date}.log"
                        
                        # Perform the rename
                        print(f"Renaming: {filename} -> {new_name}")
                        os.rename(filename, new_name)
                    else:
                        print(f"Skipping {filename}: Date string not found inside.")
            
            except Exception as e:
                print(f"Could not process {filename}: {e}")

if __name__ == "__main__":
    rename_logs_by_date()