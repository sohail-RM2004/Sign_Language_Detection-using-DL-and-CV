import os

# Files to remove (redundant/debug files)
files_to_remove = [
    "analyze_model.py",
    "debug_model.py", 
    "test_model_simple.py",
    "test_data_loading.py",
    "retrain_improved_model.py",
    "project_setup.py"
]

# Remove redundant files
for file in files_to_remove:
    if os.path.exists(file):
        os.remove(file)
        print(f"Removed: {file}")
    else:
        print(f"Not found: {file}")

# Remove duplicate requirements.txt (keep the one in sign_language_project)
if os.path.exists("requirements.txt") and os.path.exists("sign_language_project/requirements.txt"):
    os.remove("requirements.txt")
    print("Removed duplicate requirements.txt")

print("\nCleanup completed!")
print("Remaining core files:")
core_files = [
    "config.yaml",
    "config_loader.py", 
    "complete_training_script.py",
    "sign_language_model.py",
    "sign_language_preprocessing.py", 
    "realtime_detection.py",
    "gui_demo.py"
]

for file in core_files:
    if os.path.exists(file):
        print(f"+ {file}")
    else:
        print(f"- {file} (missing)")