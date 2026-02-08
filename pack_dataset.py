import os
import shutil
import random

# --- CONFIGURATION ---
# FIXED: Pointing to your actual train folder inside FanFour
SOURCE_DIR = r"Sample_data\train"
TARGET_DIR = "Final_Submission_Dataset"
CLASSES = ['clean', 'other', 'open', 'short', 'scratch', 'particle', 'dead_via', 'misalign']
IMAGES_PER_CLASS = 250


def pack_data():
    # 1. Check if Source Exists
    if not os.path.exists(SOURCE_DIR):
        print(f"❌ CRITICAL ERROR: Could not find source directory at: {os.path.abspath(SOURCE_DIR)}")
        return

    # 2. Create Target Directory
    if os.path.exists(TARGET_DIR):
        shutil.rmtree(TARGET_DIR)
    os.makedirs(TARGET_DIR)
    print(f"📁 Created destination folder: {os.path.abspath(TARGET_DIR)}")

    total_moved = 0

    # 3. Loop through classes
    for cls in CLASSES:
        src_path = os.path.join(SOURCE_DIR, cls)
        dst_path = os.path.join(TARGET_DIR, cls)

        # Check if class folder exists
        if not os.path.exists(src_path):
            print(f"⚠️ MISSING CLASS: Could not find '{cls}' inside {SOURCE_DIR}")
            continue

        os.makedirs(dst_path, exist_ok=True)

        # Get all images
        files = [f for f in os.listdir(src_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        if not files:
            print(f"⚠️ EMPTY: No images found in {cls}")
            continue

        # Select Random 250 (or all if less than 250)
        count = min(len(files), IMAGES_PER_CLASS)
        selected = random.sample(files, count)

        # Copy them
        for f in selected:
            shutil.copy2(os.path.join(src_path, f), os.path.join(dst_path, f))

        print(f"✅ Packed {len(selected)} images for class: {cls}")
        total_moved += len(selected)

    print("=" * 40)
    print(f"🚀 DONE. {total_moved} images copied to 'Final_Submission_Dataset'.")
    print(f"👉 NEXT STEP: Right-click 'Final_Submission_Dataset' -> Send to -> Compressed (zipped) folder.")


if __name__ == "__main__":
    pack_data()