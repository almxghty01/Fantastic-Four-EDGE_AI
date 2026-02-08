import cv2
import numpy as np
import os
import random
import shutil


BASE_DIR = r"D:\Desktop\pycharm\pycharmfiles\EdgeChipv1\data set"
IMG_SIZE = 224
TOTAL_IMAGES = 5000


CLASSES = [
    'clean', 'other', 'open', 'short',
    'scratch', 'particle', 'dead_via', 'misalign'
]


def create_sem_texture():
    img = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    noise_level = random.randint(20, 40)
    noise = np.random.normal(128, noise_level, (IMG_SIZE, IMG_SIZE)).astype(np.uint8)
    blur_k = random.choice([3, 5])
    img = cv2.GaussianBlur(cv2.add(img, noise), (blur_k, blur_k), 0)
    return img


def draw_circuit_lines(img):
    coords = []
    num_lines = random.randint(3, 6)
    spacing = IMG_SIZE // (num_lines + 1)
    for i in range(1, num_lines + 1):
        x = i * spacing + random.randint(-4, 4)
        color = random.randint(40, 80)
        thickness = random.randint(8, 14)
        cv2.line(img, (x, 0), (x, IMG_SIZE), (color), thickness)
        coords.append(x)
    return coords


def generate_mass_dataset():
    # 1. DELETE OLD DATA (PCB + Old Synthetic) - Clean Slate
    if os.path.exists(BASE_DIR):
        print(f"🧹 Clearing old data at {BASE_DIR}...")
        shutil.rmtree(BASE_DIR)

    # 2. Setup Folders
    for split in ['train', 'val', 'test']:
        path = os.path.join(BASE_DIR, split)
        for cls in CLASSES:
            os.makedirs(os.path.join(path, cls), exist_ok=True)

    print(f"🚀 STARTING FACTORY: 5,000 IESA-COMPLIANT IMAGES")
    print("=" * 60)

    for i in range(TOTAL_IMAGES):
        if i % 1000 == 0: print(f"   ... Fabricated {i} wafers ...")

        # Split: 70% Train, 20% Val, 10% Test
        r = random.random()
        split = 'train' if r < 0.7 else 'val' if r < 0.9 else 'test'

        label = CLASSES[i % 8]
        save_dir = os.path.join(BASE_DIR, split, label)

        img = create_sem_texture()
        tracks = draw_circuit_lines(img)

        # --- DEFECT LOGIC ---
        if label == 'clean':
            pass
        elif label == 'other':
            center_x, center_y = random.randint(50, 170), random.randint(50, 170)
            cv2.circle(img, (center_x, center_y), random.randint(30, 80), (200), -1)
        elif label == 'open':
            if tracks:
                x, y = random.choice(tracks), random.randint(20, 200)
                bg_color = int(np.mean(img[y, x - 20:x - 10]))
                cv2.circle(img, (x, y), random.randint(10, 14), (bg_color), -1)
        elif label == 'short':
            if len(tracks) > 1:
                idx = random.randint(0, len(tracks) - 2)
                cv2.line(img, (tracks[idx], 100), (tracks[idx + 1], 100), (60), random.randint(4, 10))
        elif label == 'scratch':
            pt1 = (random.randint(0, IMG_SIZE), random.randint(0, IMG_SIZE))
            pt2 = (pt1[0] + random.randint(-40, 40), pt1[1] + random.randint(-40, 40))
            cv2.line(img, pt1, pt2, (220), random.randint(1, 3))
        elif label == 'particle':
            cv2.circle(img, (random.randint(10, 210), random.randint(10, 210)), random.randint(2, 6), (255), -1)
        elif label == 'dead_via':
            if tracks: cv2.circle(img, (random.choice(tracks), random.randint(30, 190)), 14, (20), -1)
        elif label == 'misalign':
            rows, cols = img.shape
            M = np.float32([[1, 0, 25], [0, 1, 0]])
            img = cv2.warpAffine(img, M, (cols, rows))

        filename = f"{label}_{i:05d}.png"
        cv2.imwrite(os.path.join(save_dir, filename), img)

    print("✅ DATASET READY. 5,000 Images Created.")


if __name__ == "__main__":
    generate_mass_dataset()