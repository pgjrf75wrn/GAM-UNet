import os
from typing import List, Tuple

import numpy as np
from PIL import Image


def _load_paths(split_root: str) -> List[Tuple[str, str, str]]:
    images_dir = os.path.join(split_root, "images")
    masks_dir = os.path.join(split_root, "masks")
    depths_dir = os.path.join(split_root, "depth_mask")

    image_names = sorted(os.listdir(images_dir))
    mask_names = sorted(os.listdir(masks_dir))
    depth_names = sorted(os.listdir(depths_dir))

    if not (len(image_names) == len(mask_names) == len(depth_names)):
        raise ValueError("Image/mask/depth counts do not match.")

    paths = []
    for img_name, mask_name, depth_name in zip(image_names, mask_names, depth_names):
        paths.append(
            (
                os.path.join(images_dir, img_name),
                os.path.join(masks_dir, mask_name),
                os.path.join(depths_dir, depth_name),
            )
        )
    return paths


def _read_triplet(
    img_path: str,
    mask_path: str,
    depth_path: str,
    size: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    img = Image.open(img_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")
    depth = Image.open(depth_path).convert("L")

    if size is not None:
        img = img.resize(size, Image.BILINEAR)
        mask = mask.resize(size, Image.BILINEAR)
        depth = depth.resize(size, Image.BILINEAR)

    img_arr = np.array(img, dtype=np.uint8)
    mask_arr = np.array(mask, dtype=np.uint8)
    depth_arr = np.array(depth, dtype=np.uint8)

    return img_arr, mask_arr, depth_arr


def _save_split(
    out_dir: str,
    split_name: str,
    triplets: List[Tuple[str, str, str]],
    size: Tuple[int, int],
) -> None:
    images = []
    masks = []
    depths = []

    for img_path, mask_path, depth_path in triplets:
        img_arr, mask_arr, depth_arr = _read_triplet(
            img_path, mask_path, depth_path, size
        )
        images.append(img_arr)
        masks.append(mask_arr)
        depths.append(depth_arr)


    data = np.stack(images, axis=0)
    mask = np.stack(masks, axis=0)
    depth = np.stack(depths, axis=0)

    np.save(os.path.join(out_dir, f"data_{split_name}.npy"), data)
    np.save(os.path.join(out_dir, f"mask_{split_name}.npy"), mask)
    np.save(os.path.join(out_dir, f"depth_{split_name}.npy"), depth)


def _save_test_sets(
    root: str,
    out_dir: str,
    test_sets: List[str],
    size: Tuple[int, int],
) -> None:
    for name in test_sets:
        split_root = os.path.join(root, "TestDataset", name)
        paths = _load_paths(split_root)
        _save_split(out_dir, f"test_{name}", paths, size)


# Parameters
height = 256
width = 256

Dataset_add = "./data/"
Train_add = "TrainDataset"
Val_add = "ValDataset"
Test_add = "TestDataset"

Test_sets = [
    "CVC-300",
    "CVC-ClinicDB",
    "Kvasir",
    "CVC-ColonDB",
    "ETIS-LaribPolypDB",
]


def main() -> None:
    out_dir = Dataset_add
    os.makedirs(out_dir, exist_ok=True)

    size = (width, height)

    train_paths = _load_paths(os.path.join(Dataset_add, Train_add))
    val_paths = _load_paths(os.path.join(Dataset_add, Val_add))

    _save_split(out_dir, "train", train_paths, size)
    _save_split(out_dir, "val", val_paths, size)
    _save_test_sets(Dataset_add, out_dir, Test_sets, size)

    print("Saved:")
    print(os.path.join(out_dir, "data_train.npy"))
    print(os.path.join(out_dir, "mask_train.npy"))
    print(os.path.join(out_dir, "depth_train.npy"))
    print(os.path.join(out_dir, "data_val.npy"))
    print(os.path.join(out_dir, "mask_val.npy"))
    print(os.path.join(out_dir, "depth_val.npy"))
    for name in Test_sets:
        print(os.path.join(out_dir, f"data_test_{name}.npy"))
        print(os.path.join(out_dir, f"mask_test_{name}.npy"))
        print(os.path.join(out_dir, f"depth_test_{name}.npy"))


if __name__ == "__main__":
    main()
