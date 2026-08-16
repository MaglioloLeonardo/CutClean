import os
import shutil

import pandas as pd
import gdown
import PIL
import tarfile
import torch
import torchvision

from torchvision import transforms

def _read_partition_any(root):
    p_csv = os.path.join(root, "list_eval_partition.csv")
    p_txt = os.path.join(root, "list_eval_partition.txt")
    if os.path.isfile(p_csv):
        p = p_csv
    elif os.path.isfile(p_txt):
        p = p_txt
    else:
        raise FileNotFoundError("Missing list_eval_partition.csv or list_eval_partition.txt")
    try:
        df = pd.read_csv(p, sep=r"[\s,]+", engine="python")
        if df.shape[1] < 2:
            raise ValueError
        low = [c.lower() for c in df.columns]
        if "image_id" in low and ("split" in low or "partition" in low):
            rename_map = {}
            for c in df.columns:
                cl = c.lower()
                if cl == "image_id":
                    rename_map[c] = "image_id"
                elif cl == "split" or cl == "partition":
                    rename_map[c] = "split"
            df = df.rename(columns=rename_map)[["image_id", "split"]]
        else:
            df = pd.read_csv(p, sep=r"[\s,]+", engine="python", header=None, names=["image_id", "split"])
    except Exception:
        df = pd.read_csv(p, sep=r"[\s,]+", engine="python", header=None, names=["image_id", "split"])
    df["image_id"] = df["image_id"].astype(str).str.strip()
    s = df["split"].astype(str).str.strip().str.lower()
    map_vals = {"0": 0, "1": 1, "2": 2, "train": 0, "valid": 1, "val": 1, "validation": 1, "test": 2, "testing": 2}
    df["split"] = s.map(map_vals)
    df = df.dropna(subset=["split"])
    df["split"] = df["split"].astype(int)
    return df

def _read_attrs_any(root):
    p = os.path.join(root, "list_attr_celeba.csv")
    df = pd.read_csv(p, sep=r"[\s,]+", engine="python").replace(-1, 0)
    df.columns = [str(c).strip() for c in df.columns]
    lc = {c.lower(): c for c in df.columns}
    if "image_id" in lc:
        imgc = lc["image_id"]
    elif "image" in lc:
        imgc = lc["image"]
    elif "filename" in lc:
        imgc = lc["filename"]
    elif "file_name" in lc:
        imgc = lc["file_name"]
    elif "img_name" in lc:
        imgc = lc["img_name"]
    else:
        imgc = df.columns[0]
    if imgc != "image_id":
        df = df.rename(columns={imgc: "image_id"})
    df["image_id"] = df["image_id"].astype(str).str.strip()
    return df

def _merge_attrs_splits(attr_df, split_df):
    m = attr_df.merge(split_df[["image_id", "split"]], on="image_id", how="inner")
    if len(m):
        return m
    s = split_df.copy()
    a = attr_df.copy()
    if not s["image_id"].str.contains(r"\.").any():
        s["image_id"] = s["image_id"] + ".jpg"
        m = a.merge(s[["image_id", "split"]], on="image_id", how="inner")
        if len(m):
            return m
    if not a["image_id"].str.contains(r"\.").any():
        a["image_id"] = a["image_id"] + ".jpg"
        m = a.merge(split_df[["image_id", "split"]], on="image_id", how="inner")
        if len(m):
            return m
    a2 = attr_df.copy()
    s2 = split_df.copy()
    a2["image_id"] = a2["image_id"].str.replace(r"\.\w+$", "", regex=True)
    s2["image_id"] = s2["image_id"].str.replace(r"\.\w+$", "", regex=True)
    m = a2.merge(s2[["image_id", "split"]], on="image_id", how="inner")
    return m

def _resolve(df, name):
    m = {c.lower(): c for c in df.columns}
    key = str(name).lower()
    if key in m:
        return m[key]
    raise KeyError(name)

class CelebA(torch.utils.data.Dataset):
    def __init__(self, root, split="train", target="Blond_Hair", bias_attr="Male", unbiased=True, seed=None, train_proc=False):
        if seed is None:
            raise ValueError(
                "CelebA requires an explicit seed: it drives the balanced per-group "
                "subsampling, and an unseeded run would be silently irreproducible."
            )
        path = root
        if not os.path.isdir(path):
            self.download_dataset(path)
        self.split = split
        self.train_proc = train_proc
        split_df = _read_partition_any(path)
        attr_df = _read_attrs_any(path)
        merged = _merge_attrs_splits(attr_df, split_df)
        if len(merged) == 0:
            raise ValueError("Failed to align attributes with partition file.")
        code = {"train": 0, "valid": 1, "val": 1, "test": 2}[split]
        df = merged[merged["split"] == code].copy()
        male_col = _resolve(df, "Male")
        target_col = _resolve(df, target)
        bias_col = _resolve(df, bias_attr)
        for c in {male_col, target_col, bias_col}:
            df[c] = df[c].astype(int)
        df[male_col] = (1 - df[male_col]).astype(int)
        if unbiased:
            tv = set(df[target_col].unique().tolist())
            bv = set(df[bias_col].unique().tolist())
            if {0, 1}.issubset(tv) and {0, 1}.issubset(bv):
                counts = df.groupby([target_col, bias_col]).size()
                if counts.min() > 0:
                    min_size = int(counts.min())
                    groups = [g.sample(min_size, random_state=seed) for _, g in df.groupby([target_col, bias_col])]
                    df_use = pd.concat(groups, ignore_index=True)
                else:
                    df_use = df
            else:
                df_use = df
        else:
            df_use = df
        bias_conflicting_df = df_use[df_use[target_col] != df_use[bias_col]]
        self.attr_df = df_use if unbiased else bias_conflicting_df
        if len(self.attr_df) == 0:
            self.attr_df = df
        self.target = target_col
        self.bias_attr = bias_col
        self.path = path
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        T_train = transforms.Compose([transforms.Resize((224, 224)), transforms.RandomHorizontalFlip(p=0.5), transforms.ToTensor(), transforms.Normalize(mean, std)])
        T_test = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean, std)])
        self.T = T_train if (split == "train" or self.train_proc) else T_test

    def download_dataset(self, path):
        url = "https://drive.google.com/uc?id=1ebDzE4vsjPB4klNyTywjrZqGhUsFxZqb"
        output = os.path.join(path, "celeba.tar.gz")
        os.makedirs(path, exist_ok=True)
        print(f"=> Downloading CelebA dataset from {url}")
        try:
            gdown.download(url, output, quiet=False)
            if not os.path.exists(output):
                raise RuntimeError("gdown finished without producing the archive")
        except Exception as exc:
            raise RuntimeError(
                "Automatic CelebA download failed: the mirror behind the link above "
                "is unavailable. Download CelebA from the official page "
                "(https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) and place "
                "'img_align_celeba/', 'list_attr_celeba.txt' (or .csv) and "
                f"'list_eval_partition.txt' (or .csv) under {path} (see the README)."
            ) from exc
        print("=> Extracting dataset...")
        temp_extract_dir = os.path.join(path, "_temp_extract")
        os.makedirs(temp_extract_dir, exist_ok=True)
        with tarfile.open(output, "r:gz") as tar:
            tar.extractall(path=temp_extract_dir)
        for root, dirs, files in os.walk(temp_extract_dir):
            if "list_eval_partition.csv" in files:
                for item in os.listdir(root):
                    src = os.path.join(root, item)
                    dst = os.path.join(path, item)
                    if not os.path.exists(dst):
                        shutil.move(src, dst)
                break
        shutil.rmtree(temp_extract_dir)
        os.remove(output)
        print("CelebA dataset successfully downloaded and extracted to the correct location.")

    def __getitem__(self, index):
        data = self.attr_df.iloc[index]
        img_name = data["image_id"]
        bias = data[self.bias_attr]
        target_attr = data[self.target]
        image = PIL.Image.open(os.path.join(self.path, "img_align_celeba", img_name)).convert("RGB")
        return self.T(image), target_attr, bias

    def __len__(self):
        return len(self.attr_df)
