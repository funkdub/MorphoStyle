r"""Build the data/ layout that train/configs/*.yaml expect.

    python train/data_prep/prepare_data.py \
        --humanml3d /path/to/HumanML3D \
        --shapemymove /path/to/shapemymove_data \
        --style100 /path/to/100STYLE_retargeted \
        --smoodi /path/to/SMooDi

    --humanml3d    texts/ and the official train.txt / val.txt / test.txt
    --shapemymove  Xn/new_joint_vecs, Xr/new_joint_vecs, betas/<id>_shape.npy (ShapeMyMove data processing)
    --style100     new_joint_vecs/ (SMooDi's SMPL-retargeted 100STYLE)
    --smoodi       datasets/100STYLE_name_dict.txt, {train,test}_100STYLE_Filter.txt

It links the motion folders into data/ and writes the split files:

    data/HumanML3D/Xn/new_joint_vecs, Xr/new_joint_vecs, betas, texts, new_joints   (links)
    data/HumanML3D/splits/train.txt, val.txt                              (official HumanML3D split)
    data/100STYLE/new_joint_vecs                                          (link)
    data/100STYLE/id_mapping.json                                         (training clips of the 100STYLE split)
    data/100STYLE/betas/betas_shared.npy                                  (copy of assets/100STYLE_betas_shared.npy)
    data/SPLITS.json                                                      (counts, sources, checksums)

No test clip reaches training. The script stops if a test id is in a training or validation list, and it drops the
mirrored copy of every test clip ("M" + id, or the id without "M"): the official 100STYLE split puts 666 such copies in
train. --max_train / --max_val cut the HumanML3D lists for a quick check of the pipeline; leave them out for real
training.
"""
import argparse, hashlib, json, os, shutil, sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def ids_of(path):
    with open(path) as f:
        return [ln.split()[0] for ln in f if ln.strip()]


def base(i):
    return i[1:] if i.startswith("M") else i


def stop(msg):
    sys.exit(f"[prepare_data] {msg}")


def link(src, dst):
    src = os.path.abspath(src)
    if not os.path.exists(src):
        stop(f"missing input: {src}")
    if os.path.islink(dst):
        os.remove(dst)
    elif os.path.exists(dst):
        stop(f"{dst} exists and is not a link; move it away first")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    os.symlink(src, dst)


def stems(d, suffix):
    return {f[: -len(suffix)] for f in os.listdir(d) if f.endswith(suffix)}


def write_list(path, ids):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(ids) + "\n")
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--humanml3d", required=True, help="HumanML3D root with texts/, new_joints/ and train.txt, val.txt, test.txt")
    ap.add_argument("--shapemymove", required=True, help="ShapeMyMove data output with Xn/, Xr/ and betas/")
    ap.add_argument("--style100", required=True, help="retargeted 100STYLE root with new_joint_vecs/")
    ap.add_argument("--smoodi", required=True, help="SMooDi checkout (its datasets/ folder holds the 100STYLE lists)")
    ap.add_argument("--style_split", choices=["Filter", "Full"], default="Filter",
                    help="100STYLE split: the 47-style Filter split (default) or the full 100-style split")
    ap.add_argument("--style100_texts", default=None,
                    help="optional folder of <Style>_<Content>.txt captions for the 100STYLE clips")
    ap.add_argument("--out", default=os.path.join(REPO, "data"))
    ap.add_argument("--max_train", type=int, default=0, help="keep only the first N train ids (pipeline check only)")
    ap.add_argument("--max_val", type=int, default=0, help="keep only the first N val ids (pipeline check only)")
    a = ap.parse_args()

    sm_ds = os.path.join(a.smoodi, "datasets")
    H, S, out = a.humanml3d, a.shapemymove, a.out
    summary = {"sources": {}}

    # ---------------- HumanML3D: official split, restricted to clips that have Xn, Xr and betas ----------------
    if os.path.exists(os.path.join(H, "train.txt")) and os.path.exists(os.path.join(H, "test.txt")):
        train, test = ids_of(os.path.join(H, "train.txt")), ids_of(os.path.join(H, "test.txt"))
        val = ids_of(os.path.join(H, "val.txt")) if os.path.exists(os.path.join(H, "val.txt")) else None
        summary["sources"]["humanml3d_split"] = H
    elif os.path.exists(os.path.join(sm_ds, "train_humanml.txt")) and os.path.exists(os.path.join(sm_ds, "test_humanml.txt")):
        # SMooDi ships the official HumanML3D train / test lists as train_humanml.txt / test_humanml.txt
        train, test, val = ids_of(os.path.join(sm_ds, "train_humanml.txt")), ids_of(os.path.join(sm_ds, "test_humanml.txt")), None
        summary["sources"]["humanml3d_split"] = sm_ds
        print("[prepare_data] HumanML3D train.txt / test.txt not found; using SMooDi's copies of the official lists")
    else:
        stop("no HumanML3D split found: need train.txt and test.txt in --humanml3d")

    xn, xr, bt = os.path.join(S, "Xn", "new_joint_vecs"), os.path.join(S, "Xr", "new_joint_vecs"), os.path.join(S, "betas")
    avail = stems(xn, ".npy") & stems(xr, ".npy") & stems(bt, "_shape.npy")
    test_all = set(test)
    if os.path.exists(os.path.join(sm_ds, "test_humanml.txt")):   # the evaluation protocol's content list
        test_all |= set(ids_of(os.path.join(sm_ds, "test_humanml.txt")))
    test_base = {base(i) for i in test_all}
    if val is None:   # the official val list is everything else
        val = sorted(avail - set(train) - test_all)
        summary["sources"]["humanml3d_val"] = "available ids minus train minus test"
    for name, ids in (("train", train), ("val", val)):
        bad = [i for i in ids if i in test_all]
        if bad:
            stop(f"{len(bad)} HumanML3D test ids in the {name} list, e.g. {bad[:5]}")
    mirror_drop = {n: sorted(i for i in set(ids) if base(i) in test_base) for n, ids in (("train", train), ("val", val))}
    train_ids = sorted(i for i in set(train) if i in avail and base(i) not in test_base)
    val_ids = sorted(i for i in set(val) if i in avail and base(i) not in test_base)
    if set(train_ids) & set(val_ids):
        stop("train and val lists overlap")
    full_counts = {"train": len(train_ids), "val": len(val_ids), "test": len(test_all)}
    if a.max_train:
        train_ids = train_ids[: a.max_train]
    if a.max_val:
        val_ids = val_ids[: a.max_val]

    hd = os.path.join(out, "HumanML3D")
    link(xn, os.path.join(hd, "Xn", "new_joint_vecs"))
    link(xr, os.path.join(hd, "Xr", "new_joint_vecs"))
    link(bt, os.path.join(hd, "betas"))
    link(os.path.join(H, "texts"), os.path.join(hd, "texts"))
    link(os.path.join(H, "new_joints"), os.path.join(hd, "new_joints"))   # 000021.npy: reference skeleton of the feature code
    summary["humanml3d"] = {"available": len(avail), "full": full_counts, "written": {"train": len(train_ids), "val": len(val_ids)},
                            "mirror_of_test_dropped": {n: len(v) for n, v in mirror_drop.items()},
                            "train_sha256": write_list(os.path.join(hd, "splits", "train.txt"), train_ids),
                            "val_sha256": write_list(os.path.join(hd, "splits", "val.txt"), val_ids)}

    # ---------------- 100STYLE: mapping of the training clips; style / content order over all 100 styles ----------------
    names = {}
    with open(os.path.join(sm_ds, "100STYLE_name_dict.txt")) as f:
        for ln in f:
            p = ln.split()
            if len(p) >= 2:
                names[p[0]] = p[1]
    styles = sorted({n.split("_")[0] for n in names.values()})      # label order of the bundled style classifier
    contents = sorted({n.split("_")[1] for n in names.values()})
    s_train = ids_of(os.path.join(sm_ds, f"train_100STYLE_{a.style_split}.txt"))
    s_test = set(ids_of(os.path.join(sm_ds, f"test_100STYLE_{a.style_split}.txt")))
    s_test |= set(ids_of(os.path.join(sm_ds, "test_100STYLE_Filter.txt")))   # the evaluation protocol's references
    s_test_base = {base(i) for i in s_test}
    sv = os.path.join(a.style100, "new_joint_vecs")
    have = stems(sv, ".npy")
    bad = [i for i in s_train if i in s_test]
    if bad:
        stop(f"{len(bad)} 100STYLE test clips in the training split, e.g. {bad[:5]}")
    s_mirror_drop = sorted(i for i in set(s_train) if base(i) in s_test_base)
    keep = sorted(i for i in set(s_train) if i in have and i in names and base(i) not in s_test_base)
    mapping = {"id_to_style_content": {i: {"style": names[i].split("_")[0], "content": names[i].split("_")[1]} for i in keep},
               "styles": styles, "contents": contents}
    sd = os.path.join(out, "100STYLE")
    link(sv, os.path.join(sd, "new_joint_vecs"))
    os.makedirs(sd, exist_ok=True)
    with open(os.path.join(sd, "id_mapping.json"), "w") as f:
        json.dump(mapping, f)
    os.makedirs(os.path.join(sd, "betas"), exist_ok=True)
    shutil.copyfile(os.path.join(REPO, "assets", "100STYLE_betas_shared.npy"), os.path.join(sd, "betas", "betas_shared.npy"))
    if a.style100_texts:
        link(a.style100_texts, os.path.join(sd, "texts"))
    summary["style100"] = {"split": a.style_split, "train_clips": len(keep), "mirror_of_test_dropped": len(s_mirror_drop),
                           "train_styles": len({v["style"] for v in mapping["id_to_style_content"].values()}),
                           "styles_in_label_space": len(styles), "contents": len(contents), "test_clips_excluded": len(s_test)}

    with open(os.path.join(out, "SPLITS.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))
    print("[prepare_data] done; no test clip in the training or validation lists")


if __name__ == "__main__":
    main()
