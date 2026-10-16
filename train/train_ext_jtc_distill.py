"""Trainer for the MorphoStyle style branch.

The ShapeMyMove FSQ-VAE backbone stays frozen; what trains is the style
encoder, the text-guided joint routing (JTC) and the manifold-preserving style
modulator - a rank-r additive offset on the content codes, attenuated per frame
by a temporal gate.

Objectives: reconstruction and code preservation, style contrastive and
classifier terms, WGAN-GP adversarial and moment matching, soft root/foot
anchoring to the content, and the foot-contact and trajectory terms that the
fine-tuning configuration adds on top.

    python train/train_ext_jtc_distill.py --config train/configs/base.yaml --gpu 0
"""

import os, sys, argparse, time, json, csv, yaml, numpy as np
from pathlib import Path
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)                 # repo root
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset


def load_config(p): return yaml.safe_load(open(p))


class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}
    def update(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.shadow:
                self.shadow[n].lerp_(p.data, 1 - self.decay)
    def apply(self, model):
        bk = {}
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.shadow:
                bk[n] = p.data.clone(); p.data.copy_(self.shadow[n])
        return bk
    def restore(self, model, bk):
        for n, p in model.named_parameters():
            if n in bk: p.data.copy_(bk[n])


# set in main(); module-level so evaluate_val() can use it
update_jtc_extras = lambda batch: None


def main():
    global update_jtc_extras
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--max_steps", type=int, default=0, help="smoke test: stop after N steps")
    args = parser.parse_args()

    # --gpu sets CUDA_VISIBLE_DEVICES only when the shell has not set it
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda:0")
    print(f"[GPU] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '(unset)')}")
    cfg = load_config(args.config)
    torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"])

    output_dir = Path(cfg["output_dir"]) / cfg["name"]
    output_dir.mkdir(parents=True, exist_ok=True)
    yaml.dump(cfg, open(output_dir / "config.yaml", "w"))
    if cfg["paths"].get("shapemove_root"):
        sys.path.insert(0, cfg["paths"]["shapemove_root"])

    sc_mean = np.load(cfg["data"]["motion_mean_path"]).astype(np.float32)
    sc_std = np.load(cfg["data"]["motion_std_path"]).astype(np.float32)
    sc_std = np.where(sc_std < 1e-6, 1.0, sc_std)
    hml_mean = np.load(cfg["paths"]["hml3d_mean_path"]).astype(np.float32)
    hml_std = np.load(cfg["paths"]["hml3d_std_path"]).astype(np.float32)
    hml_std = np.where(np.abs(hml_std) < 1e-6, 1.0, hml_std)

    sc_mean_t = torch.from_numpy(sc_mean).to(device)
    sc_std_t = torch.from_numpy(sc_std).to(device)
    hml_mean_t = torch.from_numpy(hml_mean).to(device)
    hml_std_t = torch.from_numpy(hml_std).to(device)

    from dataset.style_dataset import Style100Dataset
    from dataset.hml3d_dataset import HumanML3DIdentityDataset, HumanML3DValDataset, CrossDomainDataset

    d = cfg["data"]; h = cfg["hml3d"]
    style100 = Style100Dataset(
        motion_dir=d["data_dir"], mapping_path=d["mapping_path"],
        betas_shared_path=d.get("betas_shared_path"),
        max_motion_length=d["max_motion_length"], min_motion_length=d["min_motion_length"],
        seed=42, window_size=d["window_size"], window_stride=d["window_stride"],
        motion_mean_path=d["motion_mean_path"], motion_std_path=d["motion_std_path"],
        neutral_style_name=d.get("neutral_style_name"),
    )
    _identity_ratio = float(h.get("identity_ratio", 1.0))
    _id_target_size = max(1, int(len(style100) * _identity_ratio))
    print(f"[Data] identity_ratio={_identity_ratio}, identity target_size={_id_target_size} (100Style={len(style100)})")
    hml3d_id = HumanML3DIdentityDataset(
        xn_dir=h["xn_dir"], xr_dir=h["xr_dir"], beta_dir=h["beta_dir"],
        split_file=h["train_split"], sc_mean=sc_mean, sc_std=sc_std,
        max_samples=h.get("max_train_samples", 50000),
        target_size=_id_target_size,
        max_len=d["max_motion_length"],
    )

    # cross-domain pairs (HumanML3D content + 100STYLE style), as in the SMooDi protocol
    if cfg.get("cross_domain", {}).get("enabled", False):
        cross_ratio = cfg["cross_domain"].get("ratio", 0.30)
        target_size = int(len(style100) * cross_ratio / (1 - cross_ratio))
        cross_ds = CrossDomainDataset(
            hml3d_identity_dataset=hml3d_id,
            style100_dir=d["data_dir"], style100_mapping_path=d["mapping_path"],
            sc_mean=sc_mean, sc_std=sc_std, target_size=target_size,
            style100_style_to_idx=style100.style_to_idx,
            max_len=d["max_motion_length"],
        )
        tcfg = cfg.get("teacher", {})
        if tcfg.get("enabled", False):
            # Distillation: part of the cross-domain budget is replaced by fixed
            # (content, style clip) pairs that carry MCM-LDM teacher outputs.
            try:
                from dataset.teacher_dataset import TeacherCrossDomainDataset, WithTeacherKeys
            except ImportError as e:
                raise SystemExit("teacher.enabled needs dataset/teacher_dataset.py and MCM-LDM teacher pairs, "
                                 "which are not part of this release; set teacher.enabled: false") from e
            n_teacher = int(target_size * float(tcfg.get("frac", 0.7)))
            teacher_ds = TeacherCrossDomainDataset(
                pairs_json=tcfg["pairs_json"], feats_dir=tcfg["feats_dir"],
                xn_dir=h["xn_dir"], xr_dir=h["xr_dir"], beta_dir=h["beta_dir"],
                sc_mean=sc_mean, sc_std=sc_std, style100_style_to_idx=style100.style_to_idx,
                max_len=d["max_motion_length"], target_size=n_teacher)
            cross_ds._len = max(1, target_size - n_teacher)
            ML = d["max_motion_length"]
            train_dataset = ConcatDataset([WithTeacherKeys(style100, ML), WithTeacherKeys(hml3d_id, ML),
                                           WithTeacherKeys(cross_ds, ML), teacher_ds])
            print(f"[Data] 100Style: {len(style100)}, HML3D: {len(hml3d_id)}, CrossDomain: {len(cross_ds)}, "
                  f"TeacherCross: {len(teacher_ds)}, Total: {len(train_dataset)}")
        else:
            train_dataset = ConcatDataset([style100, hml3d_id, cross_ds])
            print(f"[Data] 100Style: {len(style100)}, HML3D: {len(hml3d_id)}, "
                  f"CrossDomain: {len(cross_ds)}, Total: {len(train_dataset)}")
    else:
        train_dataset = ConcatDataset([style100, hml3d_id])
        print(f"[Data] 100Style: {len(style100)}, HML3D: {len(hml3d_id)}, Total: {len(train_dataset)}")

    train_loader = DataLoader(
        train_dataset, collate_fn=_mid_collate, batch_size=d.get("batch_size", 64), shuffle=True,
        num_workers=d.get("num_workers", 4), pin_memory=True, drop_last=True,
        persistent_workers=True, prefetch_factor=4,
    )

    # dual-path HumanML3D reconstruction: one pure HumanML3D batch per step with style_emb = 0,
    # so that recon(x, style=0) stays x
    dual_hml_cfg = cfg.get("dual_hml_recon", {})
    use_dual_hml = bool(dual_hml_cfg.get("enabled", False))
    lambda_dual_hml = float(dual_hml_cfg.get("weight", 5.0))
    # mode "zero": style_emb = 0 via cfg_uncond; "same_id": the content is its own style
    # reference and the output must still be the content
    dual_hml_mode = str(dual_hml_cfg.get("mode", "zero"))
    assert dual_hml_mode in ("zero", "same_id"), f"bad dual_hml_recon.mode: {dual_hml_mode}"
    if use_dual_hml:
        hml_pure_loader = DataLoader(
            hml3d_id, batch_size=d.get("batch_size", 64), shuffle=True,
            num_workers=2, pin_memory=True, drop_last=True, persistent_workers=True,
        )
        print(f"[DualPath] Pure-HML recon enabled, mode={dual_hml_mode}, "
              f"weight={lambda_dual_hml}, batch_size={d.get('batch_size', 64)}")
    else:
        hml_pure_loader = None

    val_dataset = HumanML3DValDataset(
        xn_dir=h["xn_dir"], xr_dir=h["xr_dir"], beta_dir=h["beta_dir"],
        val_split_file=h["val_split"], style100_dir=d["data_dir"],
        style100_mapping_path=d["mapping_path"],
        sc_mean=sc_mean, sc_std=sc_std, max_samples=cfg.get("val_samples", 200),
        max_len=d["max_motion_length"],
    )
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2)

    # Model
    from model.engine.motion_vae import SCMotionVAE
    v = cfg["vae"]
    vae = SCMotionVAE(
        nb_code=v["nb_code"], code_dim=v["code_dim"],
        output_emb_width=v["output_emb_width"], down_t=v["down_t"],
        stride_t=v["stride_t"], width=v["width"], depth=v["depth"],
        dilation_growth_rate=v["dilation_growth_rate"],
        activation=v["activation"], norm=v["norm"], mu=v["mu"],
        input_dim=v["input_dim"], quantizer_type=v["quantizer_type"],
        shape_dim=v["shape_dim"], level=v.get("fsq_levels"),
    )
    ckpt = torch.load(cfg["paths"]["shapemove_ckpt"], map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", ckpt)
    for pfx in ["motionvae.", "motion_model.", "vae."]:
        pkeys = {k[len(pfx):]: vv for k, vv in sd.items() if k.startswith(pfx)}
        if pkeys: vae.load_state_dict(pkeys, strict=False); break
    vae = vae.to(device)

    # two style architectures: the manifold-preserving style modulator alone ("mpsm") or
    # fed by the text-guided joint routing ("mpsm_jtc"); jtc.enabled selects the latter
    style_arch = cfg.get("style_arch", "mpsm")
    cfg_style = {**cfg["style_model"], "temporal_gate": cfg["temporal_gate"]}
    jtc_cfg = cfg.get("jtc", {})
    jtc_enabled = bool(jtc_cfg.get("enabled", False))
    vectorizer = None
    if style_arch == "mpsm_jtc" or jtc_enabled:
        from model.joint_text_clip import (
            T2MEvaluatorBundle, build_style_model_jtc, WordVectorizer,
            tokenize_caption,
        )
        t2m_ckpt = jtc_cfg.get(
            "t2m_ckpt",
            os.environ.get("T2M_EVAL_CKPT", "assets/t2m/finest.tar"))
        glove_dir = jtc_cfg.get(
            "glove_dir",
            os.environ.get("GLOVE_DIR", "assets/glove"))
        t2m_bundle = T2MEvaluatorBundle(t2m_ckpt).to(device)
        vectorizer = WordVectorizer(glove_dir, "our_vab")
        cfg_style_jtc = {
            **cfg_style,
            "jtc_top_k": jtc_cfg.get("top_k", 8),
            "jtc_joint_dim": jtc_cfg.get("joint_dim", 128),
            "jtc_fuse_hidden": jtc_cfg.get("fuse_hidden", 256),
            "jtc_init_gate": jtc_cfg.get("init_gate", 0.0),
        }
        model = build_style_model_jtc(cfg_style_jtc, vae, t2m_bundle).to(device)
        style_arch = "mpsm_jtc"
        # patch model.forward so every call in the loop receives the current batch's text tokens
        max_text_len = int(jtc_cfg.get("max_text_len", 22))
        _jtc_extras: dict = {}
        _orig_forward = model.forward
        def _wrapped_forward(*args, **kwargs):
            return _orig_forward(*args, **kwargs, **_jtc_extras)
        model.forward = _wrapped_forward

        def _update_jtc_extras(batch):
            ctexts = batch.get("content_text", None)
            if ctexts is None:
                _jtc_extras.clear(); return
            B = len(ctexts)
            wem_np = np.zeros((B, max_text_len, 300), dtype=np.float32)
            poh_np = np.zeros((B, max_text_len, 15), dtype=np.float32)
            cap_np = np.zeros((B,), dtype=np.int64)
            for i, txt in enumerate(ctexts):
                wem_i, poh_i, L_i = tokenize_caption(vectorizer, txt, max_text_len)
                wem_np[i] = wem_i; poh_np[i] = poh_i; cap_np[i] = max(L_i, 1)
            _jtc_extras["content_word_emb"] = torch.from_numpy(wem_np).to(device)
            _jtc_extras["content_pos_onehot"] = torch.from_numpy(poh_np).to(device)
            _jtc_extras["content_cap_lens"] = torch.from_numpy(cap_np).to(device)
            slen = batch.get("content_length", batch.get("target_length"))
            if slen is not None:
                _jtc_extras["style_lengths"] = slen.to(device)
        # Expose for the training loop
        update_jtc_extras = _update_jtc_extras
    else:
        from model.style_transfer import build_style_model
        model = build_style_model(cfg_style, vae).to(device)
        update_jtc_extras = lambda batch: None
    print(f"[Model] style_arch = {style_arch}")
    if style_arch != "mpsm_jtc":
        update_jtc_extras = lambda batch: None
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"[G] Trainable: {sum(p.numel() for p in trainable):,}")

    # Critic
    from model.critic import MotionCritic
    cc = cfg["critic"]
    critic = MotionCritic(input_dim=263, hidden=tuple(cc["hidden"]), use_ln=True).to(device)
    print(f"[D] Params: {sum(p.numel() for p in critic.parameters()):,}")

    t_cfg = cfg["training"]

    # shape predictor for style -> beta disentanglement (gradient reversal)
    shape_predictor = None
    _L_cfg = cfg.get("loss", {})
    if float(_L_cfg.get("lambda_shape_disentangle", 0.0)) > 0:
        from model.shape_foot_losses import ShapePredictor
        style_dim = cfg["style_model"].get("style_embed_dim", 256)
        shape_predictor = ShapePredictor(style_dim=style_dim, hidden=128, beta_dim=10).to(device)
        print(f"[shape] ShapePredictor enabled (style_dim={style_dim}, beta-dim=10)")
        trainable = list(trainable) + list(shape_predictor.parameters())

    # Foot-contact dims are constant (HML3D 263-d layout, dims 259-262)
    FOOT_CONTACT_DIMS = [259, 260, 261, 262]

    # joint_style_fuser and style_head start from scratch, so they get a higher LR
    jtc_lr_mult = float(jtc_cfg.get("new_module_lr_mult", 1.0)) if style_arch == "mpsm_jtc" else 1.0
    if jtc_lr_mult > 1.0:
        jtc_param_ids = set()
        for sm in (getattr(model, "joint_style_fuser", None), getattr(model, "style_head", None),
                   getattr(getattr(model, "style_injector", None), "temporal_mod", None)):
            if sm is not None:
                for p in sm.parameters():
                    if p.requires_grad:
                        jtc_param_ids.add(id(p))
        jtc_params = [p for p in trainable if id(p) in jtc_param_ids]
        base_params = [p for p in trainable if id(p) not in jtc_param_ids]
        g_optim = torch.optim.AdamW(
            [
                {"params": base_params, "lr": t_cfg["lr"]},
                {"params": jtc_params, "lr": t_cfg["lr"] * jtc_lr_mult},
            ],
            weight_decay=t_cfg["weight_decay"], betas=tuple(t_cfg["betas"]))
        print(f"[Optim] JTC modules: {len(jtc_params)} params at lr x{jtc_lr_mult}; "
              f"base: {len(base_params)} params at lr={t_cfg['lr']}")
    else:
        g_optim = torch.optim.AdamW(trainable, lr=t_cfg["lr"],
            weight_decay=t_cfg["weight_decay"], betas=tuple(t_cfg["betas"]))
    d_optim = torch.optim.Adam(critic.parameters(), lr=cc["lr"], betas=tuple(cc["betas"]))

    total_steps = t_cfg["epochs"] * len(train_loader)
    warmup = t_cfg.get("warmup_steps", 500)
    from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR
    w_sched = LambdaLR(g_optim, lambda s: min(1.0, s / max(warmup, 1)))
    m_sched = CosineAnnealingLR(g_optim, T_max=max(1, total_steps - warmup),
                                 eta_min=t_cfg.get("lr_min", 1e-6))
    scheduler = SequentialLR(g_optim, [w_sched, m_sched], milestones=[warmup])

    # Losses
    from model.losses import StyleMotionLoss, StyleContrastiveLoss, StyleClassifierLoss
    from model.losses_adv import (
        wgan_critic_loss, gradient_penalty, wgan_generator_loss,
        MomentMatchingLoss, RootMaskedStyleClassifierLoss,
        soft_root_loss, soft_foot_loss,
    )
    L = cfg["loss"]
    criterion = StyleMotionLoss(
        lambda_recon=L["lambda_recon"], lambda_vel=L["lambda_vel"],
        lambda_foot=L["lambda_foot"], lambda_code_recon=L["lambda_code_recon"],
        recon_type=L["recon_type"],
    ).to(device)
    contrastive_fn = StyleContrastiveLoss(temperature=0.07).to(device)

    # classifier: paths.classifier_path if it exists, else a body-only one under output_dir, else the shipped one
    clf_cache = None
    explicit_path = cfg.get("paths", {}).get("classifier_path")
    if explicit_path and Path(explicit_path).exists():
        clf_cache = Path(explicit_path)
        print(f"[Clf] Using EXPLICIT classifier: {clf_cache}")
    elif L.get("use_body_only_clf", False):
        body_ck = Path(cfg["output_dir"]) / "eval_classifiers_body.pt"
        if body_ck.exists():
            clf_cache = body_ck
            print(f"[Clf] Using body-only (root-masked) classifier: {clf_cache}")
    if clf_cache is None:
        clf_cache = Path("assets/eval_classifiers_full.pt")
        print(f"[Clf] Falling back to the default classifier: {clf_cache}")
    cc_clf = torch.load(clf_cache, map_location=device, weights_only=False)
    from auxiliary import MotionFeatureExtractor, StyleClassifier, ContentClassifier
    feat_ext = MotionFeatureExtractor(feature_dim=256).to(device)
    feat_ext.load_state_dict(cc_clf["feat_extractor"]); feat_ext.eval()
    s_clf = StyleClassifier(256, cc_clf["num_styles"]).to(device)
    s_clf.load_state_dict(cc_clf["style_clf"]); s_clf.eval()
    c_clf = ContentClassifier(256, cc_clf["num_contents"]).to(device)
    c_clf.load_state_dict(cc_clf["content_clf"]); c_clf.eval()
    for p in feat_ext.parameters(): p.requires_grad = False
    for p in s_clf.parameters(): p.requires_grad = False
    for p in c_clf.parameters(): p.requires_grad = False

    class CompClf(nn.Module):
        def __init__(self, fe, cl):
            super().__init__()
            self.fe = fe; self.cl = cl
            for p in self.parameters(): p.requires_grad = False
        def forward(self, m): return self.cl(self.fe(m))

    style_clf_loss = None
    if L.get("lambda_style_clf", 0) > 0:
        base_clf = CompClf(feat_ext, s_clf).to(device)
        if L.get("style_clf_mask_root", False):
            style_clf_loss = RootMaskedStyleClassifierLoss(
                base_clf, root_dims=tuple(cfg["root_dims"]), label_smoothing=0.1
            ).to(device)
        else:
            style_clf_loss = StyleClassifierLoss(base_clf).to(device)
    content_clf_loss = StyleClassifierLoss(CompClf(feat_ext, c_clf).to(device)).to(device) \
        if L.get("lambda_content_clf", 0) > 0 else None

    # SMooDi 47-class auxiliary classifier (loss in the "ori" feature space); also builds the
    # style_idx -> SMooDi label LUT when the aligned classifier mode is on
    smoodi_clf = None
    lambda_smoodi_clf = float(L.get("lambda_smoodi_clf", 0.0))
    needs_smoodi_lut = (lambda_smoodi_clf > 0) or (cfg.get("classifier_label_space") == "smoodi47")
    if needs_smoodi_lut:
        smoodi_root = cfg["paths"]["smoodi_root"]
        sys.path.insert(0, smoodi_root)
        from mld.models.architectures.mld_style_encoder import StyleClassification
        smoodi_clf = StyleClassification(nclasses=47).to(device)
        sd = torch.load(cfg["paths"]["smoodi_classifier"], map_location=device, weights_only=False)
        smoodi_clf.load_state_dict(sd, strict=True)
        for p in smoodi_clf.parameters(): p.requires_grad = False
        smoodi_clf.eval()
        ori_mean_np = np.load(cfg["paths"]["smoodi_ori_mean"]).astype(np.float32)
        ori_std_np = np.load(cfg["paths"]["smoodi_ori_std"]).astype(np.float32)
        ori_mean_t = torch.from_numpy(ori_mean_np).to(device)
        ori_std_t = torch.from_numpy(ori_std_np).to(device)
        # SMooDi labels: 100STYLE_name_dict_Filter.txt gives motion_id -> label_idx (0..46)
        smoodi_labels_path = os.environ.get("SMOODI_LABELS", "assets/100STYLE_name_dict_Filter.txt")
        smoodi_label_by_id = {}
        for line in open(smoodi_labels_path):
            parts = line.strip().split()
            if len(parts) >= 3: smoodi_label_by_id[parts[0]] = int(parts[2])
        smoodi_label_by_sname = {}
        import json as _json
        sty_map = _json.load(open(cfg["data"]["mapping_path"]))
        for mid, info in sty_map['id_to_style_content'].items():
            sn = info['style']
            if mid in smoodi_label_by_id:
                smoodi_label_by_sname.setdefault(sn, smoodi_label_by_id[mid])
        # build idx -> smoodi_label LUT using Style100's style_to_idx
        style_idx_to_smoodi = torch.full((max(style100.style_to_idx.values()) + 2,), -1, dtype=torch.long)
        for sn, idx in style100.style_to_idx.items():
            style_idx_to_smoodi[idx] = smoodi_label_by_sname.get(sn, -1)
        style_idx_to_smoodi = style_idx_to_smoodi.to(device)
        print(f"[SMooDi-Clf] loaded 47-class; mapped {len(smoodi_label_by_sname)} style names "
              f"-> {(style_idx_to_smoodi >= 0).sum().item()} our idxs map to SMooDi labels")

    # T2M evaluator + moment matching
    from auxiliary import T2MEvaluation
    t2m_eval = T2MEvaluation(t2m_dir=cfg["paths"]["t2m_dir"], device=device,
                              glove_dir=cfg["paths"]["glove_dir"])
    for p in t2m_eval.movement_encoder.parameters(): p.requires_grad = False
    for p in t2m_eval.motion_encoder.parameters(): p.requires_grad = False
    t2m_eval.movement_encoder.eval()
    t2m_eval.motion_encoder.train()  # cudnn RNN backward

    def t2m_encode_diff(motion_263, m_lens):
        motion_input = motion_263[:, :, :t2m_eval.dim_pose - 4]
        movement_feats = t2m_eval.movement_encoder(motion_input)
        m_lens_down = torch.clamp(m_lens // t2m_eval.unit_length, min=1)
        return t2m_eval.motion_encoder(movement_feats, m_lens_down)

    moment_loss = MomentMatchingLoss(feature_dim=512, ema=L.get("moment_ema", 0.99)).to(device)
    # optional multi-kernel MMD (higher-order moments)
    from model.losses_adv import MultiKernelMMDLoss
    mmd_loss_fn = MultiKernelMMDLoss(feature_dim=512,
                                        kernel_mul=float(L.get("mmd_kernel_mul", 2.0)),
                                        kernel_num=int(L.get("mmd_kernel_num", 5))).to(device)
    ema = EMA(model, decay=t_cfg.get("ema_decay", 0.999))

    start_epoch = 0
    if args.resume:
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        for cn in ["style_encoder", "style_injector", "linear_probe", "temporal_gate",
                   "joint_style_fuser", "style_head"]:
            k = f"{cn}_state_dict"
            if k in ck and hasattr(model, cn):
                getattr(model, cn).load_state_dict(ck[k], strict=False)
                print(f"  [Resume] loaded {cn}")
        if "critic_state_dict" in ck:
            critic.load_state_dict(ck["critic_state_dict"], strict=False)
        start_epoch = ck.get("epoch", 0) + 1

    root_dims = tuple(cfg["root_dims"])
    foot_dims = tuple(cfg["foot_dims"])
    identity_zero_style = cfg.get("identity_zero_style", True)

    lambda_body_anchor = float(L.get("lambda_body_anchor", 0.0))

    print(f"\n{'='*60}\nTraining {cfg['name']}\n"
          f"  Epochs: {t_cfg['epochs']}, G.LR: {t_cfg['lr']}, D.LR: {cc['lr']}\n"
          f"  max_motion_length: {d['max_motion_length']}\n"
          f"  n_critic: {cc['n_critic']}, lambda_gp: {cc['lambda_gp']}\n"
          f"  lambda_adv: {L['lambda_adv']} (warmup {L['adv_warmup_epochs']} ep)\n"
          f"  lambda_moment: {L['lambda_moment']}\n"
          f"  lambda_root_soft: {L['lambda_root_soft']}, lambda_foot_soft: {L['lambda_foot_soft']}\n"
          f"  lambda_body_anchor (cross-domain): {lambda_body_anchor}\n"
          f"  lambda_smoodi_clf (aux): {lambda_smoodi_clf}\n"
          f"  cross_domain: {cfg.get('cross_domain', {}).get('enabled', False)}\n"
          f"  identity_zero_style: {identity_zero_style}\n"
          f"  Style injector: {cfg.get('style_arch', 'mpsm')} (rank={cfg['style_model'].get('style_rank', 'N/A')})\n{'='*60}\n")

    best_metric = float("inf")
    log_rows = []
    metrics_interval = cfg.get("metrics_interval", 5)

    # persistent iterator for the dual-path HumanML3D loader (one batch per step)
    hml_pure_iter = iter(hml_pure_loader) if hml_pure_loader is not None else None

    # t-SNE snapshotter
    tsne_cfg = cfg.get("tsne", {"enabled": False})
    tsne_snap = None
    if tsne_cfg.get("enabled", False):
        try:
            from evaluation.tsne_snapshot import TSNESnapshotter
        except ImportError as e:
            raise SystemExit("tsne.enabled needs evaluation/tsne_snapshot.py, which is not part of this release") from e
        tsne_snap = TSNESnapshotter(
            style100_dataset=style100,
            device=device,
            out_dir=str(output_dir / tsne_cfg.get("subdir", "tsne")),
            max_styles=int(tsne_cfg.get("max_styles", 25)),
            per_style=int(tsne_cfg.get("per_style", 24)),
            seed=int(tsne_cfg.get("seed", 42)),
            tag=cfg["name"],
        )
    tsne_every_steps = int(tsne_cfg.get("every_steps", 500))
    global_step = 0

    # Joint-aware style guidance
    ja_cfg = cfg.get("joint_aware", {"enabled": False})
    joint_gate = None
    joint_guide = None
    joint_bottleneck = None
    ja_variant = ja_cfg.get("variant", "B")
    lambda_ja_guidance = float(ja_cfg.get("lambda_guidance", 0.0))
    lambda_ja_neutral = float(ja_cfg.get("lambda_neutral", 0.1))
    lambda_ja_separation = float(ja_cfg.get("lambda_separation", 0.1))
    lambda_ja_sparsity = float(ja_cfg.get("lambda_sparsity", 0.05))
    ja_sparsity_target = float(ja_cfg.get("sparsity_target", 0.3))
    ja_neutral_name = ja_cfg.get("neutral_style", "Neutral")
    ja_neutral_idx = style100.style_to_idx.get(ja_neutral_name, -1)
    if ja_cfg.get("enabled", False) and ja_variant == "B":
        from model.joint_aware_style import NeutralContrastJointGate, JointToCodeGuidance
        style_dim = cfg["style_model"].get("style_embed_dim", 256)
        code_dim = cfg["style_model"].get("code_dim", 512)
        joint_gate = NeutralContrastJointGate(style_dim=style_dim).to(device)
        joint_guide = JointToCodeGuidance(num_joints=22, code_dim=code_dim).to(device)
        trainable = list(trainable) + list(joint_gate.parameters()) + list(joint_guide.parameters())
        # Rebuild optimizer to include new parameters
        g_optim = torch.optim.AdamW(trainable, lr=t_cfg["lr"],
            weight_decay=t_cfg["weight_decay"], betas=tuple(t_cfg["betas"]))
        from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR
        w_sched = LambdaLR(g_optim, lambda s: min(1.0, s / max(warmup, 1)))
        m_sched = CosineAnnealingLR(g_optim, T_max=max(1, total_steps - warmup),
                                     eta_min=t_cfg.get("lr_min", 1e-6))
        scheduler = SequentialLR(g_optim, [w_sched, m_sched], milestones=[warmup])
        print(f"[joint-aware] Variant B enabled - neutral_idx={ja_neutral_idx} "
              f"gate_params={sum(p.numel() for p in joint_gate.parameters()):,} "
              f"guide_params={sum(p.numel() for p in joint_guide.parameters()):,}")
    elif ja_cfg.get("enabled", False) and ja_variant == "C":
        from model.joint_aware_style import InfoBottleneckJointMask, JointToCodeGuidance
        style_dim = cfg["style_model"].get("style_embed_dim", 256)
        code_dim = cfg["style_model"].get("code_dim", 512)
        joint_bottleneck = InfoBottleneckJointMask(
            style_dim=style_dim, sparsity=ja_sparsity_target).to(device)
        joint_guide = JointToCodeGuidance(num_joints=22, code_dim=code_dim).to(device)
        trainable = list(trainable) + list(joint_bottleneck.parameters()) + list(joint_guide.parameters())
        g_optim = torch.optim.AdamW(trainable, lr=t_cfg["lr"],
            weight_decay=t_cfg["weight_decay"], betas=tuple(t_cfg["betas"]))
        from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR
        w_sched = LambdaLR(g_optim, lambda s: min(1.0, s / max(warmup, 1)))
        m_sched = CosineAnnealingLR(g_optim, T_max=max(1, total_steps - warmup),
                                     eta_min=t_cfg.get("lr_min", 1e-6))
        scheduler = SequentialLR(g_optim, [w_sched, m_sched], milestones=[warmup])
        print(f"[joint-aware] Variant C (info-bottleneck) enabled")

    # t-SNE snapshot before the first step
    if tsne_snap is not None:
        try:
            tsne_snap.snapshot(
                model, epoch=start_epoch, step=0,
                extra_tags={"phase": "init",
                            "contrastive": f"{L.get('lambda_contrastive', 0):g}"})
        except Exception as _tsne_err:
            print(f"  [t-SNE init] WARN: {_tsne_err}")

    # optional validation before the first step
    if cfg.get("val_at_init", False) and start_epoch == 0:
        model.eval()
        init_metrics = evaluate_val(model, val_loader, device,
                                     sc_mean, sc_std, hml_mean, hml_std,
                                     t2m_eval, feat_ext, s_clf,
                                     style100_dataset=style100,
                                     root_dims=root_dims)
        print(f"[E0 init]  FID={init_metrics.get('FID', 0):.3f}  "
              f"SRA={init_metrics.get('SRA', 0):.1f}%  "
              f"Div={init_metrics.get('Diversity', 0):.2f}")
        log_rows.append({"epoch": 0, "step": 0, **{f"val_{k}": v for k, v in init_metrics.items()}})
    n_critic = int(cc["n_critic"])
    lambda_gp = float(cc["lambda_gp"])
    lambda_adv_target = float(L["lambda_adv"])
    adv_warmup_epochs = int(L["adv_warmup_epochs"])
    lambda_moment = float(L["lambda_moment"])
    lambda_root_soft = float(L["lambda_root_soft"])
    lambda_foot_soft = float(L["lambda_foot_soft"])

    for epoch in range(start_epoch, t_cfg["epochs"]):
        t0 = time.time()
        model.train(); critic.train()
        if hasattr(model, "frozen_vae"): model.frozen_vae.eval()
        criterion.set_epoch(epoch)

        lambda_adv = 0.0 if epoch < adv_warmup_epochs else \
            lambda_adv_target * min(1.0, (epoch - adv_warmup_epochs + 1) / max(1, adv_warmup_epochs))

        loss_acc = defaultdict(float); n_batches = 0

        for batch in train_loader:
            # drop samples with extreme normalized values (corrupt retargets) before the critic and G see them
            _dmax = float(L.get("data_max_abs", 30.0))
            _B0 = batch["target_motion"].shape[0]
            _mx = torch.maximum(batch["target_motion"].abs().flatten(1).max(1).values, batch["content_motion"].abs().flatten(1).max(1).values)
            _keep = torch.isfinite(_mx) & (_mx < _dmax)
            if not bool(_keep.all()):
                _bad = [(batch["mid"][i] if "mid" in batch else i, round(float(_mx[i]), 1)) for i in (~_keep).nonzero().flatten().tolist()]
                globals()["_DATA_DROPS"] = globals().get("_DATA_DROPS", 0) + len(_bad)
                print(f"[data-guard] dropping {len(_bad)} sample(s) {_bad} (total dropped {globals()['_DATA_DROPS']})", flush=True)
                if int(_keep.sum()) < 2: continue
                _ki = _keep.nonzero().flatten()
                batch = {k: (v[_ki] if torch.is_tensor(v) and v.dim() > 0 and v.shape[0] == _B0 else
                             ([v[i] for i in _ki.tolist()] if isinstance(v, list) and len(v) == _B0 else v)) for k, v in batch.items()}
            update_jtc_extras(batch)
            beta = batch["beta"].to(device)
            target = batch["target_motion"].to(device)
            tgt_len = batch["target_length"].to(device)
            content = batch["content_motion"].to(device)
            style = batch["style_motion"].to(device)
            is_id = batch.get("is_identity")
            if is_id is not None: is_id = is_id.to(device).bool()
            is_cross = batch.get("is_cross_domain")
            if is_cross is not None: is_cross = is_cross.to(device).bool()

            id_mask = is_id if identity_zero_style else None
            non_cross = ~is_cross if is_cross is not None else None

            # D
            d_loss_sum = gp_sum = d_real_sum = d_fake_sum = 0.0
            if lambda_adv > 0:
                real_motion = target[is_id] if (is_id is not None and is_id.any()) else content
                for _ in range(n_critic):
                    with torch.no_grad():
                        out_d = model(content_motion=content, beta=beta, style_motion=style,
                                       identity_mask=id_mask)
                        fake_motion = out_d["motion_recon"]
                    n = min(real_motion.shape[0], fake_motion.shape[0])
                    if n < 2: break
                    T = min(real_motion.shape[1], fake_motion.shape[1])
                    rm = real_motion[:n, :T]; fm = fake_motion[:n, :T]
                    d_optim.zero_grad(set_to_none=True)
                    d_loss_core, d_real, d_fake = wgan_critic_loss(critic, rm, fm)
                    gp = gradient_penalty(critic, rm, fm, lambda_gp=lambda_gp)
                    d_loss = d_loss_core + gp
                    if not torch.isfinite(d_loss):
                        print("[NaN-guard] non-finite critic loss, skipping critic step", flush=True)
                        d_optim.zero_grad(set_to_none=True); break
                    d_loss.backward()
                    torch.nn.utils.clip_grad_norm_(critic.parameters(), 5.0)
                    d_optim.step()
                    d_loss_sum += float(d_loss.item()); gp_sum += float(gp.item())
                    d_real_sum += float(d_real.item()); d_fake_sum += float(d_fake.item())

            # G
            g_optim.zero_grad(set_to_none=True)
            output = model(content_motion=content, beta=beta,
                          style_motion=style, target_motion=target,
                          identity_mask=id_mask)
            if bool(L.get("traj_passthrough", False)):
                # Trajectory passthrough: root angular + XZ velocity come from the content.
                _mr = output["motion_recon"]; _Tp = min(_mr.shape[1], content.shape[1])
                output["motion_recon"] = torch.cat([content[:, :_Tp, 0:3].detach(), _mr[:, :_Tp, 3:]], -1)

            # Skip recon/code_recon on cross-domain samples (no valid target)
            if non_cross is not None and non_cross.any() and not non_cross.all():
                recon_output = {k: (v[non_cross] if torch.is_tensor(v) and v.shape[0] == content.shape[0]
                                     else v) for k, v in output.items()}
                loss, ld = criterion(recon_output, target[non_cross], tgt_len[non_cross])
                loss = loss * (non_cross.sum().float() / content.shape[0])
            elif non_cross is None or non_cross.all():
                loss, ld = criterion(output, target, tgt_len)
            else:
                loss = torch.tensor(0.0, device=device, requires_grad=True)
                ld = {"total": 0.0, "recon": 0.0}

            # identity anchor: identity rows should reproduce the frozen VAE reconstruction
            lambda_id_anchor = float(L.get("lambda_identity_anchor", 0.0))
            if lambda_id_anchor > 0 and is_id is not None and is_id.any():
                with torch.no_grad():
                    # decode_baseline runs the frozen VAE (content -> quantize -> decode)
                    vae_baseline = model.decode_baseline(content[is_id], beta[is_id])
                T_id = min(vae_baseline.shape[1], output["motion_recon"].shape[1])
                id_diff = (output["motion_recon"][is_id, :T_id] - vae_baseline[:, :T_id]).abs()
                arange_id = torch.arange(T_id, device=device).unsqueeze(0).expand(is_id.sum().item(), -1)
                len_id = tgt_len[is_id].clamp(max=T_id)
                mask_id = (arange_id < len_id.unsqueeze(1)).float().unsqueeze(-1)
                id_anchor = (id_diff * mask_id).sum() / (mask_id.sum() * 263 + 1e-8)
                loss = loss + lambda_id_anchor * id_anchor
                ld["id_anchor"] = float(id_anchor.item())

            # Body content preservation for cross-domain samples (no recon target)
            # Anchors body dims [4:259] to content to prevent mode collapse when recon is skipped.
            if lambda_body_anchor > 0 and is_cross is not None and is_cross.any():
                T = min(output["motion_recon"].shape[1], content.shape[1])
                body_dims = list(range(4, 259))
                body_diff = (output["motion_recon"][is_cross, :T][:, :, body_dims]
                             - content[is_cross, :T][:, :, body_dims]).abs()
                # length mask
                B_c = is_cross.sum().item()
                arange = torch.arange(T, device=device).unsqueeze(0).expand(B_c, -1)
                lens_cross = tgt_len[is_cross].clamp(max=T)
                mask_c = (arange < lens_cross.unsqueeze(1)).float().unsqueeze(-1)
                body_anchor = (body_diff * mask_c).sum() / (mask_c.sum() * len(body_dims) + 1e-8)
                loss = loss + lambda_body_anchor * body_anchor
                ld["body_anchor"] = float(body_anchor.item())

            # Content-preservation extension: latent-space code preservation + global motion anchor
            lambda_code_pres = float(L.get("lambda_code_pres", 0.0))
            if lambda_code_pres > 0 and "fused_codes" in output and "content_codes" in output:
                # L1 between fused and content codes on cross-domain rows (recon rows are tied by the recon loss)
                if is_cross is not None and is_cross.any():
                    fused_c = output["fused_codes"][is_cross]
                    content_c = output["content_codes"][is_cross]
                    code_pres = (fused_c - content_c).abs().mean()
                    loss = loss + lambda_code_pres * code_pres
                    ld["code_pres"] = float(code_pres.item())

            # Distillation: MCM-LDM teacher on cross-domain rows that carry a
            # teacher output. Teacher was generated from Xn (neutral-shape content), so we
            # compare against a second forward at beta=0 (shape losses stay on the
            # beta-conditioned forward above).
            lambda_teacher = float(L.get("lambda_teacher", 0.0))
            lambda_teacher_dyn = float(L.get("lambda_teacher_dyn", 0.0))
            has_t = batch.get("has_teacher")
            if (lambda_teacher > 0 or lambda_teacher_dyn > 0) and has_t is not None and has_t.any():
                has_t = has_t.to(device).bool()
                t_mot = batch["teacher_motion"].to(device)
                t_len = batch["teacher_length"].to(device)
                out0 = model(content_motion=content, beta=torch.zeros_like(beta),
                             style_motion=style, target_motion=None, identity_mask=None)
                m0 = out0["motion_recon"]
                Tt = min(m0.shape[1], t_mot.shape[1])
                dif = (m0[has_t, :Tt] - t_mot[has_t, :Tt]).abs()
                w = torch.ones(263, device=device)
                w[0:4] = float(L.get("teacher_w_root", 1.0))
                w[4:67] = float(L.get("teacher_w_ric", 1.0))
                w[67:193] = float(L.get("teacher_w_rot", 1.0))
                w[193:259] = float(L.get("teacher_w_vel", 1.0))
                w[259:263] = float(L.get("teacher_w_foot", 1.0))
                dif = dif * w
                ar_t = torch.arange(Tt, device=device).unsqueeze(0).expand(int(has_t.sum().item()), -1)
                mk_t = (ar_t < t_len[has_t].clamp(max=Tt).unsqueeze(1)).float().unsqueeze(-1)
                teacher_l = (dif * mk_t).sum() / (mk_t.sum() * w.sum() + 1e-8)
                if lambda_teacher > 0:
                    loss = loss + lambda_teacher * teacher_l
                ld["teacher"] = float(teacher_l.item())

                # distillation, dynamics: phase-invariant matching to the teacher via (a) per-joint velocity RMS,
                # (b) the log-magnitude temporal spectrum of root-relative joint positions and (c) knee/elbow
                # angle ranges; all terms are scale-free
                if lambda_teacher_dyn > 0:
                    from model.nnutils.motion_utils import recover_from_ric as _rfr
                    s_un = m0[has_t, :Tt] * sc_std_t + sc_mean_t
                    with torch.no_grad():
                        t_un = t_mot[has_t, :Tt] * sc_std_t + sc_mean_t
                        t_xyz = _rfr(t_un, joints_num=22)
                    s_xyz = _rfr(s_un, joints_num=22)                       # [b,T,22,3]
                    mk3 = mk_t.unsqueeze(-1)                                 # [b,T,1,1]
                    s_rel = (s_xyz - s_xyz[:, :, :1]) * mk3
                    t_rel = (t_xyz - t_xyz[:, :, :1]) * mk3
                    mv = (mk_t[:, 1:] * mk_t[:, :-1])                        # [b,T-1,1]
                    sv = (s_rel[:, 1:] - s_rel[:, :-1]).pow(2).sum(-1)       # [b,T-1,22]
                    tv = (t_rel[:, 1:] - t_rel[:, :-1]).pow(2).sum(-1)
                    s_rms = ((sv * mv).sum(1) / (mv.sum(1) + 1e-6) + 1e-8).sqrt()
                    t_rms = ((tv * mv).sum(1) / (mv.sum(1) + 1e-6) + 1e-8).sqrt()
                    l_rms = ((s_rms - t_rms).abs() / (t_rms + 0.01)).mean()   # floor 1cm/frame: low-motion joints must not dominate
                    S = torch.fft.rfft(s_rel, dim=1).abs()[:, 1:]
                    Tm = torch.fft.rfft(t_rel, dim=1).abs()[:, 1:]
                    l_spec = (torch.log1p(S * 10.0) - torch.log1p(Tm * 10.0)).abs().mean()
                    def _ang(x, a, b, c):
                        u = x[:, :, a] - x[:, :, b]; v = x[:, :, c] - x[:, :, b]
                        cs = (u * v).sum(-1) / (u.norm(dim=-1) * v.norm(dim=-1) + 1e-6)
                        return torch.acos(cs.clamp(-1 + 1e-4, 1 - 1e-4))     # [b,T]
                    l_range = 0.0
                    valid = mk_t[..., 0] > 0
                    for (a, b_, c) in [(1, 4, 7), (2, 5, 8), (16, 18, 20), (17, 19, 21)]:
                        sa = _ang(s_xyz, a, b_, c); ta = _ang(t_xyz, a, b_, c)
                        s_rng = torch.where(valid, sa, sa.detach() * 0 - 9).max(1).values - torch.where(valid, sa, sa.detach() * 0 + 9).min(1).values
                        t_rng = torch.where(valid, ta, ta * 0 - 9).max(1).values - torch.where(valid, ta, ta * 0 + 9).min(1).values
                        l_range = l_range + ((s_rng - t_rng).abs() / (t_rng + 0.05)).mean() / 4.0
                    teacher_dyn = l_rms + l_spec + l_range
                    loss = loss + lambda_teacher_dyn * teacher_dyn
                    ld["teacher_dyn"] = float(teacher_dyn.item()); ld["td_rms"] = float(l_rms.item())
                    ld["td_spec"] = float(l_spec.item()); ld["td_rng"] = float(l_range.item())

            # content tempo: on cross-domain rows, match the per-leg-joint velocity RMS of the
            # beta-conditioned output to the content (relative L1), so the gait keeps the content's tempo
            lambda_tempo = float(L.get("lambda_content_tempo", 0.0))
            if lambda_tempo > 0 and is_cross is not None and is_cross.any():
                from model.nnutils.motion_utils import recover_from_ric as _rfr2
                Tc = min(output["motion_recon"].shape[1], content.shape[1])
                o_un = output["motion_recon"][is_cross, :Tc] * sc_std_t + sc_mean_t
                with torch.no_grad():
                    c_un = content[is_cross, :Tc] * sc_std_t + sc_mean_t
                    c_xyz = _rfr2(c_un, joints_num=22)
                o_xyz = _rfr2(o_un, joints_num=22)
                LEGS = [1, 2, 4, 5, 7, 8, 10, 11]
                ar_c = torch.arange(Tc, device=device).unsqueeze(0).expand(int(is_cross.sum().item()), -1)
                mkc = (ar_c < tgt_len[is_cross].clamp(max=Tc).unsqueeze(1)).float().unsqueeze(-1)   # [b,T,1]
                o_rel = (o_xyz - o_xyz[:, :, :1])[:, :, LEGS] * mkc.unsqueeze(-1)
                c_rel = (c_xyz - c_xyz[:, :, :1])[:, :, LEGS] * mkc.unsqueeze(-1)
                # low-pass (5-frame box) before measuring tempo, so decoder jitter does not inflate it
                def _lp(x):  # [b,T,J,3] -> same, avg-pool over time
                    b_, T_, J_, C_ = x.shape
                    y = x.permute(0, 2, 3, 1).reshape(b_, J_ * C_, T_)
                    y = torch.nn.functional.avg_pool1d(y, 5, stride=1, padding=2, count_include_pad=False)
                    return y.reshape(b_, J_, C_, T_).permute(0, 3, 1, 2)
                o_lp, c_lp = _lp(o_rel), _lp(c_rel)
                mvc = mkc[:, 1:] * mkc[:, :-1]
                ov = (o_lp[:, 1:] - o_lp[:, :-1]).pow(2).sum(-1); cv = (c_lp[:, 1:] - c_lp[:, :-1]).pow(2).sum(-1)
                o_rms = ((ov * mvc).sum(1) / (mvc.sum(1) + 1e-6) + 1e-8).sqrt()
                c_rms = ((cv * mvc).sum(1) / (mvc.sum(1) + 1e-6) + 1e-8).sqrt()
                # one-sided: only penalise *slower* than content (styles may add energy freely)
                tempo_l = (torch.relu(c_rms - o_rms) / (c_rms + 0.01)).mean()
                loss = loss + lambda_tempo * tempo_l
                ld["tempo"] = float(tempo_l.item())
                # smoothness: root-relative second-difference energy may not exceed the content's
                lambda_smooth = float(L.get("lambda_smooth", 0.0))
                if lambda_smooth > 0:
                    oa = (o_xyz - o_xyz[:, :, :1]) * mkc.unsqueeze(-1); ca = (c_xyz - c_xyz[:, :, :1]) * mkc.unsqueeze(-1)
                    m2 = (mkc[:, 2:] * mkc[:, 1:-1] * mkc[:, :-2])
                    o2 = (oa[:, 2:] - 2 * oa[:, 1:-1] + oa[:, :-2]).norm(dim=-1)   # [b,T-2,22]
                    c2 = (ca[:, 2:] - 2 * ca[:, 1:-1] + ca[:, :-2]).norm(dim=-1)
                    o2m = (o2 * m2).sum(1) / (m2.sum(1) + 1e-6); c2m = (c2 * m2).sum(1) / (m2.sum(1) + 1e-6)
                    smooth_l = (torch.relu(o2m - 1.2 * c2m) / (c2m + 1e-3)).mean()
                    loss = loss + lambda_smooth * smooth_l
                    ld["smooth"] = float(smooth_l.item())

            # floor: the lowest foot/ankle point of every clip should lie on the ground (+-2 cm)
            lambda_floor = float(L.get("lambda_floor", 0.0))
            if lambda_floor > 0:
                from model.nnutils.motion_utils import recover_from_ric as _rfr3
                Tf = min(output["motion_recon"].shape[1], content.shape[1])
                f_un = output["motion_recon"][:, :Tf] * sc_std_t + sc_mean_t
                f_xyz = _rfr3(f_un, joints_num=22)                                       # [B,T,22,3]
                ar_f = torch.arange(Tf, device=device).unsqueeze(0).expand(content.shape[0], -1)
                mkf = (ar_f < tgt_len.clamp(max=Tf).unsqueeze(1))                        # [B,T]
                feet_y = f_xyz[:, :, [7, 8, 10, 11], 1].min(-1).values                   # [B,T]
                feet_y = torch.where(mkf, feet_y, torch.full_like(feet_y, 1e3))
                min_y = feet_y.min(1).values                                             # [B]
                floor_l = (torch.relu(min_y - 0.02) + torch.relu(-min_y - 0.02)).mean()
                loss = loss + lambda_floor * floor_l
                ld["floor"] = float(floor_l.item())

            # Content-contact foot loss: stance timing from the content's contact labels
            lambda_cc = float(L.get("lambda_content_contact", 0.0)); lambda_cch = float(L.get("lambda_content_contact_h", 0.0))
            if lambda_cc > 0 or lambda_cch > 0:
                from model.nnutils.motion_utils import recover_from_ric as _rfr6
                Tc6 = min(output["motion_recon"].shape[1], content.shape[1])
                with torch.no_grad():
                    c_ct = (content[:, :Tc6, 259:263] * sc_std_t[259:263] + sc_mean_t[259:263] > 0.5).float()   # [B,T,4] for joints [7,10,8,11]
                    ar6 = torch.arange(Tc6, device=device).unsqueeze(0).expand(content.shape[0], -1)
                    mk6 = (ar6 < tgt_len.clamp(max=Tc6).unsqueeze(1)).float()
                    c_ct = c_ct * mk6.unsqueeze(-1)
                o_xyz6 = _rfr6(output["motion_recon"][:, :Tc6] * sc_std_t + sc_mean_t, joints_num=22)
                if o_xyz6.dim() == 3: o_xyz6 = o_xyz6.reshape(*o_xyz6.shape[:2], -1, 3)
                feet6 = o_xyz6[:, :, [7, 10, 8, 11]]                                            # [B,T,4,3] same order as labels
                if lambda_cc > 0:
                    v6 = (feet6[:, 1:, :, [0, 2]] - feet6[:, :-1, :, [0, 2]]).norm(dim=-1)      # [B,T-1,4] m/frame
                    both6 = c_ct[:, 1:] * c_ct[:, :-1]
                    cc_l = (v6 * both6).sum() / (both6.sum() + 1e-8)
                    loss = loss + lambda_cc * cc_l
                    ld["cc_vel"] = float(cc_l.item())
                if lambda_cch > 0:
                    fy6 = feet6[..., 1]
                    fl6 = torch.where(mk6.bool().unsqueeze(-1), fy6, torch.full_like(fy6, 1e3)).amin(dim=(1, 2), keepdim=True).detach()
                    cch_l = (torch.relu(fy6 - fl6 - 0.03) * c_ct).sum() / (c_ct.sum() + 1e-8)
                    loss = loss + lambda_cch * cch_l
                    ld["cc_h"] = float(cch_l.item())

            # geometric foot skate + stance duty
            lambda_sk = float(L.get("lambda_foot_skate_geo", 0.0)); lambda_duty = float(L.get("lambda_stance_duty", 0.0))
            if lambda_sk > 0 or lambda_duty > 0:
                from model.nnutils.motion_utils import recover_from_ric as _rfr5
                h_thr = float(L.get("skate_h_thr", 0.05)); tau5 = float(L.get("skate_tau", 0.02))
                Ts = min(output["motion_recon"].shape[1], content.shape[1])
                o_xyz5 = _rfr5(output["motion_recon"][:, :Ts] * sc_std_t + sc_mean_t, joints_num=22)
                if o_xyz5.dim() == 3: o_xyz5 = o_xyz5.reshape(*o_xyz5.shape[:2], -1, 3)
                ar5 = torch.arange(Ts, device=device).unsqueeze(0).expand(content.shape[0], -1)
                mk5 = (ar5 < tgt_len.clamp(max=Ts).unsqueeze(1)).float()                       # [B,T]
                feet5 = o_xyz5[:, :, [7, 8, 10, 11]]                                            # [B,T,4,3]
                fy5 = feet5[..., 1]
                floor5 = torch.where(mk5.bool().unsqueeze(-1), fy5, torch.full_like(fy5, 1e3)).amin(dim=(1, 2), keepdim=True).detach()
                w5 = torch.sigmoid((h_thr - (fy5 - floor5)) / tau5)                              # contact weight [B,T,4]
                v5 = (feet5[:, 1:, :, [0, 2]] - feet5[:, :-1, :, [0, 2]]).norm(dim=-1)          # [B,T-1,4] m/frame
                mkv5 = (mk5[:, 1:] * mk5[:, :-1]).unsqueeze(-1)
                if lambda_sk > 0:
                    sk_l = (w5[:, 1:] * w5[:, :-1] * v5 * mkv5).sum() / (mkv5.sum() * 4 + 1e-8)
                    loss = loss + lambda_sk * sk_l
                    ld["skate_geo"] = float(sk_l.item())
                if lambda_duty > 0:
                    with torch.no_grad():
                        c_xyz5 = _rfr5(content[:, :Ts] * sc_std_t + sc_mean_t, joints_num=22)
                        if c_xyz5.dim() == 3: c_xyz5 = c_xyz5.reshape(*c_xyz5.shape[:2], -1, 3)
                        cfy5 = c_xyz5[:, :, [7, 8, 10, 11], 1]
                        cfl5 = torch.where(mk5.bool().unsqueeze(-1), cfy5, torch.full_like(cfy5, 1e3)).amin(dim=(1, 2), keepdim=True)
                        cw5 = torch.sigmoid((h_thr - (cfy5 - cfl5)) / tau5)
                        c_duty = (cw5 * mk5.unsqueeze(-1)).sum(1) / (mk5.sum(1, keepdim=True) + 1e-8)   # [B,4]
                    o_duty = (w5 * mk5.unsqueeze(-1)).sum(1) / (mk5.sum(1, keepdim=True) + 1e-8)
                    duty_l = torch.relu(float(L.get("duty_ratio", 0.8)) * c_duty - o_duty).mean()
                    loss = loss + lambda_duty * duty_l
                    ld["duty"] = float(duty_l.item())

            # trajectory: keep the content's global root path
            lambda_traj = float(L.get("lambda_traj_xz", 0.0)); lambda_rrot = float(L.get("lambda_root_rot", 0.0))
            if lambda_traj > 0 or lambda_rrot > 0:
                from model.nnutils.motion_utils import recover_from_ric as _rfr4
                Tt2 = min(output["motion_recon"].shape[1], content.shape[1])
                ar_t2 = torch.arange(Tt2, device=device).unsqueeze(0).expand(content.shape[0], -1)
                mk_t2 = (ar_t2 < tgt_len.clamp(max=Tt2).unsqueeze(1)).float()
                if lambda_traj > 0:
                    o_un2 = output["motion_recon"][:, :Tt2] * sc_std_t + sc_mean_t
                    with torch.no_grad():
                        c_un2 = content[:, :Tt2] * sc_std_t + sc_mean_t
                        c_root = _rfr4(c_un2, joints_num=22)[:, :, 0, [0, 2]]
                    o_root = _rfr4(o_un2, joints_num=22)[:, :, 0, [0, 2]]
                    traj_l = ((o_root - c_root).abs().sum(-1) * mk_t2).sum() / (mk_t2.sum() + 1e-8)
                    loss = loss + lambda_traj * traj_l
                    ld["traj_xz"] = float(traj_l.item())
                if lambda_rrot > 0:
                    rr = (output["motion_recon"][:, :Tt2, 0] - content[:, :Tt2, 0]).abs()
                    rrot_l = (rr * mk_t2).sum() / (mk_t2.sum() + 1e-8)
                    loss = loss + lambda_rrot * rrot_l
                    ld["root_rot"] = float(rrot_l.item())

            lambda_motion_anchor = float(L.get("lambda_motion_anchor", 0.0))
            if lambda_motion_anchor > 0 and is_cross is not None and is_cross.any():
                # Strong anchor: full motion (including root traj dims [0:4]) on cross-domain.
                # Velocity-preserving: anchor only on temporal differences (allow mean shift but not derivative shift).
                T = min(output["motion_recon"].shape[1], content.shape[1])
                v_out = output["motion_recon"][is_cross, 1:T] - output["motion_recon"][is_cross, :T-1]
                v_con = content[is_cross, 1:T] - content[is_cross, :T-1]
                v_diff = (v_out - v_con).abs()
                B_c = is_cross.sum().item()
                arange = torch.arange(T-1, device=device).unsqueeze(0).expand(B_c, -1)
                lens_cross = tgt_len[is_cross].clamp(max=T-1)
                mask_v = (arange < lens_cross.unsqueeze(1)).float().unsqueeze(-1)
                motion_anchor = (v_diff * mask_v).sum() / (mask_v.sum() * 263 + 1e-8)
                loss = loss + lambda_motion_anchor * motion_anchor
                ld["motion_anchor"] = float(motion_anchor.item())

            # trajectory consistency loss (cumulative position match)
            lambda_traj = float(L.get("lambda_trajectory_consistency", 0.0))
            if lambda_traj > 0:
                from model.shape_foot_losses import trajectory_consistency_loss
                # anchor to content for style transfer; for identity rows content==target so equivalent
                tl, tl_parts = trajectory_consistency_loss(
                    output["motion_recon"], content, motion_length=tgt_len)
                loss = loss + lambda_traj * tl
                ld["traj_total"] = float(tl.item())
                ld["traj_yaw"] = float(tl_parts["yaw"].item())
                ld["traj_xz"]  = float(tl_parts["xz"].item())
                ld["traj_h"]   = float(tl_parts["h"].item())

            # soft root/foot: anchor to the target (default) or to the content
            rf_anchor = target if cfg.get("root_foot_anchor", "target") == "target" else content
            if lambda_root_soft > 0:
                free_h = (cfg.get("cross_root_free_height", False)
                          and is_cross is not None and is_cross.any())
                h_from_style = (cfg.get("cross_root_height_from_style", False)
                                and is_cross is not None and is_cross.any())
                if free_h or h_from_style:
                    # crouch/crawl styles must be able to lower the pelvis: on transfer rows anchor only
                    # rotation and XZ trajectory to the content; height is left free (free_h) or pulled
                    # toward the style reference's mean pelvis height (h_from_style)
                    xz_dims = tuple(d for d in root_dims if d != 3)
                    rl = soft_root_loss(output["motion_recon"][is_cross],
                                         rf_anchor[is_cross], root_dims=xz_dims,
                                         motion_length=tgt_len[is_cross]) * is_cross.float().mean()
                    if h_from_style:
                        h_out = output["motion_recon"][is_cross][:, :, 3].mean(dim=1)
                        with torch.no_grad():
                            h_sty = style[is_cross][:, :, 3].mean(dim=1)
                        rl = rl + (h_out - h_sty).abs().mean() * is_cross.float().mean()
                    nc = ~is_cross
                    if nc.any():
                        rl = rl + soft_root_loss(output["motion_recon"][nc],
                                                  rf_anchor[nc], root_dims=root_dims,
                                                  motion_length=tgt_len[nc]) * nc.float().mean()
                else:
                    rl = soft_root_loss(output["motion_recon"], rf_anchor, root_dims=root_dims,
                                         motion_length=tgt_len)
                loss = loss + lambda_root_soft * rl
                ld["root_soft"] = float(rl.item())
            if lambda_foot_soft > 0:
                fl = soft_foot_loss(output["motion_recon"], rf_anchor, foot_dims=foot_dims,
                                     motion_length=tgt_len)
                loss = loss + lambda_foot_soft * fl
                ld["foot_soft"] = float(fl.item())

            # Contrastive
            proj = output.get("projected_emb")
            gids = batch.get("contrastive_group_id")
            if proj is not None and gids is not None:
                gids = gids.to(device)
                valid = gids >= 0
                if valid.sum() > 1:
                    lc = contrastive_fn(proj[valid], gids[valid])
                    loss = loss + L.get("lambda_contrastive", 0.5) * lc
                    ld["contrastive"] = float(lc.item())

            # Style cycle: on cross-domain rows (the actual transfer task, which has
            # no recon target) the style encoder should read the reference style back
            # out of the generated motion. Reference embedding is detached so the
            # pressure lands on the generator, not on collapsing the encoder.
            lambda_style_cyc = float(L.get("lambda_style_cycle", 0.0))
            if lambda_style_cyc > 0 and is_cross is not None and is_cross.any():
                with torch.no_grad():
                    s_ref = model.style_encoder(style[is_cross])
                s_out = model.style_encoder(output["motion_recon"][is_cross])
                sc_l = 1.0 - F.cosine_similarity(s_out, s_ref, dim=-1).mean()
                loss = loss + lambda_style_cyc * sc_l
                ld["style_cycle"] = float(sc_l.item())

            # Style clf (Style100 only)
            # derived-feature mode: score the classifiers on features re-extracted from the recovered
            # joint positions (a differentiable port of HumanML3D's process_file)
            derived_sc = None
            clf_on_derived = bool(L.get("clf_on_derived", False))
            lambda_native_cons = float(L.get("lambda_native_consistency", 0.0))
            if clf_on_derived or lambda_native_cons > 0:
                from model.nnutils.hml_features_torch import derive_from_263_batch as _derive
                from model.nnutils.motion_utils import recover_from_ric as _rfr_d
                derived_sc = _derive(output["motion_recon"], tgt_len, sc_mean_t, sc_std_t, _rfr_d)
                if lambda_native_cons > 0:
                    Tn = derived_sc.shape[1]
                    ar_n = torch.arange(Tn, device=device).unsqueeze(0).expand(content.shape[0], -1)
                    mk_n = (ar_n < tgt_len.clamp(max=Tn).unsqueeze(1)).float().unsqueeze(-1)
                    # native channels must agree with what the positions imply (root/ric/rot/vel; foot excluded)
                    dif_n = (output["motion_recon"][:, :Tn, :259] - derived_sc[:, :, :259].detach()).abs()
                    cons_l = (dif_n * mk_n).sum() / (mk_n.sum() * 259 + 1e-8)
                    loss = loss + lambda_native_cons * cons_l
                    ld["native_cons"] = float(cons_l.item())
            _clf_input = derived_sc if clf_on_derived else output["motion_recon"]

            if style_clf_loss is not None and L.get("lambda_style_clf", 0) > 0:
                sidx = batch["style_idx"].to(device)
                # If classifier uses SMooDi 47-class label space, translate sidx via LUT
                if cfg.get("classifier_label_space") == "smoodi47":
                    smoodi_labels_clf = style_idx_to_smoodi[sidx.clamp(min=0)]
                    mask = (sidx >= 0) & (smoodi_labels_clf >= 0)
                    target_labels = smoodi_labels_clf
                else:
                    mask = sidx >= 0
                    target_labels = sidx
                if mask.sum() > 0:
                    sl = style_clf_loss(_clf_input[mask], target_labels[mask])
                    ramp = min(1.0, epoch / max(cfg.get("style_clf_warmup", 3), 1))
                    loss = loss + L["lambda_style_clf"] * ramp * sl
                    ld["style_clf"] = float(sl.item())

            # SMooDi 47-class auxiliary classifier loss (uses style_idx -> SMooDi label LUT)
            if smoodi_clf is not None and lambda_smoodi_clf > 0:
                sidx = batch["style_idx"].to(device)
                smoodi_labels = style_idx_to_smoodi[sidx.clamp(min=0)]   # [-1 for out-of-range]
                sm_mask = (sidx >= 0) & (smoodi_labels >= 0)
                if sm_mask.sum() > 0:
                    gen_sc_m = _clf_input[sm_mask]
                    gen_raw = gen_sc_m * sc_std_t + sc_mean_t
                    gen_ori = (gen_raw - ori_mean_t) / ori_std_t
                    logits = smoodi_clf(gen_ori, stage="Classification")
                    sm_loss = F.cross_entropy(logits, smoodi_labels[sm_mask], label_smoothing=0.1)
                    ramp = min(1.0, epoch / max(cfg.get("style_clf_warmup", 3), 1))
                    loss = loss + lambda_smoodi_clf * ramp * sm_loss
                    ld["smoodi_clf"] = float(sm_loss.item())

            # Content clf
            if content_clf_loss is not None and L.get("lambda_content_clf", 0) > 0:
                cidx = batch.get("content_idx")
                if cidx is not None:
                    cidx = cidx.to(device); mask = cidx >= 0
                    if mask.sum() > 0:
                        cl = content_clf_loss(output["motion_recon"][mask], cidx[mask])
                        loss = loss + L["lambda_content_clf"] * cl
                        ld["content_clf"] = float(cl.item())

            # Fusion regularization (on residual codes)
            fr = L.get("lambda_fusion_reg", 0)
            if fr > 0:
                res = output.get("code_residual")
                if res is not None:
                    fl2 = fr * res.pow(2).mean()
                    loss = loss + fl2
                    ld["fusion_reg"] = float(fl2.item())

            # Shape disentanglement + foot physics losses
            phys_warmup = int(L.get("physics_warmup_epochs", 0))
            phys_ramp = min(1.0, (epoch + 1) / max(phys_warmup + 1, 1))

            # 1. shape disentanglement: predictor on style_emb -> beta, via GRL
            lambda_shape_dis = float(L.get("lambda_shape_disentangle", 0.0))
            if lambda_shape_dis > 0 and shape_predictor is not None and "style_emb" in output:
                style_emb_t = output["style_emb"]
                if style_emb_t is not None:
                    true_beta = beta.to(device).float()
                    from model.shape_foot_losses import shape_disentangle_loss
                    shp_loss = shape_disentangle_loss(style_emb_t, true_beta, shape_predictor, alpha=1.0)
                    loss = loss + lambda_shape_dis * phys_ramp * shp_loss
                    ld["shape_dis"] = float(shp_loss.item())

            # 2. Foot contact MSE (on dim 259-262 of 263-rep)
            lambda_foot_contact = float(L.get("lambda_foot_contact", 0.0))
            if lambda_foot_contact > 0:
                from model.shape_foot_losses import foot_contact_loss
                fc_loss = foot_contact_loss(output["motion_recon"], target, motion_length=tgt_len)
                loss = loss + lambda_foot_contact * phys_ramp * fc_loss
                ld["foot_contact"] = float(fc_loss.item())

            # 3+4+5. xyz-based foot losses (ground, contact-velocity, leg crossing)
            need_xyz = any(float(L.get(k, 0.0)) > 0 for k in
                            ["lambda_foot_ground_y", "lambda_foot_vel_contact", "lambda_leg_cross"])
            if need_xyz:
                try:
                    from model.shape_foot_losses import (foot_ground_y_loss,
                                                     foot_vel_at_contact_loss,
                                                     leg_crossing_penalty)
                    from model.nnutils.motion_utils import recover_from_ric
                    # denormalize and recover xyz on the GPU; recover_from_ric is differentiable, so the
                    # positional losses below reach the model
                    motion_unnorm = output["motion_recon"] * sc_std_t + sc_mean_t
                    xyz_all = recover_from_ric(motion_unnorm, joints_num=22)
                    if xyz_all.dim() == 3:
                        xyz_all = xyz_all.reshape(*xyz_all.shape[:2], -1, 3)
                    xyz_all = xyz_all.float()  # [B, T, 22, 3]
                    # Contact probs from dim 259-262 of normalized output
                    contact_probs = output["motion_recon"][:, :, FOOT_CONTACT_DIMS] * sc_std_t[FOOT_CONTACT_DIMS] + sc_mean_t[FOOT_CONTACT_DIMS]
                    contact_probs = contact_probs.clamp(0, 1)

                    lambda_fg = float(L.get("lambda_foot_ground_y", 0.0))
                    if lambda_fg > 0:
                        fgy = foot_ground_y_loss(xyz_all, contact_probs, motion_length=tgt_len)
                        loss = loss + lambda_fg * phys_ramp * fgy
                        ld["foot_ground_y"] = float(fgy.item())

                    lambda_fvc = float(L.get("lambda_foot_vel_contact", 0.0))
                    if lambda_fvc > 0:
                        fvc = foot_vel_at_contact_loss(xyz_all, contact_probs, motion_length=tgt_len)
                        loss = loss + lambda_fvc * phys_ramp * fvc
                        ld["foot_vel_ct"] = float(fvc.item())

                    lambda_lc = float(L.get("lambda_leg_cross", 0.0))
                    if lambda_lc > 0:
                        lcp = leg_crossing_penalty(xyz_all, motion_length=tgt_len)
                        loss = loss + lambda_lc * phys_ramp * lcp
                        ld["leg_cross"] = float(lcp.item())

                    # anti-float and bilateral trajectory losses
                    lambda_af = float(L.get("lambda_anti_float", 0.0))
                    if lambda_af > 0:
                        from model.shape_foot_losses import anti_float_loss
                        afl = anti_float_loss(xyz_all, contact_probs, motion_length=tgt_len)
                        loss = loss + lambda_af * phys_ramp * afl
                        ld["anti_float"] = float(afl.item())

                    # shape anchor loss on all rows (target_motion gives the bone lengths)
                    lambda_sa = float(L.get("lambda_shape_anchor", 0.0))
                    if lambda_sa > 0:
                        from model.shape_foot_losses import shape_anchor_loss
                        from model.nnutils.motion_utils import recover_from_ric
                        # un-normalize on GPU, keep gradient
                        tg_unnorm = target * sc_std_t + sc_mean_t                  # [B, T, 263]
                        out_unnorm = output["motion_recon"] * sc_std_t + sc_mean_t
                        # recover_from_ric is differentiable torch op
                        tg_xyz_t  = recover_from_ric(tg_unnorm, joints_num=22)     # [B, T, 22, 3]
                        out_xyz_t = recover_from_ric(out_unnorm, joints_num=22)
                        # ensure shape [B, T, 22, 3]
                        if tg_xyz_t.dim() == 3:
                            tg_xyz_t = tg_xyz_t.reshape(*tg_xyz_t.shape[:2], -1, 3)
                            out_xyz_t = out_xyz_t.reshape(*out_xyz_t.shape[:2], -1, 3)
                        sa = shape_anchor_loss(out_xyz_t, tg_xyz_t,
                                                motion_length=tgt_len,
                                                long_bone_weight=float(L.get("lambda_sa_long_weight", 2.0)))
                        loss = loss + lambda_sa * phys_ramp * sa
                        ld["sa_anchor"] = float(sa.item())

                    # shape-control loss on identity and cross-domain rows (content = Xn, target = Xr)
                    lambda_sc = float(L.get("lambda_shape_control", 0.0))
                    sc_mask = None
                    if lambda_sc > 0:
                        if is_id is not None and is_cross is not None:
                            sc_mask = is_id | is_cross
                        elif is_id is not None:
                            sc_mask = is_id
                        elif is_cross is not None:
                            sc_mask = is_cross
                    if lambda_sc > 0 and sc_mask is not None and sc_mask.any():
                        from model.shape_foot_losses import shape_control_loss
                        idx_id = sc_mask.nonzero(as_tuple=True)[0]
                        # Anchors (Xn/Xr) need no gradient; the output slice reuses
                        # the differentiable xyz_all from above.
                        with torch.no_grad():
                            cn_unnorm = content[idx_id] * sc_std_t + sc_mean_t
                            tg_unnorm = target[idx_id] * sc_std_t + sc_mean_t
                            xn_xyz_t = recover_from_ric(cn_unnorm, joints_num=22)
                            xr_xyz_t = recover_from_ric(tg_unnorm, joints_num=22)
                            if xn_xyz_t.dim() == 3:
                                xn_xyz_t = xn_xyz_t.reshape(*xn_xyz_t.shape[:2], -1, 3)
                                xr_xyz_t = xr_xyz_t.reshape(*xr_xyz_t.shape[:2], -1, 3)
                        out_xyz_t = xyz_all[idx_id]
                        sc_d = shape_control_loss(
                            out_xyz_t, xn_xyz_t, xr_xyz_t,
                            motion_length=tgt_len[idx_id],
                            lambda_bonelen=float(L.get("lambda_sc_bonelen", 1.0)),
                            lambda_dir=float(L.get("lambda_sc_dir", 1.0)),
                            lambda_mag=float(L.get("lambda_sc_mag", 0.5)),
                        )
                        loss = loss + lambda_sc * phys_ramp * sc_d["total"]
                        ld["sc_bonelen"] = float(sc_d["l_bonelen"].item())
                        ld["sc_dir"] = float(sc_d["l_dir"].item())
                        ld["sc_mag"] = float(sc_d["l_mag"].item())
                        ld["sc_cos"] = float(sc_d["cos_sim_mean"].item())

                    lambda_bt = float(L.get("lambda_bilateral_traj", 0.0))
                    if lambda_bt > 0:
                        from model.shape_foot_losses import bilateral_trajectory_loss
                        # content xyz once, on the GPU (no gradient needed)
                        with torch.no_grad():
                            content_unnorm = content * sc_std_t + sc_mean_t
                            cnt_xyz_all = recover_from_ric(content_unnorm, joints_num=22)
                            if cnt_xyz_all.dim() == 3:
                                cnt_xyz_all = cnt_xyz_all.reshape(*cnt_xyz_all.shape[:2], -1, 3)
                        btl = bilateral_trajectory_loss(xyz_all, cnt_xyz_all,
                                                          motion_length=tgt_len)
                        loss = loss + lambda_bt * phys_ramp * btl
                        ld["bilateral_traj"] = float(btl.item())
                except Exception as _exc:
                    # xyz conversion may fail in some edge cases; skip silently after first log
                    if step == 0 and epoch == start_epoch:
                        print(f"  [shape] WARN xyz-based foot losses disabled: {_exc}")

            # joint-aware style guidance
            if joint_gate is not None and "style_emb" in output and output["style_emb"] is not None:
                # neutral-contrast joint gate
                from model.joint_aware_style import neutral_contrast_gate_loss
                jg_gate = joint_gate(style, output["style_emb"])          # [B, 22]
                sidx_full = batch["style_idx"].to(device)
                # Only include samples with a valid style_idx (Style100 rows)
                valid_ja = sidx_full >= 0
                if valid_ja.any() and ja_neutral_idx >= 0:
                    sub_gate = jg_gate[valid_ja]
                    sub_sidx = sidx_full[valid_ja]
                    ja_losses = neutral_contrast_gate_loss(
                        sub_gate, sub_sidx, ja_neutral_idx,
                        sparsity_target=ja_sparsity_target, margin=0.2)
                    ja_total = (lambda_ja_neutral * ja_losses["neutral_var"]
                                + lambda_ja_separation * ja_losses["separation"]
                                + lambda_ja_sparsity * ja_losses["sparsity"])
                    loss = loss + ja_total
                    ld["ja_neutral"] = float(ja_losses["neutral_var"].item())
                    ld["ja_sep"] = float(ja_losses["separation"].item())
                    ld["ja_spars"] = float(ja_losses["sparsity"].item())
                if lambda_ja_guidance > 0 and joint_guide is not None:
                    # guidance: mask the residual channels of fused_codes by the joint-importance projection,
                    # so that only style-critical joints move
                    chan_gate = joint_guide(jg_gate)                # [B, code_dim]
                    res = output.get("code_residual")
                    if res is not None:
                        # [B, code_dim, T']  * [B, code_dim, 1]
                        anti_leak = (res * (1.0 - chan_gate).unsqueeze(-1)).pow(2).mean()
                        loss = loss + lambda_ja_guidance * anti_leak
                        ld["ja_guide"] = float(anti_leak.item())
            if joint_bottleneck is not None and "style_emb" in output and output["style_emb"] is not None:
                # info-bottleneck mask on style_emb: attenuates the residual per joint
                mask = joint_bottleneck(output["style_emb"])        # [B, 22]
                if joint_guide is not None and lambda_ja_guidance > 0:
                    chan_gate = joint_guide(mask)
                    res = output.get("code_residual")
                    if res is not None:
                        anti_leak = (res * (1.0 - chan_gate).unsqueeze(-1)).pow(2).mean()
                        loss = loss + lambda_ja_guidance * anti_leak
                        ld["ja_guide"] = float(anti_leak.item())
                spars = joint_bottleneck.sparsity_loss(mask)
                loss = loss + lambda_ja_sparsity * spars
                ld["ja_bott_spars"] = float(spars.item())

            # Adversarial
            if lambda_adv > 0:
                g_adv = wgan_generator_loss(critic, output["motion_recon"])
                loss = loss + lambda_adv * g_adv
                ld["g_adv"] = float(g_adv.item())

            # cycle consistency: f(stylized output, zero style) ~ content; identity_mask forces
            # the zero-style path
            lambda_cycle = float(L.get("lambda_cycle", 0.0))
            cycle_warmup = int(cfg.get("cycle_warmup", 5))
            if lambda_cycle > 0 and epoch >= cycle_warmup and "motion_recon" in output:
                styled = output["motion_recon"].detach()   # detach: cycle-loss doesn't bleed into primary forward
                Bc = styled.shape[0]
                all_id = torch.ones(Bc, dtype=torch.bool, device=device)
                cycle_out = model(content_motion=styled, beta=beta,
                                    style_motion=style, identity_mask=all_id)
                cycle_recon = cycle_out["motion_recon"]
                T_cy = min(cycle_recon.shape[1], content.shape[1])
                cy_diff = (cycle_recon[:, :T_cy] - content[:, :T_cy]).abs()
                arange_cy = torch.arange(T_cy, device=device).unsqueeze(0).expand(Bc, -1)
                mask_cy = (arange_cy < tgt_len.clamp(max=T_cy).unsqueeze(1)).float().unsqueeze(-1)
                cycle_loss = (cy_diff * mask_cy).sum() / (mask_cy.sum() * 263 + 1e-8)
                loss = loss + lambda_cycle * cycle_loss
                ld["cycle"] = float(cycle_loss.item())

            # diversity regularizer: hinge-match the real batch's pairwise distance
            lambda_div = float(L.get("lambda_diversity", 0.0))
            div_warmup = int(cfg.get("diversity_warmup", 5))
            if lambda_div > 0 and epoch >= div_warmup and "motion_recon" in output:
                gen_sc_d = output["motion_recon"]
                gen_raw_d = gen_sc_d * sc_std_t + sc_mean_t
                gen_hml_d = (gen_raw_d - hml_mean_t) / hml_std_t
                T_d = gen_hml_d.shape[1]
                Tp_d = ((T_d + 3) // 4) * 4
                if Tp_d != T_d:
                    pad_d = torch.zeros(gen_hml_d.shape[0], Tp_d - T_d, 263,
                                          device=device, dtype=gen_hml_d.dtype)
                    gen_hml_d = torch.cat([gen_hml_d, pad_d], dim=1)
                gen_lens_d = tgt_len.clamp(max=Tp_d)
                gen_feats_d = t2m_encode_diff(gen_hml_d, gen_lens_d)
                Bd = gen_feats_d.shape[0]
                if Bd >= 2:
                    pw_g = torch.cdist(gen_feats_d, gen_feats_d, p=2)
                    mask_g = torch.triu(torch.ones_like(pw_g), diagonal=1).bool()
                    mean_g = pw_g[mask_g].mean()
                    # Target reference: use real identity samples' diversity
                    # (only if we have real feats this step; else skip)
                    with torch.no_grad():
                        target_dist = torch.tensor(9.0, device=device)  # ~GT level
                        if is_id is not None and is_id.any():
                            real_sc_d = target[is_id]
                            real_raw_d = real_sc_d * sc_std_t + sc_mean_t
                            real_hml_d = (real_raw_d - hml_mean_t) / hml_std_t
                            if real_hml_d.shape[1] < Tp_d:
                                pr = torch.zeros(real_hml_d.shape[0], Tp_d - real_hml_d.shape[1], 263,
                                                  device=device, dtype=real_hml_d.dtype)
                                real_hml_d = torch.cat([real_hml_d, pr], dim=1)
                            else:
                                real_hml_d = real_hml_d[:, :Tp_d]
                            real_lens_d = tgt_len[is_id].clamp(max=Tp_d)
                            if real_hml_d.shape[0] >= 2:
                                rf = t2m_encode_diff(real_hml_d, real_lens_d)
                                pw_r = torch.cdist(rf, rf, p=2)
                                mask_r = torch.triu(torch.ones_like(pw_r), diagonal=1).bool()
                                target_dist = pw_r[mask_r].mean()
                    div_loss = F.relu(target_dist - mean_g)   # only penalize when gen < real
                    loss = loss + lambda_div * div_loss
                    ld["diversity"] = float(div_loss.item())
                    ld["gen_div"] = float(mean_g.item())
                    ld["tgt_div"] = float(target_dist.item())

            # Moment matching
            if lambda_moment > 0 and is_id is not None and is_id.any():
                gen_sc = output["motion_recon"]
                gen_raw = gen_sc * sc_std_t + sc_mean_t
                gen_hml = (gen_raw - hml_mean_t) / hml_std_t
                real_sc = target[is_id]
                real_raw = real_sc * sc_std_t + sc_mean_t
                real_hml = (real_raw - hml_mean_t) / hml_std_t
                T = gen_hml.shape[1]
                T_pad = ((T + 3) // 4) * 4
                if T_pad != T:
                    pad_g = torch.zeros(gen_hml.shape[0], T_pad - T, 263,
                                        device=device, dtype=gen_hml.dtype)
                    gen_hml = torch.cat([gen_hml, pad_g], dim=1)
                    if real_hml.shape[1] < T_pad:
                        pad_r = torch.zeros(real_hml.shape[0], T_pad - real_hml.shape[1], 263,
                                            device=device, dtype=real_hml.dtype)
                        real_hml = torch.cat([real_hml, pad_r], dim=1)
                    else:
                        real_hml = real_hml[:, :T_pad]
                gen_lens = tgt_len.clamp(max=T_pad)
                real_lens = tgt_len[is_id].clamp(max=T_pad)
                gen_feats = t2m_encode_diff(gen_hml, gen_lens)
                with torch.no_grad():
                    real_feats = t2m_encode_diff(real_hml, real_lens)
                mm = moment_loss(gen_feats, real_feats)
                loss = loss + lambda_moment * mm
                ld["moment"] = float(mm.item())
                # multi-kernel MMD over the same features
                lambda_mmd = float(L.get("lambda_mmd", 0.0))
                mmd_warmup = int(cfg.get("mmd_warmup", 5))
                if lambda_mmd > 0 and epoch >= mmd_warmup:
                    mmd_val = mmd_loss_fn(gen_feats, real_feats.detach())
                    loss = loss + lambda_mmd * mmd_val
                    ld["mmd"] = float(mmd_val.item())

            # dual-path HumanML3D reconstruction (optional): style_emb = 0 on a pure HumanML3D batch,
            # the output should match the VAE baseline
            if hml_pure_iter is not None:
                try:
                    batch_hml = next(hml_pure_iter)
                except StopIteration:
                    hml_pure_iter = iter(hml_pure_loader)
                    batch_hml = next(hml_pure_iter)
                update_jtc_extras(batch_hml)
                content_h = batch_hml["content_motion"].to(device, non_blocking=True)
                beta_h = batch_hml["beta"].to(device, non_blocking=True)
                target_h = batch_hml["target_motion"].to(device, non_blocking=True)
                if dual_hml_mode == "zero":
                    # style_emb = 0 (cfg_uncond)
                    out_h = model(content_motion=content_h, beta=beta_h,
                                  style_motion=None, cfg_uncond=True)
                else:  # "same_id"
                    # style_motion = content: real style_emb, the output should still be the content
                    out_h = model(content_motion=content_h, beta=beta_h,
                                  style_motion=content_h, cfg_uncond=False)
                hml_recon = F.l1_loss(out_h["motion_recon"], target_h)
                loss = loss + lambda_dual_hml * hml_recon
                ld["hml_dual"] = float(hml_recon.item())

            ld["total"] = float(loss.item())
            if not torch.isfinite(loss):   # NaN-guard: skip the update instead of poisoning the weights
                _nan_skips = globals().get("_NAN_SKIPS", 0) + 1; globals()["_NAN_SKIPS"] = _nan_skips
                print(f"[NaN-guard] non-finite loss, skipping step (total skipped {_nan_skips})", flush=True)
                g_optim.zero_grad(set_to_none=True)
                if "_GOOD" in globals():
                    import copy as _copy
                    model.load_state_dict(globals()["_GOOD"]["m"]); critic.load_state_dict(globals()["_GOOD"]["c"])
                    g_optim.load_state_dict(_copy.deepcopy(globals()["_GOOD"]["go"])); d_optim.load_state_dict(_copy.deepcopy(globals()["_GOOD"]["do"]))
                    print("[NaN-guard] restored last good model+critic+optimizer snapshot", flush=True)
                if _nan_skips > 300: raise RuntimeError("too many non-finite losses")
                continue
            _lv = float(loss.item()); _ema = globals().get("_LOSS_EMA")
            if _ema is not None and globals().get("_GSTEP", 0) > 30 and _lv > 3.0 * _ema and _lv > _ema + 15:
                _loc = locals(); _sk = globals().get("_SPIKE_SKIPS", 0) + 1; globals()["_SPIKE_SKIPS"] = _sk
                _tl = _loc.get("tgt_len"); _ic = _loc.get("is_cross")
                try:
                    _Tq = min(output["motion_recon"].shape[1], target.shape[1])
                    _pr = (output["motion_recon"][:, :_Tq] - target[:, :_Tq]).abs().mean((1, 2)).tolist()
                    _mids = batch.get("mid", list(range(len(_pr))))
                    print("[spike-guard] per-sample recon: " + ", ".join(f"{m}:{r:.1f}" for m, r in zip(_mids, _pr)), flush=True)
                except Exception as _e:
                    print(f"[spike-guard] per-sample recon failed: {_e}", flush=True)
                print(f"[spike-guard] loss {_lv:.1f} vs ema {_ema:.1f}: skipping step (total {_sk}); "
                      f"terms={{ {', '.join(f'{k}={v:.2f}' for k, v in ld.items() if isinstance(v, float) and v > 1.0)} }}; "
                      f"len={None if _tl is None else _tl.tolist()}; cross={None if _ic is None else int(_ic.sum())}", flush=True)
                g_optim.zero_grad(set_to_none=True)
                continue
            globals()["_LOSS_EMA"] = _lv if _ema is None else 0.98 * _ema + 0.02 * _lv
            loss.backward()
            if t_cfg.get("grad_clip", 0) > 0:
                _gn = nn.utils.clip_grad_norm_(trainable, t_cfg["grad_clip"])
                ld["gnorm"] = float(_gn)   # pre-clip gradient norm: is the clip throttling training?
            g_optim.step()
            globals()["_GSTEP"] = globals().get("_GSTEP", 0) + 1
            if globals()["_GSTEP"] % 50 == 0:   # NaN-guard snapshot (weights are finite here: loss was finite before backward)
                import copy as _copy
                globals()["_GOOD"] = {"m": {k: v.detach().clone() for k, v in model.state_dict().items()},
                                      "c": {k: v.detach().clone() for k, v in critic.state_dict().items()},
                                      "go": _copy.deepcopy(g_optim.state_dict()), "do": _copy.deepcopy(d_optim.state_dict())}
            scheduler.step()
            ema.update(model)

            global_step += 1

            # periodic t-SNE snapshot (EMA weights for stability)
            if tsne_snap is not None and tsne_every_steps > 0 and \
                    (global_step == 1 or global_step % tsne_every_steps == 0):
                try:
                    backup_tsne = ema.apply(model)
                    tsne_snap.snapshot(
                        model, epoch=epoch + 1, step=global_step,
                        extra_tags={"contrastive":
                                    f"{L.get('lambda_contrastive', 0):g}"})
                    ema.restore(model, backup_tsne)
                except Exception as _tsne_err:
                    print(f"  [t-SNE] WARN: {_tsne_err}")

            if lambda_adv > 0 and n_critic > 0:
                ld["d_loss"] = d_loss_sum / n_critic
                ld["d_gp"] = gp_sum / n_critic

            for k, v in ld.items(): loss_acc[k] += v
            n_batches += 1
            if args.max_steps and n_batches >= args.max_steps:
                print(f"[smoke] stopping after {n_batches} steps; loss keys: {sorted(loss_acc)}"); break

            if n_batches % cfg.get("log_interval", 100) == 0:
                a = {k: v / n_batches for k, v in loss_acc.items()}
                lr = g_optim.param_groups[0]["lr"]
                extra = ""
                for k in ["style_clf", "root_soft", "foot_soft", "g_adv", "moment",
                           "d_loss", "cycle", "mmd", "diversity", "gen_div", "tgt_div",
                           "code_pres", "motion_anchor", "teacher", "teacher_dyn", "td_rms", "td_spec", "td_rng", "tempo", "smooth", "native_cons", "floor", "traj_xz", "root_rot", "foot_vel_ct", "foot_ground_y", "skate_geo", "duty", "cc_vel", "cc_h", "gnorm",
                           "sc_bonelen", "sc_dir", "sc_cos", "sc_mag",
                           "traj_total", "traj_yaw", "traj_xz", "traj_h",
                           "sa_anchor"]:
                    if k in a: extra += f" {k}={a[k]:.3f}"
                jg = ""
                if hasattr(model, "style_head"):
                    try:
                        jg = f" jg_norm={float(model.style_head.joint_gate):.4f}"
                    except Exception:
                        jg = ""
                print(f"  [E{epoch+1}][{n_batches}/{len(train_loader)}] "
                      f"loss={a['total']:.3f} recon={a['recon']:.3f}{extra}{jg} "
                      f"lr={lr:.2e} lambda_adv={lambda_adv:.3f}")

        avg_loss = {k: v / max(n_batches, 1) for k, v in loss_acc.items()}

        # Val
        val_metrics = {}
        if (epoch + 1) % metrics_interval == 0:
            backup = ema.apply(model)
            model.eval()
            try:
                val_metrics = evaluate_val(model, val_loader, device,
                                        sc_mean, sc_std, hml_mean, hml_std,
                                        t2m_eval, feat_ext, s_clf,
                                        style100_dataset=style100,
                                        root_dims=root_dims)
            except Exception as _ve:   # guard: FID's sqrtm raises on NaN features
                print(f"[val-guard] evaluate_val failed: {_ve}", flush=True); val_metrics = {}
            ema.restore(model, backup)
            fid = val_metrics.get("FID", float("inf"))
            # track best.pt only after the warmup epochs (default 10): early, near-identity epochs
            # have a misleadingly low FID
            best_save_after = int(cfg.get("best_save_after_epoch", 10))
            if epoch >= best_save_after and fid < best_metric:
                best_metric = fid
                save_ckpt(model, g_optim, critic, d_optim, epoch,
                          str(output_dir / "best.pt"), cfg, ema)

        t1 = time.time()
        ms = ""
        if val_metrics:
            ms = (f" | SRA={val_metrics.get('SRA', 0):.1f}% "
                  f"FID={val_metrics.get('FID', 0):.2f} "
                  f"Div={val_metrics.get('Diversity', 0):.2f}")
        print(f"Epoch {epoch+1}/{t_cfg['epochs']} ({t1-t0:.0f}s) "
              f"loss={avg_loss.get('total',0):.3f} recon={avg_loss.get('recon',0):.3f}"
              f" root_soft={avg_loss.get('root_soft',0):.3f}"
              f" d_loss={avg_loss.get('d_loss',0):.3f}{ms}")
        log_rows.append({"epoch": epoch + 1, **avg_loss, **val_metrics})
        save_ckpt(model, g_optim, critic, d_optim, epoch,
                  str(output_dir / "latest.pt"), cfg, ema)
        # periodic epoch checkpoints
        save_interval = int(cfg.get("save_interval", 20))
        if save_interval > 0 and (epoch + 1) % save_interval == 0:
            save_ckpt(model, g_optim, critic, d_optim, epoch,
                      str(output_dir / f"epoch_{epoch+1}.pt"), cfg, ema)
        if args.max_steps:   # smoke test: one shortened epoch, saved
            break

    if log_rows:
        keys = sorted(set(k for r in log_rows for k in r.keys()))
        with open(output_dir / "training_log.csv", "w", newline="") as f:
            w = csv.DictWriter(f, keys, extrasaction="ignore"); w.writeheader()
            for r in log_rows: w.writerow(r)
    print(f"\nTraining complete. Best FID: {best_metric:.4f}")


@torch.no_grad()
def evaluate_val(model, val_loader, device, sc_mean, sc_std, hml_mean, hml_std,
                 t2m_eval, feat_ext, s_clf, style100_dataset=None, root_dims=(0, 1, 2, 3)):
    from auxiliary import compute_fid, compute_diversity, compute_foot_skating_ratio, motion263_to_xyz
    gen_hml, gt_hml, lengths = [], [], []
    for batch in val_loader:
        update_jtc_extras(batch)
        content = batch["content_motion"].to(device)
        style = batch["style_motion"].to(device)
        beta = batch["beta"].to(device)
        tgt_len = batch["target_length"]
        output = model(content_motion=content, beta=beta, style_motion=style)
        recon = output["motion_recon"].cpu().numpy()
        for i in range(len(recon)):
            L = tgt_len[i].item()
            gen_raw = recon[i, :L] * sc_std + sc_mean
            tgt_raw = batch["target_motion"][i, :L].numpy() * sc_std + sc_mean
            gen_hml.append((gen_raw - hml_mean) / hml_std)
            gt_hml.append((tgt_raw - hml_mean) / hml_std)
            lengths.append(L)

    metrics = {}
    N = len(gen_hml)
    if t2m_eval is not None and N > 20:
        def enc(motions):
            mx = ((max(m.shape[0] for m in motions)+3)//4)*4
            embs = []
            for i in range(0, len(motions), 64):
                pds, ls = [], []
                for j in range(i, min(i+64, len(motions))):
                    m = motions[j]; l = min(m.shape[0], mx)
                    p = np.zeros((mx, 263), dtype=np.float32); p[:l] = m[:l]
                    pds.append(p); ls.append(l)
                b = torch.from_numpy(np.stack(pds)).float().to(device)
                bl = torch.tensor(ls).long().to(device)
                embs.append(t2m_eval.encode_motion(b, bl).cpu().numpy())
            return np.concatenate(embs, 0)
        gt_e = enc(gt_hml); gen_e = enc(gen_hml)
        metrics["FID"] = compute_fid(np.mean(gt_e,0), np.cov(gt_e,rowvar=False),
                                      np.mean(gen_e,0), np.cov(gen_e,rowvar=False))
        metrics["Diversity"] = compute_diversity(gen_e, 20)
        metrics["Diversity_gt"] = compute_diversity(gt_e, 20)
        print(f"  [Val-HML3D] FID={metrics['FID']:.2f} Div={metrics['Diversity']:.2f} "
              f"(GT:{metrics['Diversity_gt']:.2f})")

    skating = []
    for i in range(min(100, N)):
        try:
            raw = gen_hml[i] * hml_std + hml_mean
            xyz = motion263_to_xyz(raw, 22)
            skating.append(compute_foot_skating_ratio(xyz))
        except: pass
    if skating: metrics["Foot_Skating"] = float(np.mean(skating))

    # SRA with root masking (for body-only classifier)
    if style100_dataset is not None and feat_ext is not None and s_clf is not None:
        import random
        rng = random.Random(42)
        indices = list(range(len(style100_dataset))); rng.shuffle(indices)
        beta_t = torch.from_numpy(style100_dataset.shared_beta).float().unsqueeze(0).to(device)
        sra_correct = sra_total = 0
        for idx in indices[:200]:
            sample = style100_dataset[idx]
            ct = sample["content_motion"].unsqueeze(0).to(device)
            st = sample["style_motion"].unsqueeze(0).to(device)
            update_jtc_extras({
                "content_text": [sample.get("content_text", "a person performs an action")],
                "content_length": sample.get("content_length",
                                              torch.tensor(ct.shape[1])).unsqueeze(0),
            })
            out = model(content_motion=ct, beta=beta_t, style_motion=st)
            motion_for_clf = out["motion_recon"].clone()
            for d in root_dims: motion_for_clf[:, :, d] = 0.0
            feats = feat_ext(motion_for_clf)
            pred = s_clf(feats).argmax(1).item()
            if pred == sample["style_idx"].item(): sra_correct += 1
            sra_total += 1
        metrics["SRA"] = 100.0 * sra_correct / max(sra_total, 1)
        print(f"  [Val-100Style] SRA={metrics['SRA']:.1f}%")

    return metrics


def _mid_collate(items):
    """default_collate that tolerates items without a "mid" key (Style100Dataset)."""
    from torch.utils.data import default_collate
    return default_collate([{**x, "mid": str(x.get("mid", "?"))} for x in items])


def save_ckpt(model, g_optim, critic, d_optim, epoch, path, cfg, ema=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ckpt = {
        "epoch": epoch, "config": cfg,
        "style_injector_state_dict": model.style_injector.state_dict(),
        "linear_probe_state_dict": model.linear_probe.state_dict(),
    }
    # some style heads have no temporal_gate; skip it when absent
    if hasattr(model, "temporal_gate") and model.temporal_gate is not None:
        ckpt["temporal_gate_state_dict"] = model.temporal_gate.state_dict()
    if hasattr(model, "style_encoder"):
        ckpt["style_encoder_state_dict"] = model.style_encoder.state_dict()
    # JTC extras (only present on mpsm_jtc)
    if hasattr(model, "joint_style_fuser"):
        ckpt["joint_style_fuser_state_dict"] = model.joint_style_fuser.state_dict()
    if hasattr(model, "style_head"):
        ckpt["style_head_state_dict"] = model.style_head.state_dict()
    ckpt.update({
        "critic_state_dict": critic.state_dict(),
        "g_optimizer_state_dict": g_optim.state_dict(),
        "d_optimizer_state_dict": d_optim.state_dict(),
    })
    if ema: ckpt["ema_shadow"] = ema.shadow
    torch.save(ckpt, path)
    print(f"  [Saved] {path}")


if __name__ == "__main__":
    main()
