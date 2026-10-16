# Checkpoints

`shapemymove_savae.pt` goes here: the shape-conditioned FSQ-VAE that the style branch is trained on, kept frozen.
ShapeMyMove's released checkpoint has this architecture and loads with the configs as they are. Pretrained MorphoStyle
weights will follow (see the release plan in the top-level README). Training writes its own checkpoints to
`outputs/train/<name>/`.
