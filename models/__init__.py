from models.anklenet import AnkleNet
import models.trainer
import models.metrics


def create_model(
    anklenet_args=None,
    img_size=None,
    backbone_name=None,
    num_slices=32,
    num_classes=3,
    num_embeddings=512,
    dropout_rate=0.0,
):
    model = AnkleNet(
        backbone_name=backbone_name,
        image_size=img_size,
        num_classes=num_classes,
        num_slices=num_slices,
        layer_num=anklenet_args["layer_num"],
        depth=anklenet_args["depth"],
        plane_dim=num_embeddings,
        plane_patch_size=anklenet_args["plane_patch_size"],
        plane_enc_depth=2,
        plane_enc_heads=8,
        plane_enc_dim_head=8,
        plane_enc_mlp_dim=2048,
        cross_attn_depth=2,
        cross_attn_dim_head=8,
        droput=dropout_rate,
        emb_dropout=0.1,
        )

    return model
