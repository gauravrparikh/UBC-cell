from pathlib import Path

import torch

from protopnet.datasets import training_dataloaders
from rashomon_sets.protorset_factory import ProtoRSetFactory

DEFAULT_RSET_ARGS = {
    "rashomon_bound_multiplier": 1.05,
    "num_models": 5,  # Right now, I'm randomly sampling models from the ellipsoid during fit
    "reg": "l1",
    "lam": 0.0001,
    "compute_hessian_batched": False,  # Not batching for memory's sake
    "max_iter": 10,  # The max number of iterations allowed when fitting the LR,
    "directly_compute_hessian": True,
    "device": torch.device("cuda"),
}

if __name__ == "__main__":
    print("start", flush=True)
    torch.manual_seed(42)

    SAMPLE_N_PROTOs = 4

    trained_model = torch.load(
        "/usr/xtmp/jcd97/proto-rset/wandb/test/artifacts/ps9w130l/18_project_0.8552.pth",
        map_location="cuda",
    )
    print(trained_model)
    print("model loaded", flush=True)

    batch_sizes = {"train": 20, "project": 20, "val": 20}
    split_dataloaders = training_dataloaders("cub200", batch_sizes=batch_sizes)

    train_loader = split_dataloaders.project_loader
    val_loader = split_dataloaders.val_loader
    print("dataloader", flush=True)

    rset_factory = ProtoRSetFactory(
        split_dataloaders=split_dataloaders,
        initial_protopnet=trained_model,
        initial_protopnet_path=Path(
            "/usr/xtmp/jcd97/proto-rset/wandb/test/artifacts/ps9w130l/18_project_0.8552.pth"
        ),
        rashomon_set_args=DEFAULT_RSET_ARGS,
        device="cuda",
        run_complete_vislization_in_init=False,
    )

    with torch.no_grad():
        rset_factory.initial_protopnet.project(train_loader)

    old_proto_info_dict_len = len(
        rset_factory.initial_protopnet.prototype_layer.prototype_info_dict
    )
    old_num_protos = rset_factory.initial_protopnet.prototype_layer.num_prototypes
    old_cc_shape = rset_factory.initial_protopnet.prototype_prediction_head.class_connection_layer.weight.data.shape[
        1
    ]
    old_identity_shape = rset_factory.initial_protopnet.prototype_prediction_head.prototype_class_identity.shape[
        0
    ]

    rset_factory.sample_additional_prototypes(
        target_number_of_samples=SAMPLE_N_PROTOs,
        prototype_sampling="uniform_random",
        dataloader=train_loader,
    )

    assert (
        len(rset_factory.initial_protopnet.prototype_layer.prototype_info_dict)
        == SAMPLE_N_PROTOs + old_proto_info_dict_len
    )

    assert (
        rset_factory.initial_protopnet.prototype_layer.num_prototypes
        == SAMPLE_N_PROTOs + old_num_protos
    )

    assert (
        rset_factory.initial_protopnet.prototype_prediction_head.class_connection_layer.weight.data.shape[
            1
        ]
        == SAMPLE_N_PROTOs + old_cc_shape
    )
    assert (
        rset_factory.initial_protopnet.prototype_prediction_head.prototype_class_identity.shape[
            0
        ]
        == SAMPLE_N_PROTOs + old_identity_shape
    )
    assert (
        rset_factory.initial_protopnet.prototype_layer.prototype_tensors.shape[0]
        == SAMPLE_N_PROTOs + old_num_protos
    )
