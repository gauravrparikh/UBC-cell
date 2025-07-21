import torch
import time

from protopnet.datasets import training_dataloaders
from rashomon_sets.protorset_factory import ProtoRSetFactory, DEFAULT_RSET_ARGS
from rashomon_sets.linear_rashomon_set import TorchLogisticRegressionPositiveOnly

if __name__ == "__main__":
    torch.manual_seed(0)

    # First, produce our distances dataset given a trained model =================
    # trained_model = torch.load("/usr/xtmp/ppnxt/neurips2024/live/artifacts/1ehrw1zd/95_last_only_0.8906.pth")
    trained_model = torch.load(
        "/usr/xtmp/jcd97/proto-rset/wandb/test/artifacts/ps9w130l/18_project_0.8552.pth"
    )
    batch_sizes = {"train": 20, "project": 20, "val": 20}
    split_dataloaders = training_dataloaders("cub200", batch_sizes=batch_sizes)

    rset_args = DEFAULT_RSET_ARGS
    rset_args["max_iter"] = 5_000
    rset_args["directly_compute_hessian"] = True
    rset_args["rashomon_bound_multiplier"] = 1.01
    rset_args["lam"] = 0.0001
    rset_args["reg"] = "l2"
    rset_args["lr_for_opt"] = 1
    vars_to_drop = [3 * i for i in range(100)]

    print("======== Fitting with eps 1.3 =========")
    rset_args["rashomon_bound_multiplier"] = 1.3

    factory = ProtoRSetFactory(
        split_dataloaders,
        trained_model,
        rashomon_set_args=rset_args,
        correct_class_connections_only=True,
    )

    rset_preds_proba = factory.rset.rset_predict_proba(
        factory.X_val.to(factory.rset.device)
    ).cpu()
    rset_preds = factory.rset.rset_predict(factory.X_val.to(factory.rset.device)).cpu()
    opt_preds = factory.rset.optimal_model.predict_proba(
        factory.X_val.to(factory.rset.device)
    ).cpu()
    labels = factory.y_val.long()
    loss = torch.nn.NLLLoss()
    print("Optimal loss value: ", [loss(torch.log(opt_preds), labels).item()])
    print(
        "Loss values: ",
        [
            loss(torch.log(rset_preds_proba[:, :, i]), labels).item()
            for i in range(rset_preds_proba.shape[-1])
        ],
    )
    print(
        "Accuracy values: ",
        [
            (1.0 * (rset_preds[:, i] == labels)).mean()
            for i in range(rset_preds.shape[1])
        ],
    )

    new_weights = factory.rset.optimal_model.get_params().clone()
    new_weights[vars_to_drop] = 0
    opt_model_filtered = TorchLogisticRegressionPositiveOnly(
        factory.rset.optimal_model.prototype_class_identity, trained_weights=new_weights
    )
    print(opt_model_filtered.linear_weights)
    mod_opt_preds = opt_model_filtered.predict_proba(
        factory.X_val.to(factory.rset.device)
    ).cpu()
    mod_opt_preds_hard = opt_model_filtered.predict(
        factory.X_val.to(factory.rset.device)
    ).cpu()
    print(
        "Loss value with naive filtering: ",
        [loss(torch.log(mod_opt_preds), labels).item()],
    )
    print(
        "Optimal accuracy value with naive filtering: ",
        (1.0 * (mod_opt_preds_hard == labels)).mean(),
    )

    start = time.time()
    for var in vars_to_drop:
        factory.require_to_avoid_prototype(var)

    print(f"Time to compute intersection: {time.time() - start}")
    print("======== After PRUNING =========")
    rset_preds_proba = factory.rset.rset_predict_proba(
        factory.X_val.to(factory.rset.device)
    ).cpu()
    rset_preds = factory.rset.rset_predict(factory.X_val.to(factory.rset.device)).cpu()
    new_optimal_model = TorchLogisticRegressionPositiveOnly(
        factory.rset.prototype_class_identity, trained_weights=factory.rset.center
    )
    opt_preds = new_optimal_model.predict(factory.X_val.to(factory.rset.device)).cpu()
    opt_preds_proba = new_optimal_model.predict_proba(
        factory.X_val.to(factory.rset.device)
    ).cpu()
    print(factory.rset.center[vars_to_drop])
    print([i.linear_weights[vars_to_drop] for i in factory.rset.sampled_models])
    print("Optimal loss value: ", [loss(torch.log(opt_preds_proba), labels).item()])
    print("Optimal accuracy: ", [(1.0 * (opt_preds == labels)).mean()])
    print(
        "Loss values: ",
        [
            loss(torch.log(rset_preds_proba[:, :, i]), labels).item()
            for i in range(rset_preds_proba.shape[-1])
        ],
    )
    print(
        "Accuracy values: ",
        [
            (1.0 * (rset_preds[:, i] == labels)).mean()
            for i in range(rset_preds.shape[1])
        ],
    )
