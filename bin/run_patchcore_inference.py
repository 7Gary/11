import contextlib
import gc
import logging
import os
import click
import numpy as np
import torch

import patchcore.common
import patchcore.metrics
import patchcore.patchcore
import patchcore.utils

LOGGER = logging.getLogger(__name__)

_DATASETS = {
    "mvtec": ["patchcore.datasets.mvtec", "MVTecDataset"],
    "custom": ["patchcore.datasets.custom", "CustomDataset"],
}

@click.group(chain=True)
@click.argument("results_path", type=str)
@click.option("--gpu", type=int, default=[0], multiple=True, show_default=True)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("--save_segmentation_images", is_flag=True)
@click.option("--save_faf_visualizations", is_flag=True)
@click.option("--save_dica_visualizations", is_flag=True)
def main(**kwargs):
    pass

@main.result_callback()
def run(
    methods,
    results_path,
    gpu,
    seed,
    save_segmentation_images,
    save_faf_visualizations,
    save_dica_visualizations,
):
    methods = {key: item for (key, item) in methods}

    run_save_path = patchcore.utils.create_storage_folder(
        results_path, "Inference", "inference", mode="iterate"
    )

    device = patchcore.utils.set_torch_device(gpu)
    device_context = (
        torch.cuda.device("cuda:{}".format(device.index))
        if "cuda" in device.type.lower()
        else contextlib.suppress()
    )

    result_collect = []

    dataloader_iter, n_dataloaders = methods["get_dataloaders_iter"]
    dataloader_iter = dataloader_iter(seed)

    patchcore_iter, n_patchcores = methods["get_patchcore_iter"]
    patchcore_iter = patchcore_iter(device)

    if not (n_dataloaders == n_patchcores or n_patchcores == 1):
        raise ValueError("Number of PatchCore instances must be 1 or equal to number of datasets!")

    for dataloader_count, dataloaders in enumerate(dataloader_iter):
        dataset_name = getattr(dataloaders["testing"].dataset, "classnames_to_use", [None])[0] or "unknown"

        LOGGER.info(
            "Evaluating dataset [{}] ({}/{})...".format(
                dataset_name,
                dataloader_count + 1,
                n_dataloaders,
            )
        )

        patchcore.utils.fix_seeds(seed, device)

        with device_context:
            torch.cuda.empty_cache()

            if dataloader_count < n_patchcores:
                PatchCore_list = next(patchcore_iter)

            aggregator = {
                "scores": [],
                "segmentations": [],
                "faf_maps": [],
                "dica_maps": [],
                "image_names": None,
                "image_paths": None,
            }

            labels_gt = None
            masks_gt = None

            for i, PatchCore in enumerate(PatchCore_list):
                torch.cuda.empty_cache()
                LOGGER.info("Running inference with model ({}/{})".format(i + 1, len(PatchCore_list)))

                outputs = PatchCore.predict(dataloaders["testing"])

                if len(outputs) == 8:
                    scores, segmentations, labels_pred, masks_pred, image_names, image_paths, faf_maps, dica_maps = outputs
                elif len(outputs) == 7:
                    scores, segmentations, labels_pred, masks_pred, image_names, image_paths, faf_maps = outputs
                    dica_maps = None
                elif len(outputs) == 6:
                    scores, segmentations, labels_pred, masks_pred, image_names, image_paths = outputs
                    faf_maps, dica_maps = None, None
                else:
                    raise ValueError(f"Unexpected predict output length: {len(outputs)}")

                aggregator["scores"].append(scores)
                aggregator["segmentations"].append(segmentations)
                if faf_maps is not None:
                    aggregator["faf_maps"].append(faf_maps)
                if dica_maps is not None:
                    aggregator["dica_maps"].append(dica_maps)

                if i == 0:
                    aggregator["image_names"] = image_names
                    aggregator["image_paths"] = image_paths
                    labels_gt = labels_pred
                    masks_gt = masks_pred

            # === 新增：模仿原训练脚本的 per-model min-max normalize ===
            scores_list = np.array(aggregator["scores"])  # shape: (n_models, n_tiles)
            segmentations_list = np.array(aggregator["segmentations"])  # shape: (n_models, n_tiles, H, W) 或类似

            # Per-model normalize scores
            min_scores = scores_list.min(axis=-1, keepdims=True)
            max_scores = scores_list.max(axis=-1, keepdims=True)
            scores_list = (scores_list - min_scores) / (max_scores - min_scores + 1e-8)

            # Per-model normalize segmentations (reshape to flat then normalize)
            seg_flat = segmentations_list.reshape(len(segmentations_list), len(segmentations_list[0]), -1)
            min_seg = seg_flat.min(axis=-1, keepdims=True)
            max_seg = seg_flat.max(axis=-1, keepdims=True)
            seg_flat = (seg_flat - min_seg) / (max_seg - min_seg + 1e-8)
            segmentations_list = seg_flat.reshape(segmentations_list.shape)

            # Ensemble mean
            scores = scores_list.mean(axis=0)  # (n_tiles,)
            segmentations = segmentations_list.mean(axis=0)

            # === 聚合到 image-level ===
            unique_image_names = []
            image_scores = []
            image_labels = []
            image_segmentations = []
            image_paths_agg = []

            seen = set()
            for i, name in enumerate(aggregator["image_names"]):
                base_name = name.rsplit("_", 1)[0]
                if base_name not in seen:
                    seen.add(base_name)

                    tile_indices = [j for j, n in enumerate(aggregator["image_names"]) if n.rsplit("_", 1)[0] == base_name]
                    tile_scores = scores[tile_indices]
                    tile_segs = segmentations[tile_indices]

                    image_scores.append(np.max(tile_scores))  # 取 max，最常匹配高 AUROC
                    image_segmentations.append(np.mean(tile_segs, axis=0))

                    image_labels.append(labels_gt[i])

                    orig_path = aggregator["image_paths"][tile_indices[0]]
                    image_paths_agg.append(os.path.splitext(orig_path)[0] + os.path.splitext(orig_path)[1])

            scores = np.array(image_scores)
            segmentations = np.array(image_segmentations)
            anomaly_labels = image_labels
            image_paths = image_paths_agg

            LOGGER.info(f"聚合后 image-level 样本数: {len(scores)}（原图数量）")

            # 可视化
            if save_segmentation_images:
                def image_transform(image):
                    in_std = np.array(dataloaders["testing"].dataset.transform_std).reshape(-1, 1, 1)
                    in_mean = np.array(dataloaders["testing"].dataset.transform_mean).reshape(-1, 1, 1)
                    img = dataloaders["testing"].dataset.transform_img(image)
                    return np.clip((img.numpy() * in_std + in_mean) * 255, 0, 255).astype(np.uint8)

                def mask_transform(mask):
                    return dataloaders["testing"].dataset.transform_mask(mask).numpy()

                patchcore.utils.plot_segmentation_images(
                    run_save_path,
                    image_paths,
                    segmentations,
                    scores,
                    mask_paths=None,
                    image_transform=image_transform,
                    mask_transform=mask_transform,
                    save_depth=5,
                )

                if (save_faf_visualizations or save_dica_visualizations):
                    LOGGER.info("FAF/DICA 可视化暂不支持额外叠加。")

            # 指标
            LOGGER.info("Computing evaluation metrics.")
            if len(np.unique(anomaly_labels)) > 1:
                auroc = patchcore.metrics.compute_imagewise_retrieval_metrics(scores, anomaly_labels)["auroc"]
            else:
                LOGGER.warning("只有一类样本，AUROC 设为 NaN")
                auroc = np.nan

            full_pixel_auroc = np.nan
            anomaly_pixel_auroc = np.nan

            result_collect.append({
                "dataset_name": dataset_name,
                "instance_auroc": auroc,
                "full_pixel_auroc": full_pixel_auroc,
                "anomaly_pixel_auroc": anomaly_pixel_auroc,
            })

            for key, value in result_collect[-1].items():
                if key != "dataset_name":
                    LOGGER.info("{}: {:.3f}".format(key, value))

            del PatchCore_list
            gc.collect()

        LOGGER.info("\n\n--------------------\n")

    if result_collect:
        result_metric_names = ["Instance AUROC", "Full Pixel AUROC", "Anomaly Pixel AUROC"]
        result_dataset_names = [r["dataset_name"] for r in result_collect]
        result_scores = [[r["instance_auroc"], r["full_pixel_auroc"], r["anomaly_pixel_auroc"]] for r in result_collect]

        patchcore.utils.compute_and_store_final_results(
            run_save_path,
            result_scores,
            row_names=result_dataset_names,
            column_names=result_metric_names,
        )

@main.command("patch_core_loader")
@click.option("--patch_core_paths", "-p", type=str, multiple=True, required=True)
@click.option("--faiss_on_gpu", is_flag=True)
@click.option("--faiss_num_workers", type=int, default=8)
def patch_core_loader(patch_core_paths, faiss_on_gpu, faiss_num_workers):
    def get_patchcore_iter(device):
        for patch_core_path in patch_core_paths:
            loaded_patchcores = []
            gc.collect()

            faiss_files = [f for f in os.listdir(patch_core_path) if f.endswith(".faiss")]
            n_patchcores = len(faiss_files) if faiss_files else 1

            for i in range(n_patchcores):
                nn_method = patchcore.common.FaissNN(faiss_on_gpu, faiss_num_workers)
                patchcore_instance = patchcore.patchcore.PatchCore(device)

                prepend = "" if n_patchcores <= 1 else f"Ensemble-{i+1}-{n_patchcores}_"

                patchcore_instance.load_from_path(
                    load_path=patch_core_path,
                    device=device,
                    nn_method=nn_method,
                    prepend=prepend,
                )
                loaded_patchcores.append(patchcore_instance)

            yield loaded_patchcores

    return ("get_patchcore_iter", [get_patchcore_iter, len(patch_core_paths)])

@main.command("dataset")
@click.argument("name", type=str)
@click.argument("data_path", type=click.Path(exists=True, file_okay=False))
@click.option("--subdatasets", "-d", multiple=True, type=str, required=True)
@click.option("--batch_size", default=1, type=int, show_default=True)
@click.option("--num_workers", default=8, type=int, show_default=True)
@click.option("--resize", default=256, type=int, show_default=True)
@click.option("--imagesize", default=224, type=int, show_default=True)
def dataset(name, data_path, subdatasets, batch_size, num_workers, resize, imagesize):
    dataset_info = _DATASETS[name]
    dataset_library = __import__(dataset_info[0], fromlist=[dataset_info[1]])

    def get_dataloaders_iter(seed):
        for subdataset in subdatasets:
            test_dataset = dataset_library.__dict__[dataset_info[1]](
                data_path,
                classname=subdataset,
                resize=resize,
                imagesize=imagesize,
                split=dataset_library.DatasetSplit.TEST,
                seed=seed,
            )

            test_dataloader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            )
            test_dataloader.name = f"{name}_{subdataset}"

            yield {"testing": test_dataloader}

    return ("get_dataloaders_iter", [get_dataloaders_iter, len(subdatasets)])

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()