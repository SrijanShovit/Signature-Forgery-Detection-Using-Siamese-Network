import numpy as np
from tqdm import tqdm

from src.image_quality.metrics import laplacian_variance, brightness_mean, contrast_std, noise_residual, high_frequency_energy, edge_density
from src.image_quality.thresholds import ThresholdEstimator
from src.image_quality.reporter import QualityReporter


class ImageQualityAnalyzer:

    def __init__(self, dataloader, thresholds=None):

        self.dataloader = dataloader
        self.user_thresholds = thresholds
        self.records = []

    def tensor_to_numpy(self, tensor):

        img = tensor.permute(1,2,0).cpu().numpy()

        img = (img * 255).astype(np.uint8)

        return img

    def compute_metrics(self, image):

        return {
            "blur": laplacian_variance(image),
            "brightness": brightness_mean(image),
            "contrast": contrast_std(image),
            "noise": noise_residual(image),
            "hf_energy": high_frequency_energy(image),
            "edge_density": edge_density(image)
        }

    def process_image(self, image, label):

        metrics = self.compute_metrics(image)

        metrics["label"] = int(label)

        return metrics

    def run(self):

        metric_storage = {
            "blur": [],
            "brightness": [],
            "contrast": [],
            "noise": [],
            "hf_energy": [],
            "edge_density": []
        }

        for batch in tqdm(self.dataloader):

            if len(batch) == 3:
                img1, img2, label = batch
            else:
                (img1, img2), label = batch

            for i in range(img1.shape[0]):

                image1 = self.tensor_to_numpy(img1[i])
                image2 = self.tensor_to_numpy(img2[i])

                r1 = self.process_image(image1, label[i])
                r2 = self.process_image(image2, label[i])

                self.records.append(r1)
                self.records.append(r2)

                for k in metric_storage.keys():
                    metric_storage[k].append(r1[k])
                    metric_storage[k].append(r2[k])

        if self.user_thresholds is None:

            thresholds = ThresholdEstimator(metric_storage).compute()

        else:
            thresholds = self.user_thresholds

        reporter = QualityReporter(self.records)

        return {
            "thresholds": thresholds,
            "global_summary": reporter.global_summary(),
            "classwise_summary": reporter.classwise_summary(),
            "image_table": reporter.image_table()
        }