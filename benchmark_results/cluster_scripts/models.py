import torch
from chronos import ChronosPipeline

class ChronosModel:

    def __init__(self, model="base", context=100, n_samples=20, prediction_length=64, 
                 device="cpu"):
        self.model = model
        self.context = context
        self.n_samples = n_samples
        self.prediction_length = prediction_length
        self.device = device

        self.pipeline = ChronosPipeline.from_pretrained(
            f"amazon/chronos-t5-{self.model}",
            device_map=self.device,
            torch_dtype=torch.bfloat16,
        )

        self.name = f"chronos-{self.model}-context{self.context}"

    def predict(self, data):
        """
        Given a time series data, use at most self.context data points to predict the 
        next value in the series.
        """
        forecast = self.pipeline.predict(
            context=torch.tensor(data[-self.context:]),
            prediction_length=self.prediction_length,
            num_samples=self.n_samples,
        )
        return forecast