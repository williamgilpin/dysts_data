import torch
from chronos import ChronosPipeline
import math

class ChronosModel:

    def __init__(self, model="base", context=100, n_samples=20, prediction_length=64, 
                 device="cpu"):
        self.model = model
        self.context = context
        self.n_samples = n_samples
        self.prediction_length = prediction_length
        self.device = device

        # If the prediction length is greater than 64, we need to use the autoregressive
        # mode  to predict the future values.
        self.autoregressive = (prediction_length > 64)

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
        if self.autoregressive:
            forecast = self.pipeline.predict(
                context=torch.tensor(data[-self.context:]),
                prediction_length=self.prediction_length,
                num_samples=self.n_samples,
                autoregressive=True,
            )
        else:
            forecast_agg = []
            for i in range(int(math.ceil(self.prediction_length / 64))):
                print(i)
                forecast = self.pipeline.predict(
                    context=torch.tensor(current_context),
                    prediction_length=64,
                    num_samples=self.n_samples,
                )
                current_context = torch.cat(current_context, forecast)[-self.context:]
                forecast_agg.append(forecast.squeeze())

            forecast = torch.hstack(forecast_agg)[:self.prediction_length]

        return forecast
