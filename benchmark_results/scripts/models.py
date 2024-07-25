import torch
from chronos import ChronosPipeline
import math

class ChronosModel:
    """
    A wrapper around the Chronos forecase model class that makes it easier to use in 
    forecasting tasks.

    Attributes:
        model (str): The model size to use. One of "tiny", "mini", "small", "base", "large".
        context (int): The number of data points to use for context when making predictions.
        n_samples (int): The number of samples to use when making predictions.
        prediction_length (int): The number of data points to predict.
        use_gpu (bool): Whether to use a GPU for prediction.
        max_chunk (int): The maximum number of data points to predict in one go. If the 
            prediction length is greater than this value, we use the autoregressive
            mode to predict the future values.
        device (str): Deprecated. The device to use for prediction
    """

    def __init__(self, model="base", context=100, n_samples=20, prediction_length=64, 
                 use_gpu=True, max_chunk=64, device=None):
        self.model = model
        self.context = context
        self.n_samples = n_samples
        self.prediction_length = prediction_length
        self.use_gpu = use_gpu
        self.max_chunk = max_chunk

        ## If a GPU is available, use it
        if self.use_gpu:
            has_gpu = torch.cuda.is_available()
            print("has gpu: ", torch.cuda.is_available(), flush=True)
            n = torch.cuda.device_count()
            print(f"{n} devices found.", flush=True)
            if has_gpu:
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = "cpu"

        # If the prediction length is greater than 64, we need to use the autoregressive
        # mode  to predict the future values.
        self.autoregressive = (prediction_length > self.max_chunk)

        self.pipeline = ChronosPipeline.from_pretrained(
            f"amazon/chronos-t5-{self.model}",
            device_map=self.device,
            torch_dtype=torch.bfloat16,
        )

        self.name = f"chronos-{self.model}-context{self.context}"

    def predict(self, data):
        """
        Given a time series data, use the last {self.context} timepoints of data 
        to predict next value in the series.

        Args:
            data (np.ndarray): The time series data to use for prediction. The shape of
                the data should be (T, D) where T is the number of data points and D is 
                the number of dimensions.
                
        """
        # if self.autoregressive:
        #     forecast_agg = []
        #     current_context = torch.tensor(data[-self.context:])
        #     print(int(math.ceil(self.prediction_length / self.max_chunk)), flush=True)
        #     for i in range(int(math.ceil(self.prediction_length / self.max_chunk))):
        #         forecast = self.pipeline.predict(
        #             context=torch.tensor(current_context),
        #             prediction_length=self.max_chunk,
        #             num_samples=self.n_samples,
        #         )
        #         mean_forecast = forecast.mean(dim=1)
        #         current_context = torch.cat((current_context, mean_forecast), dim=1)[:, -self.context:]
        #         forecast_agg.append(forecast)
        #     forecast = torch.cat(forecast_agg, dim=2)[:self.prediction_length]
        # else:
        #     forecast = self.pipeline.predict(
        #         context=torch.tensor(data[-self.context:]),
        #         prediction_length=self.prediction_length,
        #         num_samples=self.n_samples,
        #     )
        forecast = self.pipeline.predict(
            context=torch.tensor(data[-self.context:]),
            prediction_length=self.prediction_length,
            num_samples=self.n_samples,
            limit_prediction_length = False,
        )
        return forecast

