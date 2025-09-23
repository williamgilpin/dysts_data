This secondary data repository hosts precomputed time series datasets and benchmark results for our main repository, [dysts](https://github.com/williamgilpin/dysts).

## Installation
Clone the repository and install:

```bash
git clone https://github.com/williamgilpin/dysts_data.git
cd dysts_data
pip install -e .
```


## Usage
```python
from dysts_data.datasets import load_dataset
data = load_dataset(subsets="train", data_format="numpy", standardize=True)
```

