# dist_fitter

A dynamic Python package for fitting SciPy distributions to data and generating visualizations.

## Features
* Fit 100+ continuous SciPy distributions to single or multiple datasets.
* Supports Histograms, Empirical CDFs, Probability Plots, and Q-Q plots.
* Dynamic legend formatting with standard LaTeX math symbols (e.g., $\hat{\mu}$, $\hat{\sigma}$).
* Automatically aligns and scales multi-sample probability plots for direct comparison.

## Installation

You can install the package directly from GitHub using pip:

```bash
pip install git+https://github.com/yourusername/dist-fitter.git
```

## Quick Start

Here is a simple example of how to generate data from a Gamma distribution and plot a histogram with a theoretical fit:

```python
import numpy as np
import dist_fitter

# Generate sample data
np.random.seed(42)
sample_data = np.random.gamma(shape=2.0, scale=2.0, size=1000)

# Fit and plot a Histogram
fitted_params = dist_fitter.fit_and_plot(
    data=sample_data, 
    dist_name='gamma', 
    plot_type='hist',
    labels=['My Sample']
)

print(f"Fitted Parameters: {fitted_params}")
```

## Supported Plot Types
* `'hist'`: Histogram with overlaid PDF.
* `'ecdf'`: Empirical Cumulative Distribution Function with overlaid theoretical CDF.
* `'probplot'`: Probability plot (Ordered Values vs Probability).
* `'qqplot'`: Modern Q-Q Plot (Theoretical Quantiles vs Ordered Values).

## License
This project is licensed under the MIT License.
