
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

def get_param_str(dist_name, params):
    """Helper to format parameters dynamically with standard Greek letters and hats."""
    # Explicit overrides for the user's preferred standardizations
    if dist_name == 'norm':
        return rf"$\hat{{\mu}}={params[0]:.2f}$, $\hat{{\sigma}}={params[1]:.2f}$"
    elif dist_name == 'gamma':
        return rf"$\hat{{k}}={params[0]:.2f}$, $\hat{{\mu}}={params[1]:.2f}$, $\hat{{\theta}}={params[2]:.2f}$"
    elif dist_name == 'lognorm':
        return rf"$\hat{{s}}={params[0]:.2f}$, $\hat{{\mu}}={params[1]:.2f}$, $\hat{{\sigma}}={params[2]:.2f}$"
    
    # Dynamic fallback for the other 100+ SciPy distributions
    dist = getattr(stats, dist_name, None)
    if dist is not None:
        shapes = dist.shapes
        shape_names = [s.strip() for s in shapes.split(',')] if shapes else []
        param_names = shape_names + ['loc', 'scale']
        
        # Map SciPy's standard parameter variables to LaTeX math equivalents
        tex_map = {
            'loc': r'\mu',
            'scale': r'\sigma',
            'a': r'\alpha',
            'b': r'\beta',
            'c': 'c',
            's': 's',
            'df': 'df',
            'shape': 'shape'
        }
        
        if len(params) == len(param_names):
            formatted = []
            for name, val in zip(param_names, params):
                tex_name = tex_map.get(name, name)
                formatted.append(rf"$\hat{{{tex_name}}}={val:.2f}$")
            return ", ".join(formatted)
            
    # Absolute fallback if shapes fail
    return ", ".join([rf"$\hat{{p}}_{{{i+1}}}={p:.2f}$" for i, p in enumerate(params)])

def fit_and_plot(data, dist_name='norm', bins=30, plot_type='hist', labels=None, xlim=None):
    """
    Fits a SciPy distribution to the provided data and plots the requested plot_type.
    `data` can be a single array-like sample or an iterable of multiple samples of potentially differing lengths.
    plot_type can be 'hist', 'probplot', 'ecdf', or 'qqplot'.
    xlim can be a tuple (min, max) to manually set the x-axis limits.
    """
    # Determine if `data` is a single sample or multiple samples
    try:
        if hasattr(data[0], '__len__') and not isinstance(data[0], str):
            samples = data
        else:
            samples = [data]
    except (TypeError, IndexError):
        samples = [data]

    if len(samples) > 1 and plot_type == 'qqplot':
        raise ValueError("Q-Q plots are not supported for multiple samples simultaneously. Please plot them individually or use 'probplot' instead.")

    # Retrieve the distribution object from scipy.stats
    dist = getattr(stats, dist_name)

    # Create the plot
    plt.figure(figsize=(9, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, max(10, len(samples))))

    all_params = []
    x_mins, x_maxs = [], []
    first_params = None  # Store the first distribution's parameters

    for i, sample in enumerate(samples):
        sample_arr = np.asarray(sample)
        # Fit the distribution to the data
        params = dist.fit(sample_arr)
        all_params.append(params)

        if i == 0:
            first_params = params

        color = colors[i % len(colors)]
        base_label = labels[i] if labels and i < len(labels) else f'Sample {i+1}'
        
        # Create automatic legend label with parameters
        param_str = get_param_str(dist_name, params)
        data_label = f'{base_label} ({param_str})'

        if plot_type == 'hist':
            plt.hist(sample_arr, bins=bins, density=True, alpha=0.4, color=color, edgecolor='black', label=data_label)
            xmin_hist, xmax_hist = plt.xlim()
            x = np.linspace(xmin_hist, xmax_hist, 100)
            p = dist.pdf(x, *params)
            plt.plot(x, p, color=color, linewidth=2) # Removed fit label
            plt.ylabel('Density')
            plt.xlabel('Value')

        elif plot_type == 'probplot':
            # Get probplot data
            (osm, osr), (slope, intercept, r) = stats.probplot(sample_arr, dist=dist, sparams=params)

            if i == 0:
                plot_osm = osm
                # Fit line for first sample
                x_fit = np.array([np.min(osr), np.max(osr)])
                y_fit = (x_fit - intercept) / slope
            else:
                # Convert to probabilities, then to first plot's scale
                probs_i = dist.cdf(osm, *params)
                plot_osm = dist.ppf(probs_i, *first_params)

                # Apply the same mapping to the fit line
                x_fit_orig = np.array([np.min(osr), np.max(osr)])
                osm_fit = (x_fit_orig - intercept) / slope
                probs_fit = dist.cdf(osm_fit, *params)
                y_fit = dist.ppf(probs_fit, *first_params)
                x_fit = x_fit_orig

            # Plot data (Ordered Values on X, Transformed Theoretical Quantiles on Y)
            plt.plot(osr, plot_osm, marker='.', linestyle='none', color=color, label=data_label)

            # Plot the fit line
            plt.plot(x_fit, y_fit, linestyle='-', color=color, alpha=0.6) # Removed fit label

            # Setup probability scale on Y axis (using the first sample's scale for consistency)
            if i == 0:
                probs = np.array([1, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 98, 99])
                y_ticks = dist.ppf(probs / 100.0, *first_params)
                valid = np.isfinite(y_ticks)
                plt.yticks(y_ticks[valid], probs[valid])
                plt.ylim(y_ticks[valid][0], y_ticks[valid][-1])

            plt.ylabel('Probability (%)')
            plt.xlabel('Ordered Values')

        elif plot_type == 'ecdf':
            # Empirical CDF
            x = np.sort(sample_arr)
            y = np.arange(1, len(x) + 1) / len(x)
            plt.plot(x, y, marker='.', linestyle='none', color=color, label=data_label, alpha=0.5)

            # Theoretical CDF
            xmin_cdf, xmax_cdf = x[0], x[-1]
            x_theo = np.linspace(xmin_cdf, xmax_cdf, 100)
            y_theo = dist.cdf(x_theo, *params)
            plt.plot(x_theo, y_theo, linestyle='-', color=color, linewidth=2) # Removed fit label
            plt.ylabel('Cumulative Probability')
            plt.xlabel('Value')

        elif plot_type == 'qqplot':
            # Get probplot data
            (osm, osr), (slope, intercept, r) = stats.probplot(sample_arr, dist=dist, sparams=params)

            # Plot data (Theoretical Quantiles on X, Ordered Values on Y)
            plt.plot(osm, osr, marker='.', linestyle='none', color=color, label=data_label)

            # Plot the fit line: osr = slope * osm + intercept
            x_fit = np.array([np.min(osm), np.max(osm)])
            y_fit = slope * x_fit + intercept
            plt.plot(x_fit, y_fit, linestyle='-', color=color, alpha=0.6) # Removed fit label

            plt.ylabel('Ordered Values')
            plt.xlabel('Theoretical Quantiles')
        else:
            raise ValueError("plot_type must be 'hist', 'probplot', 'ecdf', or 'qqplot'")

        # Track percentiles from the theoretical distribution for xlim
        p1 = dist.ppf(0.01, *params)
        p99 = dist.ppf(0.99, *params)
        if np.isfinite(p1) and np.isfinite(p99):
            x_mins.append(p1)
            x_maxs.append(p99)

    # Apply x-limits using user input or 1st and 99th percentiles of the theoretical distributions
    if xlim is not None:
        plt.xlim(xlim)
    elif x_mins and x_maxs:
        global_p1 = min(x_mins)
        global_p99 = max(x_maxs)
        pad = (global_p99 - global_p1) * 0.01
        if pad == 0:
            pad = 1e-5
        plt.xlim(global_p1 - pad, global_p99 + pad)

    plot_type_names = {
        'hist': 'Histogram',
        'probplot': 'Probability Plot',
        'ecdf': 'Empirical CDF',
        'qqplot': 'Q-Q Plot'
    }
    plot_name = plot_type_names.get(plot_type, plot_type.capitalize())
    
    dist_names = {
        'norm': 'Normal',
        'gamma': 'Gamma',
        'lognorm': 'Lognormal',
        'expon': 'Exponential',
        'uniform': 'Uniform',
        't': "Student's t",
        'beta': 'Beta',
        'weibull_min': 'Weibull (Min)',
        'weibull_max': 'Weibull (Max)'
    }
    # Use the dictionary or a clean dynamic fallback
    dist_display = dist_names.get(dist_name, dist_name.replace('_', ' ').title())
    
    plt.title(f"{plot_name} with {dist_display} Fit")
    plt.grid(axis='both', alpha=0.5)
    # Move legend outside if multiple samples to avoid clutter
    if len(samples) > 1:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        plt.legend()
    plt.tight_layout()
    plt.show()

    return all_params if len(all_params) > 1 else all_params[0]
