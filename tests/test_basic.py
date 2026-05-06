import numpy as np
import dist_fitter

# A test function must start with 'test_'
def test_get_param_str_norm():
    """Test the parameter formatting for a Normal distribution."""
    params = [0.0, 1.0]
    result = dist_fitter.get_param_str('norm', params)
    
    # Use 'assert' to check if the result matches expectations
    assert "\\hat{\\mu}=0.00" in result
    assert "\\hat{\\sigma}=1.00" in result

def test_fit_and_plot_output_length():
    """Test that fitting a normal distribution returns exactly 2 parameters."""
    np.random.seed(42)
    sample = np.random.normal(loc=5, scale=2, size=50)
    
    # Run the function
    params = dist_fitter.fit_and_plot(sample, dist_name='norm', plot_type='hist')
    
    # Normal distribution has loc and scale (2 parameters)
    assert len(params) == 2
