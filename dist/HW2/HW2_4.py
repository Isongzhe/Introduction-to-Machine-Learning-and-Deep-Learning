import numpy as np

class TargetFunction:
    @staticmethod
    def evaluate(x):
        """Calculate the target function f(x) = sin(πx)."""
        return np.sin(np.pi * x)

class BiasVarianceCalculator:
    def __init__(self, num_samples=10000, num_experiments=1):
        self.num_samples = num_samples 
        self.num_experiments = num_experiments # Number of experiments to run
 
    def compute_bias_variance(self, num_points):
        bias_squared_sum = 0
        variance_sum = 0

        for _ in range(self.num_experiments):
            # Generate random training points
            x_train = np.random.uniform(-1, 1, num_points)
            y_train = TargetFunction.evaluate(x_train)

            # Hypothesis: horizontal line at midpoint
            m = np.mean(y_train)

            # Calculate bias and variance
            x_test = np.random.uniform(-1, 1, self.num_samples)
            y_true = TargetFunction.evaluate(x_test)
            y_pred = np.full_like(y_true, m)

            # Calculate bias squared
            bias_squared = np.mean((y_pred - y_true) ** 2) - np.var(y_true)
            bias_squared_sum += bias_squared

            # Calculate variance
            variance = np.var(y_pred)
            variance_sum += variance

        bias_squared = bias_squared_sum / self.num_experiments
        variance = variance_sum / self.num_experiments
        error = bias_squared + variance

        return np.sqrt(bias_squared), variance, error


if __name__ == "__main__":
    calculator = BiasVarianceCalculator(num_samples=10000, num_experiments=10000)

    # Part (a): Run for 2 points
    bias_2, var_2, error_2 = calculator.compute_bias_variance(num_points=2)
    print(f'(a) Bias: {bias_2:.4f}, Variance: {var_2:.4f}, Error: {error_2:.4f}')

    # Part (b): Run for 20 points
    bias_20, var_20, error_20 = calculator.compute_bias_variance(num_points=20)
    print(f'(b) Bias: {bias_20:.4f}, Variance: {var_20:.4f}, Error: {error_20:.4f}')
