from scipy.stats import poisson, multivariate_normal
import numpy as np

# Generate 1000 samples for number of packages over a whole day
num_pkg_samples = poisson.rvs(mu=3*24, size=1000)

# Create Gaussian dist for joint Size & Weight
mu = [120,4]
sigma = [[1.5, 1], [1, 1.5]]
sw_dist = multivariate_normal(
	mean=mu, 
	cov=sigma)

# Generate 1000 samples for Size & Weight of packages
sw_samples = sw_dist.rvs(size=1000)

# Initialize result array of T* draws
sim_draws = [0] * 1000

# Calc T* for each draw
for i in range(len(num_pkg_samples)):
	# Calc T for each package that arrived that day
	for j in range(num_pkg_samples[i]):
		eps = np.random.normal(0, 5)
		sim_draws[i] += 60 + 0.6*sw_samples[i][1] + 0.2*sw_samples[i][0] + eps


# Find the mean and standard dev
mean = np.mean(sim_draws)
std = np.std(sim_draws)

print(f"Mean: {mean}")
print(f"Standard deviation: {std}")