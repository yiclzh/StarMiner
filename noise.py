import numpy as np

from transit_physics import semi_major_axis_au, transit_depth, transit_duration_hours
from light_curve import simulate_light_curve, time_days


def generate_example(time_days, noise_std, rng):
    is_transit = rng.random() < 0.5
    if is_transit:
        star_mass_solar = rng.uniform(0.5, 1.5)
        star_radius_solar = rng.uniform(0.5, 1.5)
        planet_radius_jupiter = rng.uniform(0.05, 1.5)
        period_days = rng.uniform(1, 50)
        impact_param = rng.uniform(0, 0.9)
        a_au = semi_major_axis_au(period_days, star_mass_solar)
        depth = transit_depth(planet_radius_jupiter, star_radius_solar)
        duration_hours = transit_duration_hours(period_days, star_radius_solar, a_au, impact_param)
        flux = simulate_light_curve(time_days, 0, duration_hours, depth)
        label = 1
    else:
        flux = np.ones_like(time_days)
        label = 0

    flux = flux + rng.normal(0, noise_std, size=flux.shape)
    return flux, label

def generate_dataset(n_examples, time_days, noise_std, rng):
    X = np.zeros((n_examples, len(time_days)))
    y = np.zeros(n_examples, dtype=int)

    for i in range(n_examples):
        flux, label = generate_example(time_days, noise_std, rng)
        X[i] = flux
        y[i] = label

    return X, y


rng = np.random.default_rng(0)
X, y = generate_dataset(500, time_days, noise_std=0.001, rng=rng)
print(X.shape, y.shape)      # expect (500, len(time_days)) and (500,)
print(np.bincount(y))        # expect roughly [250, 250] — a real mix, not all one label
