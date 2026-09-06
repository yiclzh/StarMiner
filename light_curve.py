import matplotlib.pyplot as plt
import numpy as np

from transit_physics import semi_major_axis_au, transit_depth, transit_duration_hours

def simulate_light_curve(time_days, t0_days, duration_hours, depth, ingress_function=0.2):
    duration_days = duration_hours / 24
    dt = np.abs(time_days - t0_days)

    ingress_days = duration_days * ingress_function
    half_total = duration_days / 2
    half_flat = half_total - ingress_days

    flux = np.ones_like(time_days)

    full_mask = dt <= half_flat
    ramp_mask = (dt > half_flat) & (dt < half_total) #ingress/egress region


    # fully in transit
    flux[full_mask] = 1 - depth
    # in ramp
    fraction = (half_total - dt) / ingress_days
    flux[ramp_mask] = 1 - depth * fraction[ramp_mask]

    plt.plot(time_days, flux)

    return flux




time_days = np.linspace(-2, 2, 1000)
t0_days = 0
duration_hours = 29.6

for depth_val in [0.02, 0.01, 0.005, 0.002]:
    flux = simulate_light_curve(time_days, t0_days, duration_hours, depth_val)
    plt.plot(time_days, flux, label=f"depth={depth_val}")

plt.xlabel("time (days)")
plt.ylabel("relative brightness")
plt.legend()
plt.show()