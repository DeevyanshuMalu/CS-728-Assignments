import numpy as np
import matplotlib.pyplot as plt
import os

task = "A1_mem_rnn_tanh_noclip_final_state"
z = np.load(f"{task}.npz")
grad_time = z["grad_time"]  # (num_checkpoints, Tstore), NaN padded
sat_time = z["sat_time"]  # (num_checkpoints, Tstore)
valid_err = z["valid_error"]  # (num_checkpoints,)
rho = z["rho_Whh"]  # (num_checkpoints,)

# choose a checkpoint (e.g., last non-empty)
g = grad_time[-1]
s = sat_time[-1]
g = g[np.isfinite(g)]
s = s[np.isfinite(s)]

os.makedirs("plots", exist_ok=True)

plt.figure()
plt.hist(np.log10(g + 1e-12), bins=60)
plt.title("log10 ||dL/dh_t||")
plt.savefig(f"plots/{task}_grad_time.png")

plt.figure()
plt.hist(s, bins=60, range=(0, 1))
plt.title("hidden saturation distance")
plt.savefig(f"plots/{task}_sat_time.png")

plt.figure()
plt.plot(valid_err)
plt.title("validation error (%)")
plt.savefig(f"plots/{task}_valid_err.png")

plt.figure()
plt.plot(rho)
plt.title("rho_Whh")
plt.savefig(f"plots/{task}_rho_Whh.png")

plt.show()
