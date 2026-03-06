import numpy as np
import matplotlib.pyplot as plt
import os

task = "A4_mem_gru_noclip_final_state"
z = np.load(f"{task}.npz")
train_nll = z["train_nll"]  # (num_checkpoints,)
grad_time = z["grad_time"]  # (num_checkpoints, Tstore), NaN padded
sat_time = z["sat_time"]  # (num_checkpoints, Tstore)
valid_err = z["valid_error"]  # (num_checkpoints,)
rho = z["rho_Whh"]  # (num_checkpoints,)
gate_z_sat_time = z["gate_z_sat_time"]  # (num_checkpoints, Tstore)
gate_r_sat_time = z["gate_r_sat_time"]  # (num_checkpoints, Tstore)

# choose a checkpoint (e.g., last non-empty)
g = grad_time[-1]
s = sat_time[-1]
z = gate_z_sat_time[-1]
r = gate_r_sat_time[-1]
g = g[np.isfinite(g)]
s = s[np.isfinite(s)]
z = z[np.isfinite(z)]
r = r[np.isfinite(r)]

os.makedirs(f"plots/{task}", exist_ok=True)

plt.figure()
plt.plot(train_nll)
plt.title("train_nll")
plt.savefig(f"plots/{task}/train_nll.png")

plt.figure()
plt.hist(np.log10(g + 1e-12), bins=60)
plt.title("log10 ||dL/dh_t||")
plt.savefig(f"plots/{task}/grad_time.png")

plt.figure()
plt.hist(s, bins=60, range=(0, 1))
plt.title("hidden saturation distance")
plt.savefig(f"plots/{task}/sat_time.png")

plt.figure()
plt.plot(valid_err)
plt.title("validation error (%)")
plt.savefig(f"plots/{task}/valid_err.png")

plt.figure()
plt.plot(rho)
plt.title("rho_Whh")
plt.savefig(f"plots/{task}/rho_Whh.png")

if "gru" in task.lower():
    plt.figure()
    plt.hist(z, bins=60, range=(0, 1))
    plt.title("gate z saturation distance")
    plt.savefig(f"plots/{task}/gate_z_sat_time.png")

    plt.figure()
    plt.hist(r, bins=60, range=(0, 1))
    plt.title("gate r saturation distance")
    plt.savefig(f"plots/{task}/gate_r_sat_time.png")

plt.show()
