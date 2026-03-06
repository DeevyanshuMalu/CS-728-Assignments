import numpy as np


def simulate(T=70, B=10000, nll=0.097, n_classes=6):
    # nll = -log(prob_correct)
    # p_correct = e^(-nll)
    p = np.exp(-nll)
    print(f"Per-token accuracy: {p:.4f}")

    # Probability that a sequence of length T is perfectly correct
    p_seq = p**T
    print(f"Per-sequence accuracy: {p_seq:.6f}")

    # Probability that at least one sequence in batch B is correct
    # P(at least one) = 1 - P(none)
    p_any = 1 - (1 - p_seq) ** B
    print(
        f"Probability that at least one sequence in batch {B} is correct: {p_any:.6f}"
    )

    return p_seq


simulate(T=70, nll=0.097)
simulate(
    T=120, nll=0.097
)  # min_length 50 + 2*10 = 70, but average length is more like 120
simulate(T=220, nll=0.097)
