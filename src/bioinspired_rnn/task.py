"""Constants of the Economic Choice task shared by the environment and the analysis."""

#: Offer types, as (drops of B, drops of A), in the order used in all figures.
OFFER_SETS = [(0, 1), (1, 2), (1, 1), (2, 1), (3, 1), (4, 1), (6, 1), (10, 1), (2, 0)]

#: Value of one drop of juice A in drops of juice B.
A_TO_B_RATIO = 2.2

#: Network variants compared in the paper: name used in the code -> label.
NETWORKS = {"ffnn": "FFNN", "rnn_std": "Standard RNN", "rnn": "Modified RNN"}

#: Environments: code -> label.
ENVIRONMENTS = {"F": "Full", "P": "Partial"}
