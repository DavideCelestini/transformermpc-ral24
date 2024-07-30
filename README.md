# Transformer-based Model Predictive Control: Trajectory Optimization via Sequence Modeling

## Transformer architecture and parameters
The transformer architecture proposed in this work is inspired by the DecisionTransformer architecture implemented in the HuggingFace library [1].
Our implementation can be found in the $\texttt{art.py}$ file in the $\texttt{decision transformer}$ folder of each specific scenario.
| Parameter description       | Value           |
| :-----                      | :----:          |
| Embedding dimension         | $384$           |
| Maximum context length      | $100$           |
| Number of layers            | $6$             |
| Number of attention heads   | $6$             |
| Batch size                  | $4$             |
| Non-linearity               | $\textrm{ReLU}$ |
| Dropout                     | $0.1$           |
| Learning rate               | $3e^{-5}$       |
| Learning rate decay         | $\textrm{None}$ |
| Gradient norm clip          | $1.0$           |
| Gradient accumulation iters | $8$             |

<sub> [1] “Huggingface’s Tranformers Library”, https://huggingface.co/docs/transformers/index. </sub>

## Open-loop training hyperparameters
| Parameter description                  | Symbol        | Value          |
| :-----                                 | :---:         | :----:         |
| Number of samples in the dataset       | $N_d$         | $400,000$      |
| Number of REL solutions in the dataset | $N_{d_{REL}}$ | $200,000$      |
| Number of SCP solutions in the dataset | $N_{d_{SCP}}$ | $200,000$      |
| Train split (%)                        |   -           | $90$           |
| Test split (%)                         |   -           | $10$           |

## Closed-loop training hyperparameters
| Parameter description                                                                          | Symbol                          | Value                                       |
| :-----                                                                                         | :---:                           | :----:                                      |
| Interaction with the environment collected at each $\text{DA\small{GGER}}$ iteration           | $\texttt{num trajectories}$     | $4,000$                                     |
| Possible values for the planning horizon for each interaction                                  | $H$                             | $[10, 20, 30, 40, 50, 60, 70, 80, 90, 100]$ |
| Initial open-loop to closed-loop ratio in the aggregated dataset                               |   -                             | $9:1$                                       |
| Train split (%)                                                                                |   -                             | $90$                                        |
| Test split (%)                                                                                 |   -                             | $10$                                        |
