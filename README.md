# KNN and RNN Performance Lab

## Purpose

This lab explores the performance of two neighborhood-based classifiers—K-Nearest Neighbors (KNN) and Radius Neighbors (RNN)—on the Wine dataset from scikit-learn. By varying the neighborhood size (`k`) and the inclusion radius, the goal is to understand how hyperparameter choices influence classification accuracy and to identify optimal settings for each model.

## Key Insights

* **KNN Outperforms RNN:** KNN achieved a maximum test accuracy of **0.806** for $k \ge 5$, whereas RNN peaked at **0.722** for radius=350.
* **Plateau in KNN Accuracy:** Accuracy increased from 0.778 at $k=1$ to 0.806 at $k=5$ and then remained stable through $k=21$, indicating that adding more neighbors beyond 5 yielded diminishing returns.
* **Decreasing RNN Performance with Larger Radius:** RNN accuracy dropped as the radius increased, likely because larger radii include points from different classes, diluting classification decisions.
* **Model Sensitivity to Density vs. Fixed Neighbors:** KNN’s fixed-$k$ vote is robust across variable data densities, while RNN’s variable neighbor count can be informative in naturally clustered data but is sensitive to scaling and radius choice.

## Challenges and Decisions

* **Parameter Selection:** Determining an appropriate radius for RNN was challenging because too-small radii left some test points without neighbors, while too-large radii introduced noisy votes.
* **Feature Scaling Considerations:** We did not apply explicit feature normalization, relying instead on relative distances. In future work, standardizing features could improve RNN stability.
* **Balancing Simplicity and Performance:** We limited our grid to five $k$ values and six radii to keep the analysis concise. Expanding the search grid or using cross-validation could refine the optimal settings further.
