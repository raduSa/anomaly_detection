Shapley values originate from cooperative game theory and provide a mathematically sound method for allocating credit for a machine learning model’s output across its input features. Rather than relying on standard model coefficients—which can be heavily skewed by the scale of the data—SHAP (SHapley Additive exPlanations) evaluates exactly how much each feature shifts the model's prediction away from the baseline average.

What They Are
Think of a machine learning prediction as a game where the input features are the "players" and the final prediction is the game's outcome. Shapley values ensure this outcome is distributed fairly based on each player's actual contribution. They offer a unified, consistent measure of feature importance that holds true across simple linear regressions, generalized additive models, and complex, non-additive architectures like XGBoost and NLP transformers.

How They Work
To evaluate a feature's impact, the SHAP algorithm measures the difference between a model's output when a feature is present versus when it is withheld.

Joining the Game: A feature "joins" the model when its specific value for a given data point is known (e.g., setting a HouseAge input to 25 years).

Sitting Out: A feature is excluded by integrating it out using conditional expected values. This effectively asks, "What would the model predict if we didn't know this specific feature's value?"

Calculating the Value: The SHAP value is the exact difference between the expected baseline prediction and the updated prediction once the feature's value is introduced and combined with the other known features.