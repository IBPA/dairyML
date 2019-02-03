## Feature importances, FFNN & XGB Combined:

##### Do the models agree?

- Spearman rank correlation between the two feature rankings is **~.33**.  
- See next page for plots
- **FFNN top 5 features:** carbs, potassium, copper, fiber, folate
- **XGB top 5 features:** calories, protein, sugar, carbs, copper

##### How the values were obtained

- XGB feature importances are taken directly from the XGB regressor
  - Corresponds to how many times a feature was used to split the data, summed across all trees in the model
- FFNN "feature importances" are the first-layer weights 
  - For each feature, the weights going from the corresponding node in the input layer to nodes in the first hidden layer are averaged, and this average value is taken as the feature importance
  - The weights in further layers of the model also contribute to the overall impact of a feature on the model output, so first-layer weights are not perfectly representative of overall feature importance. However, as far as I know first-layer weight is a commonly used heuristic for feature importance for relatively shallow networks (as in our case).



<img src="C:\Users\Gabriel\DairyML\reports\feature_importances 2-3-19.png" style="zoom:50%" />