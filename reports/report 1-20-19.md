### SRC/PCC/MI/MAE  (actual vs. predicted) for baselines and combined model

#### Baselines:

**Mean:** The prediction for each food is the mean of the training lactose values

**Median All:**  The prediction for each food is the median of the training lactose values

**Median Nonzero:** The prediction for each food is the median of the _non-zero_ training lactose values

**Perfect Clasif., Mean Regr.:** This classifier predicts 0 where the true value is 0, or the mean of the training lactose values where the true value is nonzero

#### Model:

**Bounded Lasso + LogReg: **In the case that the classifier predicts 0 lactose, the overall model output is 0. In the case that the classifier predicts non-zero, the output from the bounded lasso regressor is used.

![image](https://user-images.githubusercontent.com/28633482/51445765-a5f1e180-1cbd-11e9-87d3-cb6a0dd55f80.png)

<div style="page-break-after: always;"></div>

### Actual vs. predicted scatter plot for combined model

![image](https://user-images.githubusercontent.com/28633482/51445907-51e7fc80-1cbf-11e9-992f-0370fe3caba8.png)

<div style="page-break-after: always;"></div>

### Distribution of the lactose values

![image](https://user-images.githubusercontent.com/28633482/51445829-71caf080-1cbe-11e9-92be-c671c6ab8e54.png)