### Testing the model:

1. ##### Install Python 3.6

2. ##### Clone or download the repository

    `git clone https://github.com/g-simmons/dairyML.git` 

   or download using the green "clone or download" button on the right

3. ##### Install the required packages

   `cd DairyML`

   `pip install requirements.txt`

4. ##### Place your test data in `data/` directory

   Make sure that the test data columns match the columns of `data/training_for_GS_122118.csv`

5. ##### Run the test script

   Test script usage is: python test.py <model_path> <data_path>

   For example, from the main directory:

   `python src/test.py models/xgb_combined.model data/<test_data_filename>.csv`

6. ##### Results are stored to csv in reports/

