# iot-intrusion-detection
For the sake of passing Data Mining course

# Installation
Run the following command on your terminal.

`$ git clone https://github.com/canbatuhan/iot-intrusion-detection.git`

Also you need to install the following libraries through `pip`.

- `sklearn`
- `tensorflow`
- `keras`
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `kagglehub`

Some of these libraries are dependent, meaning that they will be installed when when the other one is installed.

# How to use it
After downloading the repository open the folder `iot-intrusion-detection/` using `vscode`. To do that you can run the following command in your terminal.

`$ code iot-intrusion-detection`

Open the file `main.ipynb`. Then, click on `Run All` button. If you want to run the code cell by cell, you can use the play button next to each cell.

# Attention
A sample dataset is given within the repository, which involves 2219201 entities. If you desire to use raw data from scratch, uncomment the second and the third cells which have the following codes.

`#SOURCE = "mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot"` \
`#path = mining.get_dataset(SOURCE)` \
`#mining.integration(path, ["attack", "normal"], "data/dataset.csv")`

It will download a dataset from kaggle, and merge the data into a single dataset. 