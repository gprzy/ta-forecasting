# Obtendo os dados
python ./src/ta_forecasting/get_data/get_data.py

# CNN

python ./src/ta_forecasting/cnn/cnn_experiment.py ./data/goog.csv # com a base original

python ./src/ta_forecasting/cnn/cnn_experiment.py ./data/goog_ta.csv # com a base de indicadores

python ./src/ta_forecasting/cnn/cnn_experiment.py ./data/goog_and_ta.csv # com a base original + indicadores

# LSTM

python ./src/ta_forecasting/lstm/lstm_experiment.py ./data/goog.csv # com a base original

python ./src/ta_forecasting/lstm/lstm_experiment.py ./data/goog_ta.csv # com a base de indicadores

python ./src/ta_forecasting/lstm/lstm_experiment.py ./data/goog_and_ta.csv # com a base original + indicadores
