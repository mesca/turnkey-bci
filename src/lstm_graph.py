from keras.models import model_from_json
from keras.utils import plot_model

model = open('../data/output/final_across_lstm_architecture.json').read()
model = model_from_json(model)
plot_model(model, to_file='../data/output/lstm_architecture.png', show_shapes=True, show_layer_names=True)