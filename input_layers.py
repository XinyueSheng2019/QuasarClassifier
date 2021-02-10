from tensorflow.keras import backend as K
import tensorflow.keras as keras
from input import *
from train import *

model = keras.models.load_model('test/lstm/group/[\'g\']/non_GP/[32, 32]/s/model_g_ep_50_bs_128.h5')
# config, path = parser_config(path='configs.txt')
# test_path = '../data/processed/unbalanced/test_set.csv'
# # data = pd.read_csv(test_path)

# X_test, Y_test = load_test_data(test_path, seed = config['seed'], features = config['features'], set_format = config['format'], preprocess = config['processed'], group_size = config['group_size'], sequence_length = config['group_num'], set_GPR = config['set_GPR']) 

# X_test, Y_test= cut_test_data(X_test, Y_test, cut_fraction=config['cut_fraction'], set_format=config['format'], features=config['features'])


inp = model.input                                           # input placeholder
outputs = [layer.output for layer in model.layers]          # all layer outputs
functors = [K.function([inp, K.learning_phase()], [out]) for out in outputs]    # evaluation functions

# Testing
# test = np.random.random(input_shape)[np.newaxis,...]
# layer_outs = [func([test, 1.]) for func in functors]
# print(layer_outs)
