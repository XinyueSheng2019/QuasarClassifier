__author__ = "Xinyue Sheng"
__copyright__ = "Copyright (C) 2020 Xinyue Sheng"
__license__ = "Public Domain"
__version__ = "1.0"

from numpy.random import seed
# seed(1)
import tensorflow as tf
from tensorflow.keras import backend as K
# tensorflow.random.set_seed(1)
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.layers import Embedding, Activation, Masking
from tensorflow.keras.layers import LSTM, Bidirectional, GRU, SimpleRNN, TimeDistributed
from tensorflow.keras.callbacks import EarlyStopping
from input import *
from plots import *
import numpy as np
import time
import csv
import os

print(tf.config.list_physical_devices('GPU'))

'''
This file is used for building and training the classifier and saving the results and models.
'''

def get_layer_output(model, x, index=-1):
    """
    get the computing result output of any layer you want, default the last layer.
    :param model: primary model
    :param x: input of primary model( x of model.predict([x])[0])
    :param index: index of target layer, i.e., layer[23]
    :return: result
    """
    layer = K.function([model.input], [model.layers[index].output])
    return layer([x])[0]


class LossHistory(tensorflow.keras.callbacks.Callback):
    '''
    This class is used for recording the loss, accuracy, AUC, f1_score value during training.
    '''
    def on_train_begin(self, logs={}):
        self.epoch_loss = []
        self.epoch_accuracy = []
        self.epoch_AUC = []
        self.epoch_f1_score = []

        self.epoch_val_loss = []
        self.epoch_val_accuracy = []
        self.epoch_val_AUC = []
        self.epoch_val_f1_score = []

        self.batch_losses = []
        self.batch_accuracy = []
        self.batch_AUC = []
        self.batch_f1_score = []
      

    def on_epoch_end(self, batch, logs={}):
        self.epoch_loss.append(logs.get('loss'))
        self.epoch_accuracy.append(logs.get('accuracy'))
        self.epoch_AUC.append(logs.get('AUC'))
        self.epoch_f1_score.append(logs.get('f1_score'))

        self.epoch_val_loss.append(logs.get('val_loss'))
        self.epoch_val_accuracy.append(logs.get('val_accuracy'))
        self.epoch_val_AUC.append(logs.get('val_AUC'))
        self.epoch_val_f1_score.append(logs.get('val_f1_score'))

    def on_batch_end(self, batch,logs={}):
        self.batch_losses.append(logs.get('loss'))
        self.batch_accuracy.append(logs.get('accuracy'))
        self.batch_AUC.append(logs.get('AUC'))
        self.batch_f1_score.append(logs.get('f1_score'))


      

def parser_config(path):
    '''
    parse the config file
    '''
    config = {}
    f = open(path, 'r').readlines()
    n = 1
    while n<len(f):
        if ':' in f[n]:
            x = f[n][:-1].split(':')
            x[1] = x[1].strip()
            config[x[0]] = x[1]
            if x[0] == 'path':
                if x[1] == '':
                    config[x[0]] = '../data/processed/balanced/final_v1.csv'
                else:
                    config[x[0]] = x[1]
            elif x[0] == 'features':
                if x[1]!= '':
                    config[x[0]] = x[1].split(',')
                else:
                    print("WARNING: empty feature list, automatically set the feature as g")
                    config[x[0]] = ['g']
            elif x[0] == 'format':
                if x[1]!= '':
                    config[x[0]] = x[1]
                else:
                    config[x[0]] = 'group'
            elif x[0] == 'processed':
                if x[1]!= '':
                    if x[1][0] == 's':
                        config[x[0]] = 's'
                    elif x[1][0] == 'n':
                        config[x[0]] = 'n'
                    elif  x[1][0] == 'd':
                        config[x[0]] = 'd'
                    else:
                        print("WARNING: wrong preprocess input, automatically set as standardization")
                        config[x[0]] = 's'
                else:
                    print("WARNING: empty preprocess input, automatically set as standardization")
                    config[x[0]] = 's'
            elif x[0] == 'hidden_layers':
                config[x[0]] = [int(l) for l in x[1].strip('[').strip(']').split(',')]
            elif x[0] == 'metrics':
                config[x[0]] = [l for l in x[1].split(',') if l != 'f1_score']
                if 'f1_score' in x[1].split(','):
                    config[x[0]].append(f1_score)
            else:
                if x[1].lower() == 'true':
                    config[x[0]] = True
                if x[1].lower() == 'false':
                    config[x[0]] = False
                if x[1] == ''or x[1] == 'None':
                    config[x[0]] = None
            if x[0] == 'seed' or x[0] == 'group_size' or x[0] == 'group_num':
                if config[x[0]] != None:
                    config[x[0]] = int(config[x[0]])
               
        n +=1
    print(config)
    # generate a file tree
    path = config['save_path']
    if os.path.exists(path) == False:
        os.mkdir(os.getcwd()+'/'+path)
    if os.path.exists(path+'/'+config['rnn_type'].lower()) == False: 
        os.mkdir(os.getcwd()+'/'+path+'/'+config['rnn_type'].lower())
    path = path+'/'+config['rnn_type'].lower()
    if os.path.exists(path+'/'+config['format']) == False:
        os.mkdir(os.getcwd()+'/'+path+'/'+config['format'])
    path = path+'/'+config['format']
    if os.path.exists(path+'/'+str(config['features'])) == False:
        os.mkdir(os.getcwd()+'/'+path+'/'+str(config['features']))
    path = path + '/'+str(config['features'])
    if os.path.exists(path+'/GP') == False and config['set_GPR'] == True:
        os.mkdir(os.getcwd()+'/'+path+'/GP')
    elif os.path.exists(path+'/non_GP') == False and config['set_GPR'] == False:
        os.mkdir(os.getcwd()+'/'+path+'/non_GP')
    if config['set_GPR'] == True:
        path = path + '/GP'
    else:
        path = path + '/non_GP'
    print(config['hidden_layers'])
    if os.path.exists(path+'/'+str(config['hidden_layers'])) == False:
        os.mkdir(os.getcwd()+'/'+path+'/'+str(config['hidden_layers']))
    path = path + '/'+str(config['hidden_layers'])
    if os.path.exists(path+'/'+config['processed']) == False:
        os.mkdir(os.getcwd()+'/'+path+'/'+config['processed'])
    path = path + '/'+config['processed']

    print(path)

    return config, path

def binary_focal_loss(gamma=2, alpha=0.25):
    '''
    Binary form of focal loss.
    
    focal_loss(p_t) = -alpha_t * (1 - p_t)**gamma * log(p_t)
        where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.
    References:
        https://arxiv.org/pdf/1708.02002.pdf
    Usage:
     model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    '''
    alpha = tf.constant(alpha, dtype=tf.float32)
    gamma = tf.constant(gamma, dtype=tf.float32)

    def binary_focal_loss_fixed(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        alpha_t = y_true*alpha + (K.ones_like(y_true)-y_true)*(1-alpha)
    
        p_t = y_true*y_pred + (K.ones_like(y_true)-y_true)*(K.ones_like(y_true)-y_pred) + K.epsilon()
        focal_loss = - alpha_t * K.pow((K.ones_like(y_true)-p_t),gamma) * K.log(p_t)
        return K.mean(focal_loss)
    return binary_focal_loss_fixed


# def f1_score(y_true, y_pred):
#     '''
#     calculate F1 value
#     '''

#     TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#     label_true = K.sum(K.round(K.clip(y_pred, 0, 1)))
#     pred_true = K.sum(K.round(K.clip(y_true, 0, 1)))

#     precision = TP/pred_true
#     recall = TP/label_true
#     f1_score = 2 * (precision * recall)/(precision + recall)

#     return f1_score


# def precision(y_true,y_pred): 
#     TP=tf.reduce_sum(y_true*tf.round(y_pred))
#     TN=tf.reduce_sum((1-y_true)*(1-tf.round(y_pred)))
#     FP=tf.reduce_sum((1-y_true)*tf.round(y_pred))
#     FN=tf.reduce_sum(y_true*(1-tf.round(y_pred)))
#     precision=TP/(TP+FP)
#     return precision
 
# def recall(y_true,y_pred): 
#     TP=tf.reduce_sum(y_true*tf.round(y_pred))
#     TN=tf.reduce_sum((1-y_true)*(1-tf.round(y_pred)))
#     FP=tf.reduce_sum((1-y_true)*tf.round(y_pred))
#     FN=tf.reduce_sum(y_true*(1-tf.round(y_pred)))
#     recall=TP/(TP+FN)
#     return recall


def build(time_sequence = int, input_dim = int, num_class = 2, hidden_layer = [256,256,256], rnn_type = 'LSTM', dropout = 0.05,  activation = 'tanh', print_model = False, predict_trend = False):
    '''
    This is the function for building a classifer model.
    Inputs:
    - time_sequence: the length of the sequence
    - input_dim: the number of dimensions for each vector
    - num_class: the number of classes/labels
    - hidden_layer: the number of layers in RNN architecture
    - rnn_type: the type of RNN, for example, LSTM, GRU, SimpleRNN
    - dropout: the fraction of objects dropped before being fed into the next layer to avoid overfitting
    - activation: applies an activation function to an output
    - print_model: True means the RNN architecture will be printed
    - predict_trend: whether use the prediction function. ***This function is being developed***
    Returns:
    - model: the designed model with fixed layers
    '''

    # tensorflow.keras.initializers.Constant(value=0)

    if rnn_type.lower() == 'lstm':
        RNN = LSTM
    elif rnn_type.lower() == 'gru':
        RNN = GRU
    else:
        RNN = SimpleRNN

    input_data = Input(shape=(time_sequence, input_dim))

    x = Masking(mask_value=0.)(input_data)

    n = 0
    while n<len(hidden_layer):
        x = RNN(units=hidden_layer[n], activation = activation, return_sequences=(n<len(hidden_layer)-1))(x) 
        x = Dropout(dropout)(x)
        n +=1
    y = Dense(units = num_class, activation = 'softmax')(x)


    model = Model([input_data],[y])

    if print_model == True:
        print(model.summary())


    return model


def train(model, path, config, batch_size = int, epochs = int, X_train = list, Y_train = list, X_test = list, Y_test = list, features = list,
    optimizer = 'Adam', lr = None, decay = False, loss_function = 'binary_crossentropy', metrics = ['accuracy','AUC']):
    '''
    train the classifier model.
    Inputs:
    - model: the designed model generated from the build function
    - batch_size: the number of sequences fed into the layer for each time
    - epoches: the times for the input data being processed
    - X_train/X_test: the train/test data with 3 dimensions: 
        objects, the number of observations in an objects, the vector in 1 observation
    - Y_train/Y_test: the label of train/test data. 
    = features: the targeted feature list
    - optimizer: the optimization method for the loss function
    - lr: learning rate
    - decay: whether the learning rate will decrease with the increasing nunmber of epochs
    - loss_function: the function used to measure the degree to which the predicted value f(x) of the model is inconsistent with the true value Y
    - metrics: the metrics during training for testing the performance of the classifier
    Return:
    - model: the trained model

    '''

    d_value = 0.0
    if lr != None:
        lr = float(lr)
    else:
        lr = 0.001
    if decay == True:
        d_value = lr/epochs
    else:
        d_value = 0


    opt = 'Adam'
    if optimizer.lower == 'adam':
        opt = tf.keras.optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=d_value)
    elif optimizer.lower == 'sgd':
        opt = tf.keras.optimizers.SGD(lr=lr, momentum=0.0, nesterov=False, name='SGD', decay=d_value)

    model.compile(loss = loss_function, optimizer = opt, metrics = metrics)

    history = LossHistory()

    print("Start training...")

    start_time = time.time()

    class_weight = {0: sum(x[1] for x in Y_train)/len(Y_train), 1: sum(x[0] for x in Y_train)/len(Y_train)}

    earlystop = EarlyStopping(monitor = 'val_loss', patience = 3)
    model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test, Y_test), verbose=1, class_weight = class_weight, callbacks=[history,earlystop])

    sum_time = time.time() - start_time

    print("--- %s seconds ---" % sum_time)

    #plot graphs: loss, accuracy, AUC

    plot_two_graph(history.epoch_val_loss, history.epoch_loss, path+'/loss_epoch.png', 'num of epochs', 'loss', 'validation_loss','training_loss')
    plot_one_graph(history.batch_losses, path+'/loss_batch.png', 'num of batches', 'loss')

    plot_two_graph(history.epoch_val_accuracy, history.epoch_accuracy, path+'/acc_epoch.png', 'num of epochs','accuracy', 'validation_acc', 'training_acc')
    plot_one_graph(history.batch_accuracy, path+'/acc_batch.png', 'num of batches', 'accuracy')

    plot_two_graph(history.epoch_val_AUC, history.epoch_AUC, path+'/AUC_epoch.png','num of epochs', 'AUC','validation_AUC','training_AUC')
    plot_one_graph(history.batch_AUC, path+'/AUC_batch.png','num of batches', 'AUC')

    scores = model.evaluate(X_test, Y_test, verbose=1)

    # record all loss, acc, AUC information
    with open(path+'/record.txt','w') as f:
        f.write('loss_epoch_train: ')
        f.write(str(history.epoch_loss))
        f.write('\nloss_epoch_valid: ')
        f.write(str(history.epoch_val_loss))
        f.write('\nloss_batch_train: ')
        f.write(str(history.batch_losses))
        f.write('\nacc_epoch_train: ')
        f.write(str(history.epoch_accuracy))
        f.write('\nacc_epoch_valid: ')
        f.write(str(history.epoch_val_accuracy))
        f.write('\nacc_batch_train: ')
        f.write(str(history.batch_accuracy))
        f.write('\nauc_epoch_train: ')
        f.write(str(history.epoch_AUC))
        f.write('\nauc_epoch_valid: ')
        f.write(str(history.epoch_val_AUC))
        f.write('\nauc_batch_train: ')
        f.write(str(history.batch_AUC))
    f.close()

    y_pred = model.predict(X_test)

    actual_value = [x[1] for x in Y_test]
    predict_value = [x[1] for x in y_pred]
    auc = plot_ROC(actual_value, predict_value, path+'/val_')

    y_pred = [np.argmax(x) for x in y_pred]
    y_true = [np.argmax(x) for x in Y_test]

    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plot_confusion_matrix(cm_normalized, path+'/val_confusion_matrix', title='confusion matrix',classes=['non-QSO', 'QSO'])

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    recall = tp/(tp+fn)
    specificity = tn/(fp+tn)
    precision = tp/(tp+fp)
    f1_score = 2*precision*recall/(precision+recall)

    key_list = ['features','format','processed','set_GPR','rnn_type','hidden_layers','dropout','batch_size','num_epochs','test_fraction','optimizer','learning_rate','decay']

    with open(config['save_path']+'/train_results.csv','a+',newline='') as csvfile:
        writer = csv.writer(csvfile)
        with open(config['save_path']+'/train_results.csv','r',newline='') as f:
            reader = csv.reader(f)
            if not [row for row in reader]:
                writer.writerow(key_list+['train_time']+model.metrics_names + ['val_recall','val_specificity','val_precision','val_f1_score']+ ['TN','FP','FN','TP'])
            writer.writerow([config[x] for x in key_list] + [str(sum_time)] + [scores[i] for i in range(len(model.metrics_names))] +[recall, specificity, precision, f1_score]+ [tn, fp, fn, tp])




    return model

def predict(model, X_test, Y_test, path, config):
    '''
    This function is used to test the classifier's performance with a confusion metrix graph and ROC graph.
    '''

    y_pred = model.predict(X_test)
    actual_value = [x[1] for x in Y_test]
    predict_value = [x[1] for x in y_pred]
    auc = plot_ROC(actual_value, predict_value, path+'/test_')
    y_pred = [np.argmax(x) for x in y_pred]
    y_true = [np.argmax(x) for x in Y_test]
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plot_confusion_matrix(cm_normalized, path+'/test_confusion_matrix', title='confusion matrix',classes=['non-QSO', 'QSO'])

    title = ','.join(config['features'])
    key_list = ['features','format','processed','set_GPR','rnn_type','hidden_layers','dropout','batch_size','num_epochs','test_fraction','optimizer','learning_rate','decay']

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    recall = tp/(tp+fn)
    specificity = tn/(fp+tn)
    precision = tp/(tp+fp)
    f1_score = 2*precision*recall/(precision+recall)

    key_list = ['features','format','processed','set_GPR','rnn_type','hidden_layers','dropout','batch_size','num_epochs','test_fraction','optimizer','learning_rate','decay']

    with open(config['save_path']+'/test_results.csv','a+',newline='') as csvfile:
        writer = csv.writer(csvfile)
        with open(config['save_path']+'/test_results.csv','r',newline='') as f:
            reader = csv.reader(f)
            if not [row for row in reader]:
                writer.writerow(key_list+['AUC','recall','specificity','precision','f1_score']+ ['TN','FP','FN','TP'])
            writer.writerow([config[x] for x in key_list]+[auc, recall, specificity, precision, f1_score]+ [tn, fp, fn, tp])


if __name__ == '__main__':

    
    config,path = parser_config(path='configs.txt')

    (X_train, X_valid, Y_train, Y_valid), (length_train, length_test, time_sequence, input_dim, num_classes) = load_data(path = config['train_path'], test_fraction = float(config['test_fraction']), seed = config['seed'], features = config['features'], set_format = config['format'], preprocess = config['processed'], group_size = config['group_size'], sequence_length = config['group_num'],set_GPR = config['set_GPR'])
    model = build(time_sequence = time_sequence, input_dim = input_dim, num_class = num_classes, hidden_layer = config['hidden_layers'], rnn_type = config['rnn_type'], dropout = float(config['dropout']),  print_model = config['plot_model'])
    model = train(model, path, config, batch_size = int(config['batch_size']), epochs = int(config['num_epochs']), X_train = X_train, Y_train = Y_train, X_test = X_valid, Y_test = Y_valid, features = config['features'], optimizer = config['optimizer'],  lr = config['learning_rate'], decay = config['decay'], loss_function = 'binary_crossentropy', metrics = config['metrics'])

    X_test, Y_test = load_test_data(config['test_path'], seed = config['seed'], features = config['features'], set_format = config['format'], preprocess = config['processed'], group_size = config['group_size'], sequence_length = config['group_num'], set_GPR = config['set_GPR']) 
    X_test, Y_test= cut_test_data(X_test, Y_test, cut_fraction=config['cut_fraction'], set_format=config['format'], features=config['features'])

    predict(model, X_test, Y_test, path, config)
    
    title = ','.join(config['features'])    
    model.save(path+'/model_'+title+'_ep_'+str(config['num_epochs'])+'_bs_'+str(config['batch_size'])+'.h5')
    print("Save model to the disk.") 

    

    

