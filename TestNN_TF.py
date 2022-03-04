## Imports
from tensorflow.keras.models import load_model
import numpy as np
# eigene Imports
import metrics_TF
import tensorflow as tf

#-----------------------------------------------------------------------------------------------------------------------
# Reihenfolge Gesten
    # 1: 3_fingers_l
    # 2: 3_fingers_r
    # 3: index_finger_l
    # 4: index_finger_r
    # 5: L_l
    # 6: L_r
    # 7: none
    # 8: peace_l
    # 9: peace_r
    # 10: rock_l
    # 11: rock_r 
    # 12: surf_l
    # 13: surf_r
    # 14: thumbs_sidedown_l
    # 15: thumbs_sidedown_r
    # 16: thumbs_side_l
    # 17: thumbs_side_r
    # 18: thumbs_up_l
    # 19: thumbs_up_r

#-----------------------------------------------------------------------------------------------------------------------

## Modell und Testdaten laden
# 1_ElmosBrightness_SplitShuffle:
#model = load_model("/home/bhtcsuperuser/Katrin/GestureClassification_ToF/Model1_ElmosBrightness/TrainedModels/1_ElmosBrightness_SplitShuffle.h5")

# a = tf.saved_model.load('./TrainedModels/18_ElmosBrightnhessTF_Mix5P15_newds_7', tags=None, options=None)

model = tf.keras.models.load_model('./TrainedModels/18_ElmosBrightnhessTF_Mix5P15_newds_7')
model.summary()

# 1_ElmosBrightness_SplitZeit:
# model = load_model('./TrainedModelsTests/A1_ElmosBrightness_Mix5P15_newds_16.h5')

test = np.load('./NumpyArrays/Elmos_Mix5P15_Test_newds.npy', allow_pickle=True)
#------------------------------------------------

## Parameter
img_size = 32
n_classes = 19
#mean = 0.4519       # MITTELWERT: MUSS NEU BESTIMMT WERDEN
#std_der = 0.2764    # STANDARDABWEICHUNG: MUSS NEU BESTIMMT WERDEN
class_names = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
               '11', '12', '13', '14', '15', '16', '17', '18', '19']

#-----------------------------------------------------------------------------------------------------------------------

## Metriken berechnen
# Daten
#   Daten[0] => Brightness
#   Daten[1] => Distance
#   Daten[2] => Label
# Testdaten

x_test = np.array([i[0] for i in test], 'float32').reshape(-1, img_size, img_size)
norm = np.max(x_test)
x_test /= norm
x_test = tf.cast(tf.constant(x_test), dtype=tf.float32) # In Tensor umwandeln
print(x_test.ndim, x_test.shape) # 3 (2078, 32, 32)

# X_train /= 65535 #255      # Pixelwerte zwischen Null und Eins (rescale)
# X_train = tf.cast(tf.constant(X_train), dtype=tf.float32) # In Tensor umwandeln
# print(X_train.ndim, X_train.shape) # 3 (6232, 32, 32)

#x_test -= mean
#x_test /= std_der

y_test = np.array([i[2] for i in test], 'float32').reshape(-1, n_classes)

y_test = tf.constant(y_test)# In Tensor umwandeln
print(y_test.ndim, y_test.shape) # 2 (2078, 19)

# Loss und Accuracy
acc_test = model.evaluate(x_test, y_test)
print("Hallo2")
acc_test = np.array(acc_test)
acc_test = acc_test.round(4)

print('')
print('Test       [loss, accuracy]', acc_test)
print('')

# Confusion Matrix
#y_pred = model.predict_classes(x_test, batch_size=1, verbose=0)
y_pred = np.argmax(model.predict(x_test), axis=-1)
#y_pred = model.predict(x_test, axis=-1)
cnf_matrix = metrics_TF.GetConfusionMatrix(metrics_TF.GetLabel(y_test), y_pred, n_classes)
metrics_TF.PlotConfusionMatrix(cnf_matrix, class_names)
#-----------------------------------------------------------------------------------------------------------------------
