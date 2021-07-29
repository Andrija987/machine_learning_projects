# Predicting health costs with regression

#%%
# Importing moduls
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns

# %%
# Importing data
dataset  = pd.read_csv('C:\Andrija_particija\Rad\Python_AI\Data_set\DNN_Regression\insurance.csv')
dataset.tail()

# %%
# Checking the data for bad values
dataset.isna().sum()

# %%
# Converting Categoric data to numerical values
# Spliting the 'sex' into two categoris
sex = dataset.pop('sex')
dataset['female'] = (sex == 'female')*1.0
dataset['male'] = (sex == 'male')*1.0
# Looking at the data set
dataset.head()

# %%
# Converting the smoker category
# yes = 1, no = 0
dataset['smoker'] = (dataset['smoker'] == 'yes')*1.0
# Checking the data
dataset.tail()

# %%
# Converting the category 'region'
categories = dataset['region'].unique()
print(categories)
region = dataset.pop('region')
for reg in categories:
  dataset[reg] = (region == reg)*1.0
dataset.tail()

# %%
# Split the data set to train and test data set in ratio 80 / 20
train_dataset = dataset.sample(frac=0.8, random_state=25)
test_dataset = dataset.drop(train_dataset.index)
# Checking the data set split ratio
print(train_dataset.shape)
print(test_dataset.shape)
# Checking the data set
print(train_dataset.head())
print(test_dataset.head())

# %%
# Ploting some of the data
sns.pairplot(train_dataset[['age', 'bmi','children','smoker', 'expenses', 'female', 'male']], diag_kind='kde')

# %%
# Look at the data statistics
train_stats = train_dataset.describe()
train_stats.pop('expenses')
train_stats = train_stats.transpose()
train_stats

# %%
# Poping the labels from the training and test data set
train_labels = train_dataset.pop('expenses') 
test_labels = test_dataset.pop('expenses')

# %%
# Checking the data sets and labels now
print(train_dataset.head())
print(test_dataset.head())
print(train_labels.head(), train_labels.shape)
print(test_labels.head(), test_labels.shape)

# %%
# Normalizing the data
def norm(x):
  return (x - train_stats['mean']) / train_stats['std']
normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)
normed_test_data.head()

# %%
normed_train_data.head()

# %%
# Building a model
def bulid_model():
  model = keras.Sequential([
      layers.Dense(64, activation=tf.nn.relu, input_shape = [len(train_dataset.keys())]),
      layers.Dense(64, activation=tf.nn.relu),
      layers.Dense(1)
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.001)

  model.compile(loss='mse',
                optimizer = optimizer,
                metrics = ['mae', 'mse'])
  
  return model

model = bulid_model()
model.summary()

# %%
# Test to see if the model produce some results
example_batch = normed_train_data[:10]
example_results = model.predict(example_batch)
example_results

# %%
# Training the model
EPOCHS = 1000

class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')

history = model.fit(
    normed_train_data, train_labels,
    epochs = EPOCHS, validation_split = 0.2, verbose = 0,
    callbacks = [PrintDot()]
)

# %%
# Checking the results
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()

# Ploting data
def plot_history(history):
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch

  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error [MPG]')
  plt.plot(hist['epoch'], hist['mae'], label='Train Error')
  plt.plot(hist['epoch'], hist['val_mae'], label='Val Error')
  plt.legend()
  #plt.ylim(1000,5000)

  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Square Error [$MPG^2$]')
  plt.plot(hist['epoch'], hist['mse'], label='Train Error')
  plt.plot(hist['epoch'], hist['val_mse'], label='Val Error')
  plt.legend()
  #plt.ylim([0,1e8])

plot_history(history)

# %%
# Trying to improve the model
model = bulid_model()

# Stoping criteria
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=100)

history = model.fit(normed_train_data, train_labels, epochs=EPOCHS,
                    validation_split=0.2, verbose=0, callbacks = [early_stop, PrintDot()])

# %%
plot_history(history)

# %%
# Checking the results again
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()

# %%
# Test model by checking how well the model generalizes using the test set.
loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=2)

print("Testing set Mean Abs Error: {:5.2f} expenses".format(mae))

if mae < 3500:
  print("The model is aceptable")
else:
  print("The Mean Abs Error must be less than 3500. Keep trying.")

# Plot predictions.
test_predictions = model.predict(normed_test_data).flatten()

a = plt.axes(aspect='equal')
plt.scatter(test_labels, test_predictions)
plt.xlabel('True values (expenses)')
plt.ylabel('Predictions (expenses)')
lims = [0, 50000]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims,lims)


# %%
# Small script for seeing predicted and true value
predicted_values = model.predict(normed_test_data).flatten()

def pred_values():
    select_column = int(input('Select column (0, 267): '))
    print(test_dataset.iloc[select_column])
    print('Predicted value:')
    print(predicted_values[select_column])
    print('True value:')
    print(test_labels.iloc[select_column])

# %%
pred_values()
