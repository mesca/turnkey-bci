# Dependencies
import mne
import numpy as np
import pandas as pd
import keras
import h5py
import json
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Activation
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from mne.decoding import Vectorizer, Scaler

np.random.seed(0) # For reproducibility


class History(Callback):

	"""Keras callback to compute AUC and log history after each epoch."""

	def __init__(self, filepath):
		self.filepath = filepath

	def on_train_begin(self, logs={}):
		self.logs = {'loss': [], 'val_loss': [], 'val_auc': []}

	def on_epoch_end(self, epoch, logs={}):
		predictions = self.model.predict(self.validation_data[0])
		auc = roc_auc_score(self.validation_data[1], predictions)
		logs['val_auc'] = auc
		self.logs['loss'].append(logs.get('loss'))
		self.logs['val_loss'].append(logs.get('val_loss'))
		self.logs['val_auc'].append(auc)
		with open(self.filepath, 'w') as fp: json.dump(self.logs, fp)
		print('val_auc: %f' % auc)


class ResetState(Callback):

	"""Keras callback to reset states after each epoch."""

	def on_epoch_end(self, epoch, logs={}):
		self.model.reset_states()


class ERP:

	"""ERP tools

	Parameters
	----------
	input_dir : string
		The input filepath
	output_dir : string
		The output filepath.
	tmin : float
		The epoch start time, in seconds, relative to the onset stimulus.
		Defaults to -.1.
	tmax : float
		The epoch stop time, in seconds, relative to the onset stimulus
		Defaults to .9.
	sblen : int
		The epoch baseline length, in samples.
		Defaults to 10.
	selen : int
		The number of samples per epoch.
		Defaults to 100.
	sfreq : float
		The sample frequency, in Hz.
		Defaults to 100.

	"""

	def __init__(self, input_dir, output_dir, tmin=-.1, tmax=.9, sblen=10, selen=100, sfreq=100.):

		self.df = None
		self.input_dir = input_dir
		self.output_dir = output_dir
		self.tmin = tmin
		self.tmax = tmax
		self.sblen = sblen
		self.selen = selen
		self.sfreq=sfreq


	def load(self, filename):

		"""Load a h5 file from disk as a Pandas dataframe.

		Parameters
		----------
		filename : string
			The input filename, relative to the input directory.

		"""

		self.df = pd.read_hdf(self.input_dir + filename, 'df')


	def save(self, filename, df=None):

		"""Save a dataframe to disk as a h5 file.

		Parameters
		----------
		filename : string
			The output filename, relative to the output directory.
		df : instance of pandas.core.frame.DataFrame
			The dataframe to save.

		"""

		df.to_hdf(self.output_dir + filename, 'df', mode='w')


	def filter(self, slicers, df=None):

		"""Filter a dataframe with index slicers.

		Examples
  	--------
  	X = filter(df, {'condition': 'A', 'participant': slice(2, 3)})
  	X = filter(df, {'condition': 'C', 'participant': [0, 5, 9]})
  	X = filter(df, {'condition': 'M', 'participant': 2, epoch: 42})
  	X = filter(df, {'participant': 42})

		Parameters
		----------
		df : instance of pandas.core.frame.DataFrame
			The dataframe to filter.
		slice : dict of slices
			The slices.

		Returns
		-------
		df : instance of pandas.core.frame.DataFrame
			The filtered dataframe.

		Notes
		-----
		See:
			https://docs.python.org/3.6/library/functions.html#slice
			http://pandas.pydata.org/pandas-docs/stable/advanced.html#using-slicers
			https://stackoverflow.com/questions/24126542/pandas-multi-index-slices-for-level-names#24126676

		"""

		if df is None: df = self.df

		indexer = [ slice(None) ] * len(df.index.levels)
		for n, idx in slicers.items():
			indexer[df.index.names.index(n)] = idx
		return df.loc[tuple(indexer), :]


	def raw(self, df=None, verbose=None):

		"""Build a MNE Raw object.

		Parameters
		----------
		df : instance of pandas.core.frame.DataFrame
			The dataframe to convert. If None, the dataframe loaded with self.load() will be used.
			Defaults to None.
		verbose : bool, string, int or None
			The verbosity level, as detailed at: https://www.martinos.org/mne/stable/generated/mne.set_log_level.html.
			Defaults to None.

		Returns
		-------
		raw : instance of mne.io.RawArray
			The MNE Raw object.

		"""

		if df is None: df = self.df

		stim = np.zeros((1, len(df)), dtype=int) # Create an empty stim channel
		targets = self.filter({'time': pd.Timedelta(0)}, df)['target'] # Filter on 0 deltas to get targets
		targets = targets.astype(int) + 1 # Convert to int and add 1 so targets are non-zero
		events = range(self.sblen, len(targets) * self.selen, self.selen) # Event indices
		for i, event in enumerate(events):
			stim[0][event] = targets[i] # Populate stim channel

		df = df.drop(['target'], axis=1) # Drop the target column
		df *= 1e-6  # Convert from uVolts to Volts

		samples = np.concatenate((df.transpose().values, stim))

		channel_names = df.columns.tolist() + ['STI 014'] # Get channels
		channel_types = ['eog'] + ['eeg'] * (len(channel_names) - 2) + ['stim'] # Channels types
		montage = mne.channels.read_montage('standard_1005')

		# http://martinos.org/mne/stable/generated/mne.create_info.html
		info = mne.create_info(channel_names, self.sfreq, ch_types=channel_types, montage=montage)

		# http://martinos.org/mne/stable/generated/mne.io.RawArray.html
		raw = mne.io.RawArray(samples, info, verbose=verbose)
		raw.set_eeg_reference([])

		return raw


	def get_Xy(self, slicers=None, df=None, dropbad=True, downsample=None, crop=None, scaling=None, verbose=None):

		"""Get training data and target values.

		Parameters
		----------
		slicers : dict of slices
			The slicers on which the data is filtered. See self.filter(). If None, the full dataset will be used.
			Defaults to None.
		df : instance of pandas.core.frame.DataFrame
			The dataframe to use as input. If None, the dataframe loaded with self.load() will be used.
			Defaults to None.
		dropbad : bool
			Whether artifacts will be rejected or not.
			Defaults to True.
		downsample : int or None.
			The resampling rate. If None, the data will not be downsampled.
			Defaults to None.
		crop : tuple of float (tmin, tmax) or None
			The crop time interval from epochs object, in seconds. If None, epochs will not be cropped.
			Defaults to None.
		scaling : dict, string or None
			The scaling method to be applied to data channel wise. See: http://martinos.org/mne/stable/generated/mne.decoding.Scaler.html
			Defaults to None.
		verbose : bool, string, int or None
			The verbosity level, as detailed at: https://www.martinos.org/mne/stable/generated/mne.set_log_level.html.
			Defaults to None.

		Returns
		-------
		X : instance of numpy.ndarray
			The training data.
		y : instance of numpy.ndarray
			The target values.
		epochs : instance of mne.epochs.Epochs
			The MNE Epochs object.

		"""

		if df is None: df = self.df
		if slicers: df = self.filter(slicers, df)
		raw = self.raw(df, verbose=verbose)
		raw.filter(0.5, 40, method='iir') # bandpass filter
		events = mne.find_events(raw, verbose=verbose)
		event_id = {'distractor': 1, 'target': 2}
		# Reject epochs were the signal exceeds 100uV in EEG channels or 200uV in the EOG channel
		reject = {'eeg': 100e-6, 'eog': 200e-6} if dropbad else None
		# See: http://martinos.org/mne/stable/generated/mne.Epochs.html
		epochs = mne.Epochs(raw, events, event_id=event_id, tmin=self.tmin, tmax=self.tmax, baseline=(self.tmin, 0), reject=reject, verbose=verbose)
		epochs.load_data()
		if dropbad: epochs.drop_bad()
		if downsample: epochs.resample(downsample, npad='auto')
		if crop: epochs.crop(*crop)
		epochs.pick_types(eeg=True)
		X = epochs.get_data()
		y = epochs.events[:, -1] == 2 # binary events
		# See: http://martinos.org/mne/stable/generated/mne.decoding.Scaler.html
		X = Scaler(epochs.info, scaling).fit_transform(X, y)
		return X, y, epochs


	def train_test_lda(self, X_train, y_train, X_test, y_test):

		"""Regulated LDA

		Parameters
		----------
		X_train: instance of numpy.ndarray
			The training data.
		y_train: instance of numpy.ndarray
			The training target values.
		X_test: instance of numpy.ndarray
			The testing data.
		y_test: instance of numpy.ndarray
			The testing target values.

		Returns
		-------
		model : instance of sklearn.pipeline.Pipeline
			The final model.
		auc : float
			The AUC score.
		"""

		model = make_pipeline(Vectorizer(), LDA(shrinkage='auto', solver='eigen'))
		model.fit(X_train, y_train)
		auc = roc_auc_score(y_test, model.predict(X_test))
		return model, auc


	def across_lda(self, prefix=''):

		"""Train Regulated LDA models accross all participants.

		Parameters
		----------
		prefix: string, optional
			The output files prefix.

		"""

		conditions = self.df.index.levels[0].tolist()
		participants = self.df.index.levels[1].tolist()
		options = { 'dropbad': False, 'downsample': 20, 'crop': (0.1, 0.8), 'scaling': None, 'verbose': False }

		keys = []
		values = []
		for c_train in conditions:
		    for c_test in conditions:
		        for p in participants:
		            index = (c_train, c_test, p)
		            print(index)
		            keys.append(index)
		            p_train = list(participants)
		            p_train.remove(p)
		            X_train, Y_train, epochs_train = self.get_Xy({'condition': c_train, 'participant': p_train}, **options)
		            X_test, Y_test, epochs_test = self.get_Xy({'condition': c_test, 'participant': p}, **options)
		            model, auc = self.train_test_lda(X_train, Y_train, X_test, Y_test)
		            values.append(auc)

		keys = pd.MultiIndex.from_tuples(keys, names=['c_train', 'c_test', 'p'])
		aucs = pd.Series(values, index=keys)
		aucs.to_hdf(self.output_dir + prefix + 'across_lda.h5', 'df', mode='w') # Save to disk


	def train_test_lstm(self, X_train, y_train, X_test, y_test, prefix='', implementation=0):

		"""Stateful LSTM network

		Parameters
		----------
		X_train: instance of numpy.ndarray
			The training data.
		y_train: instance of numpy.ndarray
			The training target values.
		X_test: instance of numpy.ndarray
			The testing data.
		y_test: instance of numpy.ndarray
			The testing target values.
		prefix: string, optional
			The output files prefix.
		implementation: int
			The Keras RNN implementation:
			- 0: CPU optimized
			- 1: RAM optimized
			- 2: GPU optimized
			Defaults to 0.

		Returns
		-------
		history : dict
			The training history.
		auc :	dict
			The last model score, the best loss model score, the best AUC model score.

		"""

		# Transpose to shape (samples, times, features)
		X_train = np.transpose(X_train, (0, 2, 1))
		X_test = np.transpose(X_test, (0, 2, 1))

		# Compensate for class imbalance
		y_len = float(len(y_train))
		class_weight = dict((i, (y_len - (y_train == i).sum()) / y_len) for i in np.unique(y_train))

		# Callbacks
		history = History(self.output_dir + prefix + 'history.json') # Performs AUC on validation data
		stop = EarlyStopping(monitor='val_loss', patience=50, mode='min') # early stopping
		checkpoint_loss = ModelCheckpoint(self.output_dir + prefix + 'model_best_loss.h5', monitor='val_loss', save_best_only=True, mode='min') # checkpoint
		checkpoint_auc = ModelCheckpoint(self.output_dir + prefix + 'model_best_auc.h5', monitor='val_auc', save_best_only=True, mode='max') # checkpoint
		reset = ResetState() # Reset state at the end of each epoch

		# Training params
		epochs = 1000
		batch_size = 32
		validation_split = 0.25

		# In a stateful network, input size must be a multiple of batch size
		valid_size = int(X_train.shape[0] * validation_split / batch_size) * batch_size
		train_size = int(X_train.shape[0] * (1 - validation_split) / batch_size) * batch_size
		X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=valid_size, train_size=train_size, stratify=y_train)

		# Network architecture
		model = Sequential()
		model.add(LSTM(
			128,
			batch_size=batch_size,
			input_shape=(X_train.shape[1], X_train.shape[2]),
			stateful=True,
			dropout=0.4,
			recurrent_dropout=0.4,
			implementation=implementation,
			return_sequences=True
		))
		model.add(LSTM(64, dropout=0.4, recurrent_dropout=0.4, implementation=implementation, return_sequences=True))
		model.add(LSTM(32, dropout=0.4, recurrent_dropout=0.4, implementation=implementation))
		model.add(Dense(1))
		model.add(Activation('sigmoid'))

		# Compilation
		model.compile(optimizer='adam', loss='binary_crossentropy')

		# Save model architecture
		architecture = model.to_json()
		with open(self.output_dir + prefix + 'architecture.json', 'w') as handler:
			handler.write(architecture)

		# Print some info
		print(model.summary())

		# Fit
		logs = model.fit(
			X_train,
			y_train,
			shuffle=True,
			batch_size=batch_size,
			epochs=epochs,
			validation_data=(X_valid, y_valid),
			class_weight=class_weight,
			callbacks=[history, checkpoint_loss, checkpoint_auc, stop, reset],
			verbose=2
		)

		# Save last model
		model.save(self.output_dir + prefix + 'model_last.h5')

		# Adjust test set size
		# TODO: create new model from saved weights with a batch size of 1 for online prediction
		test_size = int(X_test.shape[0] / batch_size) * batch_size
		X_test = X_test[0:test_size]
		y_test = y_test[0:test_size]
		print('Testing on %d samples' % X_test.shape[0])

		# Compute AUC
		auc = {}
		auc['last'] = roc_auc_score(y_test, model.predict(X_test)) # Last model
		model = load_model(self.output_dir + prefix + 'model_best_loss.h5')
		auc['best_loss'] = roc_auc_score(y_test, model.predict(X_test)) # Best loss model
		model = load_model(self.output_dir + prefix + 'model_best_auc.h5')
		auc['best_auc'] = roc_auc_score(y_test, model.predict(X_test)) # Best AUC model

		# Save AUC
		with open(self.output_dir + prefix + 'results.json', 'w') as fp: json.dump(auc, fp)
		print(auc)

		return logs.history, auc


	def across_lstm(self, prefix='', implementation=0):

		"""Train LSTM models accross all participants.

		Parameters
		----------
		prefix: string, optional
				The output files prefix.
		implementation: int
			The Keras RNN implementation:
			- 0: CPU optimized
			- 1: RAM optimized
			- 2: GPU optimized
			Defaults to 0.

		"""

		conditions = self.df.index.levels[0].tolist()
		participants = self.df.index.levels[1].tolist()
		options = { 'dropbad': False, 'downsample': 20, 'crop': (0.1, 0.8), 'scaling': 'median', 'verbose': False }
		results = []
		prepend = prefix + 'across_lstm'

		for c_train in conditions:
		    for c_test in conditions:
		        for p in participants:
		            index = (c_train, c_test, p)
		            prefix = prepend + '_'  + '-'.join(map(str, index)) + '_'
		            print(index)
		            p_train = list(participants)
		            p_train.remove(p)
		            X_train, y_train, epochs_train = self.get_Xy({'condition': c_train, 'participant': p_train}, **options)
		            X_test, y_test, epochs_test = self.get_Xy({'condition': c_test, 'participant': p}, **options)
		            history, auc = self.train_test_lstm(X_train, y_train, X_test, y_test, prefix=prefix, implementation=implementation)
		            results.append({
		            	'c_train': c_train,
		            	'c_test': c_test,
		            	'p': p,
		            	'history': history,
		            	'auc': auc
		            })

		with open(self.output_dir + prepend + '.json', 'w') as fp: json.dump(results, fp) # Save to disk
