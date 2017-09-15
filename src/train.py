import time
import erp

# Configuration
input_dir = '../data/input/'
output_dir = '../data/output/'
eeg_file = 'EEG.h5'
prefix = time.strftime('%Y%m%d-%H%M', time.gmtime()) + '_' # This will be prepended to saved files
implementation = 0 # RNN implementation: 0 for CPU, 1 for RAM, 2 for GPU

start = time.time()

# Init
run = erp.ERP(input_dir, output_dir)

# Load data
run.load(eeg_file)
print('Data loaded.')

# Build LDA models
run.across_lda(prefix=prefix)

# Build LSTM models
run.across_lstm(prefix=prefix, implementation=implementation)

stop = time.time()
elapsed = (stop - start) / 60
print('Elapsed: %f minutes.' % elapsed)