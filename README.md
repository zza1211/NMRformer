# NMRformer
use ‘pip install -r requirements.txt’ to create nmrformer environment

### Input
### The uploaded files include:
1. csv file, the first column is the chemical shift, and the second column is the 1H NMR spectrum.
2. txt file, one data per row, representing the chemical shift of the peak in the spectrum.
##### When you input peaks that cannot be recognized or noisy peaks into the model, it will affect the output of the model, so please try to filter the peaks in the input spectral peak list as much as possible.

In NMRformer.ipynb, you can load the model and input data, resulting in an output.csv file that contains the corresponding metabolites and probabilities for each peak. You can use the data in example_data for testing.

