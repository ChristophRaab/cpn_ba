# Using RSLVQ as classifier layer in a CNN

This is the code belonging to the bachelors thesis.
Note that the code uses Python 3.
Python 2 might work, but no guarantees.

## Part I

`Part1.py` contains the tests for part 1.
It can run a single test, the best 10 hyperparameters found in the thesis or the batch tests.
Shuffling the dataset is also possible.
Run `Part1.py --help` to see all parameters.
The RSLVQ implementation used by it is found in `rslvq1.py`.

For it to work, the dog image dataset must be present (see [`CodeData/Instructions.md`](CodeData/Instructions.md)).

To run the comparison network execute `Part1 comparison.py`.
This network needs the dogbreed dataset as well.

## Part II

`Part2.py` contains the networks used in part 2 of the thesis.

Making it work is a bit more difficult:
First, `python3 setup.py build` must be executed in the `nnetcopy` folder (this needs cython).
This should create a build folder.
The libraries from that folder (`pool.so`/`pool.dll` and `conv.so`/`conv.dll`) need to be copied to `nnetcopy/nnet/convnet`.
After that `Part2.py` should be usable.

When starting the file without parameters, it will run the simple RSLVQ network on the MNIST dataset (supplied with this repository).
To see all available networks run `Part2.py --help`.
The RSLVQ implementation for it is found in `rslvq2.py`.
Note that due to the lower number of features the complex networks may not be used with the MNIST dataset.
To use the dogbreed dataset it needs to be correctly set up as described above.