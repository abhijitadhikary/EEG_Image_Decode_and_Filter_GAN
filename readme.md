## Running Instructions ##
1. copy the 'uci_eeg_images_train_within.mat', 'uci_eeg_images_test_within.mat' and 'uci_eeg_images_validation_within.mat' files to data/mat folder.

2. run `./src/python main.py` to train/test/validate the model from the root directory.

3. model hyperparameters are set in utils.py get_args() method

4. Output images are saved in the output folder

5. To visualize using tensorboard run `tensorboard --logdir runs --port 6006` from the root directory