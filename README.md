# Bone Age Regression with Deep Learning

This is my code for the "Bone Age Regression" competition of I2A2 - Goiânia. I learned a lot building this pipeline from scratch and was able to experiment with different model architectures and optimizers. This was my first image regression model and it was very nice seeing my theoretic knowledge working in practice.

This competition was based on RSNA's Bone Age challenge, in which given hand X-ray images, the model should predict the patient's bone age.

<img src="docs/ex1.png" width="200" height="250"> <img src="docs/ex2.png" width="200" height="250"> <img src="docs/ex3.png" width="200" height="250">

My final solution used a ResNet50 architecture, a Rectified Adam optimizer and geometric data augmentations. This model achieved a Mean Average Error of 13.2 after 20 epochs of training, which I believe could be improved given more training time and a better preprocessing pipeline (using object detection to segment the hands and normalizing hand rotation, for example). I didn't saved all the hyperparameters I experimented with (neither their results) but you'll find in the code the ones I used for my last submission.

I used tensorboard to log the training curves and tqdm to track progress. I also used [FCMNotifier](https://github.com/bryanlincoln/fcm-notifier), a tool I made to send logs as notifications to my phone.

## Requirements

See requirements.txt.

## Usage

-   Download the requirements with `pip install -r requirements.txt`
-   Download the dataset and sample submission with `sh download_data.sh`. You may need to log in with your Kaggle account in order to do it.
-   Train the ResNet50 model with `python boneage.py`
-   Try different models and hyperparameters editing the train script or use the `boneage.ipynb` notebook to do it interactively.

## Credits

I used the vision models already implemented in torchvision with small changes (you can actually try other torchvision models by only adding `in_channels` parameter to generalize the number of input channels, since torchvision models work with RGB images).
