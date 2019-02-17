# Conditional Audio Generative Adversarial Network

My dissertation project for MSc Intelligent Technologies at Birkbeck College, University of London.

## Research Area

Generative Adversarial Networks (GANs) are a relatively recent neural network architecture that are currently an area of much research.
These architectures pit two neural networks, a generator and a discriminator, into a competitive game where each learns and adapts in a race to beat the other.
For example, a generator may learn to output images of a cat, whereas a discriminator will learn to identify pictures of cats; at the beginning neither is particularly successful, but over training time, each may improve so that eventually the generator can produce convincing pictures of cats.

## Problem

GAN research initially focussed on image generation, but more recently some research has been directed towards generation of audio samples.
Another area of research has been the design of 'conditional' GANs; these are network architectures that allow the user to specify the type of output they would like the GAN to produce (returning to our example of pictures of cats, a conditional GANs would allow the user to specify whether to create a picture of a Ginger Tom or a white Persian).

The proposal for this project was to create a GAN that produces 'conditional' audio samples; from a small set of audio recordings of spoken words 'zero' and 'one', attempt to design a GAN that would output either word on demand.
This was at the time (as far as I could tell from a literature review) a combination of two cutting edge research strands that had never been before published (if even attempted).

## Experiments

The data used was a subset of Google's 'Speech Commands Zero Through Nine' dataset, specifically focussing on the words 'zero' and 'one' as these were enough to demonstrate whether the experiment was successful, and would also minimise the amount of data used (and thus computatational cost) in the experiment.
These data files were then downsampled from the original high sample rate again to minimise the cost of the experiment.

Experiments of GANs then proceeded with the testing of a baseline unconditioned audio GAN, and two conditioned models using different methods of conditioning: concatenation and auxiliary classifier.
The concatenation method of conditioning required the combination of extra 0 and 1 channel layers onto the data tensor that was entered into the generator to determine which sample should be created, and extra outputs in the discriminator that would allow it to discriminate between different samples.
The auxiliary classifier method used a second classifier that tested whether the sample created was of the expected type, and fed into the loss function that modified the generator.

Variations of these two models were also tried with different loss functions and with the concatenation method also attempting a novel method of entering the real data sample directly into a channel, rather than appending a channel of 1s to determine the sample type.

## Results

The initial finding of the experiment was the difficulty in producing a stable audio GAN when diverging from the original model that had been published; this experiement required the size of the GAN to be reduced, and for it to handle data of a lower size, and these changes had the effect of requiring a completely different set of hyperparameters.

It was found that the concatenation method of producing a conditioned audio GANs was unlikely to be the simplest method to reach a functioning model; finding a set of hyperparameters that would allow this model to work was not possible for this experiment due to all values (even extreme ones) having little impact on the quality of generation.
Instead, it was found the auxiliary classifier method was more suitable and while it did not produce results that sounded particularly 'word-like' (within the limitations of this experiment) it did show signs of progress towards an improvement in audio quality.

## Reflection

With more time (and a better understanding than I had when setting out on this project) I would have focussed further on the problem of creating a working downsized version of an audio GANs, and tried to understand better the reasons why hyperparameters needed changing when downsizing.
The results of this research would have allowed me more insight into the factors that would be effecting the successful design of the conditioned models.

I also think that the adjustment I made to the concatenation method of conditioning, where the audio sample was directly placed into the relevant channel of the tensor, rather than a layer of 1s being placed there, could be tested on more primitive experiments with conditioning GANs (for example, with the MNIST handwritten digits dataset) to check whether this does improve on any previous methods.

## Tooling

The code for these experiments was written in Python, TensorFlow and Numpy with some other Python packages also used for minor functionality.
The experiments were run on AWS EC2 instances.

## Further Reading

The final report can be viewed here:
https://github.com/James-G-Hill/Final-Project-Intelligent-Technologies-Conditional-Audio-GANs/blob/master/Report/Final/PROJ_HillJ.pdf
