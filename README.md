# Conditional Audio Generative Adversarial Network

My final project for an MSc Intelligent Technologies at Birkbeck, University of London.

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



The final report can be viewed here:
https://github.com/James-G-Hill/Final-Project-Intelligent-Technologies-Conditional-Audio-GANs/blob/master/Report/Final/PROJ_HillJ.pdf
