# VAE + GAN Minichallenge

In this minichallenge, we pre-train a VAE that is then used to train a GAN.

## VAE

The VAE (Variational Autoencoder) model encodes image data into latent space using convolutional layers. It uses 2D convolutional layers to progressively reduce the spatial dimensions of the input. Then, a decoder part is used with 2d deconvolutional layers to increase the spatial dimensions back to the original size. The interesting thing is that the model learns to generate images of the same style as the training data from random noise. The following picture shows the model architecture:

![1688743926040](image/README/1688743926040.png)

Mathematically, we want to maximize the likelihood of the data x (in the figure) given a parameterized probability distribution $p_\theta(x) = p(x|\theta)$. This distribution is usually chosen to be a Gaussian $N(x|\mu, \phi)$ which is parameterized by $\mu$ and $\phi$ respectively. This refers to the middle section in the image.

The probability of $p_\theta(x)$ can be found by integrating over the joint probability: $p_\theta(x) = \int_z p_\theta(x, z)dz$. "z" is the latent space encoded by the "probabilistic encoder" in the image. It is usually taken to be a finite-dimensional vector of real numbers. The term can be rewritten to $p_\theta(x) = \int_z p_\theta(x|z)p(z)dz$ with the chain rule ($P(A \cap B) = P(B | A) * P(A)$). This can be modeled with the VAE model and its architecture.  $p_\theta(x|z)$ is a Gaussian distribution. $p_\theta(x)$ is a mixture of Gaussian distributions. We can also say that $p_\theta(z)$ is the prior and $p_\theta(x|z)$ is the likelihood. Using the bayesian approach, we can define $p_\theta(z|x)$ as the posterior (z given input data x). Unfortunately, computation of $p_\theta(x)$ is expensive and it is therefore necessary to approximate the posterior distribution.

The approximation is the core of the encoder model in the VAE. The posterior distribution is approximated by the encoder which calcualtes an approximation $q_\phi(z|x)$. We can also call it a "probabilistic encoder". Similarly, the "probabilistic decoder" decodes the approximated z and calculates $p_\theta(x|z)$.

### Loss

The idea is to optimize the generative model parameters $\theta$ to reduce the reconstruction error between the input and the output (x &rarr; x'). We also want to optimize $\phi$ (approximated parameters to the true parameters $\theta$) to make the approximated posterior as close as possible to the true posterior. As reconstruction loss, MSE and cross entropy are often used. A good distance loss for two distributions (approximated posterior and true posterior) is the Kullback-Leibler divergence $D_{KL}(q_\phi(z|x) || p_\theta(z|x))$. We don't know the true posterior, we just make the model train on the Kullback-Leibler divergence approximating some kind of posterior function that is assumed to have mean = 0 and standard deviation = 1. This goal makes the approximated posterior by the VAE simpler. This also simplifies the KL divergence formula.

The full loss therefore uses not only a "reconstruction term", which tries to minimalize the reconstruction error between x &rarr; x', but also the $D_{KL}$. The full loss is:

$$
loss = ||x-x'||^2 + KL[N(\mu_x, \sigma_x), N(0, 1))] = ||x-d(z))||^2 + KL[N(\mu_x, \sigma_x), N(0, 1))]
$$

#### More on KL regularisation

Like in all machine learning models, there is a risk of overfitting. The KL regularisation term in the loss makes it so the mean and standard deviation to be 1 and the mean to be 0. This prevents encoded distributions to be too far apart from each others. It also prevents distributions to be too far apart in latent space. Two points in latent space should give two similar contents when decoded (continuity) and a point sampled from latent space should give meaningful content once decoded (completeness). Continuity means that small changes in the latent space correspond to small changes in the data space. This is desirable because it means that the model can generate smooth transitions between different data points. Completeness means that every point in the latent space corresponds to a meaningful data point. This is desirable because it means that the model can generate a wide variety of different data points.

By doing the regularisation like that, it helps to prevent overfitting by discouraging the model from learning to represent each data point with a unique point in the latent space (which would correspond to a "weird" distribution overfitted on the data).

### Decoding from latent space

First a latent representation z is sampled from the prior distribution $p(z)$. Then, the data x is sampled from the conditional likelihood distribution $p(x|z)$. Since we're working with sampling which is a stochastic process that involves randomness, we need a trick called "reparameterization" for backpropagation to work. The reparameterization trick is a clever solution to this problem. Instead of sampling from the distribution directly, we do the following:

$z[i] = mu[i] + sigma[i] * eps[i]$

Where mu and sigma are output vectors from the encoder and eps is a set of values that are sampled from a gaussian distribution. "eps" is ignored in the backpropagation. Now mu and sigma are differentiable. This operation is equivalent to sampling from a Gaussian distribution with mean mu and sigma, because it scales and shifts the standard normal distribution to have the desired mean and standard deviation from the two vectors.

### Matrix dimensions:

Input = image dimensions

encoder output = mu and sigma vectors both with dimensions [batch_size, latent_dim]

eps = [batch_size, latent_dim]

z = [batch_size, latent_dim]

Each element z[i] is computed as mu[i] + sigma[i] * eps[i]


## GAN

## Data
