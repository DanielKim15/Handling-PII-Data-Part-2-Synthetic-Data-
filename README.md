# Handling-PII-Data-Part-2-Synthetic-Data-

<br />

A continuation from the first part of handling PII data, in this part I'll be discussing about using a technique called synthetic data where we can generate fake data that follows similar properties of the original data. 

<br /> 

## What is Synthetic Data?
Synthetic data is artificially generated to mimic real-world data. It is based on statistical analysis of real data. They can be useful for augmenting small real datasets for training machine learning models, running simulations when real data is not available or very minimal, testing systems and models without exposing sensitive real data, sharing datasets while preserving privacy. This is widely used within Industries that have a lot of sensitive personal data like the financial centers and healthcare industry. <br>

The key advantage of synthetic data is it looks and behaves like real data but does not contain private information. It also lets you generate large datasets even with a small sample of real data.

<br />

## What are the 5 syntehsizers in the SDV package?
<br />
For this demonstration I am going to use a python package called the Synthetic Data Vault or SDV to create synthetic data in a tabular format (aka csv/excel). Within the package there is 5 different types of models that can be used to generate data, and they each have their own strength and weakness to consider.

1. **Fast ML Preset:**
   - Produces the fastest speed to create synthetic data
   - Great for first time SDV users, can be used to get quickly started with the syntethic data

``` python

synthesizer = SingleTablePreset(
    metadata, # required
    name='FAST_ML', # required
    locales=['en_US', 'en_CA']
)

```

<br />


2. **GaussianCopulaSynthesizer:**
   - Uses classical, statistical methods to train a model and generate synthetic data
   - The most customizable one out of the 5 models
   - Can be used to work with normal distribution, binomial distribution, or other kinds depending on the data
``` python

synthesizer = GaussianCopulaSynthesizer(
    metadata, # required
    enforce_min_max_values=True,
    enforce_rounding=False,
    numerical_distributions={
        'amenities_fee': 'beta',
        'checkin_date': 'uniform'
    },
    default_distribution='norm'
)

```
<br />

3. **CTGANSynthesizer:**
   - The CTGAN Synthesizer uses GAN-based, deep learning methods to train a model and generate synthetic data
   - GANs (Generative Adversarial Networks) are a type of unsupervised machine learning model, consisting of a generator that creates data and a discriminator that differentiates between real and generated data, improving iteratively through adversarial training until the generated data is indistinguishable from real data.

``` python
synthesizer = CTGANSynthesizer(
    metadata, # required
    enforce_rounding=False,
    epochs=500,
    verbose=True
)
```
<br />

4. **TVAESynthesizer:**
   - The TVAE Synthesizer uses a variational autoencoder (VAE)-based, neural network techniques to train a model and generate synthetic data.
   - Variational Autoencoders (VAEs) are a type of generative model in machine learning that use a neural network architecture to encode input data into a latent space and then reconstruct it back to the original data, with a focus on learning the distribution of data for generating new, similar data.
   - For more info: https://www.cs.columbia.edu/~zemel/Class/Nndl-2021/files/lec13.pdf
     
``` python
synthesizer = TVAESynthesizer(
    metadata, # required
    enforce_min_max_values=True,
    enforce_rounding=False,
    epochs=500
)
```
<br />

5. **CopulaGANSynthesizer:**
   - The Copula GAN Synthesizer uses a mix classic, statistical methods and GAN-based deep learning methods to train a model and generate synthetic data.
   - Experimental, results may vary

``` python
synthesizer = CopulaGANSynthesizer(
    metadata, # required
    enforce_min_max_values=True,
    enforce_rounding=False,
    numerical_distributions={
        'amenities_fee': 'beta',
        'checkin_date': 'uniform'
    },
    epochs=500,
    verbose=True
)
```
<br />
