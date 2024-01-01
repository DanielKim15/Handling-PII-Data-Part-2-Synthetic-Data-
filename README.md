# Handling PII Data Part 2 Synthetic Data

<br />

A continuation from the first part of handling PII data, in this part I'll be discussing about using a technique called synthetic data where we can generate fake data that follows similar properties of the original data. Ideally the goal of this document is to educate the basics of synthetic data and provide a usable coding that the reader can send to their clients in case they face challenges in accessing the data due to beurocracy in PII data security. (Enough to where you'll be able to do high level analysis at least)

<br /> 

## What is Synthetic Data?
Synthetic data is artificially generated to mimic real-world data. It is based on statistical analysis of real data. They can be useful for augmenting small real datasets for training machine learning models, running simulations when real data is not available or very minimal, testing systems and models without exposing sensitive real data, sharing datasets while preserving privacy. This is widely used within Industries that have a lot of sensitive personal data like the financial centers and healthcare industry. <br>

The key advantage of synthetic data is it looks and behaves like real data but does not contain private information. It also lets you generate large datasets even with a small sample of real data.

<br />

## Real Life Example 
Continuing from Part 1 of the Handling PII Data, there was an issue my team faced with the project where we could not recieve the data in the beginning due to beurocracy issues (the lawyers from the client side were not informed or this project so they have to go through all the steps to ensure we're allowed to handle PII data). By the time we were able to obtain the entireity of the data, we only had a very short time to complete all the analysis we had planned to do before our contract was over. It greatly affected our analysis, so I'm hoping the synthetic data can be a way to avoid this issue.

<br />

## What are the 5 syntehsizers in the SDV package?
<br />
For this demonstration I am going to use a python package called the Synthetic Data Vault or SDV to create synthetic data in a tabular format (aka csv/excel). Within the package there is 5 different types of models that can be used to generate data, and they each have their own strength and weakness to consider.

<br />

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

## Preperation for the demonstration
Some context for the data we are using for the demonstration: I will be using the Emergency Connectivity Fund open source data Form 477, Covid relief pandemic program led by the FCC federal agency to ensure that schools and libraries around the nation continue to have access to technology such as computers and internet services by reimbursing them with funds. 

<br />

**The data is filtered by choosing only Form Versions that are current and choosing 7 columns from the dataset:**
   - Application Number (Identifier)
   - unique_key (Unique Identifier, Primary Key)
   - Total Funding Commitment Request Amount (Continous Variable)
   - Total Student Count (Continous Variable)
   - Monthly Recurring Unit Cost (Continous Variable)
   - Product Type (Categorical Variable)
   - Urban/ Rural Status (Binary Variable)

<br />

Another important change I made was creating a unique identifier that mixes with Funding Request Number, FRN Line Item ID, and One time Unit Cost. The unique identifier step is important because the package rely on ensuring each rows of data is unique and that that are no duplicates in the unique identifier when generating the data. If there are duplicates, you will recieve an error. If your variable is completly unique that would be ideal, but in case you do not have one be sure to create it by mixing with other variables to truly make it unique.


<br />
The last thing that's recommended before starting the test is to visualize the variables your going to synthesize in order to have a good reference point to use.
<br/>

Continuous Variables:
![image](https://github.com/DanielKim15/Handling-PII-Data-Part-2-Synthetic-Data-/assets/59937761/e9a6567b-df0a-49a5-b2d1-865bac55850f)

<br />


Categorical Variables:
![image](https://github.com/DanielKim15/Handling-PII-Data-Part-2-Synthetic-Data-/assets/59937761/4f467319-3710-40c1-abac-4d579ccaa852)

<br />

Note: The 95% quantile and above has been removed in order to get a closer look into the distrubtuion of the continous variables. The extreme skewedness beforehand made it very hard to read it.


<br />

## Demonstration

``` python
from sdv.single_table import TVAESynthesizer, GaussianCopulaSynthesizer, CTGANSynthesizer, CopulaGANSynthesizer
from sdv.lite import SingleTablePreset
from sdv.metadata import SingleTableMetadata
from sdv.evaluation.single_table import get_column_plot, evaluate_quality, get_column_pair_plot

def generate_synthetic_data(dataset, sdv_type, unique_id, num_samples):
    """
    Generates synthetic data using the specified SDV synthesizer.

    Args:
        dataset (pd.DataFrame): The dataset to base the synthetic data on.
        sdv_type (str): Type of SDV synthesizer to use ('TVAE', 'GaussianCopula', 'CopulaGAN', 'fast_ml', or 'CTGAN').
        unique_id (str): Column name to set as a primary key in the dataset.
        num_samples (int): Number of synthetic data rows to generate.

    Returns:
        pd.DataFrame: The generated synthetic dataset.
    """
    # Create metadata object and detect from dataframe
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data=dataset)
    
    # Update metadata for the unique key column
    metadata.update_column(column_name=unique_id, sdtype='id')
    metadata.set_primary_key(column_name=unique_id)
    
    # Validate metadata
    try:
        metadata.validate()
    except Exception as e:
        return f"Error: The metadata does not represent the structure and constraints of the data, try again. Details: {str(e)}"
    
    # Select the synthesizer based on sdv_type
    if sdv_type == 'TVAE':
        synthesizer = TVAESynthesizer(
                        metadata, # required
                        enforce_min_max_values=True,
                        enforce_rounding=True,
                        epochs=500) 
    elif sdv_type == 'GaussianCopula':
        synthesizer = GaussianCopulaSynthesizer(
                        metadata, 
                        enforce_min_max_values=True,
                        enforce_rounding=True,
                        #numerical_distributions={
                        #    'amenities_fee': 'beta',
                        #    'checkin_date': 'uniform'
                        #        },
                        #default_distribution='norm' #lets you determine what distrubution your working  with, read file for more information: https://docs.sdv.dev/sdv/single-table-data/modeling/synthesizers/gaussiancopulasynthesizer
                        )
    elif sdv_type == 'CTGAN':
        synthesizer = CTGANSynthesizer(
                        metadata, 
                        enforce_rounding=True,
                        epochs=500,
                        verbose=True)
    elif sdv_type == 'CopulaGAN':
        synthesizer = CopulaGANSynthesizer(
                        metadata, 
                        enforce_min_max_values=True,
                        enforce_rounding=True,
                     #   numerical_distributions={
                     #       'amenities_fee': 'beta',
                     #       'checkin_date': 'uniform'
                     #                       },
                        epochs=500,
                        verbose=True)
    elif sdv_type == 'fast_ml':
        synthesizer = SingleTablePreset(metadata, name='FAST_ML')
    else:
        return "Error: Invalid SDV synthesizer type. Choose 'TVAE', 'GaussianCopula', 'CopulaGAN', 'fast_ml', or 'CTGAN'."

    # Fit the synthesizer on the dataset
    synthesizer.fit(dataset)
    
    # Generate synthetic data
    synthetic_data = synthesizer.sample(num_rows=num_samples)
    
    return synthetic_data

# Example usage
# synthetic_dataset = generate_synthetic_data(ecf_data_sample_selected, 'TVAE', 'unique_key', 25000)

```




