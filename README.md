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


