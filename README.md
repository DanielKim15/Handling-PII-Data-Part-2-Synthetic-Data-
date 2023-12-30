# Handling-PII-Data-Part-2-Synthetic-Data-

<br />

A continuation from the first part of handling PII data, in this part I'll be discussing about using a technique called synthetic data where we can generate fake data that follows similar properties of the original data (same summary statistics, similar mean and median, distribution is similar). (Goal: Use the ECF Task 4 data and then choose a few columns, run the flowr package for synthetic data and run two test to ensure it worked (the flowr package evaluation and the histogram/distrubution test)

<br /> 

## What is Synthetic Data?
Synthetic data is artificially generated to mimic real-world data. It is based on statistical analysis of real data. They can be useful for augmenting small real datasets for training machine learning models, running simulations when real data is not available or very minimal, testing systems and models without exposing sensitive real data, sharing datasets while preserving privacy. <br>

The key advantage of synthetic data is it looks and behaves like real data but does not contain private information. It also lets you generate large datasets even with a small sample of real data.

<br />

