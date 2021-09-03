# hello-world-snorkel
Repository used to explore the [Snorkel](https://www.snorkel.org/) Python 
package which helps improve training data for build models.

From the documentation:  

    Snorkel is a system for programmatically building and managing training 
    datasets without manual labeling. In Snorkel, users can develop large training 
    datasets in hours or days rather than hand-labeling them over weeks or months.
    
    Snorkel currently exposes three key programmatic operations:
    
    Labeling data, e.g., using heuristic rules or distant supervision techniques
    
    Transforming data, e.g., rotating or stretching images to perform data augmentation
    
    Slicing data into different critical subsets for monitoring or targeted improvement
    
    Snorkel then automatically models, cleans, and integrates the resulting training 
    data using novel, theoretically-grounded techniques.


# Setup

This setup assumes Python 3 is installed and that you are using MacOS.

Create environment:  
`python3 -m venv snorkel-venv`

Activate the virtual environment:  
`source snorkel-venv/bin/activate`

To deactivate the environment:  
`deactivate`

Install dependent libraries using `pip`  

If a requirements file is provided, then you can install by running the 
following:  
`pip install -r requirements.txt`

TODO: export final requiremnts.txt when tutorial is completed

# Tutorials

## Data Labeling

I will be following the 
[Snorkel Intro Tutorial: Data Labeling](https://www.snorkel.org/use-cases/01-spam-tutorial) 
which uses the 
[YouTube comments dataset](https://archive.ics.uci.edu/ml/datasets/YouTube+Spam+Collection) 
to label whether a comment is `SPAM` or not (aka `HAM`).

**NOTE**: I copied the `utils.py` and `download_data.sh` files from 
[snorkel-tutorials](https://github.com/snorkel-team/snorkel-tutorials/tree/master/spam)
repo.

The main data labeling code for this tutorial is in the 
`./spam/tutorial-data-labeling.ipynb` jupyter notebook.

### Labeling Functions (LFs)

LFs are heuristics used to label data. There are several approaches to 
create these functions such as keyword search, pattern matching, third-party models, 
domain knowledge, and crowd sourcing the labels.

The recommended approach for developing LFs from the documentation 
is as follows:
1. Evaluate data to identify approaches for labeling the data
2. Write LFs based on (1)
3. Evaluate the performance of (2) from the training data
4. Update/improve LFs based on (3)
5. Repeat steps (3) and (4) until an ideal accuracy or coverage is reached

Note that LFs can create conflicting labels.

The labeling functions for this tutorial can be found in the 
`SpamLabelingFunctions.py`

### Keyword LFs
LFs could be as simple as keyword lookups. Instead of having to generate a new
labeling function each time a different keyword is used, Snorkel allows you to
create a template function and use the `LabelingFunction` class to generate 
multiple LFs from the template function. 

See lines 63-104 in the `SpamLabelingFunctions.py` for examples.

### Preprocessors
Preprocess data which is then used by LFs to improve the labeling accuracy.

The preprocessing functions for this tutorial can be found in the 
`SpamPreprocessors.py` which uses a third-party package to apply sentiment analysis.

You can also build complex preprocessors using spaCy (i.e. labels based on POS).
Snorkel makes this easy by providing a prebuilt LF decorator 
(`nlp_labeling_function()`). See the `has_person` LF in the 
`SpamLabelingFunctions.py` for an example.

### LFs' Performance
Several metrics are calculated to help evaluate the performance of LFs: 
* **Polarity**: Set of unique labels excluding abstains (-1) values for each LF
* **Coverage**: For each LF, the fraction of the data set with each label
* **Overlaps**: Fraction of the data set where an LF and at least one other have the same labels
* **Conflicts**: Fraction of the data set where an LF and at least one other have labels that disagree
* **Correct**: Given the ground truth, the number of correctly labeled data points per LF
* **Incorrect**: Given the ground truth, the number of incorrectly labeled data points per LF
* **Empirical Accuracy**: Given the ground truth, the overall accuracy of the LF

### Combining LFs with Label Model
Having several LFs can create conflicts when labeling data points. 
Snorkel provides a labeling model (`LabelModel`) which can apply different 
aggregation (i.e. majority voting) rules to create the final label.

Majority voting is a simple aggregation logic but it will provide misleading 
results if there is a lot of overlap and correlations between LFs. A better
approach is to use a labeling model which will weigh LFs differently.

Note that these models don't require the ground-truth and are simply using the
output of the LFs. Because of this, these label models could be used as a classifier
but will likely fail with new data if new features are required.
A better approach is to train a more generalized classifier. 

For examples on using label models, see the `tutorial-data-labeling.ipynb` and
the *Combining Labeling Function Outputs with the Label Model* section.

### Train a Classifier
We can now use the output of the label models to train a classifier. In the 
tutorial, we'll build a simple logistic regression model using Scikit-Learn.

The labeling models will produce a probability for the labels but
Scikit-Learn requires a distinct value. You can use `probs_to_pred` helper method
to convert this. Note that there isn't a threshold parameter to only classify
probabilities above a certain threshold. Instead, you'll need to use the `tol` 
parameter which is the minimum distance between probabilities and have the 
`tie_break_policy` be "abstain". For example, if you only want to label
probabilities above 0.8 for a binary classification problem, then `tol` will 
be 0.6 (0.8 - 0.2).

See the `tutorial-data-labeling.ipynb` and the *Training a Classifier* section
to see the implementation.

## Data Augmentation

I will be following the 
[Snorkel Intro Tutorial: Data Augmentation](https://www.snorkel.org/use-cases/02-spam-data-augmentation-tutorial) 
which uses the 
[YouTube comments dataset](https://archive.ics.uci.edu/ml/datasets/YouTube+Spam+Collection) 
to label whether a comment is `SPAM` or not (aka `HAM`).

The main data labeling code for this tutorial is in the 
`./spam/tutorial-data-augmentation.ipynb` jupyter notebook.

### Transformation Functions (TFs)

TFs are used to create additional data points with a class label. This is 
helpful when the training data is limited and more data points are required 
for a specific class or to capture more diverse and realistic data points which
may not be present in the training data.

For example, we can create additional text by taking text in the training data
and replacing words with synonyms. 

Like LFs, TFs also accept `Preprocessor` objects.

The TFs for this tutorial can be found in the 
`SpamTransformationFunctions.py`. Note that I did refactor the code for the POS
synonym replacement TFs. It would be nice to quickly build similar TFs by following
the same design as the keyword LFs but this isn't fully supported. You can implement
lambda mapper functions but you can only pass text as a parameter and no additional
arguments (since there is no `resources` argument). This would be a nice future
enhancement to make building similar TFs more efficient.

### Applying TFs

Use a `Policy` to define the sequence of applying TFs to data points. This includes
applying a random uniform policy (via `RandomPolicy`) or a given distribution 
(via `MeanFieldPolicy`).

After the TFs are applied and augmented data is created, you can now build models
with this expansive training set!

## Data Slicing

I will be following the 
[Snorkel Intro Tutorial: Data Slicing](https://www.snorkel.org/use-cases/03-spam-data-slicing-tutorial) 
which uses the 
[YouTube comments dataset](https://archive.ics.uci.edu/ml/datasets/YouTube+Spam+Collection) 
to label whether a comment is `SPAM` or not (aka `HAM`).

The main data slicing code for this tutorial is in the 
`./spam/tutorial-data-slicing.ipynb` jupyter notebook.

### Slicing Functions (SFs)
SFs output different slices of a data by applying binary masks to indicate
whether a data point is part of a slice or not. This is useful to evaluate 
specific segments of a data set that are more crucial to a problem (e.g. SPAM 
that has links to malicious websites). 

Similar to LFs and TFs, SFs also accepts `Preprocessor` objects.

The SFs for this tutorial can be found in the 
`SpamSlicingFunctions.py`

### Slice Performance
Using a trained model and the ground truth labels, you can evaluate the overall
performance of the model and per slice using the `Scorer` class.

### Improving Slice Performance
Through the `SliceAwareClassifier`, we can improve slice performance by turning
slices into tasks and training a multitask model via a multi-layer perceptron
(MLP) in Pytorch. 

Another approach is to over/under sample certain slices but this might not be 
feasible as the data and the number of slices scale.

See the **Improving Slice Performance** in the Jupyter notebook on how to apply
`SliceAwareClassifier`

# General Thoughts

#### Pros:  
1. General wrapper of functionality that provides consistency and quick 
evaluation of multiple LFs, TFs, and SFs.
2. The SFs are useful from an ethical AI perspective because you can evaluate
performance of different segments that might have complex rules. Also the ability
to improve slice performance using a multitask neural network is beneficial. 

#### Cons:  
1. Automating the building of multiple, similar TFs isn't possible because it
doesn't have the same structure as LFs and SFs. This would be a nice feature
to build TFs quickly and keep consistency with LFs and SFs. 
2. LFs and SFs seem very similar and it makes me wonder if different classes are
actually required. However, I haven't given this enough thought or looked into 
the code to see if this is possible so this may be more of a gut reaction and 
through time and use the distinction will make more sense. 

#### Overall:  
Snorkel is a useful library that provides consistency and efficiency when 
improving data labels and evaluating performance of different data segments. I also
really like how it relates to the ethical AI field and I could see this package
 growing more in this area and helping make ethical AI techniques more available
 (e.g. adding fairness features)

# Additional Resources

# TODOs
[ ] Run linter check: `pycodestyle`  
[ ] Run linter check: `pydocstyle`