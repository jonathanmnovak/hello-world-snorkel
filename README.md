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
`./tutorial/tutorial-data-labeling.ipynb` jupyter notebook.

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

# General Thoughts

#### Pros:  

#### Cons:  

#### Overall:  

# Additional Resources