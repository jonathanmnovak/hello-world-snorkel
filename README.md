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

### LFs' Performance
Several metrics are calculated to help evaluate the performance of LFs: 
* **Polarity**: Set of unique labels excluding abstains (-1) values for each LF
* **Coverage**: For each LF, the fraction of the data set with each label
* **Overlaps**: Fraction of the data set where an LF and at least one other have the same labels
* **Conflicts**: Fraction of the data set where an LF and at least one other have labels that disagree
* **Correct**: Given the ground truth, the number of correctly labeled data points per LF
* **Incorrect**: Given the ground truth, the number of incorrectly labeled data points per LF
* **Empirical Accuracy**: Given the ground truth, the overall accuracy of the LF

### Preprocessors
Preprocess data which is then used by LFs to improve the labeling accuracy.

The preprocessing functions for this tutorial can be found in the 
`SpamPreprocessors.py` which uses a third-party package to apply sentiment analysis.

# General Thoughts

#### Pros:  

#### Cons:  

#### Overall:  

# Additional Resources