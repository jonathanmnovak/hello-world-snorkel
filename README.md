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

# Tutorials

### Data Labeling

I will be following the 
[Snorkel Intro Tutorial: Data Labeling](https://www.snorkel.org/use-cases/01-spam-tutorial) 
which uses the 
[YouTube comments dataset](https://archive.ics.uci.edu/ml/datasets/YouTube+Spam+Collection) 
to label whether a comment is `SPAM` or not (aka `HAM`).

**NOTE**: I copied the `utils.py` file from 
[snorkel-tutorials](https://github.com/snorkel-team/snorkel-tutorials/tree/master/spam)
repo.

Data labeling code for the tutorial is in the 
`./tutorial/tutorial-data-labeling.ipynb` jupyter notebook.



# General Thoughts

#### Pros:  

#### Cons:  

#### Overall:  

# Additional Resources