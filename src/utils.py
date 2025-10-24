import numbers
import numpy as np

def is_number(x):
    return isinstance(x,(numbers.Number, np.number))