import os 
import sys
import dill

import numpy as np 
import pandas as pd 
from src.exception import CustomException

def save_object(file_path, obj):
    try:
        with open(file_path, 'wb') as file:
            dill.dump(obj, file)
    except Exception as e:
        raise CustomException(e, sys)
    
    except Exception as e:  
        raise CustomException(e, sys)