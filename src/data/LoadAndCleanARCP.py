# -*- coding: utf-8 -*-
from pathlib import Path
import numpy as np
import pandas as pd
from src.data.dataframe_utilities import StandardizeColumnNames

def LoadARCP():
    """ load local copy of ARCP data

    returns: pandas dataframe
    """

    path = Path.cwd().parent.parent
    input_loc =  path /'Data'/ 'Master Project Data'
    arc_path = input_loc / 'ARC Preparedness Data.csv'
    arc = pd.read_csv(arc_path, 
                    dtype = {'GEOID': str, 'Zip': str})

    return arc

def CleanARCP(arc):
    """ clean arcp data

    returns: pandas dataframe
    """

    arc = StandardizeColumnNames(arc)
    arc.dropna(inplace = True)
    
    # trim geoid leading saftey marks 
    arc['geoid'] = arc['geoid'].str[2:]

    return arc


def LoadAndCleanARCP():
    """ load and clean local copy of ARCP data

    returns: pandas dataframe
    """
    
    arc = LoadARCP()
    arc = CleanARCP(arc)

    return arc


if __name__ == '__main__':
   
    ARCP = LoadAndCleanARCP()