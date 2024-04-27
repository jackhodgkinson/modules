# Module of functions for data manipulation.

## DATASETS 
### Function to name datasets given suffixes 
def dataset_namer(input_name, suffix, suffix2=''):        
    global string
    if suffix2 != '':
        string = f"{input_name}_{suffix}_{suffix2}"
    else:
        string = f"{input_name}_{suffix}"
        
    return string

## DICTIONARY
### Function to split dictionaries by specified split
def dict_split(dictionary, split):
    
    new_dict = {}
    
    for key, value in dictionary.items():
        if split in key:
            name = dataset_namer(split, get_var_name(dictionary))
            new_dict[key] = value
            globals()[name] = new_dict    
            
## VARIABLES
### Function to retrieve variable name as a string
def get_var_name(input_var):
    for name, var in globals().items():
        if var is input_var:
            return name
    