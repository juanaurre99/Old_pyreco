import h5py
import numpy as np


# save model to hdf5
def save_model(model, filepath):

    #check_file = open('./tests/save_func_files/hdf5_should_content.txt', 'w')

    def recursively_save_to_group(h5file, obj, path=''):
        # iterate through all attributes of the object
        for attr_name in dir(obj):
            if attr_name.startswith('_'):
                continue  # skip private attributes
            
            attr_value = getattr(obj, attr_name)
            if callable(attr_value):
                continue  # skip methods

            # construct the new path for the attribute
            new_path = f"{path}/{attr_name}".strip('/')
            
    #        check_file.write(f"{new_path}: {attr_value}\n")
            # check the type of the attribute and save accordingly
            #if isinstance(attr_value, (str, int, float, bool)):
            if isinstance(attr_value, (str, int, float, bool, list, tuple)):
    #            pass
                h5file.attrs[new_path] = attr_value
            elif isinstance(attr_value, np.ndarray):
    #            pass
                h5file.create_dataset(new_path, data=attr_value)
            elif isinstance(attr_value, (list, tuple)):
    #            pass
                # convert lists and tuples to numpy arrays before saving
                #h5file.create_dataset(new_path, data=np.array(attr_value))
                h5file.create_dataset(new_path, data=attr_value)
            elif isinstance(attr_value, dict):
    #            pass
                # cecursively save dictionary entries
                for key, value in attr_value.items():
                    dict_path = f"{new_path}/{key}"
                    if isinstance(value, (str, int, float, bool, np.ndarray)):
                        h5file.create_dataset(dict_path, data=value)
                    else:
                        recursively_save_to_group(h5file, value, dict_path)
            else:
                # if the attribute is an object, recursively save its attributes
                recursively_save_to_group(h5file, attr_value, new_path)

    # create the HDF5 file
    with h5py.File(filepath, 'w') as h5file:
        # recursively save model attributes
        recursively_save_to_group(h5file, model)
