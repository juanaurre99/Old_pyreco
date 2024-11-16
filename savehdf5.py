import h5py
import numpy as np

def save_model(model, filepath):
    
    def recursively_save_to_group(h5file, obj, path=''):

        # define a set of known problematic attributes
        #problematic_attributes = {'metrics_fun'}  # Add more problematic attribute names as needed
        
        # iterate through all attributes of the object
        for attr_name in dir(obj):

            if attr_name.startswith('_'):
                continue  # skip private attributes

            #if attr_name in problematic_attributes:
            #    print(f"Skipping problematic attribute '{attr_name}'.")
            #    continue  # skip known problematic attributes

            attr_value = getattr(obj, attr_name)

            if callable(attr_value):
                continue  # skip methods

            # construct the new path for the attribute
            new_path = f"{path}/{attr_name}".strip('/')

            try:
                # check the type of the attribute and save accordingly
                if isinstance(attr_value, (str, int, float, bool)):
                    h5file.attrs[new_path] = attr_value

                elif isinstance(attr_value, np.ndarray):
                    h5file.create_dataset(new_path, data=attr_value)

                elif isinstance(attr_value, (list, tuple)):
                    # convert lists and tuples to numpy arrays before saving
                    h5file.create_dataset(new_path, data=np.array(attr_value))

                elif isinstance(attr_value, dict):
                    # recursively save dictionary entries

                    for key, value in attr_value.items():
                        dict_path = f"{new_path}/{key}"

                        if isinstance(value, (str, int, float, bool, np.ndarray)):
                            h5file.create_dataset(dict_path, data=value)

                        else:
                            recursively_save_to_group(grp, value, key)

                else:
                    # if the attribute is an object, recursively save its attributes
                    grp = h5file.create_group(new_path)
                    recursively_save_to_group(grp, attr_value)

            except Exception as e:
                # log the error and skip this attribute
                print(f"Skipping attribute at path '{new_path}' due to error: {e}")

    # create the HDF5 file
    with h5py.File(filepath, 'w') as h5file:
        # recursively save model attributes
        recursively_save_to_group(h5file, model)
