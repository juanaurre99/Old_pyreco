import numpy as np
import json

def save_model(model, output_file_path):

    def recursively_save_to_dict(obj, path, output_dict):
        # define problematic attribute
        problematic_attributes = {'metrics_fun'}  
        
        # iterate through all attributes of the object
        for attr_name in dir(obj):

            if attr_name.startswith('_'):
                continue  # skip private attributes

            if attr_name in problematic_attributes:
                print(f"Skipping problematic attribute '{attr_name}'.")
                continue  # skip known problematic attributes

            attr_value = getattr(obj, attr_name)

            if callable(attr_value):
                continue  # skip methods

            # construct the new path for the attribute
            new_path = f"{path}/{attr_name}".strip('/')

            try:
                # check the type of the attribute and save accordingly
                if isinstance(attr_value, (str, int, float, bool)):
                    output_dict[new_path] = attr_value

                elif isinstance(attr_value, np.ndarray):
                    output_dict[new_path] = attr_value.tolist()  # convert numpy array to list for JSON compatibility

                elif isinstance(attr_value, (list, tuple)):
                    output_dict[new_path] = attr_value

                elif isinstance(attr_value, dict):
                    output_dict[new_path] = {}  # initialize sub-dictionary

                    for key, value in attr_value.items():
                        dict_path = f"{new_path}/{key}"

                        if isinstance(value, (str, int, float, bool, np.ndarray)):
                            output_dict[new_path][key] = value if not isinstance(value, np.ndarray) else value.tolist()

                        else:
                            recursively_save_to_dict(value, dict_path, output_dict[new_path])

                else:
                    # if the attribute is an object, recursively save its attributes
                    output_dict[new_path] = {}  # initialize sub-dictionary for nested objects
                    recursively_save_to_dict(attr_value, new_path, output_dict[new_path])
                    
            except Exception as e:
                # log the error and skip this attribute
                print(f"Skipping attribute at path '{new_path}' due to error: {e}")

    output_dict = {}
    recursively_save_to_dict(model, '', output_dict)

    # create a filtered dictionary with only serializable values
    def safe_json_dump(value):
        try:
            return json.dumps(value)
        except (TypeError, OverflowError):
            return None

    # create a filtered dictionary with only serializable values
    filtered_dict = {}
    for key, value in output_dict.items():
        if safe_json_dump(value) is not None:
            filtered_dict[key] = value
        else:
            print(f"Value at '{key}' is not JSON serializable and will be skipped.")

    # save the filtered dictionary to JSON file
    with open(output_file_path, 'w') as json_file:
        json.dump(filtered_dict, json_file, indent=4)







