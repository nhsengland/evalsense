def deep_update(old_dict: dict, new_dict: dict) -> dict:
    """Recursively updates a dictionary with data from another dictionary.

    Args:
        old_dict (dict): The dictionary to update.
        new_dict (dict): The new dictionary data.
    """
    for key, value in new_dict.items():
        if isinstance(value, dict):
            old_dict[key] = deep_update(old_dict.get(key, {}), value)
        else:
            old_dict[key] = value
    return old_dict
