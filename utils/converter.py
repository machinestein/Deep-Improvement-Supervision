import json
def convert_and_merge_datasets(custom_list_file, original_arc_file, output_file):
    """
    Converts a custom dataset (list) to the standard ARC format (dict)
    and merges in the 'test' examples from the original ARC dataset.

    Args:
        custom_list_file (str): Path to the custom JSON dataset (a list of tasks).
        original_arc_file (str): Path to the original ARC JSON dataset (a dict).
        output_file (str): Path to save the final merged JSON dataset.
    """
    
    # --- 1. Load Custom Dataset (List) ---
    try:
        with open(custom_list_file, 'r') as f:
            custom_data_list = json.load(f)
        if not isinstance(custom_data_list, list):
             print(f"Error: Custom file '{custom_list_file}' is not a list.")
             return
    except FileNotFoundError:
        print(f"Error: Custom file '{custom_list_file}' not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{custom_list_file}'.")
        return

    # --- 2. Load Original ARC Dataset (Dict) ---
    try:
        with open(original_arc_file, 'r') as f:
            original_arc_data = json.load(f)
        if not isinstance(original_arc_data, dict):
             print(f"Error: Original ARC file '{original_arc_file}' is not a dict.")
             return
    except FileNotFoundError:
        print(f"Error: Original ARC file '{original_arc_file}' not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{original_arc_file}'.")
        return

    # --- 3. Process and Merge ---
    converted_data_dict = {}
    
    for task_data in custom_data_list:
        task_id = task_data.get("task_id")
        if not task_id:
            print("Warning: Skipping item in custom list with no 'task_id'")
            continue

        # --- NEW STEP: Get 'test' data from original file ---
        original_task_data = original_arc_data.get(task_id)
        if original_task_data and "test" in original_task_data:
            test_data = original_task_data["test"]
        else:
            print(f"Warning: No 'test' data found for task_id '{task_id}' in original file.")
            test_data = []

        converted_task = {
            "train": [],
            "test": test_data  # Assign the 'test' data directly
        }

        # --- Process 'train' data as before ---
        if "train_transitions" in task_data:
            for train_item in task_data["train_transitions"]:
                
                converted_train_item = {
                    "input": train_item.get("input"),
                    "output": train_item.get("output"),
                }
                
                original_steps = train_item.get("steps")
                
                if original_steps is None:
                    converted_train_item["steps"] = None 
                elif isinstance(original_steps, list):
                    processed_steps = []
                    for step_object in original_steps:
                        if isinstance(step_object, dict) and "image" in step_object:
                            processed_steps.append(step_object["image"])
                    converted_train_item["steps"] = processed_steps
                else:
                    converted_train_item["steps"] = None

                converted_task["train"].append(converted_train_item)
        
        # (No need to process 'test_transitions' from custom file anymore)

        converted_data_dict[task_id] = converted_task

    # --- 4. Save Final Merged File ---
    with open(output_file, 'w') as f:
        json.dump(converted_data_dict, f, indent=4)
    print(f"Conversion and merge complete. Final data saved to '{output_file}'")
# Example Usage:
# Assuming your custom dataset JSON is structured like the raw_response in Image 1,
# but it also has the train_transitions wrapper.
# Let's create a dummy custom_dataset.json to test this script.

# dummy_custom_data = {
#     "00576224": {
#         "train_transitions": [
#             {
#                 "train_index": 0,
#                 "input": [[7, 9, 7, 9], [4, 3, 4, 3]],
#                 "output": [[7, 9, 7, 9], [4, 3, 4, 3]],
#                 "steps": [
#                     {"step": 1, "image": [[7, 9, 7, 9], [4, 3, 4, 3]]},
#                     {"step": 2, "image": [[7, 9, 7, 9], [4, 3, 4, 3]]},
#                     {"step": 3, "image": [[7, 9, 7, 9], [4, 3, 4, 3]]},
#                     {"step": 4, "image": [[7, 9, 7, 9], [4, 3, 4, 3]]},
#                     {"step": 5, "image": [[7, 9, 7, 9], [4, 3, 4, 3]]},
#                     {"step": 6, "image": [[7, 9, 7, 9], [4, 3, 4, 3]]},
#                     {"step": 7, "image": [[7, 9, 7, 9], [4, 3, 4, 3]]},
#                     {"step": 8, "image": [[7, 9, 7, 9], [4, 3, 4, 3]]},
#                     {"step": 9, "image": [[7, 9, 7, 9], [4, 3, 4, 3]]},
#                     {"step": 10, "image": [[7, 9, 7, 9], [4, 3, 4, 3]]}
#                 ]
#             }
#         ],
#         "test_transitions": []
#     }
# }

# # Save dummy data to a file
# with open("custom_dataset.json", "w") as f:
#     json.dump(dummy_custom_data, f, indent=4)

# Run the conversion
#convert_custom_to_arc_format("data/train_with_steps.json", "train_with_steps_arc_format.json")
convert_and_merge_datasets(
    custom_list_file="data/train_with_steps.json",
    original_arc_file="kaggle/combined/arc-agi_concept_challenges.json",
    output_file="data/train_with_steps_arc_format.json"
)
# You can also imagine a custom_dataset where 'steps' are at the root of the task,
# not necessarily inside train_transitions. The script would need adjustment for that
# if it were the case, but based on your description and Image 1, it seems 'steps'
# are associated with individual train_transitions.