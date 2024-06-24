
import os

def custom_print(message, filename):
    # Get the absolute path to the current script's directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct the path to the output directory relative to the current script's directory
    output_dir = os.path.join(current_dir, '..', 'output')
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    # Construct the path to the output file
    output_file = os.path.join(output_dir, filename)
    
    # Write the message to the log file
    with open(output_file, "a") as log_file:
        print(message, file=log_file)
    # Print the message to the console
    print(message)
