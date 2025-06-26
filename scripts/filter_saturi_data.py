import json
import os
import sys

def filter_and_save_data(city_code, output_prefix):
    input_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saturi_dataset.json")
    output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"train_{output_prefix}.jsonl")

    filtered_data = []
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for entry in data:
                if entry.get("city") == float(city_code):
                    filtered_data.append(entry)

    except FileNotFoundError:
        print(f"Error: Input file not found at {input_file}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {input_file}")
        return
    except Exception as e:
        print(f"An error occurred: {e}")
        return

    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in filtered_data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print(f"Filtered data for city {city_code} saved to {output_file}")
    print(f"Total entries: {len(filtered_data)}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 filter_saturi_data.py <city_code> <output_filename_prefix>")
        sys.exit(1)

    city_code_arg = sys.argv[1]
    output_prefix_arg = sys.argv[2]
    filter_and_save_data(city_code_arg, output_prefix_arg)
