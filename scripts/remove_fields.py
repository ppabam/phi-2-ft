import json
import os
import sys

def remove_fields_from_jsonl(input_filepath, output_filepath):
    cleaned_data = []
    try:
        with open(input_filepath, 'r', encoding='utf-8') as infile:
            for line in infile:
                entry = json.loads(line.strip())
                if "province" in entry:
                    del entry["province"]
                if "city" in entry:
                    del entry["city"]
                cleaned_data.append(entry)

    except FileNotFoundError:
        print(f"Error: Input file not found at {input_filepath}")
        return
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from line: {line.strip()}. Error: {e}")
        return
    except Exception as e:
        print(f"An error occurred: {e}")
        return

    with open(output_filepath, 'w', encoding='utf-8') as outfile:
        for entry in cleaned_data:
            outfile.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print(f"Cleaned data saved to {output_filepath}")
    print(f"Total entries: {len(cleaned_data)}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 remove_fields.py <input_jsonl_filepath> <output_jsonl_filepath>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    remove_fields_from_jsonl(input_file, output_file)
