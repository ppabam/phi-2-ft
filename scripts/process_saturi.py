
import json
import os

def create_dataset():
    saturi_data = []
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    for filename in os.listdir(current_dir):
        if filename.endswith(".json") and filename != "saturi_dataset.json":
            file_path = os.path.join(current_dir, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    # Extract speaker info
                    speaker_info = data.get("speaker", [{}])[0]
                    province = speaker_info.get("residenceProvince")
                    city = speaker_info.get("residenceCity")

                    # Extract transcription sentences
                    transcription = data.get("transcription", {})
                    
                    # Check in sentences
                    sentences = transcription.get("sentences", [])
                    if sentences:
                        for sentence in sentences:
                            standard_text = sentence.get("standard")
                            dialect_text = sentence.get("dialect")
                            if standard_text and dialect_text:
                                saturi_data.append({
                                    "instruction": standard_text,
                                    "output": dialect_text,
                                    "province": province,
                                    "city": city
                                })
                        continue # Move to next file after processing sentences

                    # Fallback to top-level standard/dialect if sentences are empty
                    standard_text = transcription.get("standard")
                    dialect_text = transcription.get("dialect")

                    if standard_text and dialect_text:
                        saturi_data.append({
                            "instruction": standard_text,
                            "output": dialect_text,
                            "province": province,
                            "city": city
                        })

            except json.JSONDecodeError:
                print(f"Error decoding JSON from file: {filename}")
            except Exception as e:
                print(f"An error occurred with file {filename}: {e}")

    output_file = os.path.join(current_dir, "saturi_dataset.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(saturi_data, f, ensure_ascii=False, indent=4)
    
    print(f"Dataset created successfully at: {output_file}")
    print(f"Total entries: {len(saturi_data)}")

if __name__ == "__main__":
    create_dataset()
