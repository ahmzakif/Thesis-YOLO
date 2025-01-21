import os

input_folder = "mAP-skripsi\input\detection-results"

class_map = {
    "0": "Metal",
    "1": "Plastic"
}
for filename in os.listdir(input_folder):
    if filename.endswith(".txt"):
        file_path = os.path.join(input_folder, filename)

        with open(file_path, "r") as infile:
            lines = infile.readlines()

        with open(file_path, "w") as outfile:
            for line in lines:
                data = line.strip().split()
                if len(data) == 6:
                    xmin, ymin, xmax, ymax, confidence, class_id = data
                    class_name = class_map.get(class_id, "Unknown")

                    outfile.write(f"{class_name} {confidence} {xmin} {ymin} {xmax} {ymax}\n")

print(f"Files in {input_folder} have been updated with the correct format.")
