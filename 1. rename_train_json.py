import os
import json

# Rename train name
folder = "new_ds/train" 

for fname in os.listdir(folder):
    if ".rf." not in fname:
        continue
    # ambil bagian sebelum "_png.rf.", lalu tambahkan .png
    new_name = fname.split("_png.rf.")[0] + ".png"
    src = os.path.join(folder, fname)
    dst = os.path.join(folder, new_name)
    os.rename(src, dst)
    print(fname, "->", new_name)
    
# rename json
json_path = "new_ds/train/_annotations_coco.json"
with open(json_path) as f:
    coco = json.load(f)

for im in coco["images"]:
    if "_png.rf." in im["file_name"]:
        im["file_name"] = im["file_name"].split("_png.rf.")[0] + ".png"

with open(json_path, "w") as f:
    json.dump(coco, f)

print("json updated")
