import json

notebook_path = "Random_Forest_Plant_Recommendation_Revamped.ipynb"
with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for idx, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code" and idx > 10:
        source_text = "".join(cell["source"])
        print(f"Cell Index: {idx}")
        print("--- Source ---")
        print(source_text)
        print("--------------\n")
