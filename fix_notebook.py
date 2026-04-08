import nbformat

nb = nbformat.read("models_notebooks/dl_model.ipynb", as_version=4)

if "widgets" in nb["metadata"]:
    del nb["metadata"]["widgets"]

nbformat.write(nb, "dl_model.ipynb")
