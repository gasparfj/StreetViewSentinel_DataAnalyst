import roboflow
import pathlib as Path

# Initialize Roboflow with your API key
rf = roboflow.Roboflow(api_key="YOUR_API_KEY")

try:
    # Load the dataset
    project = rf.workspace("invasivedemography").project("opuntiagsv_test")
    dataset = project.version(9).download("yolov8")

    # The dataset will be downloaded and extracted to a local directory
    print(f"Dataset downloaded to: {dataset.location}")
    BASE_DIR = Path(dataset.location)
except Exception as e:
    print("Error downloading the dataset from Roboflow:", str(e))
    BASE_DIR = None

