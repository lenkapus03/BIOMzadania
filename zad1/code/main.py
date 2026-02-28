import os

from zad1.code.data import load_data
from app import Application

if __name__ == "__main__":
    # Load CSV data
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_folder = os.path.normpath(os.path.join(current_dir, "..", "data"))
    csv_path = os.path.join(data_folder, "iris_annotation.csv")
    load_data.load_random_records(csv_path, n=100)

    # Start the app
    app = Application()
    app.run()