from src.data_loader import load_data
from src.config import DATA_PATH
from src.preprocessing import clean_data
from src.model import train_model

def main():
    df = load_data(DATA_PATH)

    if df is not None:
        df = clean_data(df)
        train_model(df)

if __name__ == "__main__":
    main()