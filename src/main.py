from data_preparation import data_preparation
from feature_engineering import feature_engineering
from model import preparation_modeling, train_and_test_model

def main():
    data_preparation('../data/raw/data_scientist_case.csv')
    feature_engineering()
    preparation_modeling()
    train_and_test_model()
    
    print("Pipeline completed successfully.")

if __name__ == "__main__":
    main()