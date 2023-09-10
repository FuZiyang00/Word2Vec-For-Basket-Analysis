from ML_Model.Model_Training import Training
from Data_cleaning_preprocessing import DataCleaning, Preprocessing
import time
import pandas as pd

if __name__ == "__main__":
    start_time = time.time()

    # CSV files loading 
    articles_descryption = pd.read_csv("all_articles.csv", sep='~')
    sales = pd.read_csv("sales.csv", sep='~')

    # Data Cleaning 
    transactions = DataCleaning.sales_to_transactions(sales)

    # Integrate the transactions with the products descriptions and remove all the noise articles
    transactions = transactions.merge(articles_descryption, on='codice_articolo')
    print('transactions table: lenght {}'.format(len(transactions)))

    df = transactions[transactions.codice_articolo != 2990099]
    df = df[~df['descrizione_reparto'].str.contains('PROMOZIONALI')]
    df = df[~df['descrizione_articolo'].str.contains('SHOPPER')]
    df = df[~df['descrizione_articolo'].str.contains('PER TE')]
    df = df[~df['descrizione_subfamiglia'].str.contains('SHOPPERS')]
    df = df[~df['descrizione_subfamiglia'].str.contains('SACCHETTO')]
    print('transactions table after removal: lenght {}'.format(len(df)))
    df.to_csv('transactions.csv', index='False')

    # Convert the transactions to sentence 
    sentences = Preprocessing.transactions_to_sentences(df)

    # Machine Learning Model training 
    ml_model = Training.create_model(sentences)
    ml_model.save("Model.mod")

    elapsed_time = time.time() - start_time
    print('{} spent for the whole training process of P-Lite using data by clients'.format(elapsed_time))

