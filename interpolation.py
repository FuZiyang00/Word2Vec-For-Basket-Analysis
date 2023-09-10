from ML_Model.Model_Interpolation import Interpolation, Output_CSV
import pandas as pd
from gensim.models import KeyedVectors

if __name__ == "__main__":
    
    transactions = pd.read_csv("Word2Vec-For-Basket-Analysis/transactions.csv")
    articles = pd.read_csv("Word2Vec-For-Basket-Analysis/all_articles.csv", sep="~")
    model = KeyedVectors.load("Word2Vec-For-Basket-Analysis/Model.mod", mmap='r')

    # Suggestions retrieval 
    query_items = []
    while True:
        item_id = int(input("Enter the item ID: "))
        query_items.append(item_id)
        choice = input("Enter an other item (y/n): ")
        if choice.casefold() == "n":
            break

    suggestions = Interpolation.suggestions(query_items, model, transactions)
    csv = Output_CSV.csv(suggestions, articles)
    csv = Output_CSV.co_purchases_computation(transactions, csv)
    csv.to_csv('suggestions.csv', index='False')

    