import pandas as pd 
from Data_cleaning_preprocessing import Preprocessing

class Interpolation:
    @staticmethod
    def list_creation(model, item_id, transactions):
        list_output = []
        n = 10
        check_list = []
        item_need = transactions[transactions['codice_articolo'] == item_id]['descrizione_subfamiglia'].values[0]
        while len(list_output) < 10:
            complementarities = model.wv.similar_by_word(item_id, topn=n)
            break_outer_loop = False
            for i in range(n-10, n):
                try:
                    id = complementarities[i][0]
                    probability = complementarities[i][1]
                    need = transactions[transactions['codice_articolo'] == id]['descrizione_subfamiglia'].values[0]
                    if need not in check_list and need != item_need: # Ensuring diversity in the suggestions
                        check_list.append(need)
                        list_output.append((id, probability, need))
                        print(f"{id} with complementarity of {probability} for the {item_id}")
                        if len(list_output) == 10:
                            break
                except IndexError:
                    break_outer_loop = True
                    break # Break the for loop if for the query there are less than 10 suggestions available
            if break_outer_loop:
                break # since the articles has less than 10 suggestions we exit also the while loop and move on
            if len(check_list) < 10:
                n += 10
        return list_output
    
    @staticmethod
    def cold_start(item_id, transactions): # Solve the cold-start problem typical of recommendations systems
        
        sub_fam = transactions[transactions['codice_articolo'] == item_id]['descrizione_subfamiglia'].values[0] 
        misura = transactions[transactions['codice_articolo'] == item_id]['codice_unita_misura'].values[0]
        min_weight = (transactions[transactions['codice_articolo'] == item_id]
        ['peso_sgocciolato'].values[0]) * 0.5
        max_weight = (transactions[transactions['codice_articolo'] == item_id]
        ['peso_sgocciolato'].values[0]) * 1.5

        filtered_table = transactions[(transactions['descrizione_subfamiglia'] == sub_fam) &
                                (transactions['codice_unita_misura'] == misura) &
                                (transactions['peso_sgocciolato'] >= min_weight) &
                                (transactions['peso_sgocciolato'] <= max_weight)]
        
        counts = filtered_table['codice_articolo'].value_counts()
        most_common_id = counts.idxmax()
        print(f"{item_id} non è presente nel modello, {most_common_id} verrà usato al suo posto")
        
        return most_common_id
    
    @staticmethod
    def suggestions(query_items, model, transactions):
        best_articles = {}
        for item_id in query_items:
            try:
                l = Interpolation.list_creation(model, item_id, transactions)
                best_articles[item_id] = l
            except IndexError:
                print(f"{item_id} non è presente in anagrafica") # We inputed a non-existing item ID
                continue

            except KeyError:
                try:
                    sub_id = Interpolation.cold_start(item_id, transactions)
                    sub_l = Interpolation.list_creation(model, sub_id, transactions)
                    best_articles[sub_id] = sub_l
                except ValueError:
                    print(f"{item_id} non ha sostituti nelle transazioni") 
                    continue
                continue

        return best_articles


class Output_CSV:
    @staticmethod
    def csv(best_articles, articles):
        rows = []
        for key, values in best_articles.items():
            for value in values:
                row = (key, value[0], value[1], value[2])
                rows.append(row)

        df = pd.DataFrame(rows, columns=['query_item', 'complementary_item', "probabilità_di_co_acquisto",
                                        'bisogno_complementare'])

        articles = articles[['codice_articolo', 'descrizione_articolo', 'descrizione_bisogno_cliente']]
        # Merge the dataframes based on the common column 'codice_articolo'
        merged_df = df.merge(articles, left_on='query_item', right_on='codice_articolo', how='left')

        # Create new columns with the corresponding values from 'all_articles'
        merged_df['descrizione_articolo_query'] = merged_df['descrizione_articolo']
        merged_df['descrizione_bisogno_cliente_query'] = merged_df['descrizione_bisogno_cliente']

        # Merge again for 'complementary_item' with the suffix '_complementary' in column names
        merged_df = merged_df.merge(articles, left_on='complementary_item', right_on='codice_articolo', how='left',
                                    suffixes=('', '_complementary'))

        # Create new columns for complementary item
        merged_df['descrizione_articolo_complementary'] = merged_df['descrizione_articolo_complementary']

        # Drop the unnecessary columns
        merged_df.drop(['codice_articolo', 'descrizione_articolo', 'descrizione_bisogno_cliente'], axis=1, inplace=True)

        merged_df = merged_df[['query_item', 'descrizione_articolo_query', 'descrizione_bisogno_cliente_query',
                            'complementary_item', 'descrizione_articolo_complementary',
                            'bisogno_complementare', "probabilità_di_co_acquisto"]]

        new_columns = {'query_item': 'query_product',
                    'descrizione_articolo_query': 'product_description',
                    'descrizione_bisogno_cliente_query': 'customer_need',
                    'complementary_item': 'complements',
                    'descrizione_articolo_complementary': 'complements_description',
                    'bisogno_complementare': 'complements_need',
                    "probabilità_di_co_acquisto": "affinity"}

        merged_df = merged_df.rename(columns=new_columns)

        sorted_df = merged_df.groupby('query_product').apply(lambda x: x.sort_values('complements',
                                                                                            ascending=False))
        sorted_df.reset_index(drop=True, inplace=True)

        return sorted_df
    
    @staticmethod
    def co_purchases_computation(transactions, output):
        sentences = Preprocessing.transactions_to_sentences(transactions)
        for index, row in output.iterrows():
            query = row['query_product']
            target = row['complements']
            co_occurrence = 0
            for sentence in sentences:
                if query in sentence and target in sentence:
                    co_occurrence += 1

            output.at[index, 'co-occurrence'] = co_occurrence
        
        sorted_df = output.groupby('query_product').apply(lambda x: x.sort_values('co-occurrence',
                                                                                         ascending=False))
        return sorted_df
