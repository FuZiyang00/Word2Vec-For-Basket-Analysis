import numpy as np
import pandas as pd

class DataCleaning:
    @staticmethod
    def sales_to_transactions(sales_df):
        sales_df.columns = sales_df.columns.to_series().apply(lambda x: x.strip())
        sales = sales_df.groupby(['codice_negozio', 'data_scontrino', 'ora', 'numero_cassa', 'numero_scontrino']) \
            .apply(lambda group: set(group['codice_articolo'].values)) \
            .reset_index()

        print(len(sales)+1)
        sales['session_id'] = np.arange(len(sales)) + 1
        df_out = sales_df.merge(sales[['codice_negozio', 'data_scontrino', 'ora', 'numero_cassa',
                                       'numero_scontrino', 'session_id']],
                                on=['codice_negozio', 'data_scontrino', 'ora', 'numero_cassa', 'numero_scontrino'],
                                suffixes=('', '_y'))

        df_out = df_out.drop(['codice_negozio', 'data_scontrino', 'ora', 'numero_cassa', 'numero_scontrino',
                              'importo'], axis=1)
        df_out = df_out.rename(columns={"numero_pezzi": "units"})
        df_out = df_out.assign(user_id='1')
        df_out.reset_index(drop=True, inplace=True)

        df_out = df_out[df_out['units'] != 0]

        df_out = df_out.groupby(['user_id', 'codice_articolo', 'session_id'])['units'].sum()
        df_out = df_out.reset_index()

        df_out = df_out.groupby('codice_articolo').filter(lambda x: len(x) > 10)
        df_out = df_out.groupby('session_id').filter(lambda x: len(x) > 1)
        return df_out

class Preprocessing:
    @staticmethod
    def transactions_to_sentences(df):
        basket = df.groupby(['session_id'], group_keys=False) \
            .apply(lambda group: list(group['codice_articolo'].values)).reset_index()
        
        sentences = basket[0].to_list()
        return sentences