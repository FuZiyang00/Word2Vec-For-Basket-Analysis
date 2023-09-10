# Word2Vec For Basket Analysis
The goal of this project is to test the efficacy of adopting a statistical NLP technique such as Word2vec for Basket Analysis. 
The ultimate aim is to come up with a "recommendation system" that suggests the best complements to be displayed alongside a selected query product. 
Since there are no objective parameters to evaluate the "goodness" of a potential suggestion, the number of co-purchases query-suggestion has been used to assess the ML model. 

# Why Word2VeC
Word2Vec" is a well-known and widely used technique in natural language processing (NLP) for word embeddings. It is used to learn vector representations (embeddings) of words from large text corpora. These vector representations capture semantic relationships between words based on their co-occurrence patterns in the text.
The key idea behind Word2Vec is that words with similar meanings tend to occur in similar contexts.

We can notice a pretty close similarity between a sentece (text) and a grocery receipt: the words that make up a sentence give to it a meaning, whereas the products that compose a receipt determine its consume context.
By leveraging on this similarity we decided to give Word2Vec a try

# Results 
| Query | Complement | Co-purchases |
| -------- | -------- | -------- |
| carrots | bananas | 4063 |
| -------- | -------- | -------- |
| carrots | lemons | 1873 |
| -------- | -------- | -------- |
| carrots | onions | 1467 |
| -------- | -------- | -------- |
| carrots | salad | 1267 |
| -------- | -------- | -------- |
| carrots | tomatoes | 1207 |

