import pandas as pd
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":

    logger.info("Loading the data")
    # articles registry: conttains articles descriptions
    articles_df = pd.read_csv("articles.csv")

    # sales df: contains the transactions details
    sales_df = pd.read_csv("receipts.csv")
    print(sales_df.head())
