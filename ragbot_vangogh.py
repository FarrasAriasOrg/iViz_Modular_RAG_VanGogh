from chatbots.chatbot_openai_vangogh import ChatbotVanGogh
from rags.rag import RAG
from ragchain import RAGChain
from colorama import init, Fore
import pandas as pd
import numpy as np

init()

model_endpoint = "https://api.openai.com/v1/chat/completions"
api_key =  ""

def filter_dataframe(df, column_name, filter_values):
    """Filters a DataFrame based on a column's values.

    Args:
        df (pd.DataFrame): The DataFrame to filter.
        column_name (str): The name of the column for filtering.
        filter_values (pd.Series or np.ndarray): Values to match in the column.

    Returns:
        pd.DataFrame: The filtered DataFrame.

    Raises:
        ValueError: If filter_values is not a Series or array.
    """
    print("YO", filter_values)
    if not isinstance(filter_values, (pd.Series, np.ndarray)):
        raise ValueError("filter_values must be a pandas Series or numpy array")

    return df[df[column_name].isin(filter_values)]

def main():
    df = pd.read_csv("./data/cognitive_dataset_van_gogh.csv", index_col=0)
    df.reset_index(drop=True, inplace=True)  # Ensure consistent indexing

    column_to_vectorize = "context"
    bio_metadata = {
        "order": 1,
        "name": "Diary Entries", 
        "context": "Excerpts from Van Gogh's fictional diary. Use for information and lexical style." 
    }

    rag_bio = RAG("van_gogh_bio", db=df[column_to_vectorize], num_results=5, metadata=bio_metadata)
    rag_chain = RAGChain([rag_bio])

    character_data_dir = "./model_custom_inits/DefaultCharacter.json"
    model_data_dir = "./model_custom_inits/DefaultModel.json"
    VanGogh = ChatbotVanGogh(model_endpoint,character_data_dir,model_data_dir,api_key)

    while True:
        user_input = input("> ")
        results, master_prompt = rag_chain.make_master_prompt(user_input, return_result_list=True)
        #_, results = rag_bio.similarity_search(user_input)
        print(results)
        filtered = filter_dataframe(df,column_to_vectorize, results)
        print(filtered)
        #print("RAG prompt: \n" + Fore.GREEN + master_prompt + Fore.RESET)
        VanGogh.process_and_chat(user_input, filtered, "vangogh")

if __name__ == "__main__":
    main()