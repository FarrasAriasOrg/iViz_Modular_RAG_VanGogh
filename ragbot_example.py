from requests import api
from chatbots.chatbot_openai import OpenAIChatbot
from rags.rag import RAG
from ragchain import RAGChain
from colorama import init, Fore

init()

model_endpoint = "https://api.openai.com/v1/chat/completions"
api_key =  "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"
if api_key == "":
    print(Fore.RED + "Your api key is empty, if you're not getting responses, make sure to explicitly write it in ragbot_example.py" + Fore.RESET)

# Metadata for RAG models
VAN_GOGH_QUESTIONS_META = {
    "order": 3, 
    "name": "Interview Questions", 
    "context": "These are questions answered by Van Gogh himself. Use them not only for information but copy the style of writing/talking"
}
EINSTEIN_QUESTIONS_META = {
    "order": 5, 
    "name": "Einstein Facts", 
    "context": "This is not relevant to the task"
}

def main():
    """Main execution loop for the chatbot application."""

    # TODO: add a force_refactor flag and implement index folder management

    rag_e = RAG("einstein", "./data/a3", num_results=5, separator_value="$$", metadata=EINSTEIN_QUESTIONS_META)
    rag_v = RAG("vginterview", "./data/newVG2", num_results=5, metadata=VAN_GOGH_QUESTIONS_META)

    rag_chain = RAGChain([rag_v, rag_e])  # Optionally include rag_e

    character_data_dir = "./model_custom_inits/DefaultCharacter.json"
    model_data_dir = "./model_custom_inits/DefaultModel.json"
    van_gogh_bot = OpenAIChatbot(model_endpoint, character_data_dir, model_data_dir, api_key) 

    while True:
        user_input = input("> ")
        master_prompt = rag_chain.make_master_prompt(user_input)
        #print("RAG prompt:\n" + Fore.GREEN + master_prompt + Fore.RESET) 
        van_gogh_bot.chat(user_input, master_prompt, print_context=True)

if __name__ == "__main__":
    main()
