# iViz Modular RAG

**IMPORTANT: Modifications**

This version of the code corresponds to the CHI position paper: LINK

This repo was forked from https://github.com/FarrasArias/iViz_Modular_RAG. It contains a subclass extension to the OpenAIChatbot class, which was adapted for the paper. Likewise, it adds the ragbot_vangogh.py script to handle the specific changes for the paper.

**Overview**

This project implements a flexible chatbot system that utilizes Retrieval-Augmented Generation (RAG) models and the OpenAI API for natural language interactions. Key features include:

* **Multiple RAG Models:** The system supports chaining multiple RAG models, each specialized in different domains or styles of communication. This allows for more comprehensive and nuanced responses.
* **Customizable Character:** The chatbot's personality and knowledge base can be modified by providing different character and model initialization data (`.json` files).
* **OpenAI Integration:** We plan to adhere to the OpenAI API for text generation, even when using open-source models.

**Installation**

The code was developed in Python 3.11.7. 

1. **Create a virtual environment (in the command line):** 
   ```bash
   python3 -m venv env  # Create environment named 'env'
   source env/bin/activate  # Activate the environment (Linux/macOS)
   env\Scripts\activate  # Activate the environment (Windows)

2. **Install requirements:** 
   ```bash
   pip install -r requirements.txt

3. **Test the code**
   ```bash
   python ragbot_example.py

The current version runs in the command line. During the conversation, the system shows in green what the prompt given to the LLM is. 

**Class Definitions**

* **OpenAIChatbot:** Interacts with the OpenAI API for chatbot functionality.
* **Embedder:** Encodes text into embeddings and allows efficient similarity search.
* **DataProcessor:** Through `split_text()`, it splits a text file into a numpy array of substrings based on a separator or chunk length.
* **RAG:** Represents a Retrieval-Augmented Generation (RAG) model.
* **RAGChain:** Utilizes multiple RAG models for question answering, combining their strengths.

**Conventions for Development**
* Please be sure to adhere to OpenAI's API guidelines (https://openai.com/api/).
* Maintain compatibility with Hugging Face models for easy switching of embedding models and LLMs.

##Creating Your Own Chatbot
See `ragbot_example.py` for a functioning minimal example.

**Data Preparation**
* Prepare text data for each RAG model (documents, interview transcripts, etc.).
* Specify metadata for each RAG model (name, order, context instructions) in a dictionary format. See `VAN_GOGH_QUESTIONS_META` and `EINSTEIN_QUESTIONS_META` in `ragbot_example` for examples.

**Customize**
* Create new `.json` files in the `model_custom_inits` folder for different chatbot personalities and knowledge domains.

**Configuration**
* Set `model_endpoint` to the correct OpenAI API chat endpoint.
* Replace the placeholder in your code with your actual OpenAI API key.
* Provide paths to your character data and model initialization files in `character_data_dir` and `model_data_dir`.

**Add RAG Models**
```python
rag = RAG("my_rag_name", "./path/to/data", num_results=5, metadata=my_metadata)
```

**Chain RAGs**
```python
rag_chain = RAGChain([rag1, rag2, ...])
```

**Using Environment Variables for API Key**
**Important:** For security, avoid hardcoding your API key directly into the code. Instead, you can store your API key in the command line.
```bash
export OPENAI_API_KEY=your_actual_api_key
```
