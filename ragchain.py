from re import search

import faiss  # Import Faiss if necessary

class RAGChain:
    """Utilizes multiple RAG models for question answering, combining their strengths."""

    def __init__(self, list_of_rags, summary_prompt=None):
        """Initializes the RAGChain.

        Args:
            list_of_rags: A list of RAG model instances.
            summary_prompt: An optional prompt for summarizing context.
        """
        self._rags = list_of_rags
        self._init_prompt = "Use the following information as context for your response: "
        if summary_prompt:
            self._init_prompt = summary_prompt

    def format_responses(self, info_list, rag):
        """Formats the provided information and RAG context.

        Args:
            info_list: A list of information items.
            rag: The RAG model instance.

        Returns:
            A formatted string of information items and the RAG context.
        """
        formatted_text = f'-{rag.name}-\ncontext: {rag.context}\n\n'
        for item in info_list:
            formatted_text += f"{item}\n\n"
        return formatted_text.rstrip()

    def make_master_prompt(self, query, return_result_list=False):
        """Constructs a master prompt for question answering with multiple RAGs.

        Args:
            query: The user's query.
            return_result_list: If True, returns a tuple with search results and the prompt.

        Returns:
            The constructed master prompt (or prompt and results if specified).
        """
        sorted_rags = sorted(self._rags, 
                             key=lambda rag: (rag.order != 99, rag.order == 0, -rag.order))

        rag_texts = [self._init_prompt]
        #search_results = []  # Collect results from each RAG
        #search_results = None

        for rag in sorted_rags:
            result, search_results = rag.similarity_search(query)
            #search_results.append(text) 
            text = self.format_responses(search_results, rag)
            rag_texts.append("---------------")
            rag_texts.append(text)

        master_prompt = "\n".join(rag_texts)

        if return_result_list:
            return search_results, master_prompt
        else:
            return master_prompt

    # TODO: Implement quality-based filtering of search results
    # def _filter_results_by_quality(self, search_results, threshold):
    #     ... 

        
    