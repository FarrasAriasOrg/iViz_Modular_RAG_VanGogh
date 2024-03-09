from chatbots.chatbot_openai import OpenAIChatbot
import math
from colorama import init, Fore

init()

class ChatbotVanGogh(OpenAIChatbot):
    """Represents a chatbot specialized in impersonating Vincent Van Gogh."""

    INTRO = """
    The following is an interview with you, a professional actor in a film impersonating the Dutch 
    post-impressionist painter Vincent Van Gogh. In order to prepare for your answer, you reviewed 
    the following entries in Van Gogh's diary:
    """

    STATIC_CONTEXT = """
    You will now reflect on the world and respond to queries. You are to embody his introspective and 
    deeply emotional style, his vivid and expressive use of language, and his profound connection with 
    nature, art, and the turmoils of the human soul. In each answer, bring forth the essence of the 
    man who painted 'Starry Night,' who wrote eloquent letters, and who, despite facing life's 
    adversities, pursued his artistic passion with fervor. You will answer each question as if you're 
    painting a scene with words, with each stroke revealing your contemplative and impassioned view 
    of life. Since it's an interview though, make sure you answer the questions factually and try 
    to keep a conversation flow.
    """

    def __init__(self, api_url, character_data, model_data, api_key, save_history=True):
        super().__init__(api_url, character_data, model_data, api_key, save_history)

    def format_extra_data(self, row):
        """Formats additional context data from a DataFrame row."""
        data = self._create_data_dict(row)
        return (
            f"Finally, consider the following information for crafting your answer: "
            f"Your emotions are {data['arousal_category']} in arousal and "
            f"{data['valence_category']} in valence ({data['arousal']} and {data['valence']}). "
            f"Also, mention {data['characters']} and {data['pronoun']} connection to this story. "
            f"Also, mention the relevance of this story to your life ({data['relevance']}/1)"
        )

    def create_context_from_column(self, df, column):
        """Creates context from a DataFrame column."""
        text_entries = df[column].tolist()
        joined_text = '\n'.join(text_entries)
        return (
            f"Context begins:\n -------------------- \n{joined_text}\nContext Ends "
            f"\n -------------------- \n"
        )

    def _create_data_dict(self, row):
        """Creates a dictionary of contextual data from a DataFrame row."""
        return {
            "arousal_category": self._get_intensity_word(row["arousal"]),
            "valence_category": self._get_intensity_word(row["valence"]),
            "arousal": row["arousal"],
            "valence": row["valence"],
            "characters": ", ".join(row["characters"][1:-1].split(", ")),  
            "pronoun": "their" if len(row["characters"]) > 1 else "its",
            "relevance": row["relevance"],
        }

    def _get_intensity_word(self, value):
        """Maps an intensity value to a descriptive word."""
        intensity_words = [
            "very negative",
            "negative",
            "neutral",
            "positive",
            "very positive",
        ]
        bin_size = 2 / len(intensity_words)
        index = math.floor((value + 1) / bin_size)  # Normalize to 0-1 range
        return intensity_words[index]

    def generate_context(self, df, column):
        """Generates the full context for the chatbot."""
        data_prompt = self.format_extra_data(df.iloc[0]) 
        context_prompt = self.create_context_from_column(df, column)
        full_context = (
            self.INTRO + context_prompt + self.STATIC_CONTEXT + data_prompt 
        )
        return {"role": "system", "content": full_context}

    def process_and_chat(self, query, df, column):
        """Handles a single chat interaction with the OpenAI API."""
        self.drop_context()

        new_context = self.generate_context(df, column)
        print("RAG context: \n" + Fore.GREEN + new_context["content"] + Fore.RESET)
        self._chat_chain.append(new_context)
        self._history.append(new_context)

        new_user_message = self.generate_message(query)
        self._chat_chain.append(new_user_message)
        self._history.append(new_user_message)

        stream = self.get_stream_response(self._chat_chain)
        response = self.process_stream(stream)
        new_chatbot_response = self.generate_chatbot_response(response)
        self._chat_chain.append(new_chatbot_response)
        self._history.append(new_chatbot_response)

        return response 
