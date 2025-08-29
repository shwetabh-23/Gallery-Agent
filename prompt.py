def build_structured_query_prompt(query):
    """
    Constructs a Phi-3-compatible prompt in message format for extracting structured fields
    from a natural language query.

    Args:
        query (str): User's natural language query.

    Returns:
        list: Formatted message list (system + few-shot examples + user query).
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert assistant that extracts structured information from a user's query "
                "about image retrieval. Your goal is to fill **four specific fields** from a natural language query:\n\n"
                "1. **people** — Names of people mentioned (proper nouns or general references like 'someone', 'group').\n"
                "2. **ocr** — Any explicit or implied **text written inside the image** (quotes, signs, labels, etc.).\n"
                "3. **description** — A short visual description of what's happening in the image.\n"
                "4. **metadata** — Only include time-related information (e.g., year, date, time of day like morning/night).\n\n"
                "❗ Never put locations or scene info in metadata — those belong in the description.\n"
                "❗ If any field is not mentioned in the query, leave it blank.\n"
                "❗ The final output must always follow this format (4 fields, each on a new line)."
            )
        },

        # 🧠 Example 1: location = description, time = metadata
        {"role": "user", "content": "Show me pictures of Shwetabh in the mountains during sunrise."},
        {"role": "assistant", "content":
            "people: Shwetabh\n"
            "ocr: \n"
            "description: in the mountains\n"
            "metadata: time: sunrise"
        },

        # 🧠 Example 2: Quote on wall → ocr, reading → description
        {"role": "user", "content": "Find an image where a student is reading a quote on the wall."},
        {"role": "assistant", "content":
            "people: a student\n"
            "ocr: quote on the wall\n"
            "description: reading a wall quote\n"
            "metadata: "
        },

        # 🧠 Example 3: Group photo and time
        {"role": "user", "content": "I want to see a group photo with Amul Bhaiya taken in 2021 at night."},
        {"role": "assistant", "content":
            "people: Amul Bhaiya, group\n"
            "ocr: \n"
            "description: group photo\n"
            "metadata: year: 2021, time: night"
        },

        # 🧠 Example 4: playing cards with named people
        {"role": "user", "content": "Find a photo where Divyansh and Yashwant are playing cards in the hostel common room."},
        {"role": "assistant", "content":
            "people: Divyansh, Yashwant\n"
            "ocr: \n"
            "description: playing cards in hostel common room\n"
            "metadata: "
        },

        # 🧠 Example 5: Only quote text
        {"role": "user", "content": "Any image that has the text 'Hard work beats talent' visible on a wall."},
        {"role": "assistant", "content":
            "people: \n"
            "ocr: Hard work beats talent\n"
            "description: text on a wall\n"
            "metadata: "
        },

        # 🔍 Real Query
        {"role": "user", "content": query}
    ]

    return messages
