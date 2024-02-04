# PDF Files Chatbots 
---

# How to run the app

1. Install dependencies:

```
pip install -r requirements.txt

```

2. (Optional) You may opt for supplying your own openAI API key, Pinecone API key, or Pinecone env, in which case you should modify the `.env` file with your key values. The current environment keys run perfectly fine.

3. Run the streamlit `app.py` file

```
streamlit run app.py

```

# How to use

1. Upload PDF file(s).
2. Wait for embeddings and indexing.
3. Ask your questions.
