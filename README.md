### Summary
This is a simple example of a Q&A app build with langchain and streamlit, powered by GPT-3. It takes a PDF file as an input and answers questions about it.

![screenshot](./screenshot.png)

### How to install
1. Clone this repo
2. Make sure you have `pipenv` installed.
3. Activate the virtual environment with `pipenv shell`
4. Run `pipenv install` to install dependencies.
5. Create `.env` file and place your OpenAI API key there.
6. Run `streamlit run app.py` to start the app.

### How to use
1. Upload a PDF file.
2. Ask a question.
3. Wait for the answer.

### Gotchas 
- The embeddings will be created and cached when you first upload a new PDF file, creating the embeddings will consume the tokens from your OpenAI API key depending on the size of the text content in the PDF file.
