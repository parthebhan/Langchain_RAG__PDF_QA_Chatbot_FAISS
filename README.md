# **Langchain_PDF_QA_Chatbot**

# **Stop Searching, Start Talking! Get Answers from PDFs Instantly with AI**

[![Streamlit App](https://img.shields.io/badge/Streamlit_App_-Langchain_PDF_QA_Chatbot-ff69b4.svg?style=for-the-badge&logo=Streamlit)](https://langchainragpdfappchatbotfaiss-yehqlvnvwkpfrdmkps4vhp.streamlit.app/)


## Purpose

This Streamlit application allows users to upload PDF files, extract text, and ask questions about the content using AI techniques such as text embeddings, similarity search, and a conversational AI model. The app supports PDF text extraction, chunking, vector storage, and question answering, all through an interactive web interface.

### Dependencies

- **Streamlit**: For building interactive web applications.
- **PyPDF2**: For extracting text from PDF files.
- **langchain**: Provides modules for text splitting (RecursiveCharacterTextSplitter), text embeddings (HuggingFaceEmbeddings), and similarity search (FAISS).
- **langchain_groq**: Access to Groq’s conversational AI models.
- **transformers**: For using Hugging Face pipelines.
- **huggingface_hub**: For logging in to Hugging Face Hub.
- **python-docx**: For creating and manipulating DOCX files.
- **dotenv**: For loading environment variables from a .env file.

### Main Functions and Workflow

1. **get_pdf_text(pdf_docs)**:
   - **Purpose**: Extracts text from uploaded PDF files.
   - **Implementation**: Uses `PdfReader` from PyPDF2 to iterate through each PDF and concatenate text from all pages.

2. **get_text_chunks(text)**:
   - **Purpose**: Splits extracted text into manageable chunks.
   - **Implementation**: Utilizes `RecursiveCharacterTextSplitter` from langchain with specified chunk size and overlap.

3. **get_vector_store(text_chunks)**:
   - **Purpose**: Generates embeddings for text chunks and creates a vector store.
   - **Implementation**: Uses `HuggingFaceEmbeddings` for embeddings and `FAISS` for vector store creation. Handles retries in case of failures and saves the vector store as "faiss_index".

4. **get_conversational_chain()**:
   - **Purpose**: Sets up a conversational AI chain for question answering.
   - **Implementation**: Defines a prompt template and initializes `ChatGroq` with Groq’s conversational AI model.

5. **user_input(user_question)**:
   - **Purpose**: Handles user input (questions) and triggers question-answering.
   - **Implementation**: Retrieves embeddings, performs similarity search using FAISS, and uses the conversational AI chain to generate a response. Displays the answer using Streamlit's `st.write`.

6. **reset_app()**:
   - **Purpose**: Resets the application state and clears cached data.
   - **Implementation**: Clears session state, removes the vector store directory, and resets all related variables.

7. **save_chat_history_to_docx(chat_history)**:
   - **Purpose**: Saves chat history to a DOCX file.
   - **Implementation**: Uses `python-docx` to create a DOCX file from the chat history, and returns a `BytesIO` object for download.

8. **main()**:
   - **Purpose**: Sets up the Streamlit application interface.
   - **Implementation**: Configures Streamlit settings, handles PDF uploads and processing, manages user questions, and displays chat history. Provides options to reset the app and save chat history.

### Usage

1. Users upload PDF files.
2. Clicking "Submit & Process" initiates PDF processing (text extraction, chunking, embeddings).
3. Users input questions related to PDF content; AI models provide answers based on processed PDFs.
4. Users can reset the app or save chat history.

### Summary

The code integrates AI and NLP techniques to interact with PDFs via a web interface. It handles text extraction, processing, and question answering, enhancing usability and accessibility of PDF content through automation and AI capabilities.


### Author

This app was created by **`Parthebhan Pari`**.

### Notes

- **Model**: Uses Groq’s conversational AI model for generating responses.
- **API Key**: Ensure that the API key for Groq is securely handled and loaded from environment variables.
- **Security**: Ensure that API keys and sensitive data are managed securely.


### **🔗 Connect with Me**

Feel free to connect with me on :

[![portfolio](https://img.shields.io/badge/my_portfolio-000?style=for-the-badge&logo=ko-fi&logoColor=white)](https://parthebhan143.wixsite.com/datainsights)

[![LinkedIn Profile](https://img.shields.io/badge/LinkedIn_Profile-000?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/parthebhan)

[![Kaggle Profile](https://img.shields.io/badge/Kaggle_Profile-000?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/parthebhan)

[![Tableau Profile](https://img.shields.io/badge/Tableau_Profile-000?style=for-the-badge&logo=tableau&logoColor=white)](https://public.tableau.com/app/profile/parthebhan.pari/vizzes)

