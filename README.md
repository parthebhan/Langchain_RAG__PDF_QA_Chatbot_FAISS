# **Langchain_PDF_Reader**

# **Stop Searching, Start Talking! Get Answers from PDFs Instantly with AI**

[![Streamlit App](https://img.shields.io/badge/Streamlit_App_-Gemini_clone-ff69b4.svg?style=for-the-badge&logo=Streamlit)](https://langchainragpdfreaderappfaiss-myfsbfmgvenjnyczran4ch.streamlit.app/)


## Purpose

The code creates a Streamlit application (`app.py`) that allows users to upload PDF files, extract text, and ask questions about the content using AI techniques such as text embeddings, similarity search, and a conversational AI model.

### Dependencies

- **Streamlit**: For building interactive web applications.
- **PyPDF2**: For extracting text from PDF files.
- **langchain**: Provides modules for text splitting (`RecursiveCharacterTextSplitter`), text embeddings (`GoogleGenerativeAIEmbeddings`), and similarity search (`FAISS`).
- **google-generativeai**: Access to Google's Generative AI models.
- **dotenv**: Loading environment variables from a `.env` file.

### Main Functions and Workflow

1. **`get_pdf_text(pdf_docs)`**:
   - **Purpose**: Extracts text from uploaded PDF files.
   - **Implementation**: Uses `PdfReader` from PyPDF2 to iterate through each PDF and concatenate text from all pages.

2. **`get_text_chunks(text)`**:
   - **Purpose**: Splits extracted text into manageable chunks.
   - **Implementation**: Utilizes `RecursiveCharacterTextSplitter` from langchain with specified chunk size and overlap.

3. **`get_vector_store(text_chunks)`**:
   - **Purpose**: Generates embeddings for text chunks and creates a vector store.
   - **Implementation**: Uses `GoogleGenerativeAIEmbeddings` for embeddings and `FAISS` for vector store creation, saved as "faiss_index".

4. **`get_conversational_chain()`**:
   - **Purpose**: Sets up a conversational AI chain for question answering.
   - **Implementation**: Defines a prompt template and initializes `ChatGoogleGenerativeAI` with Google's Generative AI model (`gemini-1.5-pro-latest`).

5. **`user_input(user_question)`**:
   - **Purpose**: Handles user input (questions) and triggers question-answering.
   - **Implementation**: Retrieves embeddings, performs similarity search (`FAISS.load_local`), and uses the conversational AI chain (`get_conversational_chain`). Displays answer using Streamlit's `st.write`.

6. **`main()`**:
   - **Purpose**: Sets up Streamlit application interface.
   - **Implementation**: Configures Streamlit (`st.set_page_config`, `st.header`). Handles PDF uploads and processing (`get_pdf_text`, `get_text_chunks`, `get_vector_store`) on button click (`Submit & Process`).

### Usage

- Users upload PDF files.
- Clicking `Submit & Process` initiates PDF processing (text extraction, chunking, embeddings).
- Users input questions related to PDF content; AI models provide answers based on processed PDFs.

### Summary

The code integrates AI and NLP techniques to interact with PDFs via a web interface. It handles text extraction, processing, and question answering, enhancing usability and accessibility of PDF content through automation and AI capabilities.


### Author

This app was created by **`Parthebhan Pari`**.

### Notes

- This app uses the Gemini Pro model from Google's GenerativeAI API to generate responses.
- Ensure that you have a stable internet connection to interact with the Gemini Pro model.
- For security reasons, make sure to handle and store your API key securely.


### **🔗 Connect with Me**

Feel free to connect with me on :

[![portfolio](https://img.shields.io/badge/my_portfolio-000?style=for-the-badge&logo=ko-fi&logoColor=white)](https://parthebhan143.wixsite.com/datainsights)

[![LinkedIn Profile](https://img.shields.io/badge/LinkedIn_Profile-000?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/parthebhan)

[![Kaggle Profile](https://img.shields.io/badge/Kaggle_Profile-000?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/parthebhan)

[![Tableau Profile](https://img.shields.io/badge/Tableau_Profile-000?style=for-the-badge&logo=tableau&logoColor=white)](https://public.tableau.com/app/profile/parthebhan.pari/vizzes)
