# Chat With WebPage 

## Overview
This project is a **Streamlit-based Webpage chat Tool** that allows users to input a webpage URL and a custom query to retrieve response content using **LangChain** and **OpenAI GPT-4o**. The application processes webpage content, extracts relevant text, generates context-aware summaries, and answer the questions accodring the the content of the webpage.

## Features
- **Extracts webpage content** automatically.
- **Summarizes content** using OpenAI's `gpt-4o` API.
- **Retrieves relevant information** based on user queries.
- **Utilizes FAISS vector storage** for efficient retrieval.
- **Checks webpage accessibility** before processing.
- **User-friendly Streamlit UI** with a responsive layout.

## Installation
### Prerequisites
Ensure you have Python 3.12 installed and `pip` available.

### Steps
1. **Clone the repository**:
   ```bash
   git clone https://github.com/ifares-csi/Chat-With-WebPage-Agent.git
   cd chat-with-webpage
   ```
2. **Create a virtual environment (optional but recommended)**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Set up environment variables**:
   - Create a `.env` file in the root directory and add:
     ```ini
     OPENAI_API_KEY=your-openai-api-key
     LANGSMITH_API_KEY=your-langsmith-api-key
     LANGSMITH_PROJECT=your-project-name
     ```

## Usage
1. **Run the Streamlit app**:
   ```bash
   streamlit run app.py
   ```
2. **Enter a webpage URL** and a query in the provided fields.
3. **Click "Summarize"** to generate a summary.
4. **View results** displayed on the page.

## Dependencies
- **Streamlit** for UI (`streamlit`)
- **LangChain** for document processing
- **OpenAI API** for language modeling
- **FAISS** for vector-based retrieval
- **Requests** for webpage accessibility checking

## Contributing
1. **Fork the repository**.
2. **Create a feature branch**:
   ```bash
   git checkout -b feature-branch
   ```
3. **Commit changes**:
   ```bash
   git commit -m "Added new feature"
   ```
4. **Push the branch**:
   ```bash
   git push origin feature-branch
   ```
5. **Submit a Pull Request**.

## License
This project is licensed under the MIT License.

<!-- ## Contact
For questions or suggestions, feel free to reach out!

📧 Email: your-email@example.com
🔗 GitHub: [Your Repository](https://github.com/your-repo/chat-with-webpage)
 -->
