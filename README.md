# FIR Assistant

## Overview

**FIR Assistant** is a Streamlit-based AI tool designed to assist Indian law enforcement by simplifying the generation of First Information Reports (FIRs) and providing answers to legal queries. Leveraging state-of-the-art language models and a knowledge base, the assistant ensures precise, efficient, and accurate handling of incident details to streamline the legal documentation process.

---

## Features

### 1. **FIR Generation**
   - Automatically generates FIRs based on the incident details provided.
   - Utilizes Indian Penal Code (IPC) and related acts for accurate legal framing.
   - Follows a standardized FIR template for consistency.

### 2. **Legal Query Assistance**
   - Answers questions related to FIR filing procedures and applicable legal sections.
   - Accesses a custom FAISS knowledge base and real-time search results for comprehensive responses.

### 3. **Speech Recognition**
   - Converts speech from audio files into text for seamless input.

### 4. **Interactive User Interface**
   - Designed with Streamlit for a clean, user-friendly experience.
   - Two main functionalities: FIR generation and legal Q&A.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/fir-assistant.git
   cd fir-assistant
   ```

2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up the environment variables:
   - Create a `.env` file and add the following variables:
     ```
     GROQ_API_KEY=<your_groq_api_key>
     TAVILY_API_KEY=<your_tavily_api_key>
     ```

4. Ensure the FAISS index is available in the `faiss_index` directory:
   - Place the required FAISS index files in the folder.

---

## Usage

1. **Run the application:**
   ```bash
   streamlit run app.py
   ```

2. Open the application in your browser at `http://localhost:8501`.

3. **Navigate between features:**
   - Use the **FIR Generation** tab to create FIRs.
   - Use the **Legal Q&A** tab to ask legal questions.

---

## Benefits

### 1. **Efficiency**
   - Speeds up FIR generation by automating the process.
   - Saves time for law enforcement by providing immediate legal insights.

### 2. **Accuracy**
   - Reduces human error by accurately identifying applicable IPC sections and acts.
   - Ensures FIRs follow a standardized format.

### 3. **Accessibility**
   - Simplifies legal documentation for officers unfamiliar with complex legal language.
   - Provides instant legal information without requiring extensive research.

### 4. **Scalability**
   - Can handle multiple users and a variety of queries simultaneously.
   - Modular architecture allows integration with other legal systems or APIs.

---

## Limitations
- Designed for Indian law enforcement; may not apply to other jurisdictions.
- Dependent on accurate inputs for generating reliable FIRs.
- Requires proper configuration of FAISS index and API keys.

---

## Contributing
Contributions are welcome! Please fork the repository, make your changes, and submit a pull request.

---

## License
This project is licensed under the Apache License. See the LICENSE file for details.
