# Mini Semantic Search

A small search engine that finds FAQ answers by **meaning**, not exact keywords. Ask "I forgot my login" and it returns the password reset FAQ, even though the words are different.

Built with **Pandas** (data loading and cleaning) and **ChromaDB** (embeddings and vector search).

## How it works

```
FAQ file (JSON)
      |
      v
Load with Pandas
      |
      v
Clean + chunk the text
      |
      v
Embed and store in Chroma
      |
      v
Query -> print top 3 matches
```

1. **Load**: reads the FAQs into a Pandas DataFrame.
2. **Clean and chunk**: removes empty rows, duplicates, HTML tags and extra whitespace, then splits long text into chunks of up to 50 words.
3. **Embed and store**: Chroma converts each chunk into a vector using its default model (`all-MiniLM-L6-v2`) and saves it in a local vector database.
4. **Query**: your question is embedded the same way, and the 3 closest chunks are returned with a similarity score (1.00 = identical meaning).

## Project structure

```
.
├── semantic_search.py        # main script
├── MakTek_FCustomer_support_faqs_dataset.json                  # your FAQ data (columns: question, answer)
├── requirements.txt
└── README.md
```

## Setup

**Requirements:** Python 3.9 or newer.

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

# 2. (Optional) create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

`requirements.txt`:

```
pandas
chromadb
```

## Usage

### Run the script

```bash
python semantic_search.py
```

The first run downloads the embedding model, so it may take a minute. If `faqs.csv` is not found, the script uses a small built-in sample so you can try it right away.

## Example queries

These questions deliberately avoid the exact wording of the FAQs:

| Query                        | Expected top match               |
| ---------------------------- | -------------------------------- |
| `I forgot my login`        | How do I reset my password?      |
| `Where is my package?`     | How can I track my order?        |
| `Stop billing me`          | How do I cancel my subscription? |
| `Can I get my money back?` | What is your refund policy?      |

Example session (your scores will differ slightly):

```
Ask a question (or type 'exit'): I forgot my login

Top 3 matches for: 'I forgot my login'

1. Similarity: 0.62
   Q: How do I reset my password?
   A: Click 'Forgot password' on the login page and follow the email link.

2. Similarity: 0.35
   Q: How do I delete my account?
   A: Go to Settings > Account > Delete account and confirm.

3. Similarity: 0.28
   Q: How do I contact customer support?
   A: Email support@example.com or use the live chat, 9am to 5pm.
```

## Tech stack

- Python
- Pandas
- ChromaDB
