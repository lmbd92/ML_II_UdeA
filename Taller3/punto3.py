# %% [markdown]
# Tools to use in the project

# %%
import torch

# %%
import resource
import os

# %%

# %%
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertForQuestionAnswering


# %%
file_name = "./text.txt"

# Check if the file exists
if not os.path.exists(file_name):
    raise FileNotFoundError(f"The file {file_name} does not exist.")

print(f"File Size is {os.stat(file_name).st_size / (1024 * 1024)} MB")

# Use a with statement to automatically close the file when done
with open(file_name, "r", encoding="utf-8") as txt_file:
    content = txt_file.read()

# Split the content into lines
lines = content.splitlines()

print(f"file content lines {len(lines)}")

# %% [markdown]
# # TF-IDF Retriever


def segment_documents(doc, max_doc_length=20):
    # List containing full and segmented doc
    segmented_docs = []

    for lines in doc:
        # Split document by spaces to obtain a word count that roughly approximates the token count
        split_to_words = lines.split(" ")

        # If the document is longer than our maximum length, split it up into smaller segments and add them to the list
        if len(split_to_words) > max_doc_length:
            for doc_segment in range(0, len(split_to_words), max_doc_length):
                segmented_docs.append(
                    " ".join(split_to_words[doc_segment : doc_segment + max_doc_length])
                )

        # If the document is shorter than our maximum length, add it to the list
        else:
            segmented_docs.append(lines)

    return segmented_docs


def get_top_k_chunks(query, docs, k=2):
    # Initialize a vectorizer that removes English stop words
    vectorizer = TfidfVectorizer(analyzer="word", stop_words="english")

    # Create a corpus of query and chunks and convert to TF-IDF vectors
    query_and_chunks = [query] + docs
    matrix = vectorizer.fit_transform(query_and_chunks)

    # Holds our cosine similarity scores
    scores = []

    # The first vector is our query text, so compute the similarity of our query against all chunks vectors
    for i in range(1, len(query_and_chunks)):
        scores.append(cosine_similarity(matrix[0], matrix[i])[0][0])

    # Sort list of scores and return the top k highest scoring chunks
    sorted_list = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    top_doc_indices = [x[0] for x in sorted_list[:k]]
    top_chunks = [docs[x] for x in top_doc_indices]

    return top_chunks


# %% [markdown]
# # BERT-SQUAD Retriever

# %% [markdown]
# We’ll import a BERT model that has been fine-tuned on SQUAD, a task that asks the model to return the span of words most likely to contain the answer to a given question. This will serve as the reader component of our question answering system.

model = BertForQuestionAnswering.from_pretrained(
    "bert-large-uncased-whole-word-masking-finetuned-squad"
)
tokenizer = BertTokenizer.from_pretrained(
    "bert-large-uncased-whole-word-masking-finetuned-squad"
)

# %% [markdown]
# # Calling the Model


# %%
def answer_question(question, answer_text):
    input_ids = tokenizer.encode(question, answer_text, max_length=512, truncation=True)

    # ======== Set Segment IDs ========
    # Search the input_ids for the first instance of the `[SEP]` token.
    sep_index = input_ids.index(tokenizer.sep_token_id)

    # The number of segment A tokens includes the [SEP] token itself.
    num_seg_a = sep_index + 1

    # The remainder are segment B.
    num_seg_b = len(input_ids) - num_seg_a

    # Construct the list of 0s and 1s.
    segment_ids = [0] * num_seg_a + [1] * num_seg_b

    # There should be a segment_id for every input token.
    assert len(segment_ids) == len(input_ids)

    outputs = model(
        torch.tensor([input_ids]),  # The tokens representing our input text.
        token_type_ids=torch.tensor(
            [segment_ids]
        ),  # The segment IDs to differentiate question from answer_text
        return_dict=True,
    )

    start_scores = outputs.start_logits
    end_scores = outputs.end_logits

    # ======== Reconstruct Answer ========
    # Find the tokens with the highest `start` and `end` scores.
    answer_start = torch.argmax(start_scores)
    answer_end = torch.argmax(end_scores)

    # Get the string versions of the input tokens.
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    # Start with the first token.
    answer = tokens[answer_start]

    # Select the remaining answer tokens and join them with whitespace.
    for i in range(answer_start + 1, answer_end + 1):
        # If it's a subword token, then recombine it with the previous token.
        if tokens[i][0:2] == "##":
            answer += tokens[i][2:]

        # Otherwise, add a space then the token.
        else:
            answer += " " + tokens[i]

    return answer


# %% [markdown]
# # Let's try it out!

# %%
# Enter our query here
# query = "How is AI being applied in the field of healthcare, and what benefits does it offer?"
query = "What ethical concerns are associated with the use of AI in decision-making processes, and how can they be addressed?"
# query = "Can you explain the potential impact of AI on the job market, and what measures can be taken to mitigate job displacement?"

# Segment our documents
segmented_docs = segment_documents(lines, 450)

# Retrieve the top k most relevant documents to the query
candidate_docs = get_top_k_chunks(query, segmented_docs, 3)

# Return the likeliest answers from each of our top k most relevant documents in descending order
for i in candidate_docs:
    answer = answer_question(query, i)
    print("Answer: ", answer)
    print("Reference Document: ", i)
    print()
