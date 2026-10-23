# %% [markdown]
# Source notebook: L2_data.ipynb
# Cell order and explanatory text are preserved; saved outputs are omitted.
# See SOURCE_FILES.md at the repository root for runtime requirements.

# %% [markdown]
# # L2: Data Exploration for Tuning a Foundation Model

# %% [markdown]
# **Project environment setup:**
# 
# - Load credentials and relevant Python Libraries
# - If you were running this notebook locally, you would first install Vertex AI. In this classroom, this is already installed.
# 
# ```
# !pip install google-cloud-aiplatform
# ```
# - You can download the `requirements.txt` for this course from the workspace of this lab. `File --> Open...`

# %%
# Notebook cell 3
from utils import authenticate
credentials, PROJECT_ID = authenticate()

# %%
# Notebook cell 4
REGION = "us-central1"

# %% [markdown]
# - Import the [Vertex AI](https://cloud.google.com/vertex-ai) SDK.
# - The library helps to interact with the Vertex AI services in the cloud.
# - Initialize it. 

# %%
# Notebook cell 6
import vertexai

# %%
# Notebook cell 7
vertexai.init(project = PROJECT_ID,
              location = REGION,
              credentials = credentials)

# %% [markdown]
# - Import [BigQuery](https://cloud.google.com/bigquery) to use as your data warehouse.
# - Initialize the client to start interacting with the data warehouse, send SQL and retrieve data into the notebook.

# %%
# Notebook cell 9
from google.cloud import bigquery

# %%
# Notebook cell 10
bq_client = bigquery.Client(project=PROJECT_ID,
                            credentials = credentials)

# %% [markdown]
# ## Stack Overflow Public Dataset

# %% [markdown]
# - You will use [Stack Overflow Data](https://cloud.google.com/blog/topics/public-datasets/google-bigquery-public-datasets-now-include-stack-overflow-q-a) on BigQuery Public Datasets.
# - The datasets include questions, answers and metadata related to Stack Overflow questions. Within this dataset, there are tables with data.
# - Create a SQL query.

# %%
# Notebook cell 13
QUERY_TABLES = """
SELECT
  table_name
FROM
  `bigquery-public-data.stackoverflow.INFORMATION_SCHEMA.TABLES`
"""

# %% [markdown]
# - The query is asking to retrieve `table_name` of all the `TABLES`
# - Use the client to send your SQL and retrieve the data (tables names).

# %%
# Notebook cell 15
query_job = bq_client.query(QUERY_TABLES)

# %%
# Notebook cell 16
for row in query_job:
    for value in row.values():
        print(value)

# %% [markdown]
# ## Data Retrieval

# %% [markdown]
# - You'll fetch some data from the data warehouse and store it in Pandas dataframe for visualization.
# - Select all columns from  `posts_questions` and put the `LIMIT` as 3.

# %%
# Notebook cell 19
INSPECT_QUERY = """
SELECT
    *
FROM
    `bigquery-public-data.stackoverflow.posts_questions`
LIMIT 3
"""

# %%
# Notebook cell 20
import pandas as pd

# %%
# Notebook cell 21
query_job = bq_client.query(INSPECT_QUERY)

# %% [markdown]
# - Take the results of the query `-->` create an arrow table (which is part of [Apache Framework](https://arrow.apache.org/docs/index.html)) `-->` which goes into a Pandas dataframe.
# - This allows for data to be in a format which is easier to read and explore with Pandas.

# %%
# Notebook cell 23
stack_overflow_df = query_job\
    .result()\
    .to_arrow()\
    .to_pandas()
stack_overflow_df.head()

# %% [markdown]
# ### Dealing with Large Datasets
# 
# - Large datasets for LLMs often don't fit into memory.
# - Select all of the columns and rows of the table `posts_questions`.

# %%
# Notebook cell 25
QUERY_ALL = """
SELECT
    *
FROM
    `bigquery-public-data.stackoverflow.posts_questions` q
"""

# %%
# Notebook cell 26
query_job = bq_client.query(QUERY_ALL)

# %%
# Notebook cell 27
try:
    stack_overflow_df = query_job\
    .result()\
    .to_arrow()\
    .to_pandas()
except Exception as e:
    print('The DataFrame is too large to load into memory.', e)

# %% [markdown]
# **Note:** The data is too large to return, as it is not fitting into memory.

# %% [markdown]
# #### Joining Tables and Query Optimization
# 
# - When working with (large) data, query optimizing is needed in order to save time and resources.
# - Select questions as `input_text` (column 1), answers as `output_text` (column 2).
# - Take the questions from `posts_questions` and answers from `posts_answers`.
# - Join the questions and their corresponding accepted answers based on their same `unique ID`.
# - Making sure the question is about `Python`, and that it `has an answer`. And the date the question was posted is on or after `2020-01-01`
# - Limit as 10,000

# %%
# Notebook cell 30
QUERY = """
SELECT
    CONCAT(q.title, q.body) as input_text,
    a.body AS output_text
FROM
    `bigquery-public-data.stackoverflow.posts_questions` q
JOIN
    `bigquery-public-data.stackoverflow.posts_answers` a
ON
    q.accepted_answer_id = a.id
WHERE
    q.accepted_answer_id IS NOT NULL AND
    REGEXP_CONTAINS(q.tags, "python") AND
    a.creation_date >= "2020-01-01"
LIMIT
    10000
"""

# %%
# Notebook cell 31
query_job = bq_client.query(QUERY)

# %%
# Notebook cell 32
### this may take some seconds to run
stack_overflow_df = query_job.result()\
                        .to_arrow()\
                        .to_pandas()

stack_overflow_df.head(2)

# %% [markdown]
# ### Adding Instructions
# 
# - Instructions for LLMs have been shown to improve
# model performance and generalization to unseen tasks [(Google, 2022)](https://arxiv.org/pdf/2210.11416.pdf).
# - Wihtout the instruction, it is only question and answer. Model might not understand what to do.
# - With the instructions, the model gets a guideline as to what task to perform.

# %%
# Notebook cell 34
INSTRUCTION_TEMPLATE = f"""\
Please answer the following Stackoverflow question on Python. \
Answer it like you are a developer answering Stackoverflow questions.

Stackoverflow question:
"""

# %% [markdown]
# - A new column will combine `INSTRUCTION_TEMPLATE` and the question `input_text`.
# - This avoids overwritting of any existing column which might be needed.

# %%
# Notebook cell 36
stack_overflow_df['input_text_instruct'] = INSTRUCTION_TEMPLATE + ' '\
    + stack_overflow_df['input_text']

# %% [markdown]
# ### Dataset for Tuning
# 
# - Divide the data into a training and evaluation. By default, 80/20 split is used.
# - This (80/20 split) allows for more data to be used for tuning. The evaluation split is used as unseen data during tuning to evaluate performance.
# - The `random_state` parameter is used to ensure random sampling for a fair comparison. 

# %%
# Notebook cell 38
from sklearn.model_selection import train_test_split

# %%
# Notebook cell 39
train, evaluation = train_test_split(
    stack_overflow_df,
    ### test_size=0.2 means 20% for evaluation
    ### which then makes train set to be of 80%
    test_size=0.2,
    random_state=42
)

# %% [markdown]
# #### Different Datasets and Flow
# 
# - Versioning data is important.
# - It allows for reproducibility, traceability, and maintainability of machine learning models.
# - Get the timestamp.

# %%
# Notebook cell 41
import datetime

# %%
# Notebook cell 42
date = datetime.datetime.now().strftime("%H:%d:%m:%Y")

# %% [markdown]
# - Generate a `jsonl` file.
# - Name it as `tune_data_stack_overflow_python_qa-{date}`

# %%
# Notebook cell 44
cols = ['input_text_instruct','output_text']
tune_jsonl = train[cols].to_json(orient="records", lines=True)

# %%
# Notebook cell 45
training_data_filename = f"tune_data_stack_overflow_\
                            python_qa-{date}.jsonl"

# %%
# Notebook cell 46
with open(training_data_filename, "w") as f:
    f.write(tune_jsonl)

# %% [markdown]
# ## Try it Yourself! - Evaluation Set
# 
# The code above generted a `jsonl` file for the `train` set. Now, its time for you to make the `evaluation` set, which you can name as `tune_eval_data_stack_overflow_python_qa-{date}.jsonl`. The code for that is also provided to you in the drop down below, but we encourage you to try it yourself first before you look at it.
# 
# <details>
#   <summary><font size="2" color="darkgreen"><b>Code for Evaluation Set (Click to expand)</b></font></summary>
#     
# ```python
# 
# cols = ['input_text_instruct','output_text']
# ### you need to use the "evaluation" set now
# tune_jsonl = evaluation[cols].to_json(orient="records", lines=True)
# 
# ### change the file name
# ### use "tune_eval_data_stack_overflow_python_qa-{date}.jsonl"
# evaluation_data_filename = f"tune_eval_data_stack_overflow_\
#                             python_qa-{date}.jsonl"
# 
# ### write the file
# with open(evaluation_data_filename, "w") as f:
#     f.write(tune_jsonl)
# 
# ```

# %%
# Notebook cell 48
cols = ['input_text_instruct','output_text']
### you need to use the "evaluation" set now
tune_jsonl = evaluation[cols].to_json(orient="records", lines=True)

# %%
# Notebook cell 49
evaluation_data_filename = f"tune_eval_data_stack_overflow_\
                            python_qa-{date}.jsonl"

# %%
# Notebook cell 50
with open(evaluation_data_filename, "w") as f:
    f.write(tune_jsonl)
