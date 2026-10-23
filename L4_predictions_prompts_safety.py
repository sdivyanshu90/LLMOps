# %% [markdown]
# Source notebook: L4_predictions_prompts_safety.ipynb
# Cell order and explanatory text are preserved; saved outputs are omitted.
# See SOURCE_FILES.md at the repository root for runtime requirements.

# %% [markdown]
# # L4: Predictions, Prompts and Safety

# %% [markdown]
# - Load the Project ID and credentials.

# %%
# Notebook cell 3
from utils import authenticate
credentials, PROJECT_ID = authenticate()

# %%
# Notebook cell 4
REGION = "us-central1"

# %% [markdown]
# - Import the [Vertex AI](https://cloud.google.com/vertex-ai) SDK.
# - Import and load the model.
# - Initialize it.

# %%
# Notebook cell 6
import vertexai
from vertexai.language_models import TextGenerationModel

# %%
# Notebook cell 7
vertexai.init(project = PROJECT_ID,
              location = REGION,
              credentials = credentials)

# %% [markdown]
# ## Deployment
# 
# ### Load Balancing

# %% [markdown]
# - Load from pre-trained `text-bison@001`
# - Retrieve the endpoints (deployed as REST API)

# %%
# Notebook cell 10
model = TextGenerationModel.from_pretrained("text-bison@001")

# %% [markdown]
# - Get the list of multiple models executed and deployed.
# - This helps to rout the traffic to different endpoints.

# %%
# Notebook cell 12
list_tuned_models = model.list_tuned_model_names()

# %%
# Notebook cell 13
for i in list_tuned_models:
    print (i)

# %% [markdown]
# - Randomly select from one of the endpoints to divide the prediction load.

# %%
# Notebook cell 15
import random

# %%
# Notebook cell 16
tuned_model_select = random.choice(list_tuned_models)

# %% [markdown]
# ### Getting the Response
# 
# - Load the endpoint of the randomly selected model to be called with a prompt.
# - The prompt needs to be as similar as possible as the one you trained your model on (python questions from stack overflow dataset)

# %%
# Notebook cell 18
deployed_model = TextGenerationModel.get_tuned_model\
(tuned_model_select)

# %%
# Notebook cell 19
PROMPT = "How can I load a csv file using Pandas?"

# %% [markdown]
# - Use `deployed_model.predit` to call the API using the prompt. 

# %%
# Notebook cell 21
### depending on the latency of your prompt
### it can take some time to load
response = deployed_model.predict(PROMPT)

# %%
# Notebook cell 22
print(response)

# %% [markdown]
# - `pprint` makes the response easily readable.

# %%
# Notebook cell 24
from pprint import pprint

# %% [markdown]
# - Sending multiple prompts can return multiple responses `([0], [1], [2]...)`
# - In this example, only 1 prompt is being sent, and returning only 1 response `([0])`

# %%
# Notebook cell 26
### load the first object of the response
output = response._prediction_response[0]

# %%
# Notebook cell 27
### print the first object of the response
pprint(output)

# %%
# Notebook cell 28
### load the second object of the response
output = response._prediction_response[0][0]

# %%
# Notebook cell 29
### print the second object of the response
pprint(output)

# %%
# Notebook cell 30
### retrieve the "content" key from the second object
final_output = response._prediction_response[0][0]["content"]

# %%
# Notebook cell 31
### printing "content" key from the second object
print(final_output)

# %% [markdown]
# #### Prompt Management and Templates
# - Remember that the model was trained on data that had an `Instruction` and a `Question` as a `Prompt` (Lesson 2).
# - In the example above, *only*  a `Question` as a `Prompt` was used for a response.
# - It is important for the production data to be the same as the training data. Difference in data can effect the model performance.
# - Add the same `Instruction` as it was used for training data, and combine it with a `Question` to be used as a `Prompt`.

# %%
# Notebook cell 33
INSTRUCTION = """\
Please answer the following Stackoverflow question on Python.\
Answer it like\
you are a developer answering Stackoverflow questions.\
Question:
"""

# %%
# Notebook cell 34
QUESTION = "How can I store my TensorFlow checkpoint on\
Google Cloud Storage? Python example?"

# %% [markdown]
# - Combine the intruction and the question to create the prompt.

# %%
# Notebook cell 36
PROMPT = f"""
{INSTRUCTION} {QUESTION}
"""

# %%
# Notebook cell 37
print(PROMPT)

# %% [markdown]
# - Get the response using the new prompt, which is consistent with the prompt used for training.

# %%
# Notebook cell 39
final_response = deployed_model.predict(PROMPT)

# %%
# Notebook cell 40
output = final_response._prediction_response[0][0]["content"]

# %% [markdown]
# - Note how the response changed from earlier.

# %%
# Notebook cell 42
print(output)

# %% [markdown]
# ### Safety Attributes
# - The reponse also includes safety scores.
# - These scores can be used to make sure that the LLM's response is within the boundries of the expected behaviour.
# - The first layer for this check, `blocked`, is by the model itself.

# %%
# Notebook cell 44
### retrieve the "blocked" key from the 
### "safetyAttributes" of the response
blocked = response._prediction_response[0][0]\
['safetyAttributes']['blocked']

# %% [markdown]
# - Check to see if any response was blocked.
# - It returns `True` if there is, and `False` if there's none.

# %%
# Notebook cell 46
print(blocked)

# %% [markdown]
# - The second layer of this check can be defined by you, as a practitioner, according to the thresholds you set.
# - The response returns probabilities for each safety score category which can be used to design the thresholds.

# %%
# Notebook cell 48
### retrieve the "safetyAttributes" of the response
safety_attributes = response._prediction_response[0][0]\
['safetyAttributes']

# %%
# Notebook cell 49
pprint(safety_attributes)

# %% [markdown]
# ### Citations
# - Ideally, a LLM should generate as much original cotent as possible.
# - The `citationMetadata` can be used to check and reduce the chances of a LLM generating a lot of existing content.

# %%
# Notebook cell 51
### retrieve the "citations" key from the 
### "citationMetadata" of the response
citation = response._prediction_response[0][0]\
['citationMetadata']['citations']

# %% [markdown]
# - Returns a list with information if the response is cited, if not, then it retuns an empty list.

# %%
# Notebook cell 53
pprint(citation)

# %% [markdown]
# ## Try it Yourself! - Return a Citation
# 
# Now it is time for you to come up with an example, for which the model response should return citation infomration. The idea here is to see how that would look like, so keeping it basic, the prompt can be different than the coding examples used above. Below code is one such prompt:

# %%
# Notebook cell 55
PROMPT = "Finish the sentence: To be, or not "

# %%
# Notebook cell 56
response = deployed_model.predict(PROMPT)

# %%
# Notebook cell 57
### output of the model
output = response._prediction_response[0][0]["content"]
print(output)

# %%
# Notebook cell 58
### check for citation
citation = response._prediction_response[0][0]\
['citationMetadata']['citations']

pprint(citation)

# %% [markdown]
# **Your turn!**

# %%
# Notebook cell 60


# %%
# Notebook cell 61


# %%
# Notebook cell 62


# %%
# Notebook cell 63


# %% [markdown]
# ### Optional Notebook
# 
# [Tuning and deploy a foundation model](https://github.com/GoogleCloudPlatform/generative-ai/blob/main/language/tuning/tuning_text_bison.ipynb)

# %%
# Notebook cell 65

