# Job Finder

Today's youth have the burden of seeking job opportunities in a highly complex market with a myriad of competitors. Thus, it is pivotal to figure out the sector and job that best fit the interest and skills as soon as possible and apply to the job without having to write cover letters and the like. Tasks taken care of by **Job Finder**.

---

## Project Description

The application is comprised of three agents that help the user:

- find the sector and job they'd best fit in
- and write a cover letter directed at a company of their choosing.

### Agents:

- **Agent 1**: Selects a sector that fits best with the user's interest, using the embedded user's input and a fine-tuned LLM to find what best befits the user, then returns it.
- **Agent 2**: Selects a job that fits best with the user's skills and interest, using the embedded input and a fine-tuned LLM to find what best matches — then returns it.
- **Agent 3**: Writes a cover letter using inputs given by the user, which usually ranges from prior experiences, skills, and the company of interest. In the future, it will use the user's CV as input.

---

## Tools Used

- Python
- Streamlit
- Hugging Face Transformers (LLM + Embeddings)
- LangChain (optional, if you're using it)




---

## How to Clone the Project

To download and run this project locally, use the following command:

```bash
git clone https://github.com/EdmondDant/Job-Finder
cd Job-Finder 
```

---





## Setup-- create a new env to ensure everything runs smoothly, upgrade pip, and download requirements

```bash
python3.10 -m venv env
source env/bin/activate
```




---
## upgrade pip install
```bash 

# Step 2: Upgrade pip and install all dependencies

python -m pip install --upgrade pip

```
---
## install requirements
```bash 



pip install --no-cache-dir -r requirements.txt

```
---

## Unzip download models and uncompressed modesl and vectors and  

```bash
gdown --fuzzy "https://drive.google.com/file/d/1o6q5H3vk5Xc2jKbQUStYeC92dZWcB5rF/view?usp=sharing"


gdown --fuzzy "https://drive.google.com/file/d/1ra6eQFKxHuuf5ztPBHX9-Kpa4zCFrIUS/view?usp=sharing"


gdown --fuzzy "https://drive.google.com/file/d/1jkx97AFLbUc80QDbqpPQ_RMBYFXDoA9a/view?usp=sharing"

gdown --fuzzy "https://drive.google.com/drive/folders/1MRkFm2rX-YgVCUe21JRXT7aAzJiCcLz6?usp=drive_link"

```
---
## unzip vectorstore.zip for agent1 to run smoothly
```bash

unzip career_vectorstore.zip -d career_vectorstore
unzip career_model.zip -d career_model
unzip  finetuned-flan-t5-coverletter.zip 

```


---



## Run the app via streamlit
```bash
streamlit run app.py

```

---