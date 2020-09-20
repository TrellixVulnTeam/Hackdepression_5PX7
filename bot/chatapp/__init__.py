import csv
from flask import Flask, render_template, request
from chatapp.interact import download_pretrained_model, get_dataset, add_special_tokens_, sample_sequence
import torch
import random
from transformers import OpenAIGPTLMHeadModel, OpenAIGPTTokenizer
 
app = Flask(__name__)


with open('./chatapp/data/censoring.csv', 'r') as triggers_file:
    triggers = {row[0]:row[1] for row in csv.reader(triggers_file)}

max_history = 2
min_length, max_length = 1, 20
dataset_path = './chatapp/data/counsel_chat_250-tokens_full.json'
dataset_cache = './chatapp/dataset_cache'
model_checkpoint = download_pretrained_model()
device = "cpu"
seed = 0
random.seed(seed)
torch.random.manual_seed(seed)
torch.cuda.manual_seed(seed)

# Get pretrained model and tokenizer
tokenizer = OpenAIGPTTokenizer.from_pretrained(model_checkpoint)
model = OpenAIGPTLMHeadModel.from_pretrained(model_checkpoint)
model.to(device)
add_special_tokens_(model, tokenizer)

# Sample a personality
dataset = get_dataset(tokenizer, dataset_path, dataset_cache)
personalities = [dialog["personality"] for dataset in dataset.values() for dialog in dataset]
personality = random.choice(personalities)

history = []

@app.route("/")
def home():
    return render_template("botpage.html")


@app.route("/get")
def get_bot_response():
    global history
    userText = request.args.get('msg')
    if userText == '':
        return 'please give me something to reply to!'
    if 'suicide' in userText or 'kill myself' in userText or 'kill me' in userText or 'want to die' in userText:
        return 'oh no! based on what i\'m hearing, i\'m not best equipped to talk with you--a professional is, however. if you\'re \
        having unsafe thoughts, please call the US National Suicide Prevention Lifeline at 800-273-8255 or talk to someone you trust. \
        sending you love <33 in the meantime, please feel free to continue talking to me; i\'m always here to listen!'
    history.append(tokenizer.encode(userText))
    with torch.no_grad():
        out_ids = sample_sequence(personality, history, tokenizer, model, max_length, min_length, device)
    history.append(out_ids)
    history = history[-(2 * max_history + 1):]
    reply = str(tokenizer.decode(out_ids, skip_special_tokens=True))
    censored_reply = reply
    for trigger in list(reversed(triggers.keys())):
        if trigger in censored_reply and censored_reply[censored_reply.find(trigger)-1] == ' ' and censored_reply[censored_reply.find(trigger)+trigger] == ' ':
            censored_reply = censored_reply.replace(trigger, triggers[trigger])
    return censored_reply


@app.route("/resetme")
def reset_bot():
    global history, tokenizer, model, personality
    dataset_path = './chatapp/data/counsel_chat_250-tokens_full.json'
    dataset_cache = './chatapp/dataset_cache'
    model_checkpoint = download_pretrained_model()
    device = "cpu"
    seed = random.randrange(0,100)
    random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Get pretrained model and tokenizer
    tokenizer = OpenAIGPTTokenizer.from_pretrained(model_checkpoint)
    model = OpenAIGPTLMHeadModel.from_pretrained(model_checkpoint)
    model.to(device)
    add_special_tokens_(model, tokenizer)

    # Sample a personality
    dataset = get_dataset(tokenizer, dataset_path, dataset_cache)
    personalities = [dialog["personality"] for dataset in dataset.values() for dialog in dataset]
    personality = random.choice(personalities)

    history = []
    return ""

    
app.config.from_pyfile('../config.py') 

 
if __name__ == "__main__":
    app.run()
