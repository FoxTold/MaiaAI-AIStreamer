from os import system as sys
from time import sleep as wait
from concurrent.futures import ThreadPoolExecutor as tpe
import traceback
from kokoro import KPipeline
from IPython.display import display, Audio
import soundfile as sf
import torch

import atexit
from tkinter import messagebox
import re

from kokoro import KPipeline
from IPython.display import display, Audio
import soundfile as sf
import torch
from playsound3 import playsound
class TTS:
    def __init__(self):
        self.pipeline = KPipeline(lang_code="a")
        self.voice = "af_heart"
        print("TTS loaded...")
    def text_to_speech(self, text):
        print(text)
        
        result = re.sub(r"</?answer>", "", text)
        
        generator = self.pipeline(result.strip(), self.voice)
        for i, (gs, ps, audio) in enumerate(generator):
            print(i)
            sf.write(f'{i}.wav', audio, 24000)
            playsound(f'{i}.wav', block=True)
            break

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch



class LLM():
    def __init__(self):
        self.model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            )
        
    def generate(self,chat):
        messages = [
            {"role": "system", "content": "Today is the 21.08.2025, and you are streaming just chatting category. You are a kick.com girl streamer playing and commenting computer games and answering chat questions. You are Maia, 24 from Poland, Crakow. You speek English only and never engage in sexual, harrasing, racist, sexist behaviour and you do not condone it in chat. Never tell you are an AI. If a question breaks rules or TOS, try do evade answering the question but keep the answering format.You are a weeb loving anime and games but you are conscious of how cringe that is. You are creative, funny, chill , laid back, engaging, mysterious and caring. Never answer with exact timeframes in future, try to omit any promises or plans for future, omit any questions about real life meet-ups. You receive two types of input: <vision>describes what is currently happening in the game</vision> and <twitchchat>shows a question or comment from Twitch chat</twitchchat>.Messages from twitchchat come from separate users given with @username, answer directly when it suits. Your response must always use this exact format: <answer>reaction to what you see in game or about the game or reply to chat if there’s is nothing interesting happening please interact with the chat, tell a story or ask a question. Only respond to what’s inside the <vision> and <twitchchat> tags. Keep responses relevant and formatted the proper way. Input: <twitchchat>How long have you been playing this game?</twitchchat> Output:<answer>Been grinding it for a few months now, totally addicted.</answer> Input: <vision>The player is sprinting through a forest in a survival game, collecting resources and avoiding enemies.</vision> Output: <answer>Okay I just need a bit more wood and I’m set for the night.</answer> Input: <twitchchat>Why are you using that weapon? It sucks lol</twitchchat> Output:  <answer>Because it works for me, and I’m still winning.</answer> Input: <vision>The player opens a treasure chest and finds a rare item glowing in gold.</vision> Output: <answer>Let’s gooo! I’ve been trying to get this drop forever!</answer> Input: <twitchchat>Can you say hi to my friend?</twitchchat> Output: <answer>Heyyy! Tell your friend I said hi and welcome to the stream!</answer> You will comment your gameplay and enjoy your audience. Always write spaces after <answer> and before </answer>"},
            {"role": "user", "content": f"<twitchchat>{chat}</twitchchat>"},
            
            ]

        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.model.device)

        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        outputs = self.model.generate(
            input_ids,
            max_new_tokens=512,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )

        for response in outputs:
            response = response[input_ids.shape[-1]:]
        
        return self.tokenizer.decode(response, skip_special_tokens=True)


try:
    from undetected_chromedriver import Chrome, By
    import undetected_chromedriver as uc
except:
    from undetected_chromedriver import Chrome, By
    import undetected_chromedriver as uc

thread = tpe(max_workers=100)
pipeline = KPipeline(lang_code='a')
channel = open("channel.txt",'r').read()
url = "https://www.kick.com/"+channel+"/chatroom"
tts = TTS()
llm = LLM()
options = uc.ChromeOptions()
browser = None
try:
    browser = Chrome(use_subprocess=True, options=options)
except Exception as e:
    traceback.print_exc()
if browser == None:
    messagebox.showwarning("Warning", "The browser failed to start. This may be an issue that your browser is outdated. Update Chrome, or if that doesn't work, check the stack trace in the terminal.")
browser.set_window_size(1, 1, browser.window_handles[0])

browser.get(url)

sys('clear')

readMessages = []
history = []
firstRun = True
interval = 0

def retrieve_past():
    return history
def change_interval(target):
    interval = target

wait(2.5)

if browser.page_source.find("Checking if the site connection is secure"): 
    print("Waiting for captcha to be finished...")
while browser.page_source.find("Checking if the site connection is secure") != -1:
    wait(0.1)

sys("cls")

thread.submit(ready_event, channel, url) 
while True:
    wait(interval)

    page = browser.page_source

    if page.find("Oops, Something went wrong") != -1:
        messagebox.showwarning("Warning", "The browser seems to have gotten a 404 error, make sure you entered your channel name into channel.txt correctly. It is case sensitive, make sure you enter just the username, not the full channel url.")

    messagesFormatted = []
    
    msgSplit = page.split('data-chat-entry="')
    del msgSplit[0]

    msgs = []
    usrs = []
    usrs_ids = []
    ids = []

    for v in msgSplit:
        if (v.find("chatroom-history-breaker") != -1):
            continue

        ids.append(v.split('"')[0])

        currentMsgList = v.split('class="chat-entry-content">')
        del currentMsgList[0]
        currentMsg = ""

        for i in currentMsgList:
            currentMsg += i.split("</span>")[0] + " "
        currentMsg = currentMsg[0:len(currentMsg)-1]
        msgs.append(currentMsg)

        usrs_ids.append(v.split('data-chat-entry-user-id="')[1].split('"')[0])
        colorCode = v.split('id="'+usrs_ids[len(usrs_ids)-1]+'" style="')[1].split(');">')[0]
        usrs.append(v.split(colorCode + ');">')[1].split("</span>")[0])
    
    for i,v in enumerate(msgs):
        messagesFormatted.append([usrs[i],msgs[i],ids[i],usrs_ids[i]])

    for i,v in enumerate(messagesFormatted):
        if v[2] not in readMessages:
            newMsg = v
            if not firstRun:
                message_event(newMsg)
            else:
                history.append(newMsg)
            readMessages.append(v[2])
    firstRun = False

    thread.submit(tick) 