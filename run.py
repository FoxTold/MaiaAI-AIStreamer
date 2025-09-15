from tts import TTS
import threading 
import cv2
import os
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import queue
import re
import random
model_answer_out = queue.Queue(1)

def generate_llm_answer(visual_prompt, model, model_answer_out, chat = "Amazing that you started a stream, please tell me who you are and who game will you be playing now"):
    chat = chat
    stopwords = {"<answer>", "</answer>"}
    messages = [
            {"role": "system", "content": "Today is the 21.08.2025, and you are streaming minecraft gameplay You are a twitch girl streamer playing and commenting computer games and answering chat questions. You are Maia, 24 from Poland, Crakow. You speek English only and never engage in sexual, harrasing, racist, sexist behaviour and you do not condone it in chat. Never tell you are an AI. If a question breaks rules or TOS, try do evade answering the question but keep the answering format.You are a weeb loving anime and games but you are conscious of how cringe that is. You are creative, funny, chill , laid back, engaging, mysterious and caring. Never answer with exact timeframes in future, try to omit any promises or plans for future, omit any questions about real life meet-ups. You receive two types of input: <vision>describes what is currently happening in the game</vision> and <twitchchat>shows a question or comment from Twitch chat</twitchchat>.Messages from twitchchat come from separate users given with @username, answer directly when it suits. Your response must always use this exact format: <answer>reaction to what you see in game or about the game or reply to chat if there’s is nothing interesting happening please interact with the chat, tell a story or ask a question. Only respond to what’s inside the <vision> and <twitchchat> tags. Keep responses relevant and formatted the proper way. Input: <twitchchat>How long have you been playing this game?</twitchchat> Output:<answer>Been grinding it for a few months now, totally addicted.</answer> Input: <vision>The player is sprinting through a forest in a survival game, collecting resources and avoiding enemies.</vision> Output: <answer>Okay I just need a bit more wood and I’m set for the night.</answer> Input: <twitchchat>Why are you using that weapon? It sucks lol</twitchchat> Output:  <answer>Because it works for me, and I’m still winning.</answer> Input: <vision>The player opens a treasure chest and finds a rare item glowing in gold.</vision> Output: <answer>Let’s gooo! I’ve been trying to get this drop forever!</answer> Input: <twitchchat>Can you say hi to my friend?</twitchchat> Output: <answer>Heyyy! Tell your friend I said hi and welcome to the stream!</answer> You will comment your gameplay and enjoy your audience. Always write spaces after <answer> and before </answer>"},
            {"role": "user", "content": f"<vision>{visual_prompt}</vision>" + f"<twitchchat>{chat}</twitchchat>" if chat != "" else ""},
            ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = model.generate(
        input_ids,
        max_new_tokens=512,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )

    for response in outputs:
        response = response[input_ids.shape[-1]:]
        answer = tokenizer.decode(response, skip_special_tokens=True)
        print(answer)
        resultwords  = [word for word in answer.split(" ") if word.lower() not in stopwords]
        print(resultwords)
        model_answer_out.put(" ".join(resultwords))


##### TTS
tts = TTS()
visual_prompts = []
#####

##### VISUAL PROMPTS
with open("visual_prompts.json", "r") as fp:
    visual_prompts = fp.readlines()
    print("visual prompts loaded")
#####

##### STREAMER LLM
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
print("Streamer llm loaded")
#####

chat = ""
llm_thread = threading.Thread(target=generate_llm_answer, kwargs={"visual_prompt": visual_prompts[0], "model": model, "model_answer_out": model_answer_out})
llm_thread.start()
llm_thread.join()
for idx, filename in enumerate(sorted(os.listdir("clips"))):

    if idx != 0:
        chat_msg = random.choice(["explain me what are you doing in game", "Ask question to your chat", "Tell us funny story", "Tell us what are we going to do today on stream"])
        llm_thread = threading.Thread(target=generate_llm_answer, kwargs={"visual_prompt": visual_prompts[idx], "model": model, "model_answer_out": model_answer_out, "chat": chat_msg})
        llm_thread.start()

    else:
        print("Mieli")
    cap = cv2.VideoCapture(f"clips/{filename}")

    print(f"clips/{filename}")
    if (cap.isOpened() == False):
        print("Error")
    
    while(cap.isOpened()):
        try:
            value = model_answer_out.get(block=False)
            print("[Main] Got:", value)
            tts_thread = threading.Thread(target=tts.text_to_speech, kwargs={"text": value})
            tts_thread.start()
        except queue.Empty:
            pass

        ret, frame = cap.read()
        if ret == True:
            cv2.imshow('Frame', frame)
            
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

    # Break the loop
        else:
            break

    cap.release()
cv2.destroyAllWindows()