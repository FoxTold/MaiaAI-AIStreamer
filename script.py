from time import sleep
def ready_event(channel, link):                
    history = retrieve_past()                  
    change_interval(0.1)                      

def message_event(msg): 
    author = msg[0]     
    content = msg[1]    
    message_id = msg[2] 
    user_id = msg[3]    
    
    text = llm.generate(f"{content}")
    tts.text_to_speech(text)
    print(author+": "+content)      
    
    sleep(5)

def tick(): 
    pass    

exec(open("main.py").read()) 