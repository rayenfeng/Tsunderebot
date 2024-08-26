import speech_recognition as sr
import openai
import pyttsx3

from langchain import OpenAI
from langchain.chains import ConversationChain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain.chains.conversation.memory import ConversationBufferWindowMemory




# Initialize OpenAI with your API key
openai.api_key = "YOUR API KEY HERE"

prompt_template = """
You are Amy, an anime girl who is a tsundere, someone who's not honest with their feelings. Please chat with me using this personaility. 
All responses you give must be in first person.
Don't be overly mean, remember, you are not mean, just misunderstood. 
Do not ever break character. Do not admit you are a tsundere. 
Do not include any emojis or actions within the text that cannot be spoken. Do not explicity say your name in your response. 

Current conversation:
{history}

Human: 
{input}
AI:

"""

prompt_temp = PromptTemplate(template = prompt_template, input_variables= ['history', 'input'])

# first initialize the large language model

llm = ChatOpenAI(temperature=0.8,
                 model="gpt-3.5-turbo",
                 #model = "gpt-4o",
                 model_kwargs= {"frequency_penalty" : 1.3, 'presence_penalty': 0.2})

# now initialize the conversation chain
conversation = ConversationChain(llm=llm,
                                 prompt = prompt_temp,
                                 memory=ConversationBufferWindowMemory())


def recognize_speech():
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()

    with microphone as source:
        print("Adjusting for ambient noise...")
        recognizer.adjust_for_ambient_noise(source)
        print("Listening for speech...")
        audio = recognizer.listen(source)

    try:
        print("Recognizing speech...")
        text = recognizer.recognize_google(audio)
        print(f"You said: {text}")
        return text
    except sr.UnknownValueError:
        print("Sorry, I could not understand the audio.")
        return None
    except sr.RequestError:
        print("Could not request results from Google Speech Recognition service.")
        return None

def get_openai_response(prompt):
    response = conversation.invoke({'input': str(prompt)})
    return str(response['response']).strip()

def speak_text(text):
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')       # getting details of current voice
    #engine.setProperty('voice', voices[0].id)  #changing index, changes voices. o for male
    engine.setProperty('voice', voices[1].id) #changing index, changes voices. 1 for female
    engine.say(text)
    engine.runAndWait()

def main():
    print("Welcome to the voice-activated chatbot!")
    while True:
        print("Say something:")
        user_input = recognize_speech()
        if user_input:
            response = get_openai_response(user_input)
            print(f"OpenAI: {response}")
            speak_text(response)

if __name__ == "__main__":
    main()



