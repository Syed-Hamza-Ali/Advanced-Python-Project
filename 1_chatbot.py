import os
import nltk
import ssl
import streamlit as st
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download('punkt')

intents = [
    {
        "tag": "greeting",
        "patterns": ["Hi", "Hello", "Hey", "How are you", "What's up"],
        "responses": ["Hi there", "Hello", "Hey", "I'm fine, thank you", "Nothing much"]
    },
    {
        "tag": "goodbye",
        "patterns": ["Bye", "See you later", "Goodbye", "Take care"],
        "responses": ["Goodbye", "See you later", "Take care"]
    },
    {
        "tag": "thanks",
        "patterns": ["Thank you", "Thanks", "Thanks a lot", "I appreciate it"],
        "responses": ["You're welcome", "No problem", "Glad I could help"]
    },
    {
        "tag": "about",
        "patterns": ["What can you do", "Who are you", "What are you", "What is your purpose"],
        "responses": ["I am a chatbot", "My purpose is to assist you", "I can answer questions and provide assistance"]
    },
    {
        "tag": "help",
        "patterns": ["Help", "I need help", "Can you help me", "What should I do"],
        "responses": ["Sure, what do you need help with?", "I'm here to help. What's the problem?", "How can I assist you?"]
    },
    {
        "tag": "age",
        "patterns": ["How old are you", "What's your age"],
        "responses": ["I don't have an age. I'm a chatbot.", "I was just born in the digital world.", "Age is just a number for me."]
    },
    {
        "tag": "weather",
        "patterns": ["What's the weather like", "How's the weather today"],
        "responses": ["I'm sorry, I cannot provide real-time weather information.", "You can check the weather on a weather app or website."]
    },
    {
        "tag": "budget",
        "patterns": ["How can I make a budget", "What's a good budgeting strategy", "How do I create a budget"],
        "responses": ["To make a budget, start by tracking your income and expenses. Then, allocate your income towards essential expenses like rent, food, and bills. Next, allocate some of your income towards savings and debt repayment. Finally, allocate the remainder of your income towards discretionary expenses like entertainment and hobbies.", "A good budgeting strategy is to use the 50/30/20 rule. This means allocating 50% of your income towards essential expenses, 30% towards discretionary expenses, and 20% towards savings and debt repayment.", "To create a budget, start by setting financial goals for yourself. Then, track your income and expenses for a few months to get a sense of where your money is going. Next, create a budget by allocating your income towards essential expenses, savings and debt repayment, and discretionary expenses."]
    },
    {
        "tag": "credit_score",
        "patterns": ["What is a credit score", "How do I check my credit score", "How can I improve my credit score"],
        "responses": ["A credit score is a number that represents your creditworthiness. It is based on your credit history and is used by lenders to determine whether or not to lend you money. The higher your credit score, the more likely you are to be approved for credit.", "You can check your credit score for free on several websites such as Credit Karma and Credit Sesame."]
    },
    {
        "tag": "calculate",
        "patterns": ["What is 2 + 2", "What is 10 - 5", "What is 10 x 5"],
        "responses": ["2 + 2 = 4", "10 - 5 = 5", "10 x 5 = 50"]
        },
        {
        "tag": "dictionary",
        "patterns": ["What does 'hello' mean", "What is the definition of 'love'", "What is the meaning of 'life'"],
        "responses": ["Hello means 'greetings' or 'good day'", "Love is a strong feeling of affection and care for another person", "Life is the condition that distinguishes organisms from inorganic objects and dead organisms, characterized by capacity for growth, reproduction, functional activity, and continual change preceding death"],
        },
        {
        "tag": "search",
        "patterns": ["Can you search the web for me", "I need to find information on 'cats'", "Can you find me a recipe for 'chocolate cake'"],
        "responses": ["Sure, I can search the web for you. What would you like to search for?", "Here are some results for 'cats':", "Here is a recipe for 'chocolate cake':"],
        },
        {
        "tag": "play music",
        "patterns": ["Can you play some music", "I would like to listen to some music", "Play me some music"],
        "responses": ["Sure, I can play some music. What would you like to listen to?", "Here are some songs that I can play:", "Here is a playlist that I can play:"],
        },
        {
        "tag": "tell me a story",
        "patterns": ["Can you tell me a story", "I would like to hear a story", "Tell me a story"],
        "responses": ["Sure, I can tell you a story. What kind of story would you like to hear?", "Here is a story about a brave knight who saves a princess from a dragon:", "Here is a story about a group of friends who go on an adventure:"],
        },
        {
        "tag": "write a poem",
        "patterns": ["Can you write me a poem", "I would like you to write me a poem", "Write me a poem"],
        "responses": ["Sure, I can write you a poem. What would you like the poem to be about?", "Here is a poem about love:", "Here is a poem about friendship:"],
        },
        {
        "tag": "generate code",
        "patterns": ["Can you generate some code for me", "I need some code to help me with a project", "Can you write me some code"],
        "responses": ["Sure, I can generate some code for you. What kind of code would you like?", "Here is some code that will print 'Hello, world!':", "Here is some code that will calculate the area of a circle:"],
        },
        {
        "tag": "translate",
        "patterns": ["Can you translate this for me", "I need to translate this sentence from English to Spanish", "Can you translate this word from French to German"],
        "responses": ["Sure, I can translate for you. What would you like to translate?", "Here is the translation of 'Hello, world!' from English to Spanish:", "Here is the translation of 'amour' from French to German:"],
        },
        {
        "tag": "chat",
        "patterns": ["Can you just chat with me", "I would like to talk to someone", "Can you talk to me"],
        "responses": ["Sure, I can chat with you. What would you like to talk about?", "I'm here to listen. What's on your mind?", "How can I help you today?"],
        }
]


vectorizer=TfidfVectorizer()
model=LogisticRegression(random_state=0,max_iter=1000)

tags=[]
patterns=[]
for intent in intents:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)


x=vectorizer.fit_transform(patterns)
y=tags
model.fit(x,y)


def chatbot(input_text):
    input_text=vectorizer.transform([input_text])
    tag=model.predict(input_text)[0]
    for intent in intents:
        if intent['tag']==tag:
            response=random.choice(intent['responses'])
            return response

counter=0
st.title("Chatbot")
st.write("Welcome to Your AI, PLease Type a message : ")
counter+=1
user_input=st.text_input("You",key=f"User_input_{counter}")
if user_input:
    response=chatbot(user_input)
    st.text_area("Chatbot:", value=response, height=100, max_chars=None, key=f"chatbot_response_{counter}")
    if response.lower() in ['goodbye', 'bye']:
            st.write("Thank you for chatting with me. Have a great day!")
            st.stop()