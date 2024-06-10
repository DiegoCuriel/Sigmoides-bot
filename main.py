import os
from dotenv import load_dotenv
import openai
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.schema import Document
from fastapi import FastAPI
import uvicorn
import discord
from discord.ext import commands
from math import sqrt

# Cargar variables de entorno
load_dotenv()

openai.api_key = os.getenv('OPENAI_API_KEY')
discord_token = os.getenv('DISCORD_TOKEN')

# Cargar documentos
def load_documents(directory):
    documents = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
                content = file.read()
                documents.append(Document(page_content=content))
    return documents

# Utiliza una ruta relativa o una variable de entorno para la ruta de los documentos
docs_path = os.getenv('DOCS_PATH', './nutricion_y_salud')
docs = load_documents(docs_path)
print(f"Se cargaron {len(docs)} documentos.")

embeddings_model = OpenAIEmbeddings(openai_api_key=openai.api_key)

vector_store = Chroma.from_documents(documents=docs, embedding=embeddings_model, persist_directory="./chroma_db")

llm = OpenAI(api_key=openai.api_key)
qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vector_store.as_retriever())

def cosine_similarity_manual(vec1, vec2):
    dot_product = sum(p*q for p, q in zip(vec1, vec2))
    magnitude = sqrt(sum([val**2 for val in vec1])) * sqrt(sum([val**2 for val in vec2]))
    if not magnitude:
        return 0
    return dot_product / magnitude

def answer_question(query):
    print(f"Recibida pregunta: {query}")
    result = qa_chain.invoke(query)
    answer = result['result']
    print(f"Respuesta generada: {answer}")
    
    # Verificar si la respuesta está dentro del contexto de los documentos cargados
    try:
        query_embedding = embeddings_model.embed_query(query)
        answer_embedding = embeddings_model.embed_query(answer)
        print(f"Embeddings generados para la pregunta y la respuesta.")

        document_embeddings = [embeddings_model.embed_query(doc.page_content) for doc in docs]
        print(f"Embeddings generados para los documentos.")

        # Calcular la similitud coseno manualmente
        similarities = [cosine_similarity_manual(answer_embedding, doc_embedding) for doc_embedding in document_embeddings]
        max_similarity = max(similarities)
        print(f"Similitud máxima: {max_similarity}")
        
        # Umbral de similitud para considerar la respuesta válida
        threshold = 0.5
        
        if max_similarity > threshold:
            print(f"Respuesta válida encontrada con similitud: {max_similarity}")
            return answer
        else:
            print(f"No hay suficiente contexto para responder la pregunta.")
            return "No tengo suficiente contexto para responder esa pregunta ._."
    except Exception as e:
        print(f"Error durante la verificación del contexto: {e}")
        return "Hubo un error al procesar tu pregunta."

# Configuración del bot de Discord
intents = discord.Intents.default()
intents.message_content = True  # Habilitar el intent de contenido de mensajes
bot = commands.Bot(command_prefix="!", intents=intents)

@bot.event
async def on_ready():
    print(f"Bot conectado como {bot.user}")

@bot.command(name='commands')
async def commands_command(ctx):
    help_text = """
    Comandos disponibles:
    !info - Información sobre el bot
    !preguntar [pregunta] - Haz una pregunta relacionada sobre la nutrición y salud
    !devs - Información sobre los desarrolladores del bot
    """
    await ctx.send(help_text)

@bot.command(name='info')
async def info_command(ctx):
    info_text = """
    Este es un bot sigma que puede responder preguntas relacionadas con la nutrición y salud.
    
    Preguntame sobre:
    Beneficios de Diferentes Alimentos
    Dietas Balanceadas
    Ejercicio y Bienestar
    """
    await ctx.send(info_text)

@bot.command(name='preguntar')
async def preguntar(ctx, *, question):
    print(f"Pregunta recibida en Discord: {question}")
    answer = answer_question(question)
    await ctx.send(f"Pregunta: {question}\nRespuesta: {answer}")

@bot.command(name='devs')
async def devs_command(ctx):
    devs_text = """
    Actividad M7 - Discord Bot

    Hugo Alejandro Gómez Herrera - A01640856
    Diego Curiel Castellanos - A01640372
    Juan Daniel Muñoz Dueñas - A01641792
    Carlos David Amezcua Canales - A01641742
    Enrique Mora Navarro - A01635459
    """
    await ctx.send(devs_text)

# Configuración de FastAPI
app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.on_event("startup")
async def startup_event():
    await bot.start(discord_token)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
