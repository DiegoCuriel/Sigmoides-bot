const express = require('express');
const dotenv = require('dotenv');
const { Client, GatewayIntentBits } = require('discord.js');
const fs = require('fs');
const path = require('path');
const OpenAI = require('openai-api');

dotenv.config();

const app = express();
const port = process.env.PORT || 3000;

const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
const openai = new OpenAI(OPENAI_API_KEY);

// Cargar documentos
function loadDocuments(directory) {
  let documents = [];
  const files = fs.readdirSync(directory);
  files.forEach(file => {
    if (file.endsWith('.txt')) {
      const content = fs.readFileSync(path.join(directory, file), 'utf-8');
      documents.push(content);
    }
  });
  return documents;
}

const docsPath = process.env.DOCS_PATH || './nutricion_y_salud';
const docs = loadDocuments(docsPath);
console.log(`Se cargaron ${docs.length} documentos.`);

async function answerQuestion(question) {
  try {
    const gptResponse = await openai.complete({
      engine: 'davinci',
      prompt: question,
      maxTokens: 150,
      temperature: 0.9,
      topP: 1,
      presencePenalty: 0,
      frequencyPenalty: 0,
      bestOf: 1,
      n: 1,
      stream: false,
      stop: ['\n', "testing"]
    });
    return gptResponse.data.choices[0].text.trim();
  } catch (error) {
    console.error("Error al obtener la respuesta de OpenAI:", error);
    return "Hubo un error al obtener la respuesta.";
  }
}

// Configuración del bot de Discord
const client = new Client({ intents: [GatewayIntentBits.Guilds, GatewayIntentBits.GuildMessages, GatewayIntentBits.MessageContent] });

client.once('ready', () => {
  console.log(`Bot conectado como ${client.user.tag}`);
});

client.on('messageCreate', async message => {
  if (message.content.startsWith('!')) {
    const [command, ...args] = message.content.slice(1).split(' ');
    const query = args.join(' ');

    if (command === 'info') {
      message.channel.send("Este es un bot sigma que puede responder preguntas relacionadas con la nutrición y salud. ¡Preguntame lo que quieras!");
    } else if (command === 'preguntar') {
      const answer = await answerQuestion(query);
      message.channel.send(`Pregunta: ${query}\nRespuesta: ${answer}`);
    } else if (command === 'devs') {
      message.channel.send(`
        Actividad M7 - Discord Bot
        Hugo Alejandro Gómez Herrera - A01640856
        Diego Curiel Castellanos - A01640372
        Juan Daniel Muñoz Dueñas - A01641792
        Carlos David Amezcua Canales - A01641742
        Enrique Mora Navarro - A01635459
      `);
    }
  }
});

client.login(process.env.DISCORD_TOKEN);

// Configuración de Express
app.get('/', (req, res) => {
  res.json({ Hello: 'World' });
});

app.listen(port, () => {
  console.log(`Servidor corriendo en http://localhost:${port}`);
});
