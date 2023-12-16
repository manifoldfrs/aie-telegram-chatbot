import json
import logging
import os

import numpy as np
import openai
import pandas as pd
import requests
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from functions import functions, run_function
from questions import answer_question

CODE_PROMPT = """
Here are two input:output examples for code generation. Please use these and follow the styling for future requests that you think are pertinent to the request. Make sure All HTML is generated with the JSX flavoring.

// SAMPLE 1
// A Blue Box with 3 yellow cirles inside of it that have a red outline
<div style={{
  backgroundColor: 'blue',
  padding: '20px',
  display: 'flex',
  justifyContent: 'space-around',
  alignItems: 'center',
  width: '300px',
  height: '100px',
}}>
  <div style={{
    backgroundColor: 'yellow',
    borderRadius: '50%',
    width: '50px',
    height: '50px',
    border: '2px solid red'
  }}></div>
  <div style={{
    backgroundColor: 'yellow',
    borderRadius: '50%',
    width: '50px',
    height: '50px',
    border: '2px solid red'
  }}></div>
  <div style={{
    backgroundColor: 'yellow',
    borderRadius: '50%',
    width: '50px',
    height: '50px',
    border: '2px solid red'
  }}></div>
</div>

// SAMPLE 2
// A RED BUTTON THAT SAYS 'CLICK ME'
<button style={{
  backgroundColor: 'red',
  color: 'white',
  padding: '10px 20px',
  border: 'none',
  borderRadius: '50px',
  cursor: 'pointer'
}}>
  Click Me
</button>
"""

load_dotenv()

openai.api_key = os.environ["OPENAI_API_KEY"]
tg_bot_token = os.environ["TG_BOT_TOKEN"]

messages = [
    {
        "role": "system",
        "content": "You are a helpful assistant that answers questions.",
    },
    {"role": "system", "content": CODE_PROMPT},
]


logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)


df = pd.read_csv("processed/embeddings.csv", index_col=0)
df["embeddings"] = df["embeddings"].apply(eval).apply(np.array)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(
        chat_id=update.effective_chat.id, text="I'm a bot, please talk to me!"
    )


# async def chat(update: Update, context: ContextTypes.DEFAULT_TYPE):
#     messages.append({"role": "user", "content": update.message.text})
#     completion = openai.ChatCompletion.create(model="gpt-4-0314", messages=messages)
#     completion_answer = completion.choices[0]["message"]["content"]
#     messages.append({"role": "assistant", "content": completion_answer})

#     await context.bot.send_message(
#         chat_id=update.effective_chat.id, text=completion_answer
#     )


async def mozilla(update: Update, context: ContextTypes.DEFAULT_TYPE):
    answer = answer_question(df, question=update.message.text, debug=True)
    await context.bot.send_message(chat_id=update.effective_chat.id, text=answer)


async def chat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Append the user's message to the messages list"""
    messages.append({"role": "user", "content": update.message.text})
    """Generate an initial response, providing functions to enable function calling"""
    initial_response = openai.ChatCompletion.create(
        model="gpt-4", messages=messages, functions=functions
    )
    initial_response_message = initial_response.get("choices", [{}])[0].get("message")
    final_response = None
    """Check if the initial response contains a function call"""
    if initial_response_message and initial_response_message.get("function_call"):
        # Extract the function name and arguments
        name = initial_response_message["function_call"]["name"]
        args = json.loads(initial_response_message["function_call"]["arguments"])

        # Run the corresponding function
        function_response = run_function(name, args)

        # if 'svg_to_png_bytes' function, send a photo and return as there's nothing else to do
        if name == "svg_to_png_bytes":
            await context.bot.send_photo(
                chat_id=update.effective_chat.id, photo=function_response
            )
            return

        # Generate the final response
        final_response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                *messages,
                initial_response_message,
                {
                    "role": "function",
                    "name": initial_response_message["function_call"]["name"],
                    "content": json.dumps(function_response),
                },
            ],
        )
        final_answer = final_response["choices"][0]["message"]["content"]

        # Send the final response if it exists
        if final_answer:
            messages.append({"role": "assistant", "content": final_answer})
            await context.bot.send_message(
                chat_id=update.effective_chat.id, text=final_answer
            )
        else:
            # Send an error message if something went wrong
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text="Something went wrong, please try again",
            )
    else:
        # If no function call, send the initial response
        messages.append(initial_response_message)
        await context.bot.send_message(
            chat_id=update.effective_chat.id, text=initial_response_message["content"]
        )

    await context.bot.send_message(
        chat_id=update.effective_chat.id, text="What else can I help you with?"
    )


async def image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    response = openai.Image.create(prompt=update.message.text, n=1, size="1024x1024")
    image_url = response["data"][0]["url"]
    image_response = requests.get(image_url)
    await context.bot.send_photo(
        chat_id=update.effective_chat.id, photo=image_response.content
    )


async def transcribe_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Make sure we have a voice file to transcribe
    voice_id = update.message.voice.file_id
    if voice_id:
        file = await context.bot.get_file(voice_id)
        await file.download_to_drive(f"voice_note_{voice_id}.ogg")
        await update.message.reply_text("Voice note downloaded, transcribing now")
        audio_file = open(f"voice_note_{voice_id}.ogg", "rb")
        transcript = openai.Audio.transcribe("whisper-1", audio_file)
        await update.message.reply_text(f'Transcript finished:\n {transcript["text"]}')


if __name__ == "__main__":
    application = ApplicationBuilder().token(tg_bot_token).build()

    start_handler = CommandHandler("start", start)
    chat_handler = CommandHandler("chat", chat)
    mozilla_handler = CommandHandler("mozilla", mozilla)
    image_handler = CommandHandler("image", image)
    voice_handler = MessageHandler(filters.VOICE, transcribe_message)
    # code_generation_handler = CommandHandler("code", code_generation)

    application.add_handler(start_handler)
    application.add_handler(chat_handler)
    application.add_handler(mozilla_handler)
    application.add_handler(image_handler)
    application.add_handler(voice_handler)
    # application.add_handler(code_generation_handler)

    application.run_polling()
