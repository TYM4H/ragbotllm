import aiohttp
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from dotenv import load_dotenv
import os
import asyncio

load_dotenv()
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
API_URL = "http://127.0.0.1:8000"

bot = Bot(BOT_TOKEN)
dp = Dispatcher()


@dp.message(Command("start"))
async def start(msg: types.Message):
    await msg.answer("Привет! Пришли PDF, потом задавай вопросы")


@dp.message(lambda m: m.document and m.document.mime_type == "application/pdf")
async def handle_pdf(msg: types.Message):
    file = await bot.get_file(msg.document.file_id)
    file_url = f"https://api.telegram.org/file/bot{BOT_TOKEN}/{file.file_path}"

    async with aiohttp.ClientSession() as s:
        async with s.get(file_url) as resp:
            pdf_bytes = await resp.read()
        form = aiohttp.FormData()
        form.add_field("user_id", str(msg.from_user.id))
        form.add_field("file", pdf_bytes, filename="doc.pdf", content_type="application/pdf")
        async with s.post(f"{API_URL}/ingest", data=form):
            await msg.answer("Документ добавлен!")


@dp.message()
async def handle_question(msg: types.Message):
    async with aiohttp.ClientSession() as s:
        form = aiohttp.FormData()
        form.add_field("user_id", str(msg.from_user.id))
        form.add_field("question", msg.text)
        async with s.post(f"{API_URL}/query", data=form) as resp:
            data = await resp.json()
    await msg.answer(data["answer"])


if __name__ == "__main__":
    asyncio.run(dp.start_polling(bot))
