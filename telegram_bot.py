from __future__ import annotations
import logging
import os
import itertools
import asyncio

import telegram
from uuid import uuid4
from telegram import constants, BotCommandScopeAllGroupChats
from telegram import Message, MessageEntity, Update, InlineQueryResultArticle, InputTextMessageContent, BotCommand, ChatMember
from telegram.error import RetryAfter, TimedOut
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, \
    filters, InlineQueryHandler, Application, CallbackContext

from openai_helper import OpenAIHelper
from usage_tracker import UsageTracker


def message_text(message: Message) -> str:
    message_text = message.text
    if message_text is None:
        return ''
    for _, text in sorted(message.parse_entities([MessageEntity.BOT_COMMAND]).items(), key=(lambda item: item[0].offset)):
        message_text = message_text.replace(text, '').strip()
    return message_text if len(message_text) > 0 else ''


class ChatGPTTelegramBot:
    def __init__(self, config: dict, openai: OpenAIHelper):
        self.config = config
        self.openai = openai
        self.commands = [
            BotCommand(command='help', description='Помощь'),
            BotCommand(command='reset', description='Сброс диалога '),
            BotCommand(command='image', description='Генератор изображений на основе Dall-e (пример запроса /image шрек)'),
            BotCommand(command='resend', description='Изменить ответ на последнее сообщение'),
            BotCommand(command='info', description='Информация')
        ]
        self.group_commands = [
            BotCommand(command='chat', description='Чат с ботом!')
        ] + self.commands
        self.disallowed_message = "Упс, а тебе недоступен бот "
        self.usage = {}
        self.last_message = {}

    async def help(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        commands = self.group_commands if self.is_group_chat(update) else self.commands
        commands_description = [f'/{command.command} - {command.description}' for command in commands]
        help_text = 'Я поехавший чатбот напиши мне скорее!' + \
                    '\n\n' + \
                    '\n'.join(commands_description) + \
                    '\n\n' + \
                    'Можешь скинуть нюдсы, я оценю' + \
                    '\n\n' + \
                    "А тут могла бы быть Ваша реклама"
        await update.message.reply_text(help_text, disable_web_page_preview=True)

    async def info(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        help_text = 'Код взят с гитхаба, адаптирован под ру аудиторию и полностью оптимизирован.' + \
                    '\n\n' + \
                    "Лишние функции по типу распознавания речи с гс и видеосообщений удалены, возможно потом появятся!"
        await update.message.reply_text(help_text, disable_web_page_preview=True)
        
    async def resend(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not await self.is_allowed(update, context):
            logging.warning(f'Пользователю {update.message.from_user.name}  (id: {update.message.from_user.id})'
                            f' не разрешено повторно отправлять сообщения')
            await self.send_disallowed_message(update, context)
            return
        chat_id = update.effective_chat.id
        if chat_id not in self.last_message:
            logging.warning(f'Пользователь {update.message.from_user.name} (id: {update.message.from_user.id})'
                            f' не имеет ничего для повторной отправки')
            await context.bot.send_message(chat_id=chat_id, text="Вам нечего отправить")
            return
        logging.info(f'Повторная попытка отправки: {update.message.from_user.name} '
                     f'(id: {update.message.from_user.id})')
        with update.message._unfrozen() as message:
            message.text = self.last_message.pop(chat_id)
        await self.prompt(update=update, context=context)

    async def reset(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not await self.is_allowed(update, context):
            logging.warning(f'Пользователю {update.message.from_user.name} (id: {update.message.from_user.id}) '
                f'недоступен сброс диалога')
            await self.send_disallowed_message(update, context)
            return
        logging.info(f'Сброс диалога для пользователя {update.message.from_user.name} '
            f'(id: {update.message.from_user.id})...')
        chat_id = update.effective_chat.id
        reset_content = message_text(update.message)
        self.openai.reset_chat_history(chat_id=chat_id, content=reset_content)
        await context.bot.send_message(chat_id=chat_id, text='Удачно!')

    async def image(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self.config['enable_image_generation']:
            return
        chat_id = update.effective_chat.id
        image_query = message_text(update.message)
        if image_query == '':
            await context.bot.send_message(chat_id=chat_id, text='Пабрацке, пиши как в примере! (пример: /image шрек на лыжах)')
            return
        logging.info(f'Получен новый запрос на создание изображения от пользователя {update.message.from_user.name} '
            f'(id: {update.message.from_user.id})')
        async def _generate():
            try:
                image_url, image_size = await self.openai.generate_image(prompt=image_query)
                await context.bot.send_photo(
                    chat_id=chat_id,
                    reply_to_message_id=self.get_reply_to_message_id(update),
                    photo=image_url
                )
                user_id = update.message.from_user.id
                self.usage[user_id].add_image_request(image_size, self.config['image_prices'])
                if str(user_id) not in self.config['allowed_user_ids'].split(',') and 'guests' in self.usage:
                    self.usage["guests"].add_image_request(image_size, self.config['image_prices'])
            except Exception as e:
                logging.exception(e)
                await context.bot.send_message(
                    chat_id=chat_id,
                    reply_to_message_id=self.get_reply_to_message_id(update),
                    text=f'Произошла ошибка в генерации изображения: {str(e)}',
                    parse_mode=constants.ParseMode.MARKDOWN
                )
        await self.wrap_with_indicator(update, context, constants.ChatAction.UPLOAD_PHOTO, _generate)

    async def prompt(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        logging.info(f'Новое сообщение получено от пользователя {update.message.from_user.name} (id: {update.message.from_user.id})')
        chat_id = update.effective_chat.id
        user_id = update.message.from_user.id
        prompt = message_text(update.message)
        self.last_message[chat_id] = prompt
        if self.is_group_chat(update):
            trigger_keyword = self.config['group_trigger_keyword']
            if prompt.lower().startswith(trigger_keyword.lower()):
                prompt = prompt[len(trigger_keyword):].strip()
            else:
                if update.message.reply_to_message and update.message.reply_to_message.from_user.id == context.bot.id:
                    logging.info('Получение сообщения для ответа от бота...')
                else:
                    logging.warning('Сообщение не начато с ключевого слова...')
                    return
        try:
            if self.config['stream']:
                await context.bot.send_chat_action(chat_id=chat_id, action=constants.ChatAction.TYPING)
                is_group_chat = self.is_group_chat(update)
                stream_response = self.openai.get_chat_response_stream(chat_id=chat_id, query=prompt)
                i = 0
                prev = ''
                sent_message = None
                backoff = 0
                chunk = 0
                async for content, tokens in stream_response:
                    if len(content.strip()) == 0:
                        continue
                    chunks = self.split_into_chunks(content)
                    if len(chunks) > 1:
                        content = chunks[-1]
                        if chunk != len(chunks) - 1:
                            chunk += 1
                            try:
                                await self.edit_message_with_retry(context, chat_id, sent_message.message_id, chunks[-2])
                            except:
                                pass
                            try:
                                sent_message = await context.bot.send_message(
                                    chat_id=sent_message.chat_id,
                                    text=content if len(content) > 0 else "..."
                                )
                            except:
                                pass
                            continue
                    if is_group_chat:
                        cutoff = 180 if len(content) > 1000 else 120 if len(content) > 200 else 90 if len(content) > 50 else 50
                    else:
                        cutoff = 90 if len(content) > 1000 else 45 if len(content) > 200 else 25 if len(content) > 50 else 15
                    cutoff += backoff
                    if i == 0:
                        try:
                            if sent_message is not None:
                                await context.bot.delete_message(chat_id=sent_message.chat_id,
                                                                 message_id=sent_message.message_id)
                            sent_message = await context.bot.send_message(
                                chat_id=chat_id,
                                reply_to_message_id=self.get_reply_to_message_id(update),
                                text=content
                            )
                        except:
                            continue
                    elif abs(len(content) - len(prev)) > cutoff or tokens != 'not_finished':
                        prev = content
                        try:
                            use_markdown = tokens != 'not_finished'
                            await self.edit_message_with_retry(context, chat_id, sent_message.message_id,
                                                               text=content, markdown=use_markdown)
                        except RetryAfter as e:
                            backoff += 5
                            await asyncio.sleep(e.retry_after)
                            continue
                        except TimedOut:
                            backoff += 5
                            await asyncio.sleep(0.5)
                            continue
                        except Exception:
                            backoff += 5
                            continue

                        await asyncio.sleep(0.01)
                    i += 1
                    if tokens != 'not_finished':
                        total_tokens = int(tokens)
            else:
                async def _reply():
                    response, total_tokens = await self.openai.get_chat_response(chat_id=chat_id, query=prompt)
                    chunks = self.split_into_chunks(response)
                    for index, chunk in enumerate(chunks):
                        try:
                            await context.bot.send_message(
                                chat_id=chat_id,
                                reply_to_message_id=self.get_reply_to_message_id(update) if index == 0 else None,
                                text=chunk,
                                parse_mode=constants.ParseMode.MARKDOWN
                            )
                        except Exception:
                            try:
                                await context.bot.send_message(
                                    chat_id=chat_id,
                                    reply_to_message_id=self.get_reply_to_message_id(update) if index == 0 else None,
                                    text=chunk
                                )
                            except Exception as e:
                                raise e
                await self.wrap_with_indicator(update, context, constants.ChatAction.TYPING, _reply)
            try:
                # add chat request to users usage tracker
                self.usage[user_id].add_chat_tokens(total_tokens, self.config['token_price'])
                # add guest chat request to guest usage tracker
                allowed_user_ids = self.config['allowed_user_ids'].split(',')
                if str(user_id) not in allowed_user_ids and 'guests' in self.usage:
                    self.usage["guests"].add_chat_tokens(total_tokens, self.config['token_price'])
            except:
                pass
        except Exception as e:
            logging.exception(e)
            await context.bot.send_message(
                chat_id=chat_id,
                reply_to_message_id=self.get_reply_to_message_id(update),
                text=f'Ошибка в получении ответа: {str(e)}',
                parse_mode=constants.ParseMode.MARKDOWN
            )

    async def inline_query(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        query = update.inline_query.query
        if query == '':
            return
        results = [
            InlineQueryResultArticle(
                id=str(uuid4()),
                title='Спроси ChatGPT',
                input_message_content=InputTextMessageContent(query),
                description=query,
                thumb_url='https://user-images.githubusercontent.com/11541888/223106202-7576ff11-2c8e-408d-94ea-b02a7a32149a.png'
            )
        ]
        await update.inline_query.answer(results)

    async def edit_message_with_retry(self, context: ContextTypes.DEFAULT_TYPE, chat_id: int,
                                      message_id: int, text: str, markdown: bool = True):
        try:
            await context.bot.edit_message_text(
                chat_id=chat_id,
                message_id=message_id,
                text=text,
                parse_mode=constants.ParseMode.MARKDOWN if markdown else None
            )
        except telegram.error.BadRequest as e:
            if str(e).startswith("Сообщение не может быть изменено"):
                return
            try:
                await context.bot.edit_message_text(
                    chat_id=chat_id,
                    message_id=message_id,
                    text=text
                )
            except Exception as e:
                logging.warning(f'Ошибка в изменении сообщения: {str(e)}')
                raise e
        except Exception as e:
            logging.warning(str(e))
            raise e

    async def wrap_with_indicator(self, update: Update, context: CallbackContext, chat_action: constants.ChatAction, coroutine):
        task = context.application.create_task(coroutine(), update=update)
        while not task.done():
            context.application.create_task(update.effective_chat.send_action(chat_action))
            try:
                await asyncio.wait_for(asyncio.shield(task), 4.5)
            except asyncio.TimeoutError:
                pass

    async def send_disallowed_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=self.disallowed_message,
            disable_web_page_preview=True
        )

    async def error_handler(self, update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
        logging.error(f'Exception while handling an update: {context.error}')

    def is_group_chat(self, update: Update) -> bool:
        return update.effective_chat.type in [
            constants.ChatType.GROUP,
            constants.ChatType.SUPERGROUP
        ]

    async def is_user_in_group(self, update: Update, context: CallbackContext, user_id: int) -> bool:
        try:
            chat_member = await context.bot.get_chat_member(update.message.chat_id, user_id)
            return chat_member.status in [ChatMember.OWNER, ChatMember.ADMINISTRATOR, ChatMember.MEMBER]
        except telegram.error.BadRequest as e:
            if str(e) == "Пользователь не найден":
                return False
            else:
                raise e
        except Exception as e:
            raise e

    async def is_allowed(self, update: Update, context: CallbackContext) -> bool:
        if self.config['allowed_user_ids'] == '*':
            return True
        if self.is_admin(update):
            return True
        allowed_user_ids = self.config['allowed_user_ids'].split(',')
        if str(update.message.from_user.id) in allowed_user_ids:
            return True
        if self.is_group_chat(update):
            admin_user_ids = self.config['admin_user_ids'].split(',')
            for user in itertools.chain(allowed_user_ids, admin_user_ids):
                if not user.strip():
                    continue
                if await self.is_user_in_group(update, context, user):
                    logging.info(f'{user} доступ разрешен. Получение сообщения из группового чата...')
                    return True
            logging.info(f'Сообщение из группового чата от польозователя {update.message.from_user.name} '
                f'(id: {update.message.from_user.id}) не получено')
        return False

    def is_admin(self, update: Update, log_no_admin=False) -> bool:
        if self.config['admin_user_ids'] == '-':
            if log_no_admin:
                logging.info('No admin user defined.')
            return False
        admin_user_ids = self.config['admin_user_ids'].split(',')
        if str(update.message.from_user.id) in admin_user_ids:
            return True
        return False

    def get_reply_to_message_id(self, update: Update):
        if self.config['enable_quoting'] or self.is_group_chat(update):
            return update.message.message_id
        return None

    def split_into_chunks(self, text: str, chunk_size: int = 4096) -> list[str]:
        return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    async def post_init(self, application: Application) -> None:
        await application.bot.set_my_commands(self.group_commands, scope=BotCommandScopeAllGroupChats())
        await application.bot.set_my_commands(self.commands)

    def run(self):
        application = ApplicationBuilder() \
            .token(self.config['token']) \
            .proxy_url(self.config['proxy']) \
            .get_updates_proxy_url(self.config['proxy']) \
            .post_init(self.post_init) \
            .concurrent_updates(True) \
            .build()
        application.add_handler(CommandHandler('reset', self.reset))
        application.add_handler(CommandHandler('help', self.help))
        application.add_handler(CommandHandler('image', self.image))
        application.add_handler(CommandHandler('start', self.help))
        application.add_handler(CommandHandler('resend', self.resend))
        application.add_handler(CommandHandler(
            'chat', self.prompt, filters=filters.ChatType.GROUP | filters.ChatType.SUPERGROUP)
        )
        application.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), self.prompt))
        application.add_handler(InlineQueryHandler(self.inline_query, chat_types=[
            constants.ChatType.GROUP, constants.ChatType.SUPERGROUP
        ]))
        application.add_error_handler(self.error_handler)
        application.run_polling()