"""
Сервис для распознавания речи с помощью локальной модели Whisper
"""

import logging
import os
import shutil
from typing import Optional

logger = logging.getLogger(__name__)


class WhisperService:
    """
    Сервис для распознавания речи с использованием локальной модели Whisper
    Не требует интернет-соединения после загрузки модели
    """

    def __init__(self, model_name: str = "medium", ffmpeg_path: Optional[str] = None):
        """
        Инициализация сервиса Whisper

        Args:
            model_name: Размер модели Whisper (tiny, base, small, medium, large)
                       medium - хороший баланс между качеством и скоростью
            ffmpeg_path: Явный путь к FFmpeg (если None, будет автопоиск)
        """
        self.model_name = model_name
        self.model = None
        self.ffmpeg_path = ffmpeg_path
        self._setup_ffmpeg()
        self._load_model()

    def _setup_ffmpeg(self):
        """Настройка пути к FFmpeg"""
        try:
            logger.info(f"🔍 Настройка FFmpeg. Указанный путь: {self.ffmpeg_path}")
            ffmpeg_found = False

            # Если указан явный путь
            if self.ffmpeg_path:
                logger.info(f"🔍 Проверяю путь: {self.ffmpeg_path}")
                if os.path.exists(self.ffmpeg_path):
                    dir_path = os.path.dirname(self.ffmpeg_path)
                    os.environ["PATH"] = (
                        dir_path + os.pathsep + os.environ.get("PATH", "")
                    )
                    logger.info(f"✅ Используется FFmpeg из: {self.ffmpeg_path}")
                    logger.info(f"✅ Добавлена директория в PATH: {dir_path}")
                    ffmpeg_found = True
                else:
                    logger.warning(f"⚠️ Файл не найден по пути: {self.ffmpeg_path}")

            # Автопоиск FFmpeg в системе
            if not ffmpeg_found:
                ffmpeg_cmd = shutil.which("ffmpeg")
                if ffmpeg_cmd:
                    logger.info(f"✅ FFmpeg найден в системе: {ffmpeg_cmd}")
                    ffmpeg_found = True

            # Попытка найти через imageio_ffmpeg (резервный вариант)
            if not ffmpeg_found:
                try:
                    import imageio_ffmpeg

                    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
                    dir_path = os.path.dirname(ffmpeg_exe)
                    os.environ["PATH"] = (
                        dir_path + os.pathsep + os.environ.get("PATH", "")
                    )
                    logger.info(
                        f"✅ Используется FFmpeg из imageio_ffmpeg: {ffmpeg_exe}"
                    )
                    ffmpeg_found = True
                except ImportError:
                    logger.debug("imageio_ffmpeg не установлен")
                except Exception as e:
                    logger.debug(f"Не удалось получить FFmpeg из imageio: {e}")

            if not ffmpeg_found:
                logger.error(
                    "❌ FFmpeg не найден! Установите FFmpeg:\n"
                    "Windows: choco install ffmpeg (или скачайте с https://ffmpeg.org)\n"
                    "Linux: sudo apt install ffmpeg\n"
                    "MacOS: brew install ffmpeg\n"
                    "После установки ПЕРЕЗАПУСТИТЕ приложение!\n"
                    "Или укажите путь в configs/app_config.yaml: speech.ffmpeg_path"
                )
                raise RuntimeError("FFmpeg не найден в системе")

        except Exception as e:
            logger.error(f"Ошибка настройки FFmpeg: {e}")
            raise

    def _load_model(self):
        """Загрузка модели Whisper"""
        try:
            import whisper

            logger.info(f"Загрузка модели Whisper '{self.model_name}'...")
            self.model = whisper.load_model(self.model_name)
            logger.info(f"Модель Whisper '{self.model_name}' успешно загружена")
        except ImportError:
            logger.error(
                "Библиотека openai-whisper не установлена. "
                "Установите: pip install openai-whisper"
            )
            raise
        except Exception as e:
            logger.error(f"Ошибка загрузки модели Whisper: {e}")
            raise

    async def transcribe_audio(self, audio_file_path: str, language: str = "ru") -> str:
        """
        Распознавание речи из аудиофайла

        Args:
            audio_file_path: Путь к аудиофайлу (поддерживаются: mp3, ogg, wav, m4a и др.)
            language: Код языка (ru, en и т.д.)

        Returns:
            Распознанный текст
        """
        if not self.model:
            return "Ошибка: модель Whisper не загружена"

        if not os.path.exists(audio_file_path):
            return "Ошибка: файл не найден"

        try:
            # Проверяем FFmpeg еще раз перед распознаванием
            if not shutil.which("ffmpeg"):
                error_msg = (
                    "❌ FFmpeg не найден в PATH!\n\n"
                    "РЕШЕНИЕ:\n"
                    "1. Установите FFmpeg:\n"
                    "   Windows: choco install ffmpeg\n"
                    "   Или скачайте: https://ffmpeg.org/download.html\n\n"
                    "2. ПЕРЕЗАПУСТИТЕ приложение после установки!\n\n"
                    "3. Или укажите путь в configs/app_config.yaml:\n"
                    "   speech:\n"
                    "     ffmpeg_path: 'C:\\\\ffmpeg\\\\bin\\\\ffmpeg.exe'"
                )
                logger.error(error_msg)
                return "FFmpeg не найден. Установите FFmpeg и перезапустите приложение."

            logger.info(f"Распознавание аудио: {audio_file_path}")

            # Whisper автоматически конвертирует аудио в нужный формат с помощью ffmpeg
            result = self.model.transcribe(
                audio_file_path,
                language=language,
                fp16=False,  # Отключаем FP16 для совместимости с CPU
                verbose=False,
            )

            text = result["text"].strip()
            logger.info(f"Распознано: {text[:100]}...")

            if not text:
                return "Не удалось распознать речь в аудиофайле"

            return text

        except FileNotFoundError as e:
            error_msg = (
                "❌ FFmpeg не найден!\n\n"
                "УСТАНОВИТЕ FFmpeg:\n"
                "1. Windows: choco install ffmpeg\n"
                "2. ПЕРЕЗАПУСТИТЕ приложение\n"
                "3. Или укажите путь в configs/app_config.yaml"
            )
            logger.error(error_msg)
            logger.error(f"Детали ошибки: {e}", exc_info=True)
            return "FFmpeg не найден. Установите FFmpeg и перезапустите."
        except Exception as e:
            logger.error(f"Ошибка распознавания речи: {e}", exc_info=True)
            return f"Ошибка распознавания: {str(e)}"

    def get_available_models(self) -> list[str]:
        """Получить список доступных моделей Whisper"""
        return ["tiny", "base", "small", "medium", "large"]

    def get_model_info(self) -> dict:
        """Получить информацию о текущей модели"""
        return {
            "model_name": self.model_name,
            "loaded": self.model is not None,
            "requires_internet": False,
            "supported_languages": [
                "ru",
                "en",
                "uk",
                "be",
                "kk",
                "uz",
                "ky",
            ],
        }
