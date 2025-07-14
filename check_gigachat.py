import os
import requests
import uuid
from dotenv import load_dotenv

# Загружаем переменные из .env файла
load_dotenv()

# --- 1. Получение Access Token ---
credentials = os.getenv("GIGACHAT_CREDENTIALS")

if not credentials:
    raise ValueError("Не найдены GIGACHAT_CREDENTIALS в файле .env")

# URL для получения токена
auth_url = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"

# Заголовки для запроса токена
auth_headers = {
    "Content-Type": "application/x-www-form-urlencoded",
    "Accept": "application/json",
    "RqUID": str(uuid.uuid4()),
    "Authorization": f"Basic {credentials}",
}

# Тело запроса
auth_data = {"scope": "GIGACHAT_API_PERS"}

try:
    print("Шаг 1: Пытаюсь получить Access Token...")
    response = requests.post(
        auth_url, headers=auth_headers, data=auth_data, verify=False
    )
    response.raise_for_status()  # Проверяем, что нет HTTP ошибок (вроде 401, 403)

    token_data = response.json()
    access_token = token_data.get("access_token")
    print("Access Token успешно получен!")

except requests.exceptions.HTTPError as e:
    print(f"ОШИБКА при получении токена: {e.response.status_code}")
    print(f"Тело ответа: {e.response.text}")
    access_token = None
except Exception as e:
    print(f"Произошла непредвиденная ошибка: {e}")
    access_token = None


# --- 2. Запрос к GigaChat API с полученным токеном ---
if access_token:
    try:
        print("\nШаг 2: Пытаюсь получить список моделей с помощью токена...")
        models_url = "https://gigachat.devices.sberbank.ru/api/v1/models"

        api_headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {access_token}",
        }

        api_response = requests.get(models_url, headers=api_headers, verify=False)
        api_response.raise_for_status()  # Проверяем на ошибки (вроде 402)

        print("\nУСПЕХ! Ответ от GigaChat API получен:")
        print(api_response.json())

    except requests.exceptions.HTTPError as e:
        print(f"\nОШИБКА при запросе к API: {e.response.status_code}")
        print(f"Тело ответа: {e.response.text}")
        print(
            "\nДИАГНОЗ: Ваши авторизационные данные, скорее всего, верны, но у вашего аккаунта нет доступа к этой функции API. Проверьте ваш тарифный план и баланс на портале SberDevelopers."
        )
    except Exception as e:
        print(f"Произошла непредвиденная ошибка: {e}")
