#!/bin/bash
# Быстрая настройка .env для запуска evaluation

echo "🔧 Настройка .env для Safety Incident Analyzer"
echo "================================================"
echo ""

# Проверяем существование .env
if [ ! -f .env ]; then
    echo "❌ .env файл не найден. Создаю из шаблона..."
    cp .env.example .env
fi

echo "Выберите конфигурацию:"
echo "1) OpenAI (проще всего)"
echo "2) GigaChat + OpenAI (для embeddings)"
echo ""
read -p "Ваш выбор (1 или 2): " choice

if [ "$choice" = "1" ]; then
    # OpenAI вариант
    read -p "Введите OPENAI_API_KEY: " openai_key

    if [ -n "$openai_key" ]; then
        sed -i "s/LLM_PROVIDER=gigachat/LLM_PROVIDER=openai/" .env
        sed -i "s/MODEL_NAME=GigaChat/MODEL_NAME=gpt-4o-mini/" .env
        sed -i "s/YOUR_OPENAI_KEY_HERE/$openai_key/" .env
        sed -i "s/EMBEDDING_PROVIDER=openai/EMBEDDING_PROVIDER=openai/" .env
        echo "✅ OpenAI ключ настроен"
    else
        echo "❌ Ключ не введен"
        exit 1
    fi

elif [ "$choice" = "2" ]; then
    # GigaChat + OpenAI вариант
    read -p "Введите GIGACHAT_CREDENTIALS: " gigachat_key
    read -p "Введите OPENAI_API_KEY (для embeddings): " openai_key

    if [ -n "$gigachat_key" ] && [ -n "$openai_key" ]; then
        sed -i "s/YOUR_GIGACHAT_TOKEN_HERE/$gigachat_key/" .env
        sed -i "s/YOUR_OPENAI_KEY_HERE/$openai_key/" .env
        echo "✅ GigaChat и OpenAI ключи настроены"
    else
        echo "❌ Ключи не введены"
        exit 1
    fi
else
    echo "❌ Неверный выбор"
    exit 1
fi

# Опционально: LangSmith
echo ""
read -p "Добавить LANGSMITH_API_KEY? (y/n): " add_langsmith
if [ "$add_langsmith" = "y" ]; then
    read -p "Введите LANGSMITH_API_KEY: " langsmith_key
    if [ -n "$langsmith_key" ]; then
        sed -i "s/YOUR_LANGSMITH_KEY_HERE/$langsmith_key/" .env
        echo "✅ LangSmith ключ настроен"
    fi
fi

echo ""
echo "================================================"
echo "✅ Конфигурация завершена!"
echo ""
echo "Проверьте настройки:"
echo "  cat .env | grep -E '^(LLM_PROVIDER|OPENAI_API_KEY|GIGACHAT|LANGSMITH)'"
echo ""
echo "Запустить evaluation:"
echo "  python eval/run_full_evaluation.py --limit 5"
echo ""
