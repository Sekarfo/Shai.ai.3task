Для хакатона Shai.ai 
Task 3
# SupplyRisk ML Model

## Структура проекта
- `api/model_train.py` — обучение модели (HistGradientBoostingClassifier, sklearn)
- `api/app.py` — REST API для инференса (FastAPI)

## Быстрый старт

1. **Создайте виртуальное окружение и установите зависимости:**
   ```sh
   python -m venv venv
   venv\Scripts\activate  # Windows
   pip install -r requirments.txt
   ```

2. **Обучите модель:**
   ```sh
   python api/model_train.py
   # или с параметрами:
   # python api/model_train.py --cv data/shipping_dataset_expanded.csv --outdir artifacts
   ```

3. **Запустите API:**
   ```sh
   uvicorn api.app:app --host 0.0.0.0 --port 8000
   ```

4. **Проверьте работу:**
   - GET `/health` — статус и фичи
   - POST `/score` — скоринг маршрутов
   - POST `/whatif` — скоринг по произвольным фичам

## Описание модели
- Используется `HistGradientBoostingClassifier` (scikit-learn)
- Фичи: sla_days, storm_flag, fx_volatility_7d, dow, month
- Таргет: delay_flag
- Автоматический подбор порогов для бизнес-логики

## Контакты
- Tg @Tearnor
---

_Для подробностей смотрите комментарии в коде и структуру артефактов._
