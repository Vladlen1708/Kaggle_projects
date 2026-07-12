# House Prices — Advanced Regression Techniques

Предсказать цену продажи жилого дома в Эймсе (Айова) по 79 характеристикам — площадь, качество, район, год постройки и т.д.

**Kaggle:** [House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview)  
**Тип задачи:** регрессия (`SalePrice`, непрерывная)  
**Метрика:** RMSE между `log(SalePrice)` предсказанным и фактическим. Ошибка на дорогих и дешёвых домах весит одинаково → **таргет логарифмируем (`np.log1p`)** и учим модель на логарифме.  
**Результат:** public LB — Ridge **0.12462**, CatBoost **0.12556** (RMSE на log). Блендинг — в работе.  

---

## Данные

- `train.csv` — 1460 домов (с известным `SalePrice`)
- `test.csv` — 1459 домов
- 79 признаков + `Id` + таргет `SalePrice`
- `data_description.txt` — расшифровка всех признаков и категорий (лежит в репо)
- Целевая переменная: `SalePrice` в долларах

  ![Y_dist](<images/Снимок экрана 2026-07-12 225047.png>)

Признаков много, но по смыслу они группируются:

| Группа | Примеры | Замечание |
|---|---|---|
| Площади | `GrLivArea`, `TotalBsmtSF`, `1stFlrSF`, `LotArea` | сильнейшие предикторы, сильно скошены вправо |
| Качество/состояние | `OverallQual`, `OverallCond`, `ExterQual`, `KitchenQual`, `BsmtQual` | порядковые (Ex>Gd>TA>Fa>Po) |
| Год/возраст | `YearBuilt`, `YearRemodAdd`, `GarageYrBlt`, `YrSold` | из них считается возраст дома |
| Локация | `Neighborhood`, `MSZoning`, `Condition1/2` | номинальные, `Neighborhood` важен |
| Гараж / подвал | `Garage*`, `Bsmt*` | много `NA` = «нет гаража/подвала», а не пропуск |
| Прочее | `PoolQC`, `Fence`, `MiscFeature`, `Alley` | почти всегда `NA` = «нет объекта» |

## Важный нюанс про NA:

По `data_description.txt` во многих категориальных признаках `NA` — это **осмысленное значение** «объекта нет» (нет бассейна, гаража, подвала, забора, переулка), а не пропущенные данные.  Такие `NA` заполняем строкой `"None"` (категориальные) или `0` (связанные числовые: `GarageArea`, `TotalBsmtSF`, `MasVnrArea`). Настоящих пропусков мало — их заполняем отдельно (см. ниже).

## План подхода

### 1. EDA
- Распределение `SalePrice` — скошено вправо, тяжёлый правый хвост → подтверждаем необходимость `log1p`.
- Корреляции числовых признаков с таргетом: сильнейшие — `OverallQual`, `GrLivArea`, `GarageCars/Area`, `TotalBsmtSF`, `1stFlrSF`, `FullBath`, `YearBuilt`.
- Карта пропусков: у `PoolQC`, `MiscFeature`, `Alley`, `Fence`, `FireplaceQu` пропусков >40% — но это «None».
- **Выбросы:** 2 дома с `GrLivArea > 4000` и низкой ценой — общепризнанные выбросы, их удаляют из train.

### 2. Обработка пропусков
- Категориальные «нет объекта» (`PoolQC`, `Alley`, `Fence`, `MiscFeature`, `FireplaceQu`, все `Garage*`, `Bsmt*`, `MasVnrType`) → `"None"`.
- Связанные числовые (`GarageArea`, `GarageCars`, `TotalBsmtSF`, `BsmtFinSF1/2`, `BsmtUnfSF`, `BsmtFullBath`, `BsmtHalfBath`, `MasVnrArea`, `GarageYrBlt`) → `0`.
- `LotFrontage` → медиана **по группе `Neighborhood`** (у соседних участков схожая ширина фасада).
- Настоящие единичные пропуски (`MSZoning`, `Electrical`, `KitchenQual`, `Exterior1st/2nd`, `SaleType`, `Functional`, `Utilities`) → мода. `Utilities` почти константа — можно выбросить.

### 3. Feature engineering
- `TotalSF = TotalBsmtSF + 1stFlrSF + 2ndFlrSF` — суммарная площадь, обычно топ-1 по важности.
- `TotalBath = FullBath + 0.5·HalfBath + BsmtFullBath + 0.5·BsmtHalfBath`.
- `HouseAge = YrSold - YearBuilt`, `RemodAge = YrSold - YearRemodAdd`, `IsRemodeled`.
- Флаги наличия: `HasPool`, `Has2ndFloor`, `HasGarage`, `HasBsmt`, `HasFireplace`.
- `MSSubClass`, `MoSold`, `YrSold` — по факту категориальные, привести к строке.
- Порядковые качества (`ExterQual`, `BsmtQual`, `KitchenQual`, `HeatingQC`, `FireplaceQu`, ...) → числа по шкале `{Po:1, Fa:2, TA:3, Gd:4, Ex:5}`

### 4. Преобразование скошенности
- Числовые признаки с высокой асимметрией (`skew > 0.75`) → `log1p`. Линейным моделям это критично, деревьям — почти всё равно.

### 5. Кодирование категориальных
- Номинальные (`Neighborhood`, `MSZoning`, `SaleType`, ...) → one-hot-encoding.
- Порядковые — уже закодированы числами (шаг 3).
- Следить, чтобы набор колонок train и test совпадал (кодировать на объединённом наборе или через `align`).

### 6. Модели и валидация
- Валидация: `KFold` (5–10 фолдов), метрика — RMSE на `log`-таргете (совпадает с лидербордом).
- Baseline: регуляризованная линейная модель на скошенных фичах — `Ridge` / `Lasso` / `ElasticNet` (масштабировать признаки). На этой задаче линейные модели с регуляризацией очень сильны.
- Деревья/бустинг: `LightGBM` / `XGBoost` / `CatBoost` — знакомый тебе градиентный бустинг, работает без масштабирования и без one-hot (CatBoost).
- Улучшение: блендинг (среднее предсказаний линейной модели и бустинга) или стекинг. В топовых решениях именно смесь Lasso/ElasticNet + GBM даёт лучший результат.


## EDA — ключевые находки (по факту, train 1460×81)

- **Таргет скошен вправо:** `skew(SalePrice)=1.88`, медиана 163k, среднее 181k, хвост до 755k. После `log1p` skew=0.12 → логарифмирование обязательно.
- **Топ-корреляции с `SalePrice`:** `OverallQual` +0.79, `GrLivArea` +0.71, `GarageCars` +0.64, `GarageArea` +0.62, `TotalBsmtSF` +0.61, `1stFlrSF` +0.61, `FullBath` +0.56, `TotRmsAbvGrd` +0.53, `YearBuilt` +0.52. Слабо-отрицательные: `KitchenAbvGr` −0.14, `EnclosedPorch` −0.13.
- **Мультиколлинеарность:** `GarageCars`↔`GarageArea`, `TotRmsAbvGrd`↔`GrLivArea`, `1stFlrSF`↔`TotalBsmtSF` — линейным моделям мешает, лечится регуляризацией; для FE это аргумент собрать `TotalSF`.
- **Пропуски (реальные %):** `PoolQC` 99.5, `MiscFeature` 96.3, `Alley` 93.8, `Fence` 80.8, `MasVnrType` 59.7, `FireplaceQu` 47.3 — всё это «None». `LotFrontage` 17.7% → медиана по `Neighborhood`. Все `Garage*` ровно 81 (5.5%), все `Bsmt*` 37–38 — «нет гаража/подвала». Единственный настоящий единичный пропуск: `Electrical` (1).
- **Сильно скошенные числовые:** `MiscVal` +24, `PoolArea` +15, `LotArea` +12, `3SsnPorch` +10, `LowQualFinSF` +9 и др. → `log1p`/Box-Cox для линейных моделей.
- **Выбросы — важный нюанс:** домов с `GrLivArea>4000` четыре, но удалять надо только **два** с большой площадью и низкой ценой (`GrLivArea` 4676→\$184750 и 5642→\$160000 при `OverallQual=10`). Другие два (4316→\$755k, 4476→\$745k) — легитимные дорогие дома высшего качества, их оставить. Не удалять всё подряд по порогу площади.
- 43 категориальных признака; `Neighborhood` — 25 уникальных значений.

## Feature engineering

Реализовано в `domain_preprocess()`:

- `TotalSF = TotalBsmtSF + 1stFlrSF + 2ndFlrSF` — суммарная площадь.
- `TotalBath = FullBath + 0.5·HalfBath + BsmtFullBath + 0.5·BsmtHalfBath`.
- `HouseAge = YrSold − YearBuilt`, флаг `IsRemodeled` (`YearRemodAdd != YearBuilt`).
- Флаги наличия: `HasPool`, `Has2ndFloor`.
- Порядковые качества (`ExterQual/Cond`, `BsmtQual/Cond`, `HeatingQC`, `KitchenQual`, `FireplaceQu`, `GarageQual/Cond`, `PoolQC`) → числа `{None:0, Po:1, Fa:2, TA:3, Gd:4, Ex:5}`.
- `MSSubClass`, `MoSold`, `YrSold` → строки (категориальные по своему смыслу).
- `LotFrontage` - медиана по `Neighborhood`.

## Архитектура кода

- `pipeline_skeleton.py` — общий `load()` (+ дроп выбросов, `log1p` таргета) и `domain_preprocess()`; пайплайн **Ridge** через `ColumnTransformer` (skew>0.75 → `log1p`+scale; numeric → impute+scale; cat → impute+OneHot(ignore)).
- `pipeline_catboost.py` — импортирует `load`/`domain_preprocess`, **CatBoost** на нативных `cat_features` (без one-hot и скейлинга).
- Единый препроцесс на обе модели → предсказания напрямую готовы к блендингу.

## Модели и валидация

| Модель | Валидация | LB score |
|---|---|---|
| Ridge + OneHotEncoding + skew log1p | 5-fold CV | **0.12462** |
| CatBoost | 5-fold CV | **0.12556** |
| Блендинг (Ridge + CatBoost) | 5-fold CV | (еще не сделал) |

- Схема валидации: `KFold(5, shuffle=True, random_state=42)`, RMSE на `log`-таргете (метрика лидерборда).
- Подбор гиперпараметров: пока вручную.

## Результат

- Лучший public LB: **Ridge 0.12462** (Причем это Ridge регрессия, а не бустинг)
- Что сработало: log1p таргета, дроп 2 выбросов `GrLivArea>4000`, ordinal-кодирование качеств, единый препроцесс.
- Что еще можно попробовать: блендинг, доп. линейные модели, тюнинг.
- Что не сработало: бустинг не дал внушительных результатов по сравнению с обычной линейной регрессией.

## Следующие шаги (по приоритету)

1. **Блендинг Ridge + CatBoost** — среднее в log-пространстве (старт ~0.6/0.4, вес по CV), `np.expm1` на выходе. Самый дешёвый прирост: ошибки линейки и деревьев слабо скоррелированы.
2. Добавить **Lasso / ElasticNet** в бленд.
3. Тюнинг гиперпараметров бустинга (Optuna) — в последнюю очередь.

## Выводы и что улучшить

- Box-Cox вместо `log1p`, target encoding для `Neighborhood`, отбор признаков, стекинг.
- Еще роверить прирост от `RemodAge`, `HasGarage/HasBsmt/HasFireplace`, target encoding для `Neighborhood`.


## Стек

`Python` · `pandas` · `numpy` · `scikit-learn` · `catboost` · `matplotlib` · `seaborn`

