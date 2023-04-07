# rosatom-case-april
Цикл 8.1. Уроки настоящего машинного обучения: от теории к конкретным задачам / апрель

Формулировка задачи

Атомная отрасль является огромным пластом науки и технологий. Она является одной из самых передовых отраслей в мире и позволяет людям получать множество бонусов за счет ее использования, начиная от генерации электричества и заканчивая лечением многих заболеваний.

Тем не менее, наиболее важной частью этой отрасли, является соблюдение техники безопасности и отсутствие каких-либо происшествий. Иногда даже секунды могут сыграть огромную роль как при недопущении аварии, так и при ее устранении. Причем, информация об аварии может поступить не только от работников, но и от сторонних людей, случайно оказавшихся рядом с местом ЧП.

Именно поэтому вам предстоит решить задачу определения по тексту пользователей в соцсетях и их местонахождению, является ли их сообщение описанием какого-либо бедствия или нет. Вы должны будете обучить модель машинного обучения, которая будет получать на вход строку с сообщением, а на выходе сигнализировать о возможном ЧП, если посчитает, что оно произошло. Отдельным этапом работы над проектом является создание на основе модели web‑сервиса, доступного в сети интернет.

Анализ данных.

Ознакомьтесь с датасетом (будет опубликован не позже 04.04.2023).
Проанализируйте взаимосвязь слов/фраз в сообщениях и их общий смысл.
Составьте список признаков, указывающих на то, что сообщение является сообщением о ЧП или является обычным. Например, наличие восклицательного знака или восторженных оборотов может сигнализировать о том, что сообщение уведомляет получателя о том, что поисходит нечто необычное. 

Выбор алгоритма и обучение модели.

Выберите наиболее подходящий и эффективный алгоритм машинного обучения.
Обучите модель для решения задачи.
Подсчитайте итоговую точность модели по метрике accuracy. 

Оболочка.

Создайте сайт или программу, через которую можно обратиться к обученной модели и получить результат. Исходный код необходимо загрузить на github.com или прислать в виде архива.



06.04 - Добавлено начальное Flask приложение, в котором предстоит дальше создать интерфейс; Добавлена модель МО с accuracy ~70%; Сохранена модель МО в .pkl формате. Добавлен ввод и вывод через сайт

### Доработки на апрель - пересобрать модель МО на основе nlp; оформить сайт ввода/вывода до конца.
