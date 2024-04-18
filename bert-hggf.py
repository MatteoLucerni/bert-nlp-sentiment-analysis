from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline

# Tokenizer e modello
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

# Pipeline di classificazione
nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# test
result = nlp(
    "Oggi a lavoro la giornata è stata veramente stressante, il mio assistente continuava a sbagliare tutto, sembrava quasi che lo facesse apposta! Eppure cerco di trattarlo sempre benissimo quando stiamo insieme, mi sembrava di avere un bel rapporto con lui.",
    top_k=3,
)

print(result)
