from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
import os

#Auswahl des gewünschten Checkpoints
checkpoint = "M:/Finale_Ergebnisse/RoBERTa/"
files = os.listdir(checkpoint)
print("Files in checkpoint directory:", files)

try:
    #Lade den Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("facebookAI/xlm-roberta-base")
    #Lade das Modell lokal
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
    print("Model loaded successfully. Tokenizer loaded from Hugging Face.")
except Exception as e:
    print(f"Error loading model or tokenizer: {e}")
    exit()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

#Definieren der Eingabesequenzen als Teil des speziellen Testdatensatzes
input_sentences = [
    "Guten Tag, in unserem gesamten Stadtteil ist seit einer Stunde der Strom ausgefallen, und wir haben kritische medizinische Geräte, die sofort Strom benötigen. Wann können wir mit einer Wiederherstellung rechnen?",
    "Wir haben seit gestern Abend keine Fernwärme mehr, und die Temperatur sinkt rapide. Es sind minusgrade und wir benötigen dringend Unterstützung.",
    "Hallo, ich möchte gerne auf einen reinen Ökostromtarif umsteigen. Könnten Sie mir mitteilen, welche Angebote es aktuell gibt und wie der Wechsel abläuft?",
    "Ich wollte fragen, wann genau die nächste Abrechnung stattfindet und ob es eine Möglichkeit gibt, den Abrechnungszeitraum auf monatliche Abrechnungen umzustellen.",
    "Ich wünsche um ein Rückruf. Es ist dringend und geht um denn 20. Wegen einer Sperrung. Hn Kn",
    "Sehr geehrte Damen und Herren, können Sie mir die Jahresabrechnung senden? Ich benötige diese dringend aus steuerlichen Gründen. Danke und Gruß",
    "anbei sende ich Ihnen die relevanten Daten zur Beendigung des Energieliefervertrags.",
    "Hiermit möchte ich Ihnen mitteilen das ich keinen Stromvertag mit Ihnen abschließen möchte !",
    "Ach, wirklich klasse, dass ich im Winter eine Heizung habe, die nicht funktioniert, da ich keinen Strom geliefert bekomme. Was tun Sie dagegen?",
    "Wirklich super, wie Sie die Preise anheben. Kann ich Ihnen gleich mein Konto überschreiben?",
    "Guten Tag, heute sende ich Ihnen meinen aktuellen Zählerstand. Ich benötige keine dringende Bearbeitung.",
    "Ich habe keine Lust, ständig die Abrechnung zu kontrollieren. Können Sie mir die nächste korrekt schicken?",
    "Das Problem ist der Stromausfall. Das Problem betrifft meine ganze Familie. Das Problem muss gelöst werden.",
    "Die Rechnung stimmt nicht. Die Rechnung muss bitte angepasst werden. Die Rechnung ist wichtig.",
    "Ich habe eine Reklamation geschickt, und die wurde bisher nicht bearbeitet. Was ist damit?",
    "Ihr Angebot fand ich interessant, aber ich habe dazu eine Frage. Können Sie mir dazu mehr Infos geben?",
    "Das muss sofort repariert werden, sonst habe ich morgen kein Warmwasser!",
    "Das sollte bald geklärt werden, damit ich mich wieder auf Sie verlassen kann.",
    "Ich bin völlig verzweifelt! Ohne Strom bin ich hilflos. Tun Sie doch etwas!",
    "Es ärgert mich wirklich sehr, dass schon wieder eine falsche Abrechnung geschickt wurde.",
    "Sehr geehrte Damen und Herren, bitte helfen Sie mir dringend, mein Stromausfall dauert an.",
    "Sehr geehrte Damen und Herren, ich hätte gerne eine Korrektur meiner Abrechnung. Vielen Dank!"
]

#Auswertung
for sentence in input_sentences:
    inputs = tokenizer(sentence, return_tensors="pt", padding="max_length", truncation=True, max_length=128)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    probabilities = torch.softmax(logits, dim=-1).cpu().numpy()
    predicted_class = np.argmax(probabilities, axis=-1)
    confidence = np.max(probabilities, axis=-1)
    print(f"Satz: {sentence}")
    print(f"Klasse: {predicted_class[0]}, Confidence: {confidence[0]:.4f}")
    print("-" * 50)


