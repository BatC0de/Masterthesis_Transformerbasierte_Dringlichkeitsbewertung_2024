Die in diesem Verzeichnis verwendeten Skripte wurde dazu vewendet im Rahmen der Masterthesis 
"Kundenanfragen im Kundenservice: eine Transformerbasierte Dringlichkeitsbewertung"
unterschiedliche Aufgaben zu erfüllen.

NER-Anonymisierung.py
    --> Wurde verwendet, um die Rohdaten des Auftraggebers durch REGEX und NER zu anonymisieren.

Transformer_Train_Basisverfahren+erweitertes Verfahren
    --> Wurde zum Training der Basis+erweiterten Verfahrens verwendet, je nach Modell wurde der Modellname angepasst
    --> Auswertung der erzeugten Metriken

Transformer_Train_Hyperparameteroptimiert
    --> Wurde zum Training des Hyperparameteroptimierten Verfahrens verwendet, je nach Modell wurde der Modellname angepasst
    --> Auswertung der erzeugten Metriken
    -- Enthält explizite Anpassungen, um das Modell zugunsten von FNR und Recall zu optimieren

Inference.py
    --> Wird verwendet, um die erzeugten Modelle auf spezifische EIngabesequenzen zu prüfen