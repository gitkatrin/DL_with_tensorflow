# Einführung in die Regression mit neuronalen Netzen in Tensorflow

## 1. Übersicht
![image](https://user-images.githubusercontent.com/43642275/155163046-551e52c5-947b-4707-839a-815044887836.png) Quelle: https://youtu.be/tpCFfeUEGs8?t=29568
  1. **Datenaufbereitung (in Tensoren umwandeln)**
      1. Alle Daten in Zahlen umwandeln (Neuronale Netze konnen mit Strings nichts anfangen)
      2. Alle Tensoren müssen die richtige Form haben
      3. Merkmale skalieren (normalisieren oder standardisieren, neuronale Netze bevorzugen in der Regel die Normalisierung)


          | Skalierung | Prozess | Scikit-Learn Funktion | Anwendung|
          |------------|---------|-----------------------|----------|
          | Scale (auch als Normalisierung bezeichnet| Wandelt alle Werte in Werte zwischen 0 und 1 um, wobei die ursprüngliche Verteilung erhalten bleibt. | ```MinMaxScaler```  | als Standard-Skalierer mit neuronalen Netzen |
          | Standardisierung | Entfernt den Mittelwert und teilt jeden Wert durch die Standardabweichung | ```StandardScaler``` | Transformiert ein Merkmal so, dass es eine annähernde Normalverteilung aufweist (Vorsicht: dies verringert die Wirkung von Ausreißern).
  2. **Modell erstellen oder ein vortrainiertes Modell auswählen**
  3. **Das Modell an die Daten anpassen und eine Vorhersage treffen**
  4. **Modell evaluieren**
  5. **Verbesserung durch Experimentieren**
  6. **Speichern und laden Sie Ihr trainiertes Modell erneut**

## 2. Modelling in Tensorflow
  1. **Model erstellen:** Input-, Output- und Hidden Layer definieren
      - verbessern durch:
        - weitere Schichten hinzufügen
        - Anzahl der Neuronen innerhalb jeder Schicht erhöhen
        - Aktivierungsfunktionen ändern
  3. **Model kompilieren:** Loss Funktion, Optimierer und Evaluationsmetriken definieren
      - verbessern durch:
        - Optimiernunsgfunktion anpassen
        - Lernrate ändern
  5. **Model anpassen:** Model ausprobieren und Muster zwischen X und y erkennen (zwischen Features und Labeln)
      - verbessern durch:
        - mehr Epochen (länger tranieren)
        - mehr Daten (mehr Beispiele zum Lernen)

## 3. Modell Evaluieren
  - 3 Datensäze:
    1. **Trainings-Datensatz:** Das Modell lernt von diesen Daten (70-80% aller Daten)
    2. **Validierungs-Datensatz:** Das Modell wird mit diesen Daten angepasst (10-15% aller Daten)
    3. **Test-Datensatz:** Das Modell wird mit diesen Daten evaluiert. Es wird getestet, was das Modell gelernt hat (10-15% aller Daten)
  - Daten visualisieren
  - Model visualisieren
  - Training des Models visualisieren
  - Vorhersagen des Modells visualisieren
  - ```model.summary()``` gibt Modellarchitektur aus
    - **Total params:** Gesamte Anzahl der Parameter im Modell.
    - **Trainable params (patterns):** Dies sind die Parameter (Muster), die das Modell während des Trainings aktualisieren kann.
    - **Non-trainable params:** Diese Parameter werden während des Trainings nicht aktualisiert (dies ist typisch, wenn Sie bereits gelernte Muster oder Parameter von anderen Modellen während des Transfer Learnings einbringen).

  - **Regressionsbewertungsmetriken (Vorhersage des Modells bewerten)**
    - Je nach Problem, an dem Sie arbeiten, gibt es verschiedene Bewertungsmetriken, um die Leistung Ihres Modells zu bewerten.
    - Für Regressionen, sind drei der wichtigsten Metriken:
      1. **MAE (Mean Absoulte Error):**
          - mittlerer absoluter Fehler, "wie falsch ist jede Vorhersage meines Modells im Durchschnitt"
          - Verwendung: als Einstiegsmetrik für jede Regression
      2. **MSE (Mean Square Error):**
          - mittlerer quadratischer Fehler, "Quadrat der durchschnittlichen Fehler"
          - Verwendung: wenn größere Fehler stärker ins Gewicht fallen als kleinere Fehler
      3. **Huber:**
          - Kombination aus MAE und MSE
          - weniger empfindlich gegenüber Ausreißern als MSE

## 4. Modell verbessern
  1. **Mehr Daten:** mehr Beispiele für das Modell zum Trainieren (mehr Möglichkeiten zum Lernen von Mustern oder Beziehungen zwischen Merkmalen und Bezeichnungen)
  2. **Modell vergrößern (komplexeres Modell verwenden):** in Form von mehr Schichten oder mehr versteckten Einheiten in jeder Schicht
  3. **Länger Trainieren:** Modell eine größere Chance, Muster in den Daten zu finden

## 5. Modell tracken
  - Tools:
    1. **TensorBoard:** eine Komponente der TensorFlow Bibliothek, um die Modellierungsexperimente zu verfolgen
    2. **Weights & Biases:** ein Werkzeug zur Verfolgung aller Arten von Experimenten des maschinellen Lernens (wird direkt in TensorBoard eingebunden)

## 6. Modell speichern
  1. **SavedModel Format:**
      - verwenden, wenn man in TensorFlow Umgebung bleiben möchte
      - ```model.save("model_name")```
      - wird als Ordner gespeichert
  2. **HDF5 Format:**
      - verwenden, wenn man Modell auserhalb TensorFlow verwenden möchte
      - ```model.save("model_name.h5")```
      - wird als eine Datei gespeichert
