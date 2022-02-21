# Einführung in die Regression mit neuronalen Netzen in Tensorflow

## Modelling in Tensorflow
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

## Modell Evaluieren
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

## Modell verbessern
  1. **Mehr Daten:** mehr Beispiele für das Modell zum Trainieren (mehr Möglichkeiten zum Lernen von Mustern oder Beziehungen zwischen Merkmalen und Bezeichnungen)
  2. **Modell vergrößern (komplexeres Modell verwenden):** in Form von mehr Schichten oder mehr versteckten Einheiten in jeder Schicht
  3. **Länger Trainieren:** Modell eine größere Chance, Muster in den Daten zu finden
