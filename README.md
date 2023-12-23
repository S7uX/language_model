# Language Model

`language_model.py`

Ausgabe: **ARPA**

## Ausführen von `language_model.py`

1. Python-Konsole starten

```bash
python
```

2. Import

```python
from language_model import *
from arpa import *
```

3. Ausführen

```python
lines = load_text_from_file("input.txt")
language_model = generate_mle_language_model(lines, 2)
write_arpa_file(language_model, "output.lm")
```

### Verfügbare Funktionen

#### `load_text_from_file`

Parameter:

1. `file`: Dateipfad
2. `number_of_sentence_pseudo_words`: Anzahl der Satzanfang `<s>` und Satzende `</s>` Pseudowörter an den jeweiligen
   Satzanfängen und Enden. **`default = 1`**

#### Sprachmodellfunktionen

Parameter:

1. `lines`: Satzzeilen
2. `n`: Ordnung des Sprachmodells

* `generate_mle_language_model`
* `generate_add_k_language_model`: Extra Parameter `k` vorhanden.
* `generate_witten_bell_language_model`
* `generate_kneser_ney_language_model`

## Umsetzung

Für die Umsetzung wurde die Literatur im Verzeichnis `literatur/` verwendet
und die im Abschnitt [Online-Referenzen](#online-referenzen) aufgelisteten Webseiten.

Auf die Literatur wird im Code durch Zitate bezug genommen,
durch Referenzen der From `[JURAFSKY 2008, eqn. 3.12 on p. 5]` in den Kommentaren.

Oft wird eine Implementierung, mit einem Kommentar, der die implementierte Gleichung beschreibt, eingeleitet.

Beispiel:

```python
# Witten-Bell interpolation weights wb_lambda ≔ λ(w₁…wₙ₋₁) [CHEN 1998, eqn. 16 on p. 13]:
# ［1 - λ(w₁…wₙ₋₁)］= N₁₊(w₁…wₙ₋₁•) ⧸［N₁₊(w₁…wₙ₋₁•) + C(w₁…wₙ₋₁)］
wb_lambda = -1 * ((possible_extensions_count / (possible_extensions_count + history_count)) - 1)
```

## Code-Struktur

Für die jeweiligen Wahrscheinlichkeitsvarianten (MLE mit oder ohne spezielles Smoothing) sind jeweils Funktionen
vorhanden:

* `generate_mle_language_model`
* `generate_add_k_language_model`
* `generate_witten_bell_language_model`
* `generate_kneser_ney_language_model`

## Erklärungen zur Implementierung

### Notation

* $N$: $n$-gramm Größe.
* $w_{1:n}$ bedeutet $w_1, w_2, ..., w_n$ mit $N = n$.
* $w_{1:n-1}$ bedeutet $w_1, w_2, ..., w_{n-1}$ mit $N = n - 1$

Die **MLE-Implementierung** richtet sich nach der (allgemeinen) Gleichung für
N-Gramme [JURAFSKY 2008, eqn. 3.12 on p. 5]:

$$
P_{\text{ML}}(w_n|w_{n-N+1:n-1}) = \frac{C(w_{n-N+1:n-1}w_n)}{C(w_{n-N+1:n-1})}
$$

Die **Add-k-Smoothing-Implementierung** richtet sich nach der Gleichung:

$$
P_{\text{Add-k}}(w_n|w_{n-N+1:n-1}) = \frac{C(w_{n-N+1:n-1}w_n)+k}{C(w_{n-N+1:n-1})+kV}
$$

Die sich aus der Kombination der MLE-Gleichung (obendrüber)
und der Gleichung fur Add-k-Smoothing für Bigramme aus [JURAFSKY 2008, eqn. 3.26 on p. 16] ergibt.

Die Implementierungen im Code entsprechen diesen Gleichungen ohne große Umwege.
Da die Gleichungen wenig Interpretationsspielraum lassen,
wird die Implementierung nicht weiter verargumentiert.

### Witten-Bell-Smoothing

Die Implementierung von Witten-Bell hingegen lässt mehr Interpretationsspielraum,
weil Witten-Bell eine Methode zum rekursiven Interpolieren nach folgender Gleichung ist [CHEN 1998, eqn. 14 on p. 13]:

$$
P_{\text{WB}}(w_n|w_{n-N+1:n-1}) = \lambda(w_{n-N+1:n-1}) P_{\text{ML}}(w_n|w_{n-N+1:n-1}) + (1 - \lambda(w_{n-N+1:n-1}))P_
{\text{WB}}(w_n|w_{n-N++:n-1})
$$

Die Frage, die sich bei der Implementierung stellt, ist wie die Rekursion beendet wird.
Wie wird die Wahrscheinlichkeit $P(w_i)$ für Unigramme berechnet?

Chen erwähnt zwar eine Methode Unigramme (Modell erster Ordnung) über das Modell nullter Ordnung (Zerogramm) zu
interpolieren [CHEN 1998, on p. 11],
geht aber für Witten-Bell nicht näher auf die Implementierung ein.
Andere Quellen haben wir nicht gefunden.
Weshalb die Rekursion in der Implementierung bei den Unigrammen beendet wird.

### Kneser-Ney-Smoothing

Kneser-Ney ist auch rekursiv und folgendermaßen definiert [JURAFSKY 2008, p. 21]:

$$
P_{\text{KN}}(w_n|w_{n-N+1:n-1}) = \frac{\max(c_\textrm{KN}(w_{n-N+1:n})-d, 0)}{\sum_v c_\textrm{KN}(w_{n-N+1:n-1}v)} +
\lambda(w_{n-N+1:n-1})P_{\text{KN}}(w_n|w_{n-N+1:n-1})
$$

$$
\lambda(w_{n-N+1:n-1}) = \frac{d}{\sum_v C(w_{n-N+1:n-1}v)} \left| \{w' : C(w_{n-N+1:n-1}w') > 0\} \right|
$$

Wobei $N_{1+}(w_{n-N+1:n-1}•) = \left| \{w' : C(w_{n-N+1:n-1}w') > 0\} \right|$ die Anzahl an unterschiedlichen
(einzigartigen) Worten mit dem Vorgänger $w_{n-N+1:n-1}$ ist [CHEN 1998, eqn. 15 on p. 13].

$c_\textrm{KN}$ ist folgendermaßen definiert [JURAFSKY 2008, eqn 3.41 on p. 21]:

$$
c_\textrm{KN}(s) =
\begin{cases}
\textrm{count}(s) \quad \textrm{for the highest order} \\
\textrm{continuationcount}(s) \quad \textrm{for lower orders}
\end{cases}
$$

Der Continuation-Count für einen String $s$ ist die Anzahl an unterschiedlichen (einzigartigen) Wort-Kontexten für
diesen String $s$ [JURAFSKY 2008, p. 21]. Anders ausgedrückt ist $N_{1+}(•s)$ die Anzahl an einzigartigen Worten $w'$,
die $s$ vorausgehen. Chen definiert $\textrm{continuationcount}(s)$ folgendermaßen [CHEN 1998, p. 17]:

$$
N_{1+}(•s) = \left| \{w' : C(w's) > 0\} \right|
$$

Für den Discount $d$ wird der fixe Wert $0.75$ verwendet [JURAFSKY 2008 p, 20].
Unterschiedliche Werte für $d$ sind Gegenstand des hier nicht implementierten Modified-Kneser-Ney-Smoothing
[JURAFSKY 2008 p, 21].

### Rekursionsende

Am Rekursionsende werden Unigramms mit der Uniformverteilung $\frac{1}{V}$ interpoliert.
Das entspricht der oben beschriebenen Interpolation mit dem Modell nullter Ordnung.

$$
P_\textrm{KN}(w) = \frac{\max(c_\textrm{KN}(w)-d, 0)}{\sum_{w'} c_\textrm{KN}(w')} + \lambda(\epsilon) \frac{1}{V}
$$

Wobei $\epsilon$ das leere Wort ist und $V$ die Zahl einzigartiger Wörter (Mächtigkeit des Alphabets).

Für unbekannte Worte wird das Pseudowort `<unk>` mit der Wahrscheinlichkeit $\frac{\lambda(\epsilon)}{V}$ hinzugefügt
[JURAFSKY 2008 p, 21].
Das hat den Hintergrund, dass bei unbekannten Unigrammen bei einem Sprachmodell aus einer ARPA-Datei nicht auf
Backoff-Gewichte zurückgegriffen werden kann.

$$
\lambda(\epsilon) = \frac{d}{\sum_v C(\epsilon v)} \left| \{w' : C(\epsilon w') > 0\} \right| =
\frac{d}{\sum_v C(v)} \left| \{w' : C(w') > 0\} \right|
$$

## Online-Referenzen

### N-Gramm-Erzeugung

Anleitung: <https://albertauyeung.github.io/2018/06/03/generating-ngrams.html/>

### ARPA

* <https://cmusphinx.github.io/wiki/arpaformat/>
* <http://www.seas.ucla.edu/spapl/weichu/htkbook/node243_mn.html>

### Trainigsdaten

`sample_text/train.txt`: <https://github.com/joshualoehr/ngram-language-model/blob/master/data/train.txt>

## Literatur

Daniel Jurafsky, James H. Martin (2008). _Speech and Language Processing_, Chapter 3. \
<https://web.stanford.edu/~jurafsky/slp3/>

Stanley F. Chen, Joshua Goodman (1998). _An Empirical Study of Smoothing Techniques for Language Modeling_. \
<https://dash.harvard.edu/handle/1/25104739>
