#	Porovnanie analýzy toxicity v online priestore uvažovaním unimodálneho a multimodálneho prístupu

Autor: Bc. Roland Palgut

Školiteľ: prof. Ing. Kristína Machová, PhD.

Konzultant: Ing. Viliam Balara

# Systemova príručka 
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Obsah 
- train.jsonl – trénovacia množina dát
- dev_merged.jsonl – validačná množina
- test_merged.jsonl – testovacia množina
  
- BERT.py – textový model založený na BERT architektúre
- RoBERTa.py – textový model založený na RoBERTa architektúre
- ELECTRA.py – textový model ELECTRA
  
- CNN.py – jednoduchý konvolučný model pre obrazové dáta
- ResNET50.py – pokročilý CNN model ResNet50
- ViT.py – Vision Transformer model pre obrazy
  
- CLIP.py – multimodálny model CLIP
- VisBERT.py – multimodálny model VisualBERT
- LateFusion.py – implementácia neskorej fúzie modelov
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Vstupné dáta nie sú priamo súčasťou repozitára, keďže ich veľkosť presahuje limity. Z tohto dôvodu je potrebné ich stiahnuť z: https://huggingface.co/datasets/rolandpalgut/data_dp. Po stiahnutí je potrebné rozbaliť zo zip a zachovať štruktúru, kde sa všetky obrázky nachádzajú v priečinku img.

Pred samotným použitím fúzie je potrebné najskôr natrénovať jednotlivé modely. Ide o textové modely (BERT, ELECTRA, RoBERTa) a obrazové modely (SimpleCNN, ResNet50, ViT). Po ich natrénovaní a uložení je možné spustiť skript LateFusion.py, ktorý kombinuje ich výstupy.

Pred spustením fúzie je nutné upraviť cesty k modelom BERT a RoBERTa v konfigurácii TEXT_CONFIG, aby ukazovali na správne uložené checkpointy. Bez tejto úpravy skript nebude fungovať správne.

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
