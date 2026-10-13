# Algorytmy Macierzowe – Repozytorium Projektowe

## 🛠 Biblioteki i zasady implementacji
* **`ndarray`**: Używamy jej jako naszego odpowiednika NumPy z Pythona. Znacząco ułatwia zarządzanie pamięcią, wyciąganie kolumn, cięcie widoków i dzielenie macierzy na mniejsze bloki.
* **Ważne ograniczenie**: Nie możemy używać wbudowanych w `ndarray` metod algebraicznych (np. gotowego `.dot()`) jako rozwiązania naszych zadań. Biblioteka ma nam służyć tylko do obsługi struktury danych, a same algorytmy piszemy od zera.

## 📂 Struktura plików i kodu
* **`zx/lib.rs`**: Tutaj piszemy większość kodu i logiki (algorytmy). Wszystko, co ma być używane na zewnątrz, oznaczamy jako `pub`.
* **`zx/main.rs`**: Służy jako miejsce na przykład gotowego zadania. Dzięki temu zachowujemy porządek między biblioteką a kodem uruchomieniowym.

## ⏱ Pomiary: Czas vs Liczba operacji
Ponieważ musimy mierzyć i jedno, i drugie bez duplikowania algorytmów, zastosowaliśmy wzorzec z Traitem:
* Stworzyliśmy Traita, którego implementują **dwie struktury**.
* Jedna normalnie zlicza operacje.
* Druga jest pusta i ma flagę `#[inline(always)]`. Przy testach czasowych podajemy tę pustą implementację – kompilator fizycznie usuwa wywołania z kodu maszynowego, dzięki czemu zliczanie ma **zerowy narzut wydajnościowy**.

## 📊 Benchmarki i generowanie danych
* **Criterion**: Do mierzenia czasu używamy biblioteki `criterion`, która działa znacznie lepiej niż ręczne odejmowanie czasu. Automatycznie testuje różne wielkości macierzy, uśrednia wyniki i generuje gotowe wykresy HTML.
* **Dane wejściowe**: Do benchmarków generujemy macierze przy użyciu `rand_chacha`. Jest to deterministyczny, bardzo szybki algorytm – dzięki niemu wygenerujemy identyczne pseudolosowe dane na każdym systemie operacyjnym.
* **Pliki CSV**: Stosujemy je **tylko** do weryfikacji poprawności małych macierzy w testach (np. porównanie naszego wyniku ze wzorcem zapisanym z Matlaba lub biblioteki).

## 🧪 Testowanie
* **Testy jednostkowe**: Zgodnie z konwencją Rusta, piszemy je na samym dole pliku z weryfikowanym kodem (w module `#[cfg(test)]`).
* **Testy integracyjne**: Służą do sprawdzania, czy główna funkcja wykonuje zadanie poprawnie z zewnątrz. Umieszczamy je w osobnym folderze: `zx/tests/nazwa_testu.rs`.

## 📦 Zarządzanie zależnościami (Cargo)
Większość potrzebnych bibliotek jest już dodana. Główna lista pakietów znajduje się w **głównym folderze projektu (Workspace)**. 
Jeśli potrzebujesz dodać nową paczkę w konkretnym zadaniu (`z1/Cargo.toml`), zaciągaj ją z workspace'a w ten sposób:

```toml
[dependencies]
nazwa_biblioteki = { workspace = true }
```

---

## 🚀 Przydatne komendy

Uruchomienie konkretnego zadania (podmień `x` na numer zadania):
```bash
cargo run --bin zx
```

Uruchomienie testów poprawności:
```bash
cargo test
```

Uruchomienie pomiarów wydajności (benchmarków):
```bash
cargo bench
```