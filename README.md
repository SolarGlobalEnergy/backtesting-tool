# Dokumentace: Nástroj pro optimalizaci BESS a FVE

Tento dokument poskytuje podrobný popis nástroje pro backtesting a optimalizaci obchodních strategií pro bateriová úložiště energie (BESS), a to jak pro samostatné baterie, tak pro systémy kombinované s fotovoltaickou elektrárnou (FVE).

## Přehled modelů

Nástroj obsahuje dva hlavní optimalizační modely:

1.  **Samostatná Baterie (`BESS_Optimizer`)**: Optimalizuje provoz baterie, která obchoduje na základě cenových rozdílů na trhu. Tento model je zaměřen na čistě spekulativní obchodování.
2.  **Baterie s Fotovoltaikou (`FVE_BESS_Optimizer`)**: Optimalizuje společný provoz baterie a FVE. Tento model řeší, zda je výhodnější energii z FVE prodat ihned, nebo ji uložit do baterie a prodat později.

---

## Model 1: Samostatná Baterie (`BESS_Optimizer`)

- **Zdrojový soubor**: `BessOptimizer.py`
- **Spouštěcí skript**: `onlybess.ipynb`

### Popis
Tento model je určen pro scénáře, kde baterie funguje jako samostatný prvek. Jejím jediným cílem je maximalizovat zisk nákupem energie ze sítě za nízkou cenu a jejím prodejem zpět do sítě za cenu vyšší.

### Princip fungování

Optimalizace je založena na metodě **dynamického programování**. Tento přístup umožňuje najít globálně optimální strategii pro celý den.

1.  **Časové intervaly**: Den je rozdělen na 96 patnáctiminutových intervalů, pro které jsou známy ceny na trhu.
2.  **Stavy**: V každém intervalu může být baterie v různém stavu nabití (SoC – State of Charge). Pro zjednodušení výpočtu je kapacita baterie rozdělena na diskrétní úrovně (např. po 5 %).
3.  **Rozhodování**: Algoritmus pro každý časový interval a každý stav nabití zvažuje všechny možné akce a vybere tu, která vede k nejvyššímu celkovému zisku na konci dne.

#### Možné akce (`Action`)
V každém kroku se model rozhoduje mezi třemi základními akcemi:
- `charge_from_grid`: Nabíjení baterie energií ze sítě.
- `discharge_to_grid`: Vybíjení baterie a prodej energie do sítě.
- **`hold`** (držet): Nedělat nic a ponechat stav nabití beze změny.

#### Klíčová omezení
Model respektuje reálné fyzikální a provozní limity:
- **Výkonové limity (`bess_power_mw`)**: Maximální rychlost, jakou lze baterii nabíjet nebo vybíjet.
- **Kapacita (`bess_capacity_mwh`)**: Maximální množství energie, které baterie pojme.
- **Účinnost (`efficiency`)**: Ztráty energie, ke kterým dochází při nabíjení a vybíjení.
- **Limit cyklů (`max_cycles`)**: Omezení denního počtu cyklů, aby se prodloužila životnost baterie. Algoritmus automaticky vybere pouze ty nejziskovější obchody, aby limit nebyl překročen.

### Konfigurace projektu (`project_config`)
Pro spuštění analýzy je nutné definovat slovník s následujícími parametry:

- `name` (str): Název projektu.
- `bess_power_mw` (float): Maximální výkon baterie v MW.
- `bess_capacity_mwh` (float): Celková kapacita baterie v MWh.
- `export_limit_mw` (float): Maximální povolený výkon pro dodávku do sítě (obvykle shodný s `bess_power_mw`).
- `import_limit_mw` (float): Maximální povolený výkon pro odběr ze sítě (obvykle shodný s `bess_power_mw`).
- `efficiency` (float): Celková účinnost baterie v obou směrech (round-trip, např. `0.85` pro 85%).
- `max_cycles` (int): Maximální počet plných cyklů za den.
- `initial_soc_mwh` (float, volitelné): Počáteční stav nabití. Pokud není zadán, předpokládá se 50 %.

---

## Model 2: Baterie s Fotovoltaikou (`FVE_BESS_Optimizer`)

- **Zdrojový soubor**: `FveBessOptimizer.py`
- **Spouštěcí skript**: `fvebess.ipynb` (lze spustit přímo) nebo vlastní notebook

### Popis
Tento pokročilý model je určen pro optimalizaci systému, kde baterie (BESS) spolupracuje s fotovoltaickou elektrárnou (FVE). Cílem již není jen spekulace na cenách, ale inteligentní řízení toků energie mezi FVE, baterií a sítí tak, aby byl celkový zisk co nejvyšší.

### Princip fungování

Model opět využívá **dynamické programování**, ale rozhodovací proces je mnohem komplexnější. Kód umožňuje dva základní režimy provozu, které se liší ve strategii řízení FVE a baterie.

#### Režimy provozu

1.  **Integrovaný režim (Integrated Mode)**
    - **Popis**: Toto je hlavní a nejpokročilejší strategie. Model v tomto režimu aktivně rozhoduje, co je v danou chvíli nejvýhodnější udělat s energií z FVE. FVE a BESS jsou řízeny jako jeden celek s cílem maximalizovat celkový zisk.
    - **Logika**: Využívá tzv. **substituční strategii**. Místo okamžitého prodeje energie z FVE model zvažuje její uložení do baterie, pokud očekává v budoucnu vyšší prodejní cenu. Tím "nahrazuje" (substituuje) levný prodej za dražší.
    - **Implementace**: V kódu tomuto režimu odpovídá metoda `_solve_with_discretized_dp`.

2.  **Nezávislý režim (Independent Mode)**
    - **Popis**: V tomto režimu fungují FVE a BESS jako dva oddělené systémy.
    - **Logika**:
        - **FVE**: Veškerá vyrobená energie z FVE se okamžitě prodává do sítě za aktuální cenu.
        - **BESS**: Baterie funguje čistě spekulativně. Nakupuje energii ze sítě, když je levná, a prodává ji, když je drahá, přesně jako v **Modelu 1**. Ignoruje přitom zcela existenci FVE.
    - **Implementace**: Tato logika je obsažena v metodě `_solve_with_discretized_dp_pure_bess`. Ačkoliv není ve výchozím stavu aktivní, lze ji použít pro srovnávací analýzy, aby se ukázala přidaná hodnota integrovaného řízení.

#### Klíčový koncept: Substituční strategie
Hlavní myšlenkou je **substituce** (nahrazení). Místo toho, aby se vyrobená energie z FVE okamžitě prodala za aktuální (potenciálně nízkou) cenu na denním trhu (DT), model zvažuje, zda je výhodnější ji:
1.  **Uložit do baterie**: Pokud se očekává, že cena na vnitrodenním trhu (VDT) bude později během dne výrazně vyšší.
2.  **Prodat ihned**: Pokud jsou aktuální ceny dostatečně vysoké nebo pokud je baterie plná.

Tímto způsobem model "nahrazuje" levný prodej z FVE za dražší prodej z baterie v pozdějším čase. Zisk vzniká z rozdílu mezi cenou, za kterou by se prodalo z FVE, a cenou, za kterou se později prodá z baterie.

#### Vstupní data
Tento model vyžaduje širší sadu dat:
- `dt_prices_eur`: Ceny na **denním trhu** (DT). Používají se jako referenční cena pro prodej z FVE.
- `vdt_prices_eur`: Ceny na **vnitrodenním trhu** (VDT). Používají se pro spekulativní obchodování (nákup/prodej ze sítě) a pro ocenění prodeje z baterie.
- `fve_generation`: Časová řada výroby FVE v MW pro daný den.

#### Možné akce (`Action`)
Sada možných akcí je rozšířena o operace s FVE:
- `charge_from_grid`: Nabíjení baterie ze sítě (spekulativní, když je cena na VDT nízká).
- `discharge_to_grid`: Vybíjení baterie do sítě (prodej za cenu VDT).
- `charge_from_fve`: Nabíjení baterie přímo z FVE (substituce).
- `sell_fve_direct`: Přímý prodej vyrobené energie z FVE do sítě za cenu DT.
- **`hold`** (držet): Ponechat energii v baterii.

#### Finanční model
Zisk se skládá ze dvou hlavních složek:
1.  **Příjem z FVE**: Standardní příjem z prodeje energie z FVE na denním trhu, ze kterého si provozovatel bere fixní provizi.
2.  **Dodatečný zisk z BESS (Spread)**: Zisk dosažený díky chytrému řízení baterie (substituce a spekulace na VDT). Tento zisk se dělí mezi klienta a provozovatele (např. v poměru 70/30).

### Konfigurace projektu (`project_config`)
Konfigurační slovník pro tento model obsahuje dodatečné parametry:
- Všechny parametry z Modelu 1 (`name`, `bess_power_mw`, `bess_capacity_mwh`, atd.).
- `fve_power_mw` (float): Instalovaný výkon FVE v MW.
- `fve_scale_factor` (float): Koeficient pro škálování výroby FVE (např. `1.0` pro 100% využití nominálních dat o výrobě).

---

## Jak spustit vlastní analýzu

Spuštění analýzy vyžaduje přípravu dat a správnou konfiguraci.

1.  **Příprava dat**:
    - **Cenová data**: Připravte si Excel soubor s historickými cenami elektřiny. Pro Model 1 stačí ceny z vnitrodenního trhu (VDT). Pro Model 2 budete potřebovat ceny z denního (DT) i vnitrodenního trhu (VDT).
    - **Data o výrobě FVE** (pouze pro Model 2): Připravte si CSV nebo Excel soubor s daty o výrobě vaší FVE v 15minutových intervalech.

2.  **Výběr a úprava spouštěcího skriptu**:
    - Pro **Model 1** použijte a upravte `onlybess.ipynb`.
    - Pro **Model 2** použijte a upravte `fvebess.ipynb`.

3.  **Úprava konfigurace**:
    - V příslušném skriptu najděte slovník `PROJECTS_CONFIG`.
    - Upravte parametry tak, aby odpovídaly vašemu projektu (kapacita, výkon, účinnost atd.).

4.  **Spuštění a výsledky**:
    - Spusťte Jupyter Notebook nebo Python skript.
    - Po dokončení analýzy naleznete v adresáři projektu detailní Excel reporty s finančními výsledky, provozními statistikami a denními grafy.
