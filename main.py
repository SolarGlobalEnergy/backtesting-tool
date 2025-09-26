import pandas as pd
import warnings
import numpy as np
import os

def load_bess_data():
    """Load data for BESS optimizer"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        VDT = (pd.read_excel('data/data.xlsx', sheet_name='VDT (R2W)')).set_index('datetime', inplace=False)
        VDT = VDT[VDT.index <= '2025-06-30']
        VDT['Vážený průměr cen (EUR/MWh)'].fillna(method='ffill', inplace=True)
    return VDT

def load_fve_bess_data():
    try:
        """Load data for FVE+BESS optimizer (based on the second file)"""
        VDT = (pd.read_excel('data/data.xlsx', sheet_name='VDT (R2W)')).set_index('datetime', inplace=False)
        DT = (pd.read_excel('data/data.xlsx', sheet_name='DT (R2W)')).set_index('datetime', inplace=False)
        FVE = (pd.read_excel('data/data.xlsx', sheet_name='FVE (R2W)')).set_index('datetime', inplace=False)
        DT = DT[(DT.index >= '2025-01-01') & (DT.index <= '2025-06-28')]
        VDT = VDT[(VDT.index <= '2025-06-30')]
        FVE = FVE[(FVE.index <= '2025-06-30')]
        VDT['Vážený průměr cen (EUR/MWh)'].fillna(method='ffill', inplace=True)
            
        return DT, VDT, FVE
        
    except FileNotFoundError as e:
        print(f"Chyba: Soubor nebyl nalezen: {e}")
        print("Pro FVE+BESS optimizaci potřebujete následující soubory:")
        print("- data/data.xlsx")
        return None, None, None

if __name__ == '__main__':
    print("="*60)
    print("  ENERGY OPTIMIZER - VÝBĚR TYPU OPTIMIZACE")
    print("="*60)
    print("1. BESS Optimizer (pouze BESS)")
    print("2. FVE + BESS Optimizer (FVE + BESS kombinace)")
    print("="*60)
    
    # Choose optimizer type
    while True:
        optimizer_choice = input("Vyberte typ optimizace (1 nebo 2): ").strip()
        if optimizer_choice in ['1', '2']:
            break
        print("Neplatná volba! Zadejte 1 nebo 2.")
    
    # Load data based on optimizer choice
    print("\nNačítání dat...")
    
    if optimizer_choice == '1':
        try:
            VDT = load_bess_data()
            print("✅ Data pro BESS optimizaci načtena úspěšně")
        except Exception as e:
            print(f"❌ Chyba při načítání dat: {e}")
            input("Stiskněte Enter pro ukončení...")
            exit()
    else:
        try:
            DT, VDT, FVE = load_fve_bess_data()
            if DT is None:
                input("Stiskněte Enter pro ukončení...")
                exit()
            print("✅ Data pro FVE+BESS optimizaci načtena úspěšně")
        except Exception as e:
            print(f"❌ Chyba při načítání dat: {e}")
            input("Stiskněte Enter pro ukončení...")
            exit()
    
    # Common input collection
    print("\n" + "="*50)
    print("  KONFIGURACE PROJEKTU")
    print("="*50)
    
    project_name = input("Název projektu: ").strip()
    if not project_name:
        project_name = "Unnamed_Project"
    
    # BESS parameters
    print("\n--- BESS parametry ---")
    user_input = input("Instalovaný výkon BESS [MW]: ")
    try:
        bess_power_mw = float(user_input)
    except:
        print("❌ Neprávný formát vstupu!")
        input("Stiskněte Enter pro ukončení...")
        exit()

    user_input = input("Instalovaná kapacita BESS [MWh]: ")
    try:
        bess_capacity_mwh = float(user_input)
    except:
        print("❌ Neprávný formát vstupu!")
        input("Stiskněte Enter pro ukončení...")
        exit()
    
    user_input = input("Rezervovaný výkon OM - export [MW]: ")
    try:
        export_limit_mw = float(user_input)
    except:
        print("❌ Neprávný formát vstupu!")
        input("Stiskněte Enter pro ukončení...")
        exit()
    
    user_input = input("Rezervovaná kapacita v OM - import [MW]: ")
    try:
        import_limit_mw = float(user_input)
    except:
        print("❌ Neprávný formát vstupu!")
        input("Stiskněte Enter pro ukončení...")
        exit()
    
    print("\n--- Provozní parametry ---")
    user_input = input("Účinnost BESS (předvolená hodnota je 0.85): ").strip()
    if user_input:
        try:
            efficiency = float(user_input)
            if not 0.1 <= efficiency <= 1.0:
                print("⚠️  Účinnost nastavena na 0.85 (hodnota mimo rozsah 0.1-1.0)")
                efficiency = 0.85
        except:
            print("⚠️  Neplatná hodnota, použita předvolená účinnost 0.85")
            efficiency = 0.85
    else:
        efficiency = 0.85
    
    user_input = input("Maximální denní počet cyklů (předvolená hodnota je 2): ").strip()
    if user_input:
        try:
            max_cycles = float(user_input)
            if max_cycles < 0:
                print("⚠️  Počet cyklů nastaven na 2 (záporná hodnota)")
                max_cycles = 2
        except:
            print("⚠️  Neplatná hodnota, použity předvolené 2 cykly")
            max_cycles = 2
    else:
        max_cycles = 2

    # FVE specific inputs
    if optimizer_choice == '2':
        print("\n--- FVE parametry ---")
        user_input = input("Instalovaný výkon FVE [MW]: ")
        try:
            fve_power_mw = float(user_input)
        except:
            print("❌ Neprávný formát vstupu!")
            input("Stiskněte Enter pro ukončení...")
            exit()
        fve_scale_factor = fve_power_mw / 1.6
        
        print("\nRežim provozu:")
        print("1. fve_bess_integrated (FVE a BESS integrované - substituce)")
        print("2. fve_bess_independent (FVE a BESS nezávislé - čistý BESS)")
        
        while True:
            mode_choice = input("Vyberte režim (1 nebo 2): ").strip()
            if mode_choice == '1':
                operation_mode = 'fve_bess_integrated'
                break
            elif mode_choice == '2':
                operation_mode = 'fve_bess_independent'
                break
            print("Neplatná volba! Zadejte 1 nebo 2.")

    # Report options
    print("\n--- Možnosti reportů ---")
    user_input = input("Chci denní reporty [A/N]: ").strip().lower()
    daily_reports = user_input == 'a'

    user_input = input("Chci vidět grafy v Excelu [A/N]: ").strip().lower()
    with_plots = user_input == 'a'
    
    # Import appropriate optimizer and functions
    try:
        if optimizer_choice == '1':
            print("\n📦 Importování BESS Optimizer...")
            from BessOptimizer import analyze_projects, BESS_Optimizer, export_full_report_to_excel
            
            PROJECTS_CONFIG = {
                project_name: {
                    'name': project_name,
                    'bess_power_mw': bess_power_mw,
                    'bess_capacity_mwh': bess_capacity_mwh,
                    'export_limit_mw': export_limit_mw,
                    'import_limit_mw': import_limit_mw,
                    'efficiency': efficiency,
                    'max_cycles': max_cycles
                }
            }
            
            print(f"\n🚀 Spouštění BESS optimizace pro projekt '{project_name}'...")
            print(f"   • Výkon: {bess_power_mw} MW")
            print(f"   • Kapacita: {bess_capacity_mwh} MWh")
            print(f"   • Max cykly/den: {max_cycles}")
            
            results = analyze_projects(VDT, projects_config=PROJECTS_CONFIG, 
                                     analysis_period_days=365, show_daily_reports=daily_reports)
            
            for i in results.keys():
                optimizer_instance = BESS_Optimizer(PROJECTS_CONFIG[i])
                project_key_to_export = i
                project_data_to_export = results[project_key_to_export]
                excel_filename = f"report_{project_key_to_export}_BESS.xlsx"
                
                print(f"\n📊 Vytváření Excel reportu: {excel_filename}")
                export_full_report_to_excel(project_data_to_export, VDT, optimizer_instance, 
                                          excel_filename, with_plots=with_plots)
        
        else:  # FVE+BESS Optimizer
            print("\n📦 Importování FVE+BESS Optimizer...")
            from FveBessOptimizer import analyze_projects, FVE_BESS_Optimizer, export_full_report_to_excel
            
            PROJECTS_CONFIG = {
                project_name: {
                    'name': project_name,
                    'fve_power_mw': fve_power_mw,
                    'fve_scale_factor': fve_scale_factor,
                    'bess_power_mw': bess_power_mw,
                    'bess_capacity_mwh': bess_capacity_mwh,
                    'export_limit_mw': export_limit_mw,
                    'import_limit_mw': import_limit_mw,
                    'efficiency': efficiency,
                    'max_cycles': max_cycles,
                    'operation_mode': operation_mode
                }
            }
            
            print(f"\n🚀 Spouštění FVE+BESS optimizace pro projekt '{project_name}'...")
            print(f"   • FVE výkon: {fve_power_mw} MW (faktor: {fve_scale_factor})")
            print(f"   • BESS výkon: {bess_power_mw} MW, kapacita: {bess_capacity_mwh} MWh")
            print(f"   • Režim: {operation_mode}")
            print(f"   • Max cykly/den: {max_cycles}")
            
            results = analyze_projects(DT, VDT, FVE, projects_config=PROJECTS_CONFIG,
                                     analysis_period_days=365, show_daily_reports=daily_reports)
            
            for i in results.keys():
                optimizer_instance = FVE_BESS_Optimizer(PROJECTS_CONFIG[i])
                project_key_to_export = i
                project_data_to_export = results[project_key_to_export]
                excel_filename = f"report_{project_key_to_export}_FVE_BESS.xlsx"
                
                print(f"\n📊 Vytváření Excel reportu: {excel_filename}")
                export_full_report_to_excel(project_data_to_export, VDT, optimizer_instance, 
                                          excel_filename, with_plots=with_plots)

    except ImportError as e:
        print(f"❌ Chyba při importování: {e}")
        print("Ujistěte se, že máte v adresáři soubory:")
        if optimizer_choice == '1':
            print("- BessOptimizer.py")
        else:
            print("- FveBessOptimizer.py")
        input("Stiskněte Enter pro ukončení...")
        exit()
    
    except Exception as e:
        print(f"❌ Chyba při optimizaci: {e}")
        input("Stiskněte Enter pro ukončení...")
        exit()

    print("\n" + "="*60)
    print("  ✅ OPTIMIZACE ÚSPĚŠNĚ DOKONČENA!")
    print("="*60)
    
    # Display results summary
    if results:
        for project_key, result in results.items():
            annual_income = result.get('annual_client_income_eur', 0) + result.get('annual_our_income_eur', 0)
            days_analyzed = result.get('days_analyzed', 0)
            print(f"\n📈 Projekt: {project_key}")
            print(f"   • Analyzované dny: {days_analyzed}")
            print(f"   • Celkový roční příjem: {annual_income:,.0f} EUR")
            if days_analyzed > 0:
                print(f"   • Průměrný denní příjem: {annual_income/days_analyzed:.0f} EUR/den")
    
    print("\n" + "="*60)
    input("Stiskněte Enter pro ukončení...")
