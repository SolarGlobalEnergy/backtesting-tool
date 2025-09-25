import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import timedelta
import logging
from dataclasses import dataclass
import io
import warnings


@dataclass
class Action:
    """
    A class to store actions in a single interval.

    Args:
        charge_from_grid (float): The amount of energy to charge from the grid.
        discharge_to_grid (float): The amount of energy to discharge to the grid.
    """
    charge_from_grid: float = 0.0
    discharge_to_grid: float = 0.0

    def __post_init__(self):
        epsilon = 1e-9
        assert all(x >= -epsilon for x in [self.charge_from_grid, self.discharge_to_grid])

    def __hash__(self):
        return hash((self.charge_from_grid, self.discharge_to_grid))

    def __eq__(self, other):
        return (isinstance(other, Action) and
                self.charge_from_grid == other.charge_from_grid and
                self.discharge_to_grid == other.discharge_to_grid)

class BESS_Optimizer:
    """
    Optimizes BESS (Battery Energy Storage System) trading strategies.
    """
    def __init__(self, project_config: Dict):
        """
        Initializes the BESS_Optimizer.

        Args:
            project_config (Dict): A dictionary containing the project configuration.
        """
        self.config = project_config
        self.logger = logging.getLogger(__name__)

        self.CLIENT_SHARE = 0.70
        self.OUR_SHARE = 0.30
        self.INTERVAL_HOURS = 0.25
        self.CZK_EUR_RATE = 24.15
        self.DP_SOC_STEP = 0.05
        
        self.bess_capacity_mwh = self.config['bess_capacity_mwh']
        self.bess_power_mw = self.config['bess_power_mw']
        self.soc0_mwh = self.config.get('initial_soc_mwh', self.bess_capacity_mwh * 0.5)
        self.export_limit_mw = self.config.get('export_limit_mw', 1)
        self.import_limit_mw = self.config.get('import_limit_mw', 1)
        self.max_cycles = self.config.get('max_cycles', 2)

        total_efficiency = self.config.get('efficiency', 0.85)
        self.charge_efficiency = np.sqrt(total_efficiency)
        self.discharge_efficiency = np.sqrt(total_efficiency)

    def _enforce_cycle_limit(self, dp_strategy: List[Action], vdt_prices: List[float]) -> Tuple[List[Action], float]:
        """
        Enforces the daily cycle limit on the BESS strategy.

        This method prioritizes the most profitable charge and discharge intervals
        to stay within the allowed daily cycle limit. It creates a new strategy
        of the same length as the input, replacing non-selected intervals with
        'hold' actions (no charge or discharge).

        Args:
            dp_strategy (List[Action]): The dynamic programming strategy.
            vdt_prices (List[float]): The VDT prices for the day.

        Returns:
            Tuple[List[Action], float]: A tuple containing the modified strategy
                                     and the actual number of cycles performed.
        """
    
        if self.max_cycles <= 0 or self.bess_capacity_mwh <= 0:
            return dp_strategy, 0

        max_energy = self.max_cycles * self.bess_capacity_mwh

        interval_profits = []
        for i, action in enumerate(dp_strategy):
            profit_per_mwh = 0
            if action.discharge_to_grid > 0:
                profit_per_mwh = vdt_prices[i]
            elif action.charge_from_grid > 0:
                profit_per_mwh = -vdt_prices[i]

            interval_profits.append({
                'index': i,
                'action': action,
                'profit_per_mwh': profit_per_mwh,
                'charge_energy': action.charge_from_grid * self.INTERVAL_HOURS,
                'discharge_energy': action.discharge_to_grid * self.INTERVAL_HOURS
            })

        discharge_intervals = sorted(
            [ip for ip in interval_profits if ip['discharge_energy'] > 0],
            key=lambda x: x['profit_per_mwh'], reverse=True
        )
        charge_intervals = sorted(
            [ip for ip in interval_profits if ip['charge_energy'] > 0],
            key=lambda x: x['profit_per_mwh']
        )

        # Select the best intervals within the cycle limit
        selected_intervals = set()
        current_charge = 0
        current_discharge = 0

        # Alternate between charging and discharging for a balanced approach
        discharge_idx = 0
        charge_idx = 0

        while (discharge_idx < len(discharge_intervals) or charge_idx < len(charge_intervals)):
            # Add the best discharge if we can
            if discharge_idx < len(discharge_intervals):
                interval = discharge_intervals[discharge_idx]
                if current_discharge + interval['discharge_energy'] <= max_energy:
                    selected_intervals.add(interval['index'])
                    current_discharge += interval['discharge_energy']
                    discharge_idx += 1
                else:
                    discharge_idx = len(discharge_intervals)

            # Add the best charge if we can
            if charge_idx < len(charge_intervals):
                interval = charge_intervals[charge_idx]
                if current_charge + interval['charge_energy'] <= max_energy:
                    selected_intervals.add(interval['index'])
                    current_charge += interval['charge_energy']
                    charge_idx += 1
                else:
                    charge_idx = len(charge_intervals)

            if (discharge_idx >= len(discharge_intervals) and charge_idx >= len(charge_intervals)):
                break

        modified_strategy = []
        cumulative_charge = 0
        cumulative_discharge = 0
        
        for i, action in enumerate(dp_strategy):
            if i in selected_intervals:
                # Keep the original action
                modified_strategy.append(action)
                cumulative_charge += action.charge_from_grid * self.INTERVAL_HOURS
                cumulative_discharge += action.discharge_to_grid * self.INTERVAL_HOURS
            else:
                # For unselected intervals, add an empty action (hold)
                modified_strategy.append(Action())

        actual_cycles = ((cumulative_charge + cumulative_discharge) /
                        2) / self.bess_capacity_mwh if self.bess_capacity_mwh > 0 else 0

        return modified_strategy, actual_cycles
    
    def optimize_day_strategy_dp(self, vdt_prices_eur: np.array, start_soc: float = 0, end_soc: float = None) -> Dict:
        """
        Optimization with correct profit sharing and respect for the cycle limit.

        Args:
            vdt_prices_eur (np.array): An array of VDT prices in EUR.
            start_soc (float): The starting state of charge.
            end_soc (float): The ending state of charge.

        Returns:
            Dict: A dictionary containing the optimization results.
        """

        if len(vdt_prices_eur) != 96:
            raise ValueError("All arrays must contain 96 elements")

        vdt_optimization = self._optimize_vdt_trading(vdt_prices_eur.tolist(), start_soc, end_soc)
            
        if 'dp_strategy' in vdt_optimization:
            validated_strategy = self._validate_strategy_physically(vdt_optimization['dp_strategy'])
                
            strategy_report = self._convert_dp_strategy_to_report_format_safe(
                validated_strategy, vdt_prices_eur.tolist(),start_soc
            )
            vdt_optimization['strategy'] = strategy_report

        vdt_spread = vdt_optimization['total_additional_profit_eur']
        vdt_breakdown = {
            'client_vdt_spread_revenue': vdt_spread * self.CLIENT_SHARE,
            'our_vdt_spread_commission': vdt_spread * self.OUR_SHARE
        }

        total_client_income = vdt_breakdown['client_vdt_spread_revenue']
        total_our_income = vdt_breakdown['our_vdt_spread_commission']

        total_results = {
            'financial_results': {
                'client_vdt_spread_revenue_eur': round(vdt_breakdown['client_vdt_spread_revenue'], 2),
                'our_vdt_spread_commission_eur': round(vdt_breakdown['our_vdt_spread_commission'], 2),
                'client_total_income_eur': round(total_client_income, 2),
                'our_total_commission_eur': round(total_our_income, 2),
                'client_total_income_czk': round(total_client_income * self.CZK_EUR_RATE, 2),
                'our_total_commission_czk': round(total_our_income * self.CZK_EUR_RATE, 2),
            },
            'operational_summary': {
                'vdt_strategy': vdt_optimization['strategy'],
                'actual_cycles': vdt_optimization.get('actual_cycles', 0),
                'cycle_limit_enforced': vdt_optimization.get('cycle_limit_enforced', False),
                'start_soc_mwh': start_soc
            },
            'trading_report': self._generate_trading_report(vdt_optimization['strategy']),
            'algorithm_used': "dp",
            'dp_states_used': vdt_optimization.get('dp_states_used', 'N/A')
        }

        return total_results
    
    def _generate_trading_report(self, strategy: List[Dict]) -> Dict:
        """
        Generates a detailed trading report from the strategy.

        Args:
            strategy (List[Dict]): The trading strategy, where each entry represents
                                an interval.

        Returns:
            Dict: A dictionary containing the list of active trades with profit
                  and SoC information.
        """
        active_trades = []
        cumulative_profit = 0

        for interval_data in strategy:
            action = interval_data['action']['dp_action']
            vdt_price = interval_data['vdt_price_eur_mwh']

            period_profit = self._calculate_action_profit_dp(action, vdt_price)

            if abs(period_profit) > 0.01:
                cumulative_profit += period_profit
                trade_type = "hold"
                if action.discharge_to_grid > 0:
                    trade_type = "discharge"
                elif action.charge_from_grid > 0:
                    trade_type = "charge"

                active_trades.append({
                    'period': interval_data['interval'],
                    'vdt_price': vdt_price,
                    'action': trade_type,
                    'profit': period_profit,
                    'soc': interval_data['soc_mwh'],
                    'cumulative': cumulative_profit
                })
        return {'active_trades': active_trades}

    def _optimize_vdt_trading(self, vdt_prices: List[float],
                              start_soc: float = 0.0, end_soc: float = None) -> Dict:
        """
        Wrapper for running DP with subsequent cycle correction.

        Args:
            vdt_prices (List[float]): The VDT prices.
            start_soc (float): The starting state of charge.
            end_soc (float): The ending state of charge.

        Returns:
            Dict: A dictionary containing the optimization results.
        """
        soc_step = self.DP_SOC_STEP
        soc_states = np.arange(0, self.bess_capacity_mwh + soc_step, soc_step)

        result = self._solve_with_discretized_dp(vdt_prices,  start_soc, end_soc, soc_states, soc_step)

        original_strategy = result['dp_strategy']

        modified_strategy, actual_cycles = self._enforce_cycle_limit(original_strategy, vdt_prices)
        strategy_report = self._convert_dp_strategy_to_report_format_safe(modified_strategy, vdt_prices, start_soc)

        total_profit = sum(self._calculate_action_profit_dp(a, vdt_prices[i])
                           for i, a in enumerate(modified_strategy))

        return {
            'total_additional_profit_eur': float(total_profit),
            'strategy': strategy_report,
            'final_soc': result['final_soc'],
            'dp_states_used': result['dp_states_used'],
            'actual_cycles': actual_cycles,
            'cycle_limit_enforced': actual_cycles <= self.max_cycles,
            'dp_strategy': modified_strategy  
        }

    def _solve_with_discretized_dp(self, vdt_prices, start_soc, end_soc, soc_states, soc_step):
        """
        DP solution for the substitution strategy (without cycle limits).

        Args:
            vdt_prices (List[float]): The VDT prices.
            start_soc (float): The starting state of charge.
            end_soc (float): The ending state of charge.
            soc_states (np.ndarray): An array of SoC states.
            soc_step (float): The step size for the SoC states.

        Returns:
            Dict: A dictionary containing the DP solution.
        """
        n_intervals = len(vdt_prices)
        n_states = len(soc_states)

        positive_prices = [p for p in vdt_prices if p > 0]
        if len(positive_prices) > 0:
            buy_threshold = np.percentile(positive_prices, 25)
            sell_threshold = np.percentile(positive_prices, 75)
        else:
            buy_threshold = min(vdt_prices) if vdt_prices else 0
            sell_threshold = max(vdt_prices) if vdt_prices else 100

        dp_value = np.full((n_intervals + 1, n_states), -np.inf)
        dp_parent = np.full((n_intervals + 1, n_states), -1, dtype=int)
        dp_action = {}

        start_state_idx = self._find_closest_state_idx(start_soc, soc_states)
        dp_value[0, start_state_idx] = 0.0

        for t in range(n_intervals):
            for s in range(n_states):
                if dp_value[t, s] == -np.inf:
                    continue

                actions = self._generate_dp_actions(soc_states[s])

                for action in actions:
                    new_soc = self._calculate_new_soc_dp(
                        soc_states[s], action)
                    if not (0 <= new_soc <= self.bess_capacity_mwh + 0.001):
                        continue

                    new_soc = max(0.0, min(new_soc, self.bess_capacity_mwh))
                    new_state_idx = self._find_closest_state_idx(
                        new_soc, soc_states)

                    profit = self._calculate_action_profit_dp(
                        action, vdt_prices[t])

                    penalty = self._get_interpolation_penalty(
                        new_soc, soc_states[new_state_idx], soc_step)
                    new_value = dp_value[t, s] + profit - penalty

                    if new_value > dp_value[t + 1, new_state_idx]:
                        dp_value[t + 1, new_state_idx] = new_value
                        dp_parent[t + 1, new_state_idx] = s
                        dp_action[(t, s, new_state_idx)] = action

        if end_soc is None:
            end_state_idx = np.argmax(dp_value[n_intervals])
        else:
            end_state_idx = self._find_closest_state_idx(end_soc, soc_states)
            if dp_value[n_intervals, end_state_idx] == -np.inf:
                end_state_idx = np.argmax(dp_value[n_intervals])
        if dp_value[n_intervals, end_state_idx] == -np.inf:
            raise ValueError("Optimal strategy not found")

        dp_strategy = self._reconstruct_dp_strategy(
            dp_parent, dp_action, n_intervals, end_state_idx)

        return {
            'dp_strategy': dp_strategy,
            'final_soc': soc_states[end_state_idx],
            'dp_states_used': n_states,
        }

    def _generate_dp_actions(self, current_soc: float) -> List[Action]:
        """
        Generates actions based on the substitution strategy.

        Args:
            current_soc (float): The current state of charge.

        Returns:
            List[Action]: A list of possible actions.
        """
        actions = []

        remaining_capacity = max(0, self.bess_capacity_mwh - current_soc)
        max_charge_mwh = min(
            remaining_capacity / self.charge_efficiency, self.bess_capacity_mwh * self.INTERVAL_HOURS)
        max_discharge_mwh = min(current_soc, self.bess_capacity_mwh * self.INTERVAL_HOURS)

        is_large_battery = self.bess_capacity_mwh > 10 or self.bess_power_mw > 8

        max_discharge_mw = min(max_discharge_mwh / self.INTERVAL_HOURS, self.export_limit_mw)
        if max_discharge_mwh > 0:
            if not True:  # is_large_battery:
                discharge_options = np.linspace(0, max_discharge_mw, 5)
                for discharge_mw in discharge_options[1:]:
                    if discharge_mw <= self.export_limit_mw:
                        actions.append(Action(discharge_to_grid=discharge_mw))
            else:
                actions.append(Action(discharge_to_grid=max_discharge_mw))


        charge_power_limit = min(max_charge_mwh / self.INTERVAL_HOURS, self.import_limit_mw)
        if max_charge_mwh > 0 and charge_power_limit > 0:
            if not True:  # is_large_battery:
                charge_options = np.linspace(0, charge_power_limit, 5)
                for charge_mw in charge_options[1:]:
                    actions.append(Action(charge_from_grid=charge_mw))
            else:
                actions.append(Action(charge_from_grid=charge_power_limit))

        actions.append(Action())

        return list(set(actions))

    def _validate_strategy_physically(self, strategy: List[Action]) -> List[Action]:
        """
        Post-processing validation of the strategy - corrects physically impossible states.

        Args:
            strategy (List[Action]): The trading strategy.

        Returns:
            List[Action]: The corrected trading strategy.
        """
        corrected_strategy = []
        
        for i, action in enumerate(strategy):
            corrected_action = Action()
              
            
            if action.discharge_to_grid > self.export_limit_mw:
                corrected_action.discharge_to_grid = self.export_limit_mw
            else:
                corrected_action.discharge_to_grid = action.discharge_to_grid
            
            corrected_action.charge_from_grid = action.charge_from_grid
            
            corrected_strategy.append(corrected_action)
        
        return corrected_strategy

    def _calculate_action_profit_dp(self, action: Action, vdt_price: float) -> float:
        """
        Calculates the profit of an action.

        Args:
            action (Action): The action taken.
            vdt_price (float): The VDT price.

        Returns:
            float: The profit of the action.
        """
        discharge_revenue = action.discharge_to_grid * vdt_price * self.INTERVAL_HOURS
        grid_purchase_cost = action.charge_from_grid * vdt_price * self.INTERVAL_HOURS
        
        return discharge_revenue - grid_purchase_cost

    def _calculate_new_soc_dp(self, current_soc: float, action: Action) -> float:
        """
        Calculates the new state of charge.

        Args:
            current_soc (float): The current state of charge.
            action (Action): The action taken.

        Returns:
            float: The new state of charge.
        """
        charge_energy = action.charge_from_grid * self.INTERVAL_HOURS * self.charge_efficiency

        discharge_energy = (action.discharge_to_grid * self.INTERVAL_HOURS) / self.discharge_efficiency

        return current_soc + charge_energy - discharge_energy

    def _find_closest_state_idx(self, soc: float, soc_states: np.ndarray) -> int:
        """
        Finds the index of the closest state of charge.

        Args:
            soc (float): The state of charge.
            soc_states (np.ndarray): An array of SoC states.

        Returns:
            int: The index of the closest SoC state.
        """
        soc = max(0, min(soc, soc_states[-1]))
        return np.argmin(np.abs(soc_states - soc))

    def _get_interpolation_penalty(self, actual_soc: float, discrete_soc: float, soc_step: float) -> float:
        """
        Calculates the interpolation penalty.

        Args:
            actual_soc (float): The actual state of charge.
            discrete_soc (float): The discrete state of charge.
            soc_step (float): The step size for the SoC states.

        Returns:
            float: The interpolation penalty.
        """
        return abs(actual_soc - discrete_soc) / soc_step * 0.01

    def _reconstruct_dp_strategy(self, dp_parent: np.ndarray, dp_action: Dict,
                                 n_intervals: int, end_state_idx: int) -> List[Action]:
        """
        Reconstructs the optimal strategy from the DP tables.

        Args:
            dp_parent (np.ndarray): The DP parent table.
            dp_action (Dict): The DP action table.
            n_intervals (int): The number of intervals.
            end_state_idx (int): The index of the end state.

        Returns:
            List[Action]: The reconstructed strategy.
        """
        strategy = [Action()] * n_intervals
        current_state = end_state_idx
        for t in range(n_intervals - 1, -1, -1):
            parent_state = dp_parent[t + 1, current_state]
            if parent_state != -1:
                action_key = (t, parent_state, current_state)
                if action_key in dp_action:
                    strategy[t] = dp_action[action_key]
                current_state = parent_state
        return strategy

    def _convert_dp_strategy_to_report_format_safe(self, dp_strategy: List[Action], vdt_prices, start_soc) -> List[Dict]:
        """
        Converts the DP strategy to a safe report format.

        Args:
            dp_strategy (List[Action]): The DP strategy.
            vdt_prices (List[float]): The VDT prices.
            start_soc (float): The starting state of charge.

        Returns:
            List[Dict]: The strategy in report format.
        """
        strategy = []
        current_soc = start_soc
        
        for t, action in enumerate(dp_strategy):
            corrected_action = Action(
                charge_from_grid=action.charge_from_grid,
                discharge_to_grid=action.discharge_to_grid
            )
            
            max_possible_discharge_mwh = current_soc  
            max_possible_discharge_mw = max_possible_discharge_mwh / self.INTERVAL_HOURS
            
            if corrected_action.discharge_to_grid > max_possible_discharge_mw:
                corrected_action = Action(
                    charge_from_grid=corrected_action.charge_from_grid,
                    discharge_to_grid=max_possible_discharge_mw 
                )
            
            charge_energy = corrected_action.charge_from_grid * self.charge_efficiency * self.INTERVAL_HOURS
            discharge_energy = corrected_action.discharge_to_grid * self.INTERVAL_HOURS / self.discharge_efficiency
            
            current_soc = max(0.0, min(self.bess_capacity_mwh, current_soc + charge_energy - discharge_energy))

            strategy.append({
                'interval': t + 1,
                'vdt_price_eur_mwh': vdt_prices[t],
                'action': {'dp_action': corrected_action},
                'soc_mwh': current_soc,
            })
        
        return strategy
    
    def print_daily_trading_report(self, day_results, day_number=None):
        """
        Prints the daily trading report. This part of the output remains in Czech.

        Args:
            day_results (Dict): The results for the day.
            day_number (int): The day number.
        """
        report = day_results.get('trading_report', {})
        trades = report.get('active_trades', [])
        financials = day_results['financial_results']
        operational = day_results.get('operational_summary', {})

        day_str = f" DEN {day_number}" if day_number else ""
        print(f"\n{'='*95}")
        print(f"💰 OBCHODNÍ REPORT💰")
        print(f"{'='*95}")

        print("--- FINANČNÍ VÝSLEDKY ---")
        print(
            f"💰 Celkový příjem BESS: {(financials.get('client_vdt_spread_revenue_eur', 0) + financials.get('our_vdt_spread_commission_eur', 0)):.2f} EUR")
 
        actual_cycles = operational.get('actual_cycles', 0)
        print(f"\n--- INFORMACE O CYKLECH ---")
        print(f"🔄 Počet cyklů: {actual_cycles:.2f}")

        print("\n--- ROZPIS PŘÍJMŮ ---")

        print(
            f"  Klient (VDT): {financials.get('client_vdt_spread_revenue_eur', 0):.2f} EUR")
        print(
            f"  My (VDT):     {financials.get('our_vdt_spread_commission_eur', 0):.2f} EUR")

        if trades:
            print(f"\n--- AKTIVNÍ OPERACE ---")
            print("Perioda | VDT cena | Akce             | Zisk    |   SOC   ")
            print("--------|----------|------------------|---------|----------")

            for trade in trades:
                action_map = {
                    'block_charge': 'nabíjení',
                    'block_discharge': 'vybíjení',
                    'hold': 'Držení'
                }
                action_text = action_map.get(trade['action'], trade['action'])

                print(f"{trade['period']:6d}  | {trade['vdt_price']:8.2f} | "
                      f"{action_text:16s} | {trade['profit']:7.2f} | {trade['soc']:8.3f}")

        print(f"{'='*95}")


def generate_plot_for_excel(timestamps, vdt_price_data,
                            bess_discharge_data, soc_data, charge_grid_data,
                            export_limit, bess_capacity):
    """
    Generates a plot for the Excel report.

    Args:
        timestamps (List): A list of timestamps.
        vdt_price_data (List[float]): The VDT price data.
        bess_discharge_data (List[float]): The BESS discharge data.
        soc_data (List[float]): The SoC data.
        charge_grid_data (List[float]): The grid charge data.
        export_limit (float): The export limit.
        bess_capacity (float): The BESS capacity.

    Returns:
        io.BytesIO: The plot image data.
    """
    
    vdt_price_data = np.array(vdt_price_data, dtype=np.float64)
    bess_discharge_data = np.array(bess_discharge_data, dtype=np.float64)
    soc_data = np.array(soc_data, dtype=np.float64)
    charge_grid_data = np.array(charge_grid_data, dtype=np.float64)
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig = plt.figure(figsize=(18, 12))

    ax1 = fig.add_subplot(3, 1, 1)
    ax2 = fig.add_subplot(3, 1, 2, sharex=ax1)
    ax3 = fig.add_subplot(3, 1, 3, sharex=ax1)

    date_str = timestamps[0].strftime('%d.%m.%Y')
    x_axis = np.arange(len(timestamps))
    bar_width = 1 

    ax1.text(0.5, 1.05, f'Datum: {date_str}', transform=ax1.transAxes, 
             ha='center', va='bottom', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Výkon (MW)', fontsize=12)
    
    ax1.bar(x_axis, bess_discharge_data, width=bar_width, align='edge',
            label='Dodávka z BESS', color='royalblue', alpha=0.7)
    
    negative_charge = [-x for x in charge_grid_data]
    ax1.bar(x_axis, negative_charge, width=bar_width, align='edge',
            label='Nabíjení ze sítě', color='red', alpha=0.7)
    
    ax1.axhline(y=export_limit, color='darkgreen', linestyle='--',
                linewidth=2, label=f'Export limit ({export_limit} MW)')
    
    import_limit = export_limit
    ax1.axhline(y=-import_limit, color='darkred', linestyle='--',
                linewidth=2, label=f'Import limit ({import_limit} MW)')
    
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, linestyle='--', linewidth=0.5)
    ax1.axhline(y=0, color='black', linewidth=0.5)
    plt.setp(ax1.get_xticklabels(), visible=False)

    ax2.text(0.5, 1.05, f'Datum: {date_str}', transform=ax2.transAxes, 
             ha='center', va='bottom', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Cena VDT (EUR/MWh)', fontsize=12)
    
    ax2.plot(x_axis, vdt_price_data, label='Cena VDT', 
             color='darkred', linewidth=2.5)
    
    hourly_indices = list(range(0, len(timestamps), 4))
    if len(hourly_indices) > 0:
        hourly_prices = [vdt_price_data[i] for i in hourly_indices if i < len(vdt_price_data)]
        ax2.scatter(hourly_indices, hourly_prices,
                    color='darkred', s=20, zorder=5, marker='o')
    
    ax2.fill_between(x_axis, 0, vdt_price_data,
                     color='darkred', alpha=0.1)
    
    ax2.legend(loc='upper left', fontsize=11)
    ax2.grid(True, linestyle='--', linewidth=0.5)
    plt.setp(ax2.get_xticklabels(), visible=False)

    ax3.text(0.5, 1.05, f'Datum: {date_str}', transform=ax3.transAxes, 
             ha='center', va='bottom', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Stav nabití (MWh)', fontsize=12)
    ax3.set_xlabel('Čas', fontsize=12)
    
    ax3.plot(x_axis, soc_data, label='Stav nabití (SOC)',
             color='purple', linewidth=2.5, drawstyle='steps-post')
    
    ax3.axhline(y=bess_capacity, color='grey', linestyle=':',
                linewidth=2, label=f'Max kapacita ({bess_capacity} MWh)')
    
    ax3.legend(loc='upper left', fontsize=9)
    ax3.set_ylim(0, bess_capacity * 1.1)
    ax3.grid(True, linestyle='--', linewidth=0.5)

    tick_positions = np.arange(0, len(timestamps) + 1, 4) 
    tick_labels = [f"{h:02d}" for h in range(25)] 
    
    ax3.set_xticks(tick_positions)
    ax3.set_xticklabels(tick_labels[:len(tick_positions)]) 
    plt.setp(ax3.get_xticklabels(), rotation=0)

    plt.subplots_adjust(hspace=0.3, top=0.94, bottom=0.08)

    img_data = io.BytesIO()
    plt.savefig(img_data, format='png', bbox_inches='tight', dpi=100)
    plt.close(fig)
    img_data.seek(0)

    return img_data

def export_full_report_to_excel(project_results: dict, original_vdt_data: pd.DataFrame, optimizer_instance: BESS_Optimizer,
                                filename: str, with_plots: bool = False):
    """
    Exports the full report to an Excel file. This part of the output remains in Czech.

    Args:
        project_results (dict): The project results.
        original_vdt_data (pd.DataFrame): The original VDT data.
        optimizer_instance (BESS_Optimizer): The BESS_Optimizer instance.
        filename (str): The name of the Excel file.
        with_plots (bool): Whether to include plots in the Excel file.
    """

    project_name = project_results['config']['name']
    print(
        f"\nCreating Excel report '{filename}' for project '{project_name}'...")

    with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
        workbook = writer.book

        # Sheet 1: Annual report
        worksheet_year = workbook.add_worksheet('year')
        config_df = pd.DataFrame.from_dict(
            project_results['config'], orient='index', columns=['Value'])
        worksheet_year.write('A1', 'Konfigurace projektu',
                             workbook.add_format({'bold': True}))
        config_df.to_excel(writer, sheet_name='year', startrow=2, startcol=0)

        format_zvyrazneni3 = workbook.add_format({'bg_color': "#BEF994", 'bold': True, 'border': 1, 'border_color': '#839671'})
        format_neutral = workbook.add_format({'bg_color': "#FFE681"})
        format_vypocet = workbook.add_format({'font_color': '#ED7D31', 'bold': True})
        format_spatne_czk = workbook.add_format({'bg_color': '#FFC7CE', 'font_color': '#9C0006', 'num_format': '#,##0.00 "Kč"'})
        euro_format = workbook.add_format({'num_format': '€ #,##0.00'})
        format_neutral_euro = workbook.add_format({'bg_color': '#FFEB9C', 'num_format': '€ #,##0.00'})

        annual_report_df = create_detailed_report(project_results)

        neutral_keys = [
            'Celkový příjem klienta (EUR)', 
            'Náš celkový příjem (EUR)', 
            'Celkový příjem BESS (EUR)',
            'Naše provize z FVE (EUR)',
            'Cykly BESS'
        ]
        

        worksheet_year.write('E1', 'Roční souhrnný report',
                             workbook.add_format({'bold': True}))
        # annual_report_df.T.to_excel(
        #     writer, sheet_name='year', startrow=2, startcol=4, header=False)

        annual_report_dict = create_detailed_report(project_results).to_dict('records')[0]

        row = 2
        for key, value in annual_report_dict.items():
            key_cell_format = None
            value_cell_format = None

            if key == 'Projekt':
                key_cell_format = format_zvyrazneni3
                value_cell_format = format_zvyrazneni3
            elif key in neutral_keys:
                key_cell_format = format_neutral
                value_cell_format = format_neutral_euro if '(EUR)' in key else format_neutral
            elif '(EUR)' in key:
                value_cell_format = euro_format

            worksheet_year.write(row, 4, key, key_cell_format)
            worksheet_year.write(row, 5, value, value_cell_format)
            row += 1

        worksheet_year.write('A4', 'name', format_zvyrazneni3)
        worksheet_year.write('B4', project_results['config']['name'], format_zvyrazneni3)
        
        worksheet_year.write('G1', 'Kurz[EUR/CZK]', format_vypocet)
        worksheet_year.write('H1', 25, format_vypocet)

        worksheet_year.write_formula('G5', '=F5*H1', format_spatne_czk)
        worksheet_year.write_formula('G6', '=F6*H1', format_spatne_czk)
        worksheet_year.write_formula('G7', '=F7*H1', format_spatne_czk)

        worksheet_year.set_column('A:B', 25)
        worksheet_year.set_column('E:F', 30)
        worksheet_year.set_column('G:H', 20)

        if with_plots:
            worksheet_daily = workbook.add_worksheet('daily')

            current_row = 1
            export_limit = project_results['config']['export_limit_mw']
            import_limit = project_results['config']['import_limit_mw']
            bess_capacity = project_results['config']['bess_capacity_mwh']

            for i, day_data in enumerate(project_results['daily_results']):

                daily_df = create_daily_report_df(
                    day_data, i, project_results['config'], optimizer_instance)
                daily_df.to_excel(writer, sheet_name='daily',
                                  startrow=current_row, startcol=0)

                start_idx = i * 96
                end_idx = (i + 1) * 96
                day_timestamps = original_vdt_data.index[start_idx:end_idx]

                bess_discharge_data = [d['action']['dp_action'].discharge_to_grid
                                       for d in day_data['operational_summary']['vdt_strategy']]
                soc_data = [d['soc_mwh']
                            for d in day_data['operational_summary']['vdt_strategy']]
                charge_grid_data = [
                    d['action']['dp_action'].charge_from_grid for d in day_data['operational_summary']['vdt_strategy']]

                vdt_price_data = original_vdt_data[
                    'Vážený průměr cen (EUR/MWh)'].values[start_idx:end_idx]

                image_data = generate_plot_for_excel(
                    day_timestamps, vdt_price_data,
                    bess_discharge_data, soc_data, charge_grid_data, export_limit, bess_capacity)

                worksheet_daily.insert_image(f'D{current_row + 1}', f'day_{i+1}_plot.png',
                                             {'image_data': image_data})

                current_row += 55

                worksheet_daily.set_column('A:B', 35)
                worksheet_daily.set_column('D:M', 12)

    print(f"✅ Report '{filename}' was successfully created!")

def create_detailed_report(project_results: dict) -> pd.DataFrame:
    """
    Creates a detailed report. This part of the output remains in Czech.

    Args:
        project_results (dict): The project results.

    Returns:
        pd.DataFrame: A DataFrame containing the detailed report.
    """
    config = project_results['config']
    daily_results = project_results['daily_results']

    client_bess_income = sum(
        d['financial_results']['client_vdt_spread_revenue_eur'] for d in daily_results)
    our_bess_commission = sum(
        d['financial_results']['our_vdt_spread_commission_eur'] for d in daily_results)
    total_bess_income = client_bess_income + our_bess_commission

    total_efficiency = config['efficiency']
    charge_efficiency = np.sqrt(total_efficiency)
    discharge_efficiency = np.sqrt(total_efficiency)
    bess_capacity = config['bess_capacity_mwh']

    total_charged = project_results['total_charged_to_bess_mwh']
    total_discharged = project_results['total_discharged_from_bess_mwh']

    charging_losses = total_charged * (1 - charge_efficiency)
    discharging_losses = (
        total_discharged / discharge_efficiency) - total_discharged
    efficiency_losses_mwh = charging_losses + discharging_losses
    bess_cycles = ((total_charged + total_discharged) / 2) / \
        bess_capacity if bess_capacity > 0 else 0

    project_summary = {
        'Projekt': config['name'],
        'Analyzované dny': project_results['days_analyzed'],
        'Celkový příjem klienta (EUR)': round(project_results['annual_client_income_eur'], 2),
        'Náš celkový příjem (EUR)': round(project_results['annual_our_income_eur'], 2),
        'Celkový příjem BESS (EUR)': round(total_bess_income, 2),
        'CELKEM Fyzicky prodáno (MWh)': round(project_results['total_delivered_to_grid_mwh'], 2),
        'Nakoupeno ze sítě (MWh)': round(project_results['total_purchased_from_grid_mwh'], 2),
        'Ztráty účinnosti BESS (MWh)': round(efficiency_losses_mwh, 2),
        'Cykly BESS': round(bess_cycles, 2),
    }

    return pd.DataFrame([project_summary])

def create_daily_report_df(day_data: dict, day_index: int, project_config: dict,
                           optimizer_instance: BESS_Optimizer) -> pd.DataFrame:
    """
    Creates a daily report DataFrame. This part of the output remains in Czech.

    Args:
        day_data (dict): The data for the day.
        day_index (int): The index of the day.
        project_config (dict): The project configuration.
        optimizer_instance (BESS_Optimizer): The BESS_Optimizer instance.

    Returns:
        pd.DataFrame: A DataFrame containing the daily report.
    """

    delivered, purchased, charged, discharged = 0, 0, 0, 0
    interval_hours = optimizer_instance.INTERVAL_HOURS
    soc0mwh = day_data['operational_summary'].get('start_soc_mwh', 0)
    soc24mwh = day_data['operational_summary']['vdt_strategy'][-1]['soc_mwh']
    for interval in day_data['operational_summary']['vdt_strategy']:
        action = interval['action']['dp_action']
        delivered += action.discharge_to_grid * interval_hours
        purchased += action.charge_from_grid * interval_hours
        charged += action.charge_from_grid * interval_hours
        discharged += action.discharge_to_grid * interval_hours

    charge_efficiency = optimizer_instance.charge_efficiency
    discharge_efficiency = optimizer_instance.discharge_efficiency
    bess_capacity = project_config['bess_capacity_mwh']

    charging_losses = charged * (1 - charge_efficiency)
    discharging_losses = (discharged / discharge_efficiency) - discharged
    total_losses = charging_losses + discharging_losses

    if bess_capacity > 0:
        cycles = ((charged + discharged) / 2) / bess_capacity
    else:
        cycles = 0

    financials = day_data['financial_results']
    report_data = {
        f"DEN {day_index + 1}": {
            '--- Finance (EUR) ---': '',
            'Příjem klienta (VDT)': round((financials['client_vdt_spread_revenue_eur'] + financials['our_vdt_spread_commission_eur']), 2),
            # 'Naše provize (VDT)': round(financials['our_vdt_spread_commission_eur'], 2),
            'CELKEM Klient': round(financials['client_total_income_eur'], 2),
            'CELKEM My': round(financials['our_total_commission_eur'], 2),
            '--- Energie (MWh) ---': '',
            'Dodávka do sítě': round(delivered, 2),
            'Nákup ze sítě': round(purchased, 2),
            '--- BESS (MWh) ---': '',
            'Vybíjení do sítě': round(discharged, 2),
            'Ztráty účinnosti': round(total_losses, 2),
            'Cykly': round(cycles, 2),
            'SOC na začátku': round(soc0mwh, 2),
            'SOC na konci': round(soc24mwh, 2)
        }
    }

    df = pd.DataFrame(report_data)
    return df

def analyze_projects(vdt_prices_data, projects_config: Dict,
                     analysis_period_days=365,  show_daily_reports=False):
    """
    Analyzes projects.

    Args:
        vdt_prices_data (pd.DataFrame): The VDT prices data.
        projects_config (Dict): The projects configuration.
        analysis_period_days (int): The analysis period in days.
        show_daily_reports (bool): Whether to show daily reports.

    Returns:
        Dict: A dictionary containing the analysis results.
    """

    results = {}
    projects_to_analyze = {}

    projects_to_analyze = projects_config
    if projects_to_analyze is None:
        print("No project configuration provided.")

    for project_key, config in projects_to_analyze.items():

        optimizer = BESS_Optimizer(config)

        daily_results = []

        total_client_income_eur, total_our_income_eur = 0, 0
        total_delivered_to_grid_mwh, total_purchased_from_grid_mwh = 0, 0
        total_charged_to_bess_mwh, total_discharged_from_bess_mwh = 0, 0
        successful_days = 0
        current_soc = optimizer.soc0_mwh
        max_days = min(analysis_period_days, len(vdt_prices_data) // 96)

        for day in range(max_days):
            start_idx = day * 96
            end_idx = (day + 1) * 96

            if end_idx > len(vdt_prices_data):
                break

            try:
                vdt_day = vdt_prices_data.iloc[start_idx:
                                               end_idx]['Vážený průměr cen (EUR/MWh)'].values

                day_result = optimizer.optimize_day_strategy_dp(vdt_day, start_soc=current_soc)
  
                daily_results.append(day_result)

                if day_result['operational_summary']['vdt_strategy']:
                    current_soc = day_result['operational_summary']['vdt_strategy'][-1]['soc_mwh']

                total_client_income_eur += day_result['financial_results']['client_total_income_eur']
                total_our_income_eur += day_result['financial_results']['our_total_commission_eur']
                successful_days += 1

                for interval_data in day_result['operational_summary']['vdt_strategy']:
                    action = interval_data['action']['dp_action']
                    interval_hours = optimizer.INTERVAL_HOURS
                    total_delivered_to_grid_mwh += action.discharge_to_grid * interval_hours
                    total_purchased_from_grid_mwh += action.charge_from_grid * interval_hours
                    total_charged_to_bess_mwh += action.charge_from_grid * interval_hours
                    total_discharged_from_bess_mwh += action.discharge_to_grid * interval_hours

                if show_daily_reports and (day < 3 or day % 30 == 0):
                    optimizer.print_daily_trading_report(
                        day_result, day_number=day + 1)

            except Exception as e:
                print(f"Error on day {day}: {e}")
                continue

        if successful_days > 0:
            results[project_key] = {
                'config': config,
                'daily_results': daily_results,
                'days_analyzed': successful_days,
                'annual_client_income_eur': total_client_income_eur,
                'annual_our_income_eur': total_our_income_eur,
                'annual_client_income_czk': total_client_income_eur * optimizer.CZK_EUR_RATE,
                'annual_our_income_czk': total_our_income_eur * optimizer.CZK_EUR_RATE,
                'avg_daily_client_eur': total_client_income_eur / successful_days,
                'avg_daily_our_eur': total_our_income_eur / successful_days,
                'total_delivered_to_grid_mwh': total_delivered_to_grid_mwh,
                'total_purchased_from_grid_mwh': total_purchased_from_grid_mwh,
                'total_charged_to_bess_mwh': total_charged_to_bess_mwh,
                'total_discharged_from_bess_mwh': total_discharged_from_bess_mwh,
            }

            print(f"\nSuccessfully analyzed {successful_days} days")
            print(f"Annual BESS income: {(total_client_income_eur + total_our_income_eur):,.0f} EUR")


    return results

if __name__ == '__main__':
    # vdt2025 = pd.read_excel("data/OTE_2025.xlsx", sheet_name="VDT (EUR)", skiprows=5, usecols="A:L")

    # # Rename columns to English for consistency
    # vdt2025.rename(columns={
    #     'Časový interval': 'time_interval',
    #     'Den': 'day',
    #     'Vážený průměr cen (EUR/MWh)': 'weighted_avg_price_eur_mwh'
    # }, inplace=True)

    # # Create datetime column from the day and time interval
    # start_time = vdt2025['time_interval'].str.split('-').str[0]
    # vdt2025['datetime'] = pd.to_datetime(vdt2025['day'].astype(str) + ' ' + start_time)

    # # Select and reorder columns, keeping only what's necessary
    # vdt2025 = vdt2025[['datetime', 'weighted_avg_price_eur_mwh']]

    # vdt2025.sort_values('datetime', inplace=True)
    # vdt2025.drop_duplicates(subset=['datetime'], keep='first', inplace=True)
    # vdt2025.set_index('datetime', inplace=True)

    # vdt2025.fillna(method='ffill', inplace=True)

    # start_date = '2025-01-01 00:00:00'
    # end_date = '2025-07-31 23:59:59'
    # VDT = vdt2025.copy()
    # VDT = VDT.sort_index()

    # VDT = VDT.loc[start_date:end_date]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        VDT = (pd.read_excel('data.xlsx', sheet_name='VDT (R2W)')).set_index('datetime', inplace=False)
        VDT = VDT[VDT.index <= '2025-06-30']
        VDT['Vážený průměr cen (EUR/MWh)'].fillna(method='ffill', inplace=True)

    project_name = input("Název projektu: ")
    
    user_input = input("Instalovaný výkon BESS [MW]: ")
    try:
        bess_power_mw = float(user_input)
    except:
        print("Neprávný formát vstupu!")

    user_input = input("Instalovaná kapacita BESS [MWh]: ")
    try:
        bess_capacity_mwh = float(user_input)
    except:
        print("Neprávný formát vstupu!")
    
    user_input = input("Rezervovaný výkon OM [MW]: ")
    try:
        export_limit_mw = float(user_input)
    except:
        print("Neprávný formát vstupu!")
    
    user_input = input("Rezervovaná kapacita v OM [MW]: ")
    try:
        import_limit_mw = float(user_input)
    except:
        print("Neprávný formát vstupu!")
    
    user_input = input("Účinnost BESS (předvolená hodnota je 0.85): ")
    efficiency = 0.85 or float(user_input)
    
    user_input = input("Maximální denní počet cyklů (předvolená hodnota je 2): ")
    max_cycles = 2 or float(user_input)

    user_input = input("Chci denní reporty [A/N]: ")
    daily_reports = user_input.lower() == 'a'

    user_input = input("Chci vidět grafy [A/N]: ")
    with_plots = user_input.lower() == 'a'
    
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

    results = analyze_projects(VDT, projects_config=PROJECTS_CONFIG,analysis_period_days=365, show_daily_reports=daily_reports)

    for i in results.keys():
        optimizer_instance = BESS_Optimizer(PROJECTS_CONFIG[i])

        project_key_to_export = i
        project_data_to_export = results[project_key_to_export]
                
        excel_filename = f"full_report_{project_key_to_export}.xlsx"
                
        export_full_report_to_excel(project_data_to_export,VDT, optimizer_instance, excel_filename, with_plots=with_plots)

    input("Ukončete stisknutím klávesy Enter.")
