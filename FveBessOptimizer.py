import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import timedelta
import logging
from dataclasses import dataclass
import io


@dataclass
class Action:
    """
    A class to store actions in a single interval.

    Args:
        charge_from_fve (float): The amount of energy to charge from FVE.
        charge_from_grid (float): The amount of energy to charge from the grid.
        discharge_to_grid (float): The amount of energy to discharge to the grid.
        sell_fve_direct (float): The amount of energy to sell from FVE directly.
    """
    charge_from_fve: float = 0.0
    charge_from_grid: float = 0.0
    discharge_to_grid: float = 0.0
    sell_fve_direct: float = 0.0

    def __post_init__(self):
        epsilon = 1e-9
        assert all(x >= -epsilon for x in [self.charge_from_fve, self.charge_from_grid,
                                           self.discharge_to_grid, self.sell_fve_direct])

    def __hash__(self):
        return hash((self.charge_from_fve, self.charge_from_grid, self.discharge_to_grid, self.sell_fve_direct))

    def __eq__(self, other):
        return (isinstance(other, Action) and
                self.charge_from_fve == other.charge_from_fve and
                self.charge_from_grid == other.charge_from_grid and
                self.discharge_to_grid == other.discharge_to_grid and
                self.sell_fve_direct == other.sell_fve_direct)

class FVE_BESS_Optimizer:
    """
    Optimizes FVE and BESS (Battery Energy Storage System) trading strategie
    """
    def __init__(self, project_config: Dict):
        """
        Initializes the FVE_BESS_Optimizer.

        Args:
            project_config (Dict): A dictionary containing the project configuration.
        """
        self.config = project_config
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s')
        

        self.operation_mode = project_config.get('operation_mode', 'fve_bess_integrated')  

        self.BASE_COMMISSION_EUR_MWH = 13
        self.CLIENT_SHARE = 0.70
        self.OUR_SHARE = 0.30
        self.INTERVAL_HOURS = 0.25
        self.CZK_EUR_RATE = 25
        self.DP_SOC_STEP = 0.05
        self.soc0_mwh = 0.0

        total_efficiency = self.config.get('efficiency', 0.85)
        self.charge_efficiency = np.sqrt(total_efficiency)
        self.discharge_efficiency = np.sqrt(total_efficiency)

    def _enforce_cycle_limit(self, dp_strategy: List[Action], bess_capacity: float,
                             max_cycles: float, vdt_prices: List[float]) -> Tuple[List[Action], float]:
        """
        Adjusts the strategy to respect the cycle limit.
        Returns the adjusted strategy and the actual number of cycles.
        """
        if max_cycles <= 0 or bess_capacity <= 0:
            return dp_strategy, 0

        cumulative_charge = 0
        cumulative_discharge = 0
        modified_strategy = []

        max_energy = max_cycles * bess_capacity

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
                'charge_energy': (action.charge_from_fve + action.charge_from_grid) * self.INTERVAL_HOURS,
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
                    # We can't add any more discharges
                    discharge_idx = len(discharge_intervals)

            # Add the best charge if we can
            if charge_idx < len(charge_intervals):
                interval = charge_intervals[charge_idx]
                if current_charge + interval['charge_energy'] <= max_energy:
                    selected_intervals.add(interval['index'])
                    current_charge += interval['charge_energy']
                    charge_idx += 1
                else:
                    # We can't add any more charges
                    charge_idx = len(charge_intervals)

            if (discharge_idx >= len(discharge_intervals) and charge_idx >= len(charge_intervals)):
                break

        for i, action in enumerate(dp_strategy):
            if i in selected_intervals:
                modified_strategy.append(action)
                cumulative_charge += (action.charge_from_fve +
                                      action.charge_from_grid) * self.INTERVAL_HOURS
                cumulative_discharge += action.discharge_to_grid * self.INTERVAL_HOURS
            else:
                # Replace with an action without charging/discharging from/to the grid, but FVE->BESS remains
                # if it was a substitution (charge_from_fve > 0 and VDT < DT)
                modified_action = Action(
                    charge_from_fve=action.charge_from_fve,  # FVE substitution remains
                    charge_from_grid=0.0,  # No speculative charging
                    discharge_to_grid=0.0,  # No speculative discharging
                    sell_fve_direct=action.sell_fve_direct +
                    action.charge_from_grid  # Direct sale
                )
                modified_strategy.append(modified_action)
                cumulative_charge += modified_action.charge_from_fve * self.INTERVAL_HOURS

        # Calculate the actual number of cycles (average of charging and discharging)
        actual_cycles = ((cumulative_charge + cumulative_discharge) /
                         2) / bess_capacity if bess_capacity > 0 else 0

        return modified_strategy, actual_cycles
    
    def _calculate_dt_revenue_breakdown(self, fve_generation_scaled: np.array, dt_prices_eur: np.array) -> Dict:
        """
        Calculates the breakdown of revenue from DT based on a fixed commission.

        Args:
            fve_generation_scaled (np.array): The scaled FVE generation.
            dt_prices_eur (np.array): The DT prices in EUR.

        Returns:
            Dict: A dictionary with the client's DT revenue and our DT commission.
        """
        total_client_dt_revenue = 0
        total_our_dt_commission = 0

        for i in range(len(dt_prices_eur)):
            energy_mwh = fve_generation_scaled[i] * self.INTERVAL_HOURS
            dt_price = dt_prices_eur[i]

            if dt_price < 0:
                client_revenue = 0
                our_commission = 0
            else:
                total_dt_revenue = energy_mwh * dt_price

                if dt_price >= self.BASE_COMMISSION_EUR_MWH:
                    our_commission = energy_mwh * self.BASE_COMMISSION_EUR_MWH
                    client_revenue = total_dt_revenue - our_commission
                else:
                    our_commission = 0
                    client_revenue = total_dt_revenue

            total_client_dt_revenue += client_revenue
            total_our_dt_commission += our_commission

        return {
            'client_dt_revenue': total_client_dt_revenue,
            'our_dt_commission': total_our_dt_commission
        }

    def optimize_day_strategy_dp(self, dt_prices_eur: np.array, vdt_prices_eur: np.array,
                                 fve_generation: np.array, start_soc: float = 0, end_soc: float = None) -> Dict:
        """
        Optimization with correct profit sharing and respect for the cycle limit.

        Args:
            dt_prices_eur (np.array): The DT prices in EUR.
            vdt_prices_eur (np.array): The VDT prices in EUR.
            fve_generation (np.array): The FVE generation.
            start_soc (float): The starting state of charge.
            end_soc (float): The ending state of charge.

        Returns:
            Dict: A dictionary with the optimization results.
        """

        if len(dt_prices_eur) != 96:
            raise ValueError("All arrays must contain 96 elements")

        bess_power = self.config['bess_power_mw']
        bess_capacity = self.config['bess_capacity_mwh']
        efficiency = self.config.get('efficiency', 0.95)
        export_limit = self.config['export_limit_mw']
        import_limit = self.config['import_limit_mw']
        max_cycles = self.config.get('max_cycles', 1.0)
        fve_generation_scaled = fve_generation * \
            self.config['fve_scale_factor']

        try:
            # STEP 1: DP optimization
            vdt_optimization = self._optimize_vdt_trading(
                dt_prices_eur.tolist(), vdt_prices_eur.tolist(), fve_generation_scaled.tolist(),
                bess_power, bess_capacity, efficiency, export_limit, import_limit,
                start_soc, end_soc, max_cycles
            )   
            current_trading_report = vdt_optimization.get('trading_report', {'active_trades': []})
            # STEP 2: PHYSICAL VALIDATION - new check
            if 'dp_strategy' in vdt_optimization:
                validated_strategy = self._validate_strategy_physically(
                    vdt_optimization['dp_strategy'], 
                    fve_generation_scaled.tolist(),
                    export_limit
                )
                
                # Recalculate the report with the validated strategy
                if validated_strategy != vdt_optimization['dp_strategy']:
                    strategy_report, trading_report = self._convert_dp_strategy_to_report_format_safe(
                        validated_strategy, dt_prices_eur.tolist(), vdt_prices_eur.tolist(), 
                        fve_generation_scaled.tolist(), bess_capacity, efficiency, start_soc
                    )
                    vdt_optimization['strategy'] = strategy_report
                    vdt_optimization['trading_report'] = trading_report
                    current_trading_report = trading_report

        finally:
            pass
        # STEP 2: Profit sharing
        dt_breakdown = self._calculate_dt_revenue_breakdown(
            fve_generation_scaled, dt_prices_eur)

        vdt_spread = vdt_optimization['total_additional_profit_eur']
        vdt_breakdown = {
            'client_vdt_spread_revenue': vdt_spread * self.CLIENT_SHARE,
            'our_vdt_spread_commission': vdt_spread * self.OUR_SHARE
        }

        total_client_income = dt_breakdown['client_dt_revenue'] + \
            vdt_breakdown['client_vdt_spread_revenue']
        total_our_income = dt_breakdown['our_dt_commission'] + \
            vdt_breakdown['our_vdt_spread_commission']

        total_results = {
            'financial_results': {
                'client_dt_revenue_eur': round(dt_breakdown['client_dt_revenue'], 2),
                'our_dt_commission_eur': round(dt_breakdown['our_dt_commission'], 2),
                'client_vdt_spread_revenue_eur': round(vdt_breakdown['client_vdt_spread_revenue'], 2),
                'our_vdt_spread_commission_eur': round(vdt_breakdown['our_vdt_spread_commission'], 2),
                'client_total_income_eur': round(total_client_income, 2),
                'our_total_commission_eur': round(total_our_income, 2),
                'client_total_income_czk': round(total_client_income * self.CZK_EUR_RATE, 2),
                'our_total_commission_czk': round(total_our_income * self.CZK_EUR_RATE, 2),
            },
            'operational_summary': {
                'total_fve_generation_mwh': round(np.sum(fve_generation_scaled) * self.INTERVAL_HOURS, 3),
                'vdt_strategy': vdt_optimization['strategy'],
                'actual_cycles': vdt_optimization.get('actual_cycles', 0),
                'cycle_limit_enforced': vdt_optimization.get('cycle_limit_enforced', False),
                'start_soc_mwh': start_soc
            },
            'trading_report': current_trading_report,
            'algorithm_used': "dp_substitution_v2_cycles",
            'dp_states_used': vdt_optimization.get('dp_states_used', 'N/A')
        }

        return total_results

    def _optimize_vdt_trading(self, dt_prices: List[float], vdt_prices: List[float],
                          fve_generation_scaled: List[float], bess_power: float,
                          bess_capacity: float, efficiency: float,
                          export_limit: float, import_limit: float,
                          start_soc: float = 0.0, end_soc: float = None,
                          max_cycles: float = 1.0) -> Dict:
        """
        Optimizes intraday (VDT) trading by running a DP algorithm and applying a cycle limit.

        This function serves as a wrapper for a multi-step process:
        1. It selects and runs a dynamic programming (DP) algorithm based on the
        `operation_mode` to find an initial optimal strategy.
        2. It applies a post-processing step to enforce the specified `max_cycles` limit.
        3. It generates final reports based on the cycle-corrected strategy.

        Returns:
            A dictionary with the optimization results, including profit,
            strategy reports, final SoC, and cycle usage information.
        """
        soc_step = self.DP_SOC_STEP
        soc_states = np.arange(0, bess_capacity + soc_step, soc_step)

        # 1. Select and run the DP algorithm based on the operation mode.
        if self.operation_mode == 'fve_bess_independent':
            result = self._solve_with_discretized_dp_fve_bess(
                dt_prices, vdt_prices, fve_generation_scaled, bess_power,
                bess_capacity, efficiency, export_limit, import_limit,
                start_soc, end_soc, soc_states, soc_step
            )
        else:
            result = self._solve_with_discretized_dp(
                dt_prices, vdt_prices, fve_generation_scaled, bess_power,
                bess_capacity, efficiency, export_limit, import_limit,
                start_soc, end_soc, soc_states, soc_step
            )

        # 2. Get the initial strategy from the DP result.
        original_strategy = result['dp_strategy']

        # 3. Apply the cycle limit constraint.
        modified_strategy, actual_cycles = self._enforce_cycle_limit(
            original_strategy, bess_capacity, max_cycles, vdt_prices
        )

        # 4. Generate reports using the cycle-corrected strategy.
        strategy_report, trading_report = self._convert_dp_strategy_to_report_format_safe(
            modified_strategy, dt_prices, vdt_prices, fve_generation_scaled,
            bess_capacity, efficiency, start_soc
        )

        # 5. Extract total profit from the report (avoids recalculation).
        active_trades = trading_report.get('active_trades', [])
        total_profit = active_trades[-1]['cumulative'] if active_trades else 0.0

        return {
            'total_additional_profit_eur': float(total_profit),
            'strategy': strategy_report,
            'trading_report': trading_report,
            'final_soc': result['final_soc'],
            'dp_states_used': result['dp_states_used'],
            'actual_cycles': actual_cycles,
            'cycle_limit_enforced': actual_cycles <= max_cycles,
            'dp_strategy': modified_strategy
        }

    def _solve_with_discretized_dp(self, dt_prices, vdt_prices, fve_generation_scaled,
                                   bess_power, bess_capacity, efficiency, export_limit,
                                   import_limit, start_soc, end_soc, soc_states, soc_step):
        """DP solution for the substitution strategy (without cycle limits)."""
        n_intervals = len(dt_prices)
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
                # FVE + BESS
                actions = self._generate_dp_actions(
                    soc_states[s], fve_generation_scaled[t], bess_power, bess_capacity,
                    export_limit, import_limit, dt_prices[t], vdt_prices[t], buy_threshold, sell_threshold, t
                )

                for action in actions:
                    new_soc = self._calculate_new_soc_dp(
                        soc_states[s], action, efficiency)
                    if not (0 <= new_soc <= bess_capacity + 0.001):
                        continue

                    new_soc = max(0.0, min(new_soc, bess_capacity))
                    new_state_idx = self._find_closest_state_idx(
                        new_soc, soc_states)

                    profit = self._calculate_action_profit_dp(
                        action, vdt_prices[t], dt_prices[t])

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
    
    def _generate_dp_actions_fve_bess(self, current_soc: float, fve_gen: float, bess_power: float,
                                        bess_capacity: float, export_limit: float,
                                        import_limit: float, vdt_price: float, buy_threshold: float, sell_threshold: float) -> List[Action]:
        """Pure BESS operation - no substitution, with dynamic thresholds."""
        actions = []

        remaining_capacity = max(0, bess_capacity - current_soc)
        max_charge_mwh = min(remaining_capacity / self.charge_efficiency, bess_power * self.INTERVAL_HOURS)
        max_discharge_mwh = min(current_soc, bess_power * self.INTERVAL_HOURS)

        # FVE is ALWAYS sold directly - no substitution
        actions.append(Action(sell_fve_direct=fve_gen))


        # Charging from the grid at low VDT prices
        if max_charge_mwh > 0 and vdt_price <= buy_threshold:
            charge_power_limit = min(max_charge_mwh / self.INTERVAL_HOURS, import_limit)
            if charge_power_limit > 0:
                charge_options = np.linspace(0, charge_power_limit, 4)
                for charge_mw in charge_options:
                    # IMPORTANT: Check against export limit
                    if fve_gen + 0 <= export_limit:  # FVE does not exceed export limit
                        actions.append(Action(
                            charge_from_grid=charge_mw,
                            sell_fve_direct=fve_gen
                        ))

        if max_discharge_mwh > 0 and vdt_price >= sell_threshold:
            discharge_options = np.linspace(0, max_discharge_mwh / self.INTERVAL_HOURS, 4)
            for discharge_mw in discharge_options:
                if fve_gen + discharge_mw <= export_limit:
                    actions.append(Action(
                        discharge_to_grid=discharge_mw,
                        sell_fve_direct=fve_gen
                    ))

        return list(set(actions))

    def _solve_with_discretized_dp_fve_bess(self, dt_prices, vdt_prices, fve_generation_scaled,
                                            bess_power, bess_capacity, efficiency, export_limit,
                                            import_limit, start_soc, end_soc, soc_states, soc_step):
        """DP with pure BESS and dynamic thresholds from full-day prices."""
        
        # Calculate dynamic thresholds from full-day VDT prices
        positive_prices = [p for p in vdt_prices if p > 0]
        if len(positive_prices) > 0:
            buy_threshold = np.percentile(positive_prices, 25)
            sell_threshold = np.percentile(positive_prices, 75)
        else:
            buy_threshold = min(vdt_prices) if vdt_prices else 0
            sell_threshold = max(vdt_prices) if vdt_prices else 100

        n_intervals = len(dt_prices)
        n_states = len(soc_states)

        dp_value = np.full((n_intervals + 1, n_states), -np.inf)
        dp_parent = np.full((n_intervals + 1, n_states), -1, dtype=int)
        dp_action = {}

        start_state_idx = self._find_closest_state_idx(start_soc, soc_states)
        dp_value[0, start_state_idx] = 0.0

        for t in range(n_intervals):
            for s in range(n_states):
                if dp_value[t, s] == -np.inf:
                    continue

                # Use the special action generator with the passed thresholds
                actions = self._generate_dp_actions_fve_bess(
                    soc_states[s], fve_generation_scaled[t], bess_power, bess_capacity,
                    export_limit, import_limit, vdt_prices[t], 
                    buy_threshold, sell_threshold
                )

                for action in actions:
                    new_soc = self._calculate_new_soc_dp(soc_states[s], action, efficiency)
                    if not (0 <= new_soc <= bess_capacity + 0.001):
                        continue

                    new_soc = max(0.0, min(new_soc, bess_capacity))
                    new_state_idx = self._find_closest_state_idx(new_soc, soc_states)

                    profit = self._calculate_fve_bess_profit_dp(action, vdt_prices[t])
                    penalty = self._get_interpolation_penalty(new_soc, soc_states[new_state_idx], soc_step)

                    new_value = dp_value[t, s] + profit - penalty

                    if new_value > dp_value[t + 1, new_state_idx]:
                        dp_value[t + 1, new_state_idx] = new_value
                        dp_parent[t + 1, new_state_idx] = s
                        dp_action[(t, s, new_state_idx)] = action

        end_state_idx = self._find_closest_state_idx(end_soc, soc_states)
        if dp_value[n_intervals, end_state_idx] == -np.inf:
            end_state_idx = np.argmax(dp_value[n_intervals])
        if dp_value[n_intervals, end_state_idx] == -np.inf:
            raise ValueError("Optimal strategy not found")

        dp_strategy = self._reconstruct_dp_strategy(dp_parent, dp_action, n_intervals, end_state_idx)

        return {
            'dp_strategy': dp_strategy,
            'final_soc': soc_states[end_state_idx],
            'dp_states_used': n_states,
            'buy_threshold': buy_threshold,
            'sell_threshold': sell_threshold
        }

    def _calculate_fve_bess_profit_dp(self, action: Action, vdt_price: float) -> float:
        """Calculates ONLY BESS profit - ignores FVE substitutions."""
        # Only grid operations of BESS
        discharge_revenue = action.discharge_to_grid * vdt_price * self.INTERVAL_HOURS
        grid_charge_cost = action.charge_from_grid * vdt_price * self.INTERVAL_HOURS
        return discharge_revenue - grid_charge_cost
    
    def _generate_dp_actions(self, current_soc: float, fve_gen: float, bess_power: float,
                              bess_capacity: float, export_limit: float,
                              import_limit: float, dt_price: float, vdt_price: float, buy_threshold: float, sell_threshold: float, current_period: int) -> List[Action]:
        actions = []
        remaining_capacity = max(0, bess_capacity - current_soc)
        max_charge_mwh = min(remaining_capacity / self.charge_efficiency, bess_power * self.INTERVAL_HOURS)
        max_discharge_mwh = min(current_soc, bess_power * self.INTERVAL_HOURS)

        if dt_price < 0 and fve_gen > 0:
            fve_to_charge_mw = min(fve_gen, max_charge_mwh / self.INTERVAL_HOURS)
            
            remaining_fve_mw = fve_gen - fve_to_charge_mw
            
            fve_to_sell_mw = min(remaining_fve_mw, export_limit)
            
            actions.append(Action(
                charge_from_fve=fve_to_charge_mw,
                sell_fve_direct=fve_to_sell_mw
            ))
            return actions
        
        if fve_gen > 0:
            if fve_gen <= export_limit:
                actions.append(Action(sell_fve_direct=fve_gen))
            else:
                excess_fve = fve_gen - export_limit
                if excess_fve * self.INTERVAL_HOURS <= max_charge_mwh:
                    actions.append(Action(
                        sell_fve_direct=export_limit, 
                        charge_from_fve=excess_fve
                    ))
                else:
                    max_fve_to_bess = max_charge_mwh / self.INTERVAL_HOURS
                    actions.append(Action(
                        sell_fve_direct=export_limit,
                        charge_from_fve=max_fve_to_bess
                    ))
        
        if fve_gen > 0  and max_charge_mwh > 0 and vdt_price < dt_price:
            fve_to_bess_options = np.linspace(0, min(fve_gen, max_charge_mwh / self.INTERVAL_HOURS), 5)
            for fve_to_bess_mw in fve_to_bess_options[1:]:
                remaining_fve = fve_gen - fve_to_bess_mw
                if remaining_fve <= export_limit:
                    actions.append(Action(
                        charge_from_fve=fve_to_bess_mw,
                        sell_fve_direct=remaining_fve
                    ))

        if max_charge_mwh > 0:
            charge_power_limit = min(max_charge_mwh / self.INTERVAL_HOURS, import_limit)
            if charge_power_limit > 0:
                charge_options = np.linspace(0, charge_power_limit, 5)
                for charge_mw in charge_options[1:]: 
                    if fve_gen <= export_limit:
                        actions.append(Action(
                            charge_from_grid=charge_mw,
                            sell_fve_direct=fve_gen
                        ))
        
        if max_discharge_mwh > 0:
            discharge_options = np.linspace(0, max_discharge_mwh / self.INTERVAL_HOURS, 5)
            for discharge_mw in discharge_options[1:]: 
                total_delivery = fve_gen + discharge_mw
                if total_delivery <= export_limit: 
                    actions.append(Action(
                        discharge_to_grid=discharge_mw,
                        sell_fve_direct=fve_gen
                    ))
        
        return list(set(actions))

    def _validate_strategy_physically(self, strategy: List[Action], fve_generation: List[float], 
                                 export_limit: float) -> List[Action]:
        """
        Post-processing validation of the strategy - corrects physically impossible states.
        """
        corrected_strategy = []
        
        for i, (action, fve_gen) in enumerate(zip(strategy, fve_generation)):
            corrected_action = Action()
            
            if fve_gen > 0:
                corrected_action.sell_fve_direct = min(action.sell_fve_direct, fve_gen)
                corrected_action.charge_from_fve = min(action.charge_from_fve, 
                                                    fve_gen - corrected_action.sell_fve_direct)
            else:
                corrected_action.sell_fve_direct = 0
                corrected_action.charge_from_fve = 0
            
            total_delivery = corrected_action.sell_fve_direct + action.discharge_to_grid
            if total_delivery > export_limit:
                available_for_bess = max(0, export_limit - corrected_action.sell_fve_direct)
                corrected_action.discharge_to_grid = min(action.discharge_to_grid, available_for_bess)
            else:
                corrected_action.discharge_to_grid = action.discharge_to_grid
            
            corrected_action.charge_from_grid = action.charge_from_grid
            
            corrected_strategy.append(corrected_action)
        
        return corrected_strategy

    def _calculate_action_profit_dp(self, action: Action, vdt_price: float, dt_price: float) -> float:
        """Calculates the ADDITIONAL profit from VDT operations."""
        substitution_profit = action.charge_from_fve * \
            (dt_price - vdt_price) * self.INTERVAL_HOURS
        discharge_revenue = action.discharge_to_grid * vdt_price * self.INTERVAL_HOURS
        grid_purchase_cost = action.charge_from_grid * vdt_price * self.INTERVAL_HOURS
        
        return substitution_profit + discharge_revenue - grid_purchase_cost

    def _calculate_new_soc_dp(self, current_soc: float, action: Action, efficiency: float) -> float:
        charge_energy = ((action.charge_from_fve + action.charge_from_grid) *
                         self.INTERVAL_HOURS * self.charge_efficiency)

        discharge_energy = (action.discharge_to_grid *
                            self.INTERVAL_HOURS) / self.discharge_efficiency

        return current_soc + charge_energy - discharge_energy

    def _find_closest_state_idx(self, soc: float, soc_states: np.ndarray) -> int:
        soc = max(0, min(soc, soc_states[-1]))
        return np.argmin(np.abs(soc_states - soc))

    def _get_interpolation_penalty(self, actual_soc: float, discrete_soc: float, soc_step: float) -> float:
        return abs(actual_soc - discrete_soc) / soc_step * 0.01

    def _reconstruct_dp_strategy(self, dp_parent: np.ndarray, dp_action: Dict,
                                 n_intervals: int, end_state_idx: int) -> List[Action]:
        """Reconstructs the optimal strategy from the DP tables."""
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

    def _convert_dp_strategy_to_report_format_safe(self, dp_strategy: List[Action],
                                               dt_prices, vdt_prices,
                                               fve_generation_scaled,
                                               bess_capacity, efficiency, start_soc) -> Tuple[List[Dict], Dict]:
        """
        Converts the DP strategy into a report format + generates a trading analysis.
        Returns: (strategy, trading_report)
        """
        strategy = []
        current_soc = start_soc
        
        active_trades = []
        cumulative_profit = 0
        
        for t, action in enumerate(dp_strategy):
            corrected_action = Action(
                sell_fve_direct=action.sell_fve_direct,
                charge_from_fve=action.charge_from_fve,
                charge_from_grid=action.charge_from_grid,
                discharge_to_grid=action.discharge_to_grid
            )
            
            max_possible_discharge_mwh = current_soc  
            max_possible_discharge_mw = max_possible_discharge_mwh / self.INTERVAL_HOURS
            
            if corrected_action.discharge_to_grid > max_possible_discharge_mw:
                corrected_action = Action(
                    sell_fve_direct=corrected_action.sell_fve_direct,
                    charge_from_fve=corrected_action.charge_from_fve,
                    charge_from_grid=corrected_action.charge_from_grid,
                    discharge_to_grid=max_possible_discharge_mw 
                )
            
            charge_energy = (corrected_action.charge_from_fve + corrected_action.charge_from_grid) * \
                self.charge_efficiency * self.INTERVAL_HOURS
            discharge_energy = (corrected_action.discharge_to_grid * self.INTERVAL_HOURS) / \
                self.discharge_efficiency
            
            current_soc = max(0.0, min(bess_capacity, current_soc + charge_energy - discharge_energy))

            interval_data = {
                'interval': t + 1,
                'dt_price_eur_mwh': dt_prices[t],
                'vdt_price_eur_mwh': vdt_prices[t],
                'fve_generation_mw': fve_generation_scaled[t],
                'action': {'dp_action': corrected_action},
                'soc_mwh': current_soc,
            }
            strategy.append(interval_data)
            
            period_profit = self._calculate_action_profit_dp(
                corrected_action, vdt_prices[t], dt_prices[t])

            if abs(period_profit) > 0.01:
                cumulative_profit += period_profit
                trade_type = "hold"
                if corrected_action.charge_from_fve > 0:
                    trade_type = "substitution"
                elif corrected_action.discharge_to_grid > 0:
                    trade_type = "discharge"
                elif corrected_action.charge_from_grid > 0:
                    trade_type = "charge"

                active_trades.append({
                    'period': t + 1,
                    'dt_price': dt_prices[t],
                    'vdt_price': vdt_prices[t],
                    'action': trade_type,
                    'profit': period_profit,
                    'soc': current_soc,
                    'cumulative': cumulative_profit
                })
        
        trading_report = {'active_trades': active_trades}
        return strategy, trading_report

    def print_daily_trading_report(self, day_results, day_number=None):
        """Prints the block strategy report. This part of the output remains in Czech."""
        report = day_results.get('trading_report', {})
        trades = report.get('active_trades', [])
        financials = day_results['financial_results']
        operational = day_results.get('operational_summary', {})

        day_str = f" DEN {day_number}" if day_number else ""
        print(f"\n{'='*95}")
        print(f"💰 OBCHODNÍ REPORT - BLOKOVÁ STRATEGIE{day_str} 💰")
        print(f"{'='*95}")

        print("--- FINANČNÍ VÝSLEDKY ---")
        print(
            f"💰 Celkový příjem klienta: {financials.get('client_total_income_eur', 0):.2f} EUR")
        print(
            f"💼 Náš celkový příjem:     {financials.get('our_total_commission_eur', 0):.2f} EUR")

        actual_cycles = operational.get('actual_cycles', 0)
        print(f"\n--- INFORMACE O CYKLECH ---")
        print(f"🔄 Počet cyklů: {actual_cycles:.2f}")

        print("\n--- ROZPIS PŘÍJMŮ ---")
        print(
            f"  Klient (DT):  {financials.get('client_dt_revenue_eur', 0):.2f} EUR")
        print(
            f"  My (DT):      {financials.get('our_dt_commission_eur', 0):.2f} EUR")
        print(
            f"  Klient (VDT): {financials.get('client_vdt_spread_revenue_eur', 0):.2f} EUR")
        print(
            f"  My (VDT):     {financials.get('our_vdt_spread_commission_eur', 0):.2f} EUR")

        if trades:
            print(f"\n--- AKTIVNÍ OPERACE ---")
            print("Perioda | DT cena | VDT cena | Akce             | Zisk    |   SOC   ")
            print("--------|---------|----------|------------------|---------|----------")

            for trade in trades[:30]:
                action_map = {
                    'block_charge': 'Blok nabíjení',
                    'block_discharge': 'Blok vybíjení',
                    'substitution': 'Substituce FVE',
                    'hold': 'Držení'
                }
                action_text = action_map.get(trade['action'], trade['action'])

                print(f"{trade['period']:6d} | {trade['dt_price']:7.2f} | {trade['vdt_price']:8.2f} | "
                      f"{action_text:16s} | {trade['profit']:7.2f} | {trade['soc']:8.3f}")

            if len(trades) > 30:
                print(f"... a dalších {len(trades) - 30} operací ...")
        print(f"{'='*95}")

def generate_plot_for_excel(timestamps, fve_data, dt_price_data, vdt_price_data,
                            bess_discharge_data, soc_data,
                            charge_fve_data, charge_grid_data,
                            export_limit, import_limit, bess_capacity):
    """
    Visualization for BESS_Optimizer in the style of FVE_BESS_Optimizer.
    Three graphs arranged one below the other.
    """
    fve_data = np.array(fve_data, dtype=np.float64)
    dt_price_data = np.array(dt_price_data, dtype=np.float64)
    vdt_price_data = np.array(vdt_price_data, dtype=np.float64)
    bess_discharge_data = np.array(bess_discharge_data, dtype=np.float64)
    soc_data = np.array(soc_data, dtype=np.float64)
    charge_fve_data = np.array(charge_fve_data, dtype=np.float64)
    charge_grid_data = np.array(charge_grid_data, dtype=np.float64)
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle('Denní graf operací FVE a BESS', fontsize=16, y=0.96)

    ax1 = fig.add_subplot(3, 1, 1)
    ax2 = fig.add_subplot(3, 1, 2, sharex=ax1)
    ax3 = fig.add_subplot(3, 1, 3, sharex=ax1)

    date_str = timestamps[0].strftime('%d.%m.%Y')
    
    x_axis = np.arange(len(timestamps))
    bar_width = 1 

    ax1.text(0.5, 1.05, f'Datum: {date_str}', transform=ax1.transAxes, 
             ha='center', va='bottom', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Výkon (MW)', fontsize=12)
    
    ax1.plot(x_axis, fve_data, label='Výroba FVE', color='gold', linewidth=2, linestyle='--' , drawstyle='steps-post')
    
    fve_direct_sale = fve_data - charge_fve_data
    ax1.fill_between(x_axis, 0, fve_direct_sale, 
                 label='Prodej z FVE do sítě', color='limegreen', alpha=0.7, step='post')

    total_export = fve_direct_sale + bess_discharge_data

    ax1.fill_between(x_axis, fve_direct_sale, total_export, 
                    label='Dodávka z BESS', color='royalblue', alpha=0.7, step='post')

    negative_grid_charge = -np.array(charge_grid_data) 
    negative_fve_charge = -np.array(charge_fve_data)

    ax1.fill_between(x_axis, 0, negative_grid_charge,
                    label='Nabíjení ze sítě', color='red', alpha=0.7, step='post')

    total_charge = negative_grid_charge + negative_fve_charge

    ax1.fill_between(x_axis, negative_grid_charge, total_charge,
                    label='Nabíjení z FVE', color='goldenrod', alpha=0.7, step='post')

    ax1.axhline(y=export_limit, color='darkgreen', linestyle='--',
                linewidth=2, label=f'Export limit ({export_limit} MW)')
    ax1.axhline(y=-import_limit, color='darkred', linestyle='--',
                linewidth=2, label=f'Import limit ({import_limit} MW)')
    
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax1.axhline(y=0, color='black', linewidth=0.5)
    plt.setp(ax1.get_xticklabels(), visible=False)
    ax2.text(0.5, 1.05, f'Datum: {date_str}', transform=ax2.transAxes, 
             ha='center', va='bottom', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Cena (EUR/MWh)', fontsize=12)
    
    ax2.plot(x_axis, vdt_price_data, label='Cena VDT', 
             color='darkred', linewidth=2.5, marker='o', markersize=3)
    ax2.fill_between(x_axis, vdt_price_data, color='darkred', alpha=0.1)
    
    ax2.plot(x_axis, dt_price_data, label='Cena DT',
             color='darkblue', linewidth=2.0, marker='.', markersize=4)
    ax2.fill_between(x_axis, dt_price_data, color='darkblue', alpha=0.1)
    
    ax2.legend(loc='upper left', fontsize=11, frameon=True, shadow=True)
    ax2.grid(True, which='major', linestyle='-', linewidth=0.7, alpha=0.7)
    ax2.grid(True, which='minor', linestyle=':', linewidth=0.4, alpha=0.5)
    ax2.minorticks_on()
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
    ax3.grid(True, which='both', linestyle='--', linewidth=0.5)

    tick_positions = np.arange(0, len(timestamps) + 1, 4) 
    tick_labels = [f"{h:02d}" for h in range(25)] 
    
    ax3.set_xticks(tick_positions)
    ax3.set_xticklabels(tick_labels[:len(tick_positions)])
    plt.setp(ax3.get_xticklabels(), rotation=0, ha='center')

    plt.subplots_adjust(hspace=0.3, top=0.92, bottom=0.08)

    img_data = io.BytesIO()
    plt.savefig(img_data, format='png', bbox_inches='tight', dpi=100)
    plt.close(fig)
    img_data.seek(0)

    return img_data

def export_full_report_to_excel(project_results: dict, original_dt_data: pd.DataFrame,
                                original_vdt_data: pd.DataFrame, optimizer_instance: FVE_BESS_Optimizer,
                                filename: str, with_plots: bool = False):
    """
    Creates an Excel report with modified graphs. This part of the output remains in Czech.
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

        format_zvyrazneni3 = workbook.add_format({'bg_color': "#A5DB7E", 'bold': True, 'border': 1, 'border_color': '#839671'})
        format_neutral = workbook.add_format({'bg_color': "#FFE681"})
        format_vypocet = workbook.add_format({'font_color': '#ED7D31', 'bold': True})
        format_spatne_czk = workbook.add_format({'bg_color': '#FFC7CE', 'font_color': '#9C0006', 'num_format': '#,##0.00 "Kč"'})
        euro_format = workbook.add_format({'num_format': '€ #,##0.00'})
        format_neutral_euro = workbook.add_format({'bg_color': '#FFEB9C', 'num_format': '€ #,##0.00'})

        annual_report_df = create_detailed_report(project_results)

        neutral_keys = [
            'Příjem klienta z FVE (EUR)', 
            'Naše provize z FVE (EUR)', 
            'Celkový příjem z BESS (EUR)',
            'Celkový příjem z BESS (EUR)',
            'Počet cyklů BESS'
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

            # Определяем формат на основе ключа
            if key == 'Projekt':
                key_cell_format = format_zvyrazneni3
                value_cell_format = format_zvyrazneni3
            elif key in neutral_keys:
                key_cell_format = format_neutral
                value_cell_format = format_neutral_euro if '(EUR)' in key else format_neutral
            elif '(EUR)' in key:
                value_cell_format = euro_format

            # Записываем ключ и значение с определенными форматами
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
                day_timestamps = original_dt_data.index[start_idx:end_idx]
                # Data for the graph
                fve_data = [d['fve_generation_mw']
                            for d in day_data['operational_summary']['vdt_strategy']]
                bess_discharge_data = [d['action']['dp_action'].discharge_to_grid
                                       for d in day_data['operational_summary']['vdt_strategy']]
                soc_data = [d['soc_mwh']
                            for d in day_data['operational_summary']['vdt_strategy']]
                charge_fve_data = [
                    d['action']['dp_action'].charge_from_fve for d in day_data['operational_summary']['vdt_strategy']]
                charge_grid_data = [
                    d['action']['dp_action'].charge_from_grid for d in day_data['operational_summary']['vdt_strategy']]

                dt_price_data = original_dt_data['Marginální cena ČR (EUR/MWh)'].values[start_idx:end_idx]
                vdt_price_data = original_vdt_data[
                    'Vážený průměr cen (EUR/MWh)'].values[start_idx:end_idx]

                # Create the graph with the new functions
                image_data = generate_plot_for_excel(
                    day_timestamps, fve_data, dt_price_data, vdt_price_data,
                    bess_discharge_data, soc_data, charge_fve_data, charge_grid_data, export_limit, import_limit, bess_capacity)

                worksheet_daily.insert_image(f'D{current_row + 1}', f'day_{i+1}_plot.png',
                                             {'image_data': image_data})

                current_row += 55

                worksheet_daily.set_column('A:B', 35)
                worksheet_daily.set_column('D:M', 12)

    print(f"✅ Report '{filename}' was successfully created!")

def create_detailed_report(project_results: dict) -> pd.DataFrame:
    """
    Creates a detailed annual report with aggregated income from BESS. This part of the output remains in Czech.
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

    total_sold_direct_fve_mwh = project_results['total_delivered_to_grid_mwh'] - \
        total_discharged

    project_summary = {
        'Projekt': config['name'],
        'Analyzováno dní': project_results['days_analyzed'],
        # 'Celkový příjem klienta (EUR)': round(project_results['annual_client_income_eur'], 2),
        # 'Náš celkový příjem (EUR)': round(project_results['annual_our_income_eur'], 2),
        'Příjem klienta z FVE (EUR)': round(sum(d['financial_results']['client_dt_revenue_eur'] for d in daily_results), 2),
        'Naše provize z FVE (EUR)': round(sum(d['financial_results']['our_dt_commission_eur'] for d in daily_results), 2),
        'Celkový příjem z BESS (EUR)': round(total_bess_income, 2),
        'Celková výroba FVE (MWh)': round(project_results['total_fve_generation_mwh'], 2),
        'Fyzicky prodáno přímo z FVE (MWh)': round(total_sold_direct_fve_mwh, 2),
        'Fyzicky prodáno z BESS (MWh)': round(total_discharged, 2),
        'CELKEM fyzicky prodáno (MWh)': round(project_results['total_delivered_to_grid_mwh'], 2),
        'Nakoupeno ze sítě do BESS (MWh)': round(project_results['total_purchased_from_grid_mwh'], 2),
        'Nabito do BESS z FVE (MWh)': round(project_results['total_fve_to_bess_mwh'], 2),
        'CELKEM nabito do BESS (MWh)': round(total_charged, 2),
        'Ztráty účinnosti BESS (MWh)': round(efficiency_losses_mwh, 2),
        'Počet cyklů BESS': round(bess_cycles, 2),
    }

    return pd.DataFrame([project_summary])

def create_daily_report_df(day_data: dict, day_index: int, project_config: dict,
                           optimizer_instance: FVE_BESS_Optimizer) -> pd.DataFrame:
    """Creates a daily report DataFrame. This part of the output remains in Czech."""

    fve_gen, delivered, purchased, charged, discharged, fve_to_bess = 0, 0, 0, 0, 0, 0
    interval_hours = optimizer_instance.INTERVAL_HOURS
    soc0mwh = day_data['operational_summary'].get('start_soc_mwh', 0)
    soc24mwh = day_data['operational_summary']['vdt_strategy'][-1]['soc_mwh']
    for interval in day_data['operational_summary']['vdt_strategy']:
        action = interval['action']['dp_action']
        fve_gen += interval['fve_generation_mw'] * interval_hours
        delivered += (action.sell_fve_direct +
                      action.discharge_to_grid) * interval_hours
        purchased += action.charge_from_grid * interval_hours
        charged += (action.charge_from_fve +
                    action.charge_from_grid) * interval_hours
        discharged += action.discharge_to_grid * interval_hours
        fve_to_bess += action.charge_from_fve * interval_hours
        

        
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
            'Příjem klienta (DT)': round(financials['client_dt_revenue_eur'], 2),
            'Naše provize (DT)': round(financials['our_dt_commission_eur'], 2),
            'Příjem (VDT)': round((financials['client_vdt_spread_revenue_eur'] + financials['our_vdt_spread_commission_eur']), 2),
            # 'Naše provize (VDT)': round(financials['our_vdt_spread_commission_eur'], 2),
            # 'CELKEM Klient': round(financials['client_total_income_eur'], 2),
            # 'CELKEM My': round(financials['our_total_commission_eur'], 2),
            '--- Energie (MWh) ---': '',
            'Výroba FVE': round(fve_gen, 2),
            'Dodávka do sítě': round(delivered, 2),
            'Nákup ze sítě': round(purchased, 2),
            '--- BESS (MWh) ---': '',
            'Nabíjení z FVE': round(fve_to_bess, 2),
            'Vybíjení do sítě': round(discharged, 2),
            'Ztráty účinnosti': round(total_losses, 2),
            'Cykly': round(cycles, 2),
            'SOC na začátku': round(soc0mwh, 2),
            'SOC na konci': round(soc24mwh, 2)
        }
    }

    df = pd.DataFrame(report_data)
    return df

def analyze_projects(dt_prices_data, vdt_prices_data, fve_generation_data, projects_config: Dict,
                     analysis_period_days=365,  show_daily_reports=False):
    """
    Analyzes projects.

    Args:
        dt_prices_data (pd.DataFrame): The DT prices data.
        vdt_prices_data (pd.DataFrame): The VDT prices data.
        fve_generation_data (pd.DataFrame): The FVE generation data.
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

        optimizer = FVE_BESS_Optimizer(config)

        daily_results = []

        total_client_income_eur, total_our_income_eur = 0, 0
        total_fve_generation_mwh, total_delivered_to_grid_mwh, total_purchased_from_grid_mwh = 0, 0, 0
        total_charged_to_bess_mwh, total_discharged_from_bess_mwh, total_fve_to_bess_mwh = 0, 0, 0
        successful_days = 0
        available_days = min(len(dt_prices_data), len(vdt_prices_data), len(fve_generation_data))
        current_soc = optimizer.soc0_mwh
        max_days = min(analysis_period_days, available_days // 96)

        

        for day in range(max_days):
            start_idx = day * 96
            end_idx = (day + 1) * 96

            if end_idx > len(dt_prices_data):
                break

            try:
                dt_day = dt_prices_data.iloc[start_idx:
                                             end_idx]['Marginální cena ČR (EUR/MWh)'].values

                vdt_day = vdt_prices_data.iloc[start_idx:
                                               end_idx]['Vážený průměr cen (EUR/MWh)'].values
                
                fve_day = fve_generation_data.iloc[start_idx:
                                                   end_idx]['Výroba [kW]'].values / 1000
                day_result = optimizer.optimize_day_strategy_dp(dt_day, vdt_day, fve_day, start_soc=current_soc)

  
                daily_results.append(day_result)

                if day_result['operational_summary']['vdt_strategy']:
                    current_soc = day_result['operational_summary']['vdt_strategy'][-1]['soc_mwh']

                total_client_income_eur += day_result['financial_results']['client_total_income_eur']
                total_our_income_eur += day_result['financial_results']['our_total_commission_eur']
                successful_days += 1

                for interval_data in day_result['operational_summary']['vdt_strategy']:
                    action = interval_data['action']['dp_action']
                    interval_hours = optimizer.INTERVAL_HOURS
                    total_fve_generation_mwh += interval_data['fve_generation_mw'] * \
                        interval_hours
                    total_delivered_to_grid_mwh += (
                        action.sell_fve_direct + action.discharge_to_grid) * interval_hours
                    total_purchased_from_grid_mwh += action.charge_from_grid * interval_hours
                    total_charged_to_bess_mwh += (
                        action.charge_from_fve + action.charge_from_grid) * interval_hours
                    total_discharged_from_bess_mwh += action.discharge_to_grid * interval_hours
                    total_fve_to_bess_mwh += action.charge_from_fve * interval_hours

                if show_daily_reports:
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
                'total_fve_generation_mwh': total_fve_generation_mwh,
                'total_delivered_to_grid_mwh': total_delivered_to_grid_mwh,
                'total_purchased_from_grid_mwh': total_purchased_from_grid_mwh,
                'total_charged_to_bess_mwh': total_charged_to_bess_mwh,
                'total_discharged_from_bess_mwh': total_discharged_from_bess_mwh,
                'total_fve_to_bess_mwh': total_fve_to_bess_mwh,
            }

            print(f"\nSuccessfully analyzed {successful_days} days")
            print(f"Annual income: {(total_client_income_eur + total_our_income_eur):,.0f} EUR")

    return results

if __name__ == '__main__':
    # Note: The column names in the following lines are in Czech because they match the source Excel files.
    fve_df = pd.read_csv('data/FVE_Vresovice_2024_15min_extrapolated.csv', sep=',')
    fve_df['datetime'] = pd.to_datetime(fve_df['DT'], format='ISO8601')
    fve_df = fve_df.set_index('datetime')

    fve_df['Výroba [kW]'] = fve_df['FVE power [MW]'] * 1000

    fve_df = fve_df[['Výroba [kW]']]    
    FVE = fve_df[fve_df.index.year == 2024]

    vdt2024 = pd.read_excel("data/OTE_2024.xlsx", sheet_name="VDT (EUR)", skiprows=5, usecols="A:L")
    dt2024 = pd.read_excel("data/OTE_2024.xlsx", sheet_name="DT ČR", skiprows=5, usecols="A:L")

    vdt2024['datetime'] = pd.to_datetime(vdt2024['Den'])
    vdt2024['datetime'] = vdt2024['datetime'] + pd.to_timedelta(vdt2024['Hodina'] - 1, unit='h')

    vdt2024.sort_values('datetime', inplace=True)
    vdt2024.drop_duplicates(subset=['datetime'], keep='first', inplace=True)

    vdt2024 = vdt2024[['datetime', 'Vážený průměr cen (EUR/MWh)']]
    vdt2024.set_index('datetime', inplace=True)

    vdt_15min = vdt2024.resample('15T').ffill()

    vdt_15min.reset_index(inplace=True)
    vdt_15min['Den'] = vdt_15min['datetime'].dt.normalize()
    vdt_15min['Perioda'] = vdt_15min.groupby(vdt_15min['datetime'].dt.date).cumcount() + 1

    vdt2024_formatted = vdt_15min[['datetime', 'Den', 'Perioda', 'Vážený průměr cen (EUR/MWh)']]
    vdt2024_formatted.fillna(method='ffill', inplace=True)

    VDT = vdt2024_formatted[vdt2024_formatted['datetime'].dt.year == 2024]
    VDT.set_index('datetime', inplace=True)

    dt2024['datetime'] = pd.to_datetime(dt2024['Den'])

    dt2024['datetime'] = dt2024['datetime'] + pd.to_timedelta(dt2024['Hodina'] - 1, unit='h')

    dt2024.sort_values('datetime', inplace=True)
    dt2024.drop_duplicates(subset=['datetime'], keep='first', inplace=True)

    dt2024_subset = dt2024[['datetime', 'Marginální cena ČR (EUR/MWh)']]
    dt2024_subset.set_index('datetime', inplace=True)

    dt_15min = dt2024_subset.resample('15min').ffill()

    dt_15min.reset_index(inplace=True)

    dt_15min['Den'] = dt_15min['datetime'].dt.normalize()
    dt_15min['Perioda'] = dt_15min.groupby(dt_15min['datetime'].dt.date).cumcount() + 1

    dt2024_formatted = dt_15min[['datetime', 'Den', 'Perioda', 'Marginální cena ČR (EUR/MWh)']]

    last_known_price = 83.2 

    new_rows_data = [
        {
            'datetime': pd.to_datetime('2024-12-31 23:15:00'), 
            'Den': pd.to_datetime('2024-12-31'), 
            'Perioda': 94, 
            'Marginální cena ČR (EUR/MWh)': last_known_price
        },
        {
            'datetime': pd.to_datetime('2024-12-31 23:30:00'), 
            'Den': pd.to_datetime('2024-12-31'), 
            'Perioda': 95, 
            'Marginální cena ČR (EUR/MWh)': last_known_price
        },
        {
            'datetime': pd.to_datetime('2024-12-31 23:45:00'), 
            'Den': pd.to_datetime('2024-12-31'), 
            'Perioda': 96, 
            'Marginální cena ČR (EUR/MWh)': last_known_price
        }
    ]

    new_rows_df = pd.DataFrame(new_rows_data)

    DT = pd.concat([dt2024_formatted, new_rows_df], ignore_index=True)
    DT.set_index('datetime', inplace=True)

    PROJECTS_CONFIG = {
        'LG ': {
            'name': 'LG ',
            'fve_power_mw': 0,           
            'fve_scale_factor': 0,  
            'bess_power_mw': 12,
            'bess_capacity_mwh': 29,
            'export_limit_mw': 12,      
            'import_limit_mw': 12,      
            'efficiency': 0.85,
            'max_cycles': 2          
        },
    }


    results = analyze_projects(DT, VDT, FVE, projects_config=PROJECTS_CONFIG,
                                analysis_period_days=1, show_daily_reports=True)
