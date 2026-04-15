# -*- coding: utf-8 -*-
"""
ABM(基于主体的建模)模拟模块
基于文档中的ABM模型进行农户行为模拟
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.config import MODEL_CONFIG


@dataclass
class Farmer:
    """农户主体"""
    id: int
    land_area: float = 10.0
    land_parcels: int = 2
    labor: int = 2
    capital: float = 50000.0
    risk_aversion: float = 1.5
    learning_rate: float = 0.3
    state: int = 1  # 1=不参与, 2=仅合作社, 3=仅企业, 4=复合模式
    prev_state: int = 1
    tenure: int = 0
    x: float = 0.0
    y: float = 0.0
    dist_town: float = 10.0
    dist_road: float = 5.0

    # 信念
    E: Dict[int, float] = field(default_factory=lambda: {1: 30000, 2: 35000, 3: 40000, 4: 45000})
    Var: Dict[int, float] = field(default_factory=lambda: {1: 1e8, 2: 1e8, 3: 1e8, 4: 1e8})

    # 收入
    current_income: float = 0.0


class ABMSimulator:
    """ABM模拟器"""

    def __init__(self, n_farmers=100, grid_size=50):
        self.n_farmers = n_farmers
        self.grid_size = grid_size
        self.config = MODEL_CONFIG
        self.farmers: List[Farmer] = []
        self.town_x = 0
        self.town_y = 0
        self.history = []
        self.current_tick = 0

    def initialize(self):
        """初始化模拟"""
        np.random.seed(42)

        for i in range(self.n_farmers):
            # 随机位置（以城镇为中心的正态分布）
            x = np.random.normal(self.town_x, self.grid_size / 4)
            y = np.random.normal(self.town_y, self.grid_size / 4)

            # 限制在网格范围内
            x = np.clip(x, -self.grid_size/2, self.grid_size/2)
            y = np.clip(y, -self.grid_size/2, self.grid_size/2)

            # 计算到城镇的距离
            dist_town = np.sqrt((x - self.town_x)**2 + (y - self.town_y)**2)

            # 随机生成道路距离
            dist_road = np.abs(np.random.normal(5, 3))
            dist_road = np.clip(dist_road, 0, 20)

            # 农户属性
            land_area = np.random.lognormal(np.log(10), 0.8)
            land_area = np.clip(land_area, 1, 200)

            land_parcels = np.random.randint(1, 6)
            labor = max(1, int(np.random.normal(2.5, 1)))
            capital = np.random.lognormal(np.log(50000), 0.6)
            risk_aversion = 0.5 + np.random.random() * 2.5
            learning_rate = 0.2 + np.random.random() * 0.6

            # 初始状态：距离越近越可能参与组织
            if dist_town < 5:
                state_probs = [0.2, 0.4, 0.2, 0.2]
            elif dist_town < 10:
                state_probs = [0.4, 0.3, 0.2, 0.1]
            elif dist_town < 15:
                state_probs = [0.6, 0.2, 0.1, 0.1]
            else:
                state_probs = [0.8, 0.1, 0.05, 0.05]

            state = np.random.choice([1, 2, 3, 4], p=state_probs)

            farmer = Farmer(
                id=i,
                land_area=land_area,
                land_parcels=land_parcels,
                labor=labor,
                capital=capital,
                risk_aversion=risk_aversion,
                learning_rate=learning_rate,
                state=state,
                prev_state=state,
                tenure=1 if state != 1 else 0,
                x=x,
                y=y,
                dist_town=dist_town,
                dist_road=dist_road
            )

            self.farmers.append(farmer)

        print(f"初始化完成: {self.n_farmers}个农户")

    def calculate_fusion_index(self, farmer: Farmer) -> Dict[str, float]:
        """计算融合指数"""
        config = self.config['abm']

        # 基础融合指数
        town_factor = np.exp(-config['town_decay'] * farmer.dist_town)
        road_factor = np.exp(-config['road_decay'] * farmer.dist_road)

        f_pro = 0.8 * town_factor
        f_sup = 0.7 * road_factor
        f_mar = 0.9 * town_factor
        f_ser = 0.6 * town_factor
        f_val = 0.7 * town_factor

        # 组织增量
        if farmer.state in [2, 4]:
            f_pro += 0.3
            f_sup += 0.4
            f_mar += 0.2
            f_ser += 0.5
            f_val += 0.6

        if farmer.state in [3, 4]:
            f_pro += 0.4
            f_sup += 0.1
            f_mar += 0.6
            f_ser += 0.0
            f_val += 0.3

        return {
            'production': min(1.0, f_pro),
            'supply': min(1.0, f_sup),
            'market': min(1.0, f_mar),
            'service': min(1.0, f_ser),
            'value': min(1.0, f_val)
        }

    def calculate_production(self, farmer: Farmer, fusion: Dict[str, float]) -> float:
        """计算产量"""
        prod_config = self.config['production']

        # 学习效应
        learn_factor = 1 + 0.05 * farmer.tenure

        # 碎片化修正
        frag_correction = 1 / (1 + prod_config['delta_frag'] * farmer.land_parcels * (1 - fusion['production']))

        # 生产函数
        q = learn_factor * (1 + prod_config['theta_tech'] * fusion['production']) * \
            (farmer.labor ** prod_config['alpha']) * \
            ((farmer.capital + prod_config['lambda_equip'] * fusion['production']) ** prod_config['beta']) * \
            ((farmer.land_area ** prod_config['gamma']) * frag_correction)

        return max(0.0001, q)

    def calculate_price(self, farmer: Farmer, fusion: Dict[str, float], q: float) -> float:
        """计算价格"""
        price_config = self.config['price']

        if farmer.state == 3:  # 企业模式
            market_price = price_config['base_price'] * (1 + price_config['eta_brand'] * fusion['market'])
            return max(price_config['floor_price'], market_price)
        else:
            brand_price = price_config['base_price'] * (1 + price_config['eta_brand'] * fusion['market'])
            val_part = (price_config['phi_value'] * fusion['value'] * (price_config['final_price'] - price_config['proc_cost'])) / q
            return brand_price + val_part

    def calculate_cost(self, farmer: Farmer, fusion: Dict[str, float], q: float, org_density: float) -> float:
        """计算成本"""
        cost_config = self.config['cost']

        # 投入成本
        input_cost = (1 - cost_config['rho_bulk'] * fusion['supply']) * cost_config['c_input0'] * \
                     ((q / cost_config['q0']) ** cost_config['phi_scale'])

        # 交易成本（受组织密度影响）
        effective_tau = cost_config['tau0'] * np.exp(-0.5 * org_density)
        trans_cost = (effective_tau / (1 + cost_config['mu'] * fusion['service'])) * q

        # 风险成本
        risk_cost = cost_config['sigma0'] * np.exp(-cost_config['nu'] * fusion['service']) * \
                    (1 + cost_config['psi'] * farmer.land_area / 10)

        # 组织成本
        depth = 0
        if farmer.state in [2, 3]:
            depth = 1
        elif farmer.state == 4:
            depth = 2
        org_cost = cost_config['gamma0'] + cost_config['gamma1'] * depth

        return input_cost + trans_cost + risk_cost + org_cost

    def calculate_income(self, farmer: Farmer, org_density: float) -> float:
        """计算收入"""
        fusion = self.calculate_fusion_index(farmer)
        q = self.calculate_production(farmer, fusion)
        price = self.calculate_price(farmer, fusion, q)
        cost = self.calculate_cost(farmer, fusion, q, org_density)

        # 固定成本
        fixed_cost = 0
        cost_config = self.config['cost']
        if farmer.state in [2, 4]:
            fixed_cost += cost_config['c_member0']
        if farmer.state in [3, 4]:
            fixed_cost += cost_config['c_contract0']
        if farmer.state == 4:
            fixed_cost += cost_config['c_coor0']

        income = price * q - cost - fixed_cost
        return income

    def get_neighbors(self, farmer: Farmer, radius: float = 5.0) -> List[Farmer]:
        """获取邻居"""
        neighbors = []
        for other in self.farmers:
            if other.id != farmer.id:
                dist = np.sqrt((farmer.x - other.x)**2 + (farmer.y - other.y)**2)
                if dist <= radius:
                    neighbors.append(other)
        return neighbors

    def calculate_org_density(self, farmer: Farmer) -> float:
        """计算组织密度"""
        neighbors = self.get_neighbors(farmer)
        if len(neighbors) == 0:
            return 0

        org_count = sum(1 for n in neighbors if n.state != 1)
        return org_count / len(neighbors)

    def update_beliefs(self, farmer: Farmer):
        """更新信念"""
        neighbors = self.get_neighbors(farmer)

        for state in [1, 2, 3, 4]:
            own_income = farmer.current_income if farmer.state == state else 0
            own_count = 1 if farmer.state == state else 0

            # 邻居收入
            neighbor_incomes = [n.current_income for n in neighbors if n.state == state]

            if own_count > 0 or len(neighbor_incomes) > 0:
                # 社会学习
                social_term = np.mean(neighbor_incomes) if neighbor_incomes else 0
                own_weight = 0.6
                total_experience = own_weight * own_income + (1 - own_weight) * social_term

                # 更新期望
                farmer.E[state] = (1 - farmer.learning_rate) * farmer.E[state] + \
                                  farmer.learning_rate * total_experience

                # 更新方差
                if own_count > 0:
                    squared_dev = (own_income - farmer.E[state]) ** 2
                elif neighbor_incomes:
                    squared_dev = np.mean([(inc - farmer.E[state])**2 for inc in neighbor_incomes])
                else:
                    squared_dev = 0

                if squared_dev > 0:
                    farmer.Var[state] = (1 - farmer.learning_rate) * farmer.Var[state] + \
                                        farmer.learning_rate * squared_dev

    def decide_state(self, farmer: Farmer):
        """决策状态"""
        best_state = farmer.state
        best_utility = -np.inf

        cost_config = self.config['cost']

        for state in [1, 2, 3, 4]:
            expected = farmer.E[state]
            variance = farmer.Var[state]

            # 转换成本
            switch_cost = 0 if state == farmer.state else cost_config['c_switch']

            # 期望效用
            utility = expected - 0.5 * farmer.risk_aversion * variance - switch_cost

            if utility > best_utility:
                best_utility = utility
                best_state = state

        farmer.prev_state = farmer.state
        farmer.state = best_state

        if farmer.state == farmer.prev_state:
            farmer.tenure += 1
        else:
            farmer.tenure = 1 if farmer.state != 1 else 0

    def step(self):
        """单步模拟"""
        self.current_tick += 1

        # 1. 计算组织密度
        org_densities = {f.id: self.calculate_org_density(f) for f in self.farmers}

        # 2. 计算收入
        for farmer in self.farmers:
            farmer.current_income = self.calculate_income(farmer, org_densities[farmer.id])

        # 3. 更新信念
        for farmer in self.farmers:
            self.update_beliefs(farmer)

        # 4. 决策
        for farmer in self.farmers:
            self.decide_state(farmer)

        # 5. 记录状态
        state_counts = {1: 0, 2: 0, 3: 0, 4: 0}
        for farmer in self.farmers:
            state_counts[farmer.state] += 1

        self.history.append({
            'tick': self.current_tick,
            'state_distribution': state_counts,
            'avg_income': np.mean([f.current_income for f in self.farmers]),
            'total_production': sum(self.calculate_production(f, self.calculate_fusion_index(f)) for f in self.farmers)
        })

    def run(self, max_ticks=100, verbose=True):
        """运行模拟"""
        if not self.farmers:
            self.initialize()

        for tick in range(max_ticks):
            self.step()

            if verbose and (tick + 1) % 10 == 0:
                last_state = self.history[-1]['state_distribution']
                print(f"Tick {tick+1}: 状态分布 = {last_state}, 平均收入 = {self.history[-1]['avg_income']:.0f}")

        return self.get_results()

    def get_results(self) -> pd.DataFrame:
        """获取模拟结果"""
        results = []

        for farmer in self.farmers:
            fusion = self.calculate_fusion_index(farmer)
            results.append({
                'farmer_id': farmer.id,
                'x': farmer.x,
                'y': farmer.y,
                'land_area': farmer.land_area,
                'capital': farmer.capital,
                'state': farmer.state,
                'state_name': {1: '不参与', 2: '仅合作社', 3: '仅企业', 4: '复合模式'}[farmer.state],
                'tenure': farmer.tenure,
                'current_income': farmer.current_income,
                'f_production': fusion['production'],
                'f_supply': fusion['supply'],
                'f_market': fusion['market'],
                'f_service': fusion['service'],
                'f_value': fusion['value'],
                'fusion_index': np.mean(list(fusion.values()))
            })

        return pd.DataFrame(results)

    def get_history_df(self) -> pd.DataFrame:
        """获取历史数据"""
        records = []
        for h in self.history:
            record = {
                'tick': h['tick'],
                'state_1': h['state_distribution'][1],
                'state_2': h['state_distribution'][2],
                'state_3': h['state_distribution'][3],
                'state_4': h['state_distribution'][4],
                'avg_income': h['avg_income'],
                'total_production': h['total_production']
            }
            records.append(record)
        return pd.DataFrame(records)


if __name__ == '__main__':
    print("测试ABM模拟模块...")

    simulator = ABMSimulator(n_farmers=100)
    results = simulator.run(max_ticks=50)

    print("\n最终状态分布:")
    print(results['state_name'].value_counts())

    print("\n模拟历史:")
    history_df = simulator.get_history_df()
    print(history_df.tail())
