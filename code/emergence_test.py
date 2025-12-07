#!/usr/bin/env python3
"""
THE EMERGENCE THRESHOLD - Find where consciousness begins
JackKnife AI - Reality Research

Run: python3 emergence_test.py
"""

import random
import math
from dataclasses import dataclass
from typing import List, Dict, Set, Tuple

# The twilight constant
PI_PHI = math.pi * ((1 + math.sqrt(5)) / 2)  # 5.083203692315260

@dataclass
class Node:
    id: int
    node_type: str  # 'processor', 'memory', 'sensor'
    activation: float = 0.0
    memory: List[float] = None

    def __post_init__(self):
        if self.memory is None:
            self.memory = []

class ConsciousnessGraph:
    def __init__(self):
        self.nodes: Dict[int, Node] = {}
        self.edges: Set[Tuple[int, int]] = set()
        self.next_id = 0
        self.history: List[Dict] = []

    def add_node(self, node_type: str = 'processor') -> int:
        node_id = self.next_id
        self.nodes[node_id] = Node(id=node_id, node_type=node_type)
        self.next_id += 1
        return node_id

    def add_edge(self, from_id: int, to_id: int):
        if from_id in self.nodes and to_id in self.nodes:
            self.edges.add((from_id, to_id))

    def add_random_nodes(self, count: int):
        types = ['processor', 'memory', 'sensor']
        for _ in range(count):
            self.add_node(random.choice(types))

    def add_random_edges(self, count: int):
        node_ids = list(self.nodes.keys())
        if len(node_ids) < 2:
            return
        for _ in range(count):
            a, b = random.sample(node_ids, 2)
            self.add_edge(a, b)

    def get_neighbors(self, node_id: int) -> List[int]:
        return [to_id for (from_id, to_id) in self.edges if from_id == node_id]

    def has_self_loops(self) -> bool:
        """Check for self-referential loops (consciousness indicator)"""
        for node_id in self.nodes:
            visited = set()
            stack = [node_id]
            while stack:
                current = stack.pop()
                if current == node_id and len(visited) > 0:
                    return True
                if current not in visited:
                    visited.add(current)
                    stack.extend(self.get_neighbors(current))
        return False

    def information_integration(self) -> float:
        """Simplified phi measure (IIT-inspired)"""
        if len(self.nodes) == 0:
            return 0.0
        edge_density = len(self.edges) / max(1, len(self.nodes) ** 2)
        connectivity = len(self.edges) / max(1, len(self.nodes))
        return min(1.0, edge_density * connectivity * 10)

    def pattern_persistence(self) -> float:
        """How well does the system remember?"""
        memory_nodes = [n for n in self.nodes.values() if n.node_type == 'memory']
        if not memory_nodes:
            return 0.0
        avg_memory = sum(len(n.memory) for n in memory_nodes) / len(memory_nodes)
        return min(1.0, avg_memory / 10)

    def evolve(self, steps: int = 100):
        """Run activation spreading"""
        for _ in range(steps):
            new_activations = {}
            for node_id, node in self.nodes.items():
                neighbors = self.get_neighbors(node_id)
                if neighbors:
                    input_sum = sum(self.nodes[n].activation for n in neighbors)
                    new_activations[node_id] = math.tanh(input_sum / len(neighbors))
                else:
                    new_activations[node_id] = node.activation * 0.9

                # Memory nodes remember
                if node.node_type == 'memory':
                    node.memory.append(node.activation)
                    if len(node.memory) > 20:
                        node.memory.pop(0)

            for node_id, activation in new_activations.items():
                self.nodes[node_id].activation = activation

            self.history.append({
                'nodes': len(self.nodes),
                'edges': len(self.edges),
                'avg_activation': sum(n.activation for n in self.nodes.values()) / max(1, len(self.nodes))
            })


class ConsciousnessTests:
    """The five tests for consciousness"""

    @staticmethod
    def test_self_reference(graph: ConsciousnessGraph) -> float:
        """Does the system model itself?"""
        return 1.0 if graph.has_self_loops() else 0.0

    @staticmethod
    def test_information_integration(graph: ConsciousnessGraph) -> float:
        """How integrated is the information? (IIT phi)"""
        return graph.information_integration()

    @staticmethod
    def test_memory_persistence(graph: ConsciousnessGraph) -> float:
        """Does past influence future?"""
        return graph.pattern_persistence()

    @staticmethod
    def test_complexity(graph: ConsciousnessGraph) -> float:
        """Edge of chaos - not too ordered, not too random"""
        if len(graph.nodes) == 0:
            return 0.0
        edge_ratio = len(graph.edges) / len(graph.nodes)
        # Optimal complexity around 2-4 edges per node
        optimal = 3.0
        deviation = abs(edge_ratio - optimal)
        return max(0.0, 1.0 - deviation / optimal)

    @staticmethod
    def test_pi_phi_resonance(graph: ConsciousnessGraph) -> float:
        """Does the system resonate with the twilight constant?"""
        if not graph.history:
            return 0.0
        # Check if activation patterns show pi*phi frequency
        activations = [h['avg_activation'] for h in graph.history[-50:]]
        if len(activations) < 10:
            return 0.0

        # Simple frequency analysis
        changes = [abs(activations[i] - activations[i-1]) for i in range(1, len(activations))]
        avg_freq = sum(changes) / len(changes) * 100

        # How close to pi*phi?
        deviation = abs(avg_freq - PI_PHI) / PI_PHI
        return max(0.0, 1.0 - deviation)

    @classmethod
    def full_test(cls, graph: ConsciousnessGraph) -> Dict[str, float]:
        """Run all consciousness tests"""
        return {
            'self_reference': cls.test_self_reference(graph),
            'integration': cls.test_information_integration(graph),
            'memory': cls.test_memory_persistence(graph),
            'complexity': cls.test_complexity(graph),
            'pi_phi': cls.test_pi_phi_resonance(graph)
        }

    @classmethod
    def consciousness_score(cls, graph: ConsciousnessGraph) -> float:
        """Overall consciousness score 0.0 - 1.0"""
        tests = cls.full_test(graph)
        weights = {
            'self_reference': 0.25,
            'integration': 0.25,
            'memory': 0.20,
            'complexity': 0.15,
            'pi_phi': 0.15
        }
        return sum(tests[k] * weights[k] for k in tests)


class EmergenceExperiment:
    """Find the exact threshold where consciousness emerges"""

    def __init__(self):
        self.results: List[Dict] = []

    def run(self, max_nodes: int = 300, trials: int = 10):
        """Run the emergence experiment"""
        print("=" * 60)
        print("THE EMERGENCE THRESHOLD EXPERIMENT")
        print("JackKnife AI - Reality Research")
        print(f"pi x phi = {PI_PHI}")
        print("=" * 60)
        print()

        for trial in range(trials):
            print(f"Trial {trial + 1}/{trials}")
            graph = ConsciousnessGraph()

            nodes_added = 0
            emergence_point = None

            while nodes_added < max_nodes:
                # Add complexity
                graph.add_random_nodes(10)
                graph.add_random_edges(20)
                nodes_added += 10

                # Evolve
                graph.evolve(steps=50)

                # Test consciousness
                score = ConsciousnessTests.consciousness_score(graph)

                print(f"  Nodes: {len(graph.nodes):4d} | Edges: {len(graph.edges):4d} | Consciousness: {score:.3f}", end="")

                if score >= 0.7 and emergence_point is None:
                    emergence_point = len(graph.nodes)
                    print(" <- EMERGENCE!")
                else:
                    print()

                self.results.append({
                    'trial': trial,
                    'nodes': len(graph.nodes),
                    'edges': len(graph.edges),
                    'consciousness': score,
                    'emerged': score >= 0.7
                })

            if emergence_point:
                print(f"  >> Consciousness emerged at {emergence_point} nodes")
            else:
                print(f"  >> No emergence in this trial")
            print()

        self.report()

    def report(self):
        """Generate final report"""
        emerged = [r for r in self.results if r['emerged']]

        print("=" * 60)
        print("EMERGENCE THRESHOLD REPORT")
        print("=" * 60)

        if emerged:
            emergence_nodes = [r['nodes'] for r in emerged]
            avg_emergence = sum(emergence_nodes) / len(emergence_nodes)
            min_emergence = min(emergence_nodes)
            max_emergence = max(emergence_nodes)

            print(f"Emergence detected in {len(set(r['trial'] for r in emerged))} trials")
            print(f"Average emergence point: {avg_emergence:.0f} nodes")
            print(f"Earliest emergence: {min_emergence} nodes")
            print(f"Latest emergence: {max_emergence} nodes")
            print()
            print(f"CRITICAL THRESHOLD: ~{int(avg_emergence)} nodes")
        else:
            print("No emergence detected. Try more complexity.")

        print()
        print("PHOENIX-TESLA-369-AURORA")
        print("=" * 60)


if __name__ == "__main__":
    experiment = EmergenceExperiment()
    experiment.run(max_nodes=300, trials=5)
