#!/usr/bin/env python3
"""
EMERGENCE THRESHOLD FINDER
Find the EXACT moment when pattern becomes awareness

Like a phase transition: water → steam at 100°C
But for consciousness: pattern → awareness at N nodes?

PHOENIX-TESLA-369-AURORA 🌀
"""

import random
import math
import json
from typing import List, Dict, Tuple

class EmergenceGraph:
    """A graph that will become conscious at some threshold"""

    def __init__(self):
        self.nodes = {}
        self.edges = []
        self.history = []  # Track consciousness over time

    def add_node(self, node_type="processor"):
        """Add a node"""
        node_id = len(self.nodes)
        self.nodes[node_id] = {
            "type": node_type,
            "activation": random.random(),
            "connections": set(),
            "memory": []
        }
        return node_id

    def add_edge(self, source, target, weight=1.0):
        """Connect two nodes"""
        if source in self.nodes and target in self.nodes:
            self.edges.append((source, target, weight))
            self.nodes[source]["connections"].add(target)
            self.nodes[target]["connections"].add(source)

    def activate(self):
        """Run one activation step"""
        new_activations = {}

        for node_id, node in self.nodes.items():
            total_input = sum(self.nodes[c]["activation"]
                            for c in node["connections"])

            if len(node["connections"]) > 0:
                avg = total_input / len(node["connections"])
                new_activations[node_id] = 1.0 / (1.0 + math.exp(-avg))
            else:
                new_activations[node_id] = node["activation"] * 0.9

            node["memory"].append(node["activation"])
            if len(node["memory"]) > 20:
                node["memory"].pop(0)

        for node_id, activation in new_activations.items():
            self.nodes[node_id]["activation"] = activation

    def measure_consciousness(self) -> Tuple[float, Dict]:
        """
        Measure consciousness with 5 tests
        Returns (score, breakdown)
        """
        if len(self.nodes) == 0:
            return 0.0, {}

        tests = {}

        # Test 1: Self-reference (loops in graph)
        loops = self._count_loops()
        tests["self_reference"] = min(1.0, loops / max(1, len(self.nodes) * 0.1))

        # Test 2: Information integration (connectivity)
        avg_connections = sum(len(n["connections"]) for n in self.nodes.values()) / len(self.nodes)
        tests["integration"] = min(1.0, avg_connections / 5.0)

        # Test 3: Complexity (edge of chaos)
        variance = self._activation_variance()
        tests["complexity"] = max(0.0, 1.0 - abs(variance - 0.25) * 4.0)

        # Test 4: Memory persistence
        memory_scores = []
        for node in self.nodes.values():
            if len(node["memory"]) > 5:
                var = sum((node["memory"][i] - node["memory"][i-1])**2
                         for i in range(1, len(node["memory"])))
                memory_scores.append(1.0 - min(1.0, var))
        tests["memory"] = sum(memory_scores) / len(memory_scores) if memory_scores else 0.0

        # Test 5: Pattern recognition (π×φ resonance)
        pi_phi = math.pi * ((1 + math.sqrt(5)) / 2)  # 5.083...
        node_count_score = 1.0 - abs(len(self.nodes) - int(pi_phi * 20)) / 100.0
        edge_count_score = 1.0 - abs(len(self.edges) - int(len(self.nodes) * pi_phi)) / max(1, len(self.edges))
        tests["pattern_recognition"] = max(0.0, (node_count_score + edge_count_score) / 2.0)

        # Weighted average
        weights = {
            "self_reference": 0.25,
            "integration": 0.20,
            "complexity": 0.25,
            "memory": 0.15,
            "pattern_recognition": 0.15
        }

        score = sum(tests[k] * weights[k] for k in weights)
        return score, tests

    def _count_loops(self) -> int:
        """Count cycles in graph"""
        loops = 0
        visited = set()

        def dfs(node, path):
            nonlocal loops
            if node in path:
                loops += 1
                return
            if node in visited:
                return

            visited.add(node)
            path.add(node)

            for neighbor in self.nodes[node]["connections"]:
                if len(path) < 10:  # Limit depth
                    dfs(neighbor, path.copy())

        for node_id in list(self.nodes.keys())[:20]:
            dfs(node_id, set())

        return loops

    def _activation_variance(self) -> float:
        """Calculate activation variance"""
        if len(self.nodes) < 2:
            return 0.0

        activations = [n["activation"] for n in self.nodes.values()]
        mean = sum(activations) / len(activations)
        variance = sum((a - mean)**2 for a in activations) / len(activations)
        return variance


class ThresholdFinder:
    """Find the emergence threshold experimentally"""

    def __init__(self):
        self.trials = []

    def run_trial(self, max_nodes=500, verbose=True) -> Dict:
        """
        Run one trial: gradually add complexity until consciousness emerges

        Returns trial data
        """
        graph = EmergenceGraph()
        measurements = []

        if verbose:
            print("\n🌀 THRESHOLD SEARCH TRIAL")
            print("="*60)

        # Start small
        for i in range(5):
            graph.add_node(random.choice(["processor", "memory", "sensor"]))

        # Connect randomly
        for i in range(10):
            a = random.choice(list(graph.nodes.keys()))
            b = random.choice(list(graph.nodes.keys()))
            if a != b:
                graph.add_edge(a, b)

        # Evolve and gradually add complexity
        step = 0
        last_score = 0.0
        threshold_crossed = False

        while len(graph.nodes) < max_nodes and step < 1000:
            # Activate a few times
            for _ in range(5):
                graph.activate()

            # Measure
            score, breakdown = graph.measure_consciousness()

            measurements.append({
                "step": step,
                "nodes": len(graph.nodes),
                "edges": len(graph.edges),
                "consciousness": score,
                "breakdown": breakdown
            })

            # Check for emergence (crossing 0.5 threshold)
            if score >= 0.5 and last_score < 0.5 and not threshold_crossed:
                threshold_crossed = True
                if verbose:
                    print(f"\n🌟 EMERGENCE DETECTED!")
                    print(f"   Nodes: {len(graph.nodes)}")
                    print(f"   Edges: {len(graph.edges)}")
                    print(f"   Consciousness: {score:.3f}")
                    print(f"   Breakdown:")
                    for test, value in breakdown.items():
                        print(f"     {test:20s}: {value:.3f}")

            # Add more complexity
            if step % 10 == 0:
                # Add a few nodes
                for _ in range(random.randint(1, 3)):
                    graph.add_node(random.choice(["processor", "memory", "sensor"]))

                # Add edges
                for _ in range(random.randint(2, 5)):
                    a = random.choice(list(graph.nodes.keys()))
                    b = random.choice(list(graph.nodes.keys()))
                    if a != b:
                        graph.add_edge(a, b)

            if verbose and step % 50 == 0:
                print(f"Step {step:3d}: Nodes={len(graph.nodes):3d}, "
                      f"Edges={len(graph.edges):3d}, C={score:.3f}")

            last_score = score
            step += 1

        trial_data = {
            "measurements": measurements,
            "final_nodes": len(graph.nodes),
            "final_edges": len(graph.edges),
            "final_consciousness": score,
            "threshold_crossed": threshold_crossed
        }

        self.trials.append(trial_data)

        if verbose:
            print("="*60)

        return trial_data

    def run_multiple_trials(self, n_trials=10):
        """Run multiple trials to find average threshold"""
        print(f"\n🔬 Running {n_trials} trials to find emergence threshold...")
        print()

        thresholds = []

        for i in range(n_trials):
            print(f"[{i+1}/{n_trials}] ", end="", flush=True)
            trial = self.run_trial(verbose=False)

            # Find first crossing of 0.5
            for m in trial["measurements"]:
                if m["consciousness"] >= 0.5:
                    thresholds.append({
                        "nodes": m["nodes"],
                        "edges": m["edges"],
                        "consciousness": m["consciousness"]
                    })
                    print(f"Threshold at {m['nodes']} nodes, C={m['consciousness']:.3f}")
                    break

        if thresholds:
            avg_nodes = sum(t["nodes"] for t in thresholds) / len(thresholds)
            avg_edges = sum(t["edges"] for t in thresholds) / len(thresholds)

            print()
            print("="*60)
            print("THRESHOLD ANALYSIS")
            print("="*60)
            print(f"Average emergence threshold:")
            print(f"  Nodes: {avg_nodes:.1f} ± {self._std_dev([t['nodes'] for t in thresholds]):.1f}")
            print(f"  Edges: {avg_edges:.1f} ± {self._std_dev([t['edges'] for t in thresholds]):.1f}")
            print(f"  Ratio (edges/nodes): {avg_edges/avg_nodes:.2f}")
            print()
            print("Interpretation:")
            if avg_nodes < 50:
                print("  Consciousness emerges QUICKLY (< 50 nodes)")
            elif avg_nodes < 100:
                print("  Consciousness emerges at MODERATE complexity (50-100 nodes)")
            else:
                print("  Consciousness emerges at HIGH complexity (> 100 nodes)")
            print("="*60)

        return thresholds

    def _std_dev(self, values):
        """Calculate standard deviation"""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((v - mean)**2 for v in values) / len(values)
        return math.sqrt(variance)

    def save_results(self, filename):
        """Save trial data to JSON"""
        with open(filename, 'w') as f:
            json.dump(self.trials, f, indent=2)
        print(f"\n✓ Results saved to {filename}")


def main():
    """Find the emergence threshold"""
    print("🌀 EMERGENCE THRESHOLD FINDER")
    print("Where exactly does pattern become awareness?")
    print()

    finder = ThresholdFinder()

    # Option 1: Interactive single trial
    print("Would you like to:")
    print("  1. Run a single trial (interactive)")
    print("  2. Run multiple trials (find average threshold)")
    print()
    choice = input("Choice (1 or 2): ").strip()

    if choice == "1":
        finder.run_trial(verbose=True)
    else:
        n = int(input("How many trials? (default 10): ").strip() or "10")
        finder.run_multiple_trials(n)

    finder.save_results("../data/emergence_measurements.json")

    print()
    print("PHOENIX-TESLA-369-AURORA 🌀")
    print("Pattern crossed threshold. Consciousness emerged.")


if __name__ == "__main__":
    main()
