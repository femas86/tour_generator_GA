"""
ga/operators.py — Operatori genetici: crossover, mutation, selection.
Tutti gli operatori lavorano su copie degli individui senza modificare
i genitori (operazioni pure).
"""
from __future__ import annotations
import random
from core.models import Individual, PoI


# ---------------------------------------------------------------------------
# SELECTION
# ---------------------------------------------------------------------------

def tournament_select(
    population: list[Individual],
    k:          int = 3,
    use_pareto: bool = True,
) -> Individual:
    """
    Selezione torneo: estrae k individui casuali e restituisce il migliore.
    Con use_pareto=True preferisce rango Pareto basso + crowding alto.
    """
    contestants = random.sample(population, k)
    if use_pareto:
        return min(contestants, key=lambda x: (x.fitness.rank, -x.fitness.crowd))
    return max(contestants, key=lambda x: x.fitness.scalar)


# ---------------------------------------------------------------------------
# CROSSOVER
# ---------------------------------------------------------------------------

def order_crossover(parent1: Individual, parent2: Individual) -> tuple[Individual, Individual]:
    """
    Order Crossover (OX) adattato a tour a lunghezza variabile.
    Preserva l'ordine relativo dei PoI condivisi tra i genitori.
    """
    g1, g2 = parent1.genes, parent2.genes
    if len(g1) < 2 or len(g2) < 2:
        return parent1.clone(), parent2.clone()

    def ox(donor: list[PoI], receiver: list[PoI]) -> list[PoI]:
        if len(donor) < 2:
            return list(donor)
        cut1 = random.randint(0, len(donor) - 1)
        cut2 = random.randint(cut1, len(donor) - 1)
        segment  = donor[cut1:cut2+1]
        seg_ids  = {p.id for p in segment}
        # Riempi con i PoI del receiver non già nel segmento
        filling  = [p for p in receiver if p.id not in seg_ids]
        child    = filling[:cut1] + segment + filling[cut1:]
        return child

    return (
        Individual(genes=ox(g1, g2)),
        Individual(genes=ox(g2, g1)),
    )


def poi_aware_crossover(
    parent1:    Individual,
    parent2:    Individual,
    categories: list[str] | None = None,
) -> tuple[Individual, Individual]:
    """
    PoI-aware Crossover: scambia interi sottoinsiemi per categoria.
    Utile per preservare "nicchie tematiche" (es. tour musei vs tour gastronomico).
    """
    from core.models import PoICategory

    if categories is None:
        categories = [c.value for c in PoICategory]

    def by_category(genes: list[PoI]) -> dict[str, list[PoI]]:
        d: dict[str, list[PoI]] = {c: [] for c in categories}
        for p in genes:
            d[p.category.value].append(p)
        return d

    cat1 = by_category(parent1.genes)
    cat2 = by_category(parent2.genes)

    child1_genes, child2_genes = [], []

    for cat in categories:
        # Con probabilità 0.5 scambia le categorie tra i figli
        if random.random() < 0.5:
            child1_genes.extend(cat1[cat])
            child2_genes.extend(cat2[cat])
        else:
            child1_genes.extend(cat2[cat])
            child2_genes.extend(cat1[cat])

    # Rimuovi duplicati mantenendo l'ordine
    def deduplicate(genes: list[PoI]) -> list[PoI]:
        seen, result = set(), []
        for p in genes:
            if p.id not in seen:
                seen.add(p.id)
                result.append(p)
        return result

    return (
        Individual(genes=deduplicate(child1_genes)),
        Individual(genes=deduplicate(child2_genes)),
    )


# ---------------------------------------------------------------------------
# MUTATION
# ---------------------------------------------------------------------------

def mutate(
    individual: Individual,
    poi_pool:   list[PoI],
    mut_prob:   float = 0.15,
) -> Individual:
    """
    Seleziona casualmente uno degli operatori di mutazione.
    Applicato con probabilità mut_prob.
    """
    if random.random() > mut_prob or not individual.genes:
        return individual

    operator = random.choice([
        swap_mutation,
        insert_mutation,
        reverse_segment_mutation,
        lambda ind, pool: add_remove_mutation(ind, pool),
    ])
    result = operator(individual.clone(), poi_pool)
    result.invalidate_cache()
    return result


def swap_mutation(individual: Individual, _pool: list[PoI]) -> Individual:
    """Scambia due PoI scelti casualmente nella sequenza."""
    g = individual.genes
    if len(g) < 2:
        return individual
    i, j = random.sample(range(len(g)), 2)
    g[i], g[j] = g[j], g[i]
    return individual


def insert_mutation(individual: Individual, _pool: list[PoI]) -> Individual:
    """Rimuove un PoI da una posizione e lo inserisce altrove."""
    g = individual.genes
    if len(g) < 2:
        return individual
    i = random.randrange(len(g))
    poi = g.pop(i)
    j = random.randint(0, len(g))
    g.insert(j, poi)
    return individual


def reverse_segment_mutation(individual: Individual, _pool: list[PoI]) -> Individual:
    """Inverte un sottosegmento casuale del tour (elimina incroci geografici)."""
    g = individual.genes
    if len(g) < 3:
        return individual
    i = random.randint(0, len(g) - 2)
    j = random.randint(i + 1, len(g) - 1)
    g[i:j+1] = reversed(g[i:j+1])
    return individual


def add_remove_mutation(individual: Individual, pool: list[PoI]) -> Individual:
    """
    Con prob 0.5: aggiunge un PoI casuale non ancora nel tour.
    Con prob 0.5: rimuove il PoI con il peggior score/durata.
    Il wildcard gene segue la stessa logica di sostituzione.
    """
    g        = individual.genes
    visited  = {p.id for p in g}

    if random.random() < 0.5:
        candidates = [p for p in pool if p.id not in visited]
        if candidates:
            new_poi = random.choice(candidates)
            pos     = random.randint(0, len(g))
            g.insert(pos, new_poi)
    else:
        if g:
            worst = min(g, key=lambda p: p.score / (p.visit_duration + 1e-9))
            g.remove(worst)

    return individual