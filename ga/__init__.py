"""The GA stage (work item: the discrete memetic GA).

Modules:
    params            frozen-typed GA parameters (the smoke set; the
                      operator rates freeze at item 6's pre-registration)
    space             the configuration space: the pair table, the row
                      code (individual identity == landscape row identity),
                      the NN table, the validity fence
    oracle_provider   the certified-landscape lookup adapter (oracle mode)
    ga                the engine: run_ga(fitness_callable, space, params,
                      seed) -> GaRunResult

The engine is a black-box search: it calls ONLY the injected fitness
callable and never reads a landscape file or a provider's internals.
"""