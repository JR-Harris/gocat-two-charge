"""GOCAT-2charge fitness providers.

`contract.py` defines the provider contract and shared boundary
mechanics; provider modules (stub, later xtb / orca) bind
FITNESS_ID, PROVIDER_META, eval_batch_full, eval_batch to a
module-level `provider` instance.
"""