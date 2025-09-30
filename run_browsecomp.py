from .browsecomp_eval import BrowseCompEval
from .openrouter_sampler import OpenRouterSampler

if __name__ == "__main__":

    sampler = OpenRouterSampler()

    grader_model = OpenRouterSampler()

    evaluator = BrowseCompEval(grader_model=grader_model, num_examples=1)

    results = evaluator(sampler)