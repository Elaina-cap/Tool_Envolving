from .browsecomp_eval_agent import BrowseCompEval
from .openrouter_sampler_agent import OpenRouterSampler,OpenRouterGrader

if __name__ == "__main__":

    sampler = OpenRouterSampler()

    grader_model = OpenRouterGrader()

    evaluator = BrowseCompEval(grader_model=grader_model, num_examples=1)

    results = evaluator(sampler)