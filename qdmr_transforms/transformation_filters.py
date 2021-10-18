from qdmr_transforms.qdmr_distribution import qdmr_program_encoding, qdmr_dataset_distribution
from qdmr_transforms.qdmr_example import get_transform_base_info
from qdmr_transforms.utils import extract_ref_idx

TRANSFORM_FILTERS = ["data_programs", "data_operators", "time_diff_sum", "self_diff", "single_noun_phrase"]

class TransformFilter:
    """
    Filters QDMR transforms based on various features:
    distribution QDMR operator, program in the original dataset, etc.
    """

    def __init__(self,\
                 filters,
                 qdmr_data=None, \
                 operator_dist_threshold=None\
                 ):
        assert set(filters).issubset(set(TRANSFORM_FILTERS))
        if "data_programs" in filters or "data_operators" in filters:
            assert qdmr_data
        if "operator_dist_threshold" in filters:
            assert operator_dist_threshold >= 0 and operator_dist_threshold <= 100
        if qdmr_data:
            qdmr_dist = qdmr_dataset_distribution(qdmr_data)
            self.data_programs = qdmr_dist["programs"]
            self.data_operators = qdmr_dist["operators"]
            self.op_distribution = qdmr_dist["distribution"]
            self.op_threshold = operator_dist_threshold
        self.filter_programs = "data_programs" in filters
        self.filter_operators = "data_operators" in filters
        self.filter_time_diff_sum = "time_diff_sum" in filters
        self.filter_self_diff = "self_diff" in filters
        self.filter_single_noun_phrase = "single_noun_phrase" in filters

    def filter_out(self, qdmr_example):
        results = self.transformed_qdmr_filters(qdmr_example)
        for filter_type in results:
            if results[filter_type] == False:
                return True
        return False

    def transformed_qdmr_filters(self, qdmr_example):
        """
        Return dict of filters of whether the transformed QDMR
        should be filtered (True) or not (False).

        E.g., d["data_programs"]=True, d["time_diff_sum"]=False
            means the transform example should be removed due to the "time_diff_sum" filter
        """
        results = {}
        include_transforms = ["append_boolean_step"]
        results["program"] = qdmr_program_encoding(qdmr_example.steps)
        if self.filter_programs:
            results["data_programs"] = self.program_filter(qdmr_example,
                                                           include_transforms=include_transforms)
        if self.filter_operators:
            results["data_operators"] = self.operator_filter(qdmr_example,
                                                             threshold_pct=self.op_threshold,
                                                             include_transforms=include_transforms)
        if self.filter_time_diff_sum:
            results["time_diff_sum"] = self.time_diff_sum_filter(qdmr_example)
        if self.filter_self_diff:
            results["self_diff"] = self.self_diff_filter(qdmr_example)
        if self.filter_single_noun_phrase:
            results["single_noun_phrase"] = self.single_noun_phrase_filter(qdmr_example)
        return results

    def program_filter(self, qdmr_example, include_transforms=None):
        """
        Returns whether the transformed QDMR should be filtered
        because its program structure (encoding) is not part of
        the base QDMR dataset distribution

        Parameters
        ----------
        qdmr_example : QDMRExample
            The transformed QDMR example
        include_transforms: list
            Set of prefixes of transformations we want to include regardless of the filter

        Returns
        -------
        bool
            Flag whether qdmr should be kept (True) or filtered (False)
        """
        if include_transforms:
            transform_full_id = qdmr_example.transform
            for include_prefix in include_transforms:
                if transform_full_id.startswith(include_prefix):
                    return True
        transformed_program = qdmr_program_encoding(qdmr_example.steps)
        prog_encoding = ' '.join(transformed_program)
        return prog_encoding in self.data_programs

    def operator_filter(self, qdmr_example, threshold_pct=None, include_transforms=None):
        """
        Returns whether the transformed QDMR should be filtered
        because it contains operators not present in the base QDMR dataset
        distribution based on the threshold percentage.

        Parameters
        ----------
        qdmr_example : QDMRExample
            The transformed QDMR example
        threshold : int
            Threshold percentage of program prevalence in data.
            If threshold is empty, filter based on operator occurrence in base dataset.
        include_transforms: list
            Set of prefixes of transformations we want to include regardless of the filter

        Returns
        -------
        bool
            Flag whether qdmr should be kept (True) or filtered (False)
        """
        if include_transforms:
            transform_full_id = qdmr_example.transform
            for include_prefix in include_transforms:
                if transform_full_id.startswith(include_prefix):
                    return True
        transformed_program = qdmr_program_encoding(qdmr_example.steps)
        valid_operators = True
        for op in transformed_program:
            if op not in self.data_operators:
                return False
        if threshold_pct:
            for op in transformed_program:
                op_data_pct = self.op_distribution[op] * 100
                if op_data_pct < threshold_pct:
                    return False
        return True

    def time_diff_sum_filter(self, qdmr_example):
        """
        Returns whether the transformed QDMR should be filtered
        because it transformed a difference of dates (year/month/day) to sum
        E.g.:
            "year Jon was born; year Jane was born; difference of #1, #2"
            --> "year Jon was born; year Jane was born; sum of #1, #2"

        Parameters
        ----------
        qdmr_example : QDMRExample
            The transformed QDMR example

        Returns
        -------
        bool
            Flag whether qdmr should be kept (True) or filtered (False)
        """
        time_triggers = ["when", "date", "year", "years", "month", "months", "day", "days", "hour", "hours"]
        diff_to_sum = "difference-sum"
        assert qdmr_example.transform
        _, transform_info = get_transform_base_info(qdmr_example.transform)
        if diff_to_sum in transform_info:
            for trigger in time_triggers:
                if trigger in qdmr_example.qdmr:
                    return False
        return True

    def self_diff_filter(self, qdmr_example):
        """
        Returns whether the transformed QDMR should be filtered
        because its transform resulted in a self difference
        E.g.:
            "difference of #1, #2" --> "difference of #1, #1"

        Parameters
        ----------
        qdmr_example : QDMRExample
            The transformed QDMR example

        Returns
        -------
        bool
            Flag whether qdmr should be kept (True) or filtered (False)
        """
        for step in qdmr_example.steps:
            if step.operator == "arithmetic":
                arith = step.arguments[0]
                args = step.arguments[1:]
                if arith == "difference" and len(args) > 1:
                    if args[0] == args[1]:
                        return False
                    idx = extract_ref_idx(args[0]) - 1
                    other_idx = extract_ref_idx(args[1]) - 1
                    steps_text_list = qdmr_example.qdmr_steps_text()
                    return steps_text_list[idx] != steps_text_list[other_idx]
        return True

    def single_noun_phrase_filter(self, qdmr_example):
        """
        Returns whether the transformed QDMR should be filtered
        because its transform resulted in a single step QDMR that is a single noun phrase
        E.g.:
            "teams", "ethnic groups", etc.

        Parameters
        ----------
        qdmr_example : QDMRExample
            The transformed QDMR example

        Returns
        -------
        bool
            Flag whether qdmr should be kept (True) or filtered (False)
        """
        if len(qdmr_example.steps) == 1:
            step = qdmr_example.steps[0]
            phrase = step.arguments[0]
            if len(phrase.split()) < 3:
                return False
        return True

    def single_noun_phrase_filter(self, qdmr_example):
        """
        Returns whether the transformed QDMR should be filtered
        because its transform resulted in a single step QDMR that is a single noun phrase
        E.g.:
            "teams", "ethnic groups", etc.

        Parameters
        ----------
        qdmr_example : QDMRExample
            The transformed QDMR example

        Returns
        -------
        bool
            Flag whether qdmr should be kept (True) or filtered (False)
        """
        if len(qdmr_example.steps) == 1:
            step = qdmr_example.steps[0]
            phrase = step.arguments[0]
            if len(phrase.split()) < 3:
                return False
        return True
