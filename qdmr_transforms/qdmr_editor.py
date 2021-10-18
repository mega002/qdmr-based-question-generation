from qdmr_transforms.qdmr_identifier import *


class QDMREditor(object):
    def __init__(self, qdmr_text):
        self.qdmr_steps = {}
        steps_list = parse_decomposition(qdmr_text)
        for i in range(len(steps_list)):
            self.qdmr_steps[i + 1] = steps_list[i]

    def get_step(self, step_num):
        assert step_num in self.qdmr_steps.keys()
        return self.qdmr_steps[step_num]

    def replace_step(self, step_num, step):
        self.qdmr_steps[step_num] = step

    def add_new_step(self, step_num, step):
        new_steps = {}
        new_steps[step_num] = step
        for i in self.qdmr_steps.keys():
            orig_step = self.qdmr_steps[i]
            if i < step_num:
                new_steps[i] = orig_step
            elif i >= step_num:
                new_steps[i + 1] = self.refs_one_up(orig_step, step_num, len(self.qdmr_steps))
        self.qdmr_steps = new_steps

    def refs_one_up(self, qdmr_text, start_idx, end_idx):
        target_refs_map = {}
        for i in range(start_idx, end_idx + 1):
            target_refs_map["#%s" % i] = "#%s" % (i + 1)
        new_qdmr_step = ""
        for tok in qdmr_text.split():
            if tok in target_refs_map.keys():
                new_qdmr_step += "%s " % target_refs_map[tok]
            else:
                new_qdmr_step += "%s " % tok
        return new_qdmr_step.strip()

    def remove_step(self, step_num):
        new_steps = {}
        for i in self.qdmr_steps.keys():
            orig_step = self.qdmr_steps[i]
            if i < step_num:
                new_steps[i] = orig_step
            elif i > step_num:
                new_steps[i - 1] = self.adjust_remove_refs(orig_step, step_num, len(self.qdmr_steps))
        self.qdmr_steps = new_steps
        cleaned_steps_list = self.remove_unused_steps()
        self.qdmr_steps = {}
        for i in range(len(cleaned_steps_list)):
            self.qdmr_steps[i + 1] = cleaned_steps_list[i]

    def adjust_remove_refs(self, qdmr_text, rem_idx, end_idx):
        target_refs_map = {}
        for i in range(rem_idx, end_idx + 1):
            rem_idx_ref = self.type_step_ref(qdmr_text, rem_idx)
            if i == rem_idx and rem_idx_ref is not None:
                target_refs_map["#%s" % i] = "#%s" % rem_idx_ref
            else:
                target_refs_map["#%s" % i] = "#%s" % (i - 1)
        new_qdmr_step = ""
        for tok in qdmr_text.split():
            if tok in target_refs_map.keys():
                new_qdmr_step += "%s " % target_refs_map[tok]
            else:
                new_qdmr_step += "%s " % tok
        return new_qdmr_step.strip()

    def type_step_ref(self, qdmr_text, step_idx):
        qdmr_text = self.get_qdmr_text()
        builder = QDMRProgramBuilder(qdmr_text)
        builder.build()
        step = builder.steps[step_idx - 1]
        if step.operator not in ["filter", "comparative", "superlative"]:
            return None
        ref = step.arguments[0] if step.operator in ["filter", "comparative"] else step.arguments[1]
        return int(ref.replace("#", ""))

    def step_type_phrases(self):
        qdmr_text = self.get_qdmr_text()
        builder = QDMRProgramBuilder(qdmr_text)
        builder.build()
        type_phrases = {}
        for i in range(len(builder.steps)):
            step = builder.steps[i]
            op = step.operator
            if op == "select":
                type_phrases[i + 1] = step.arguments[0]
            elif op == "project":
                ref_phrase, ref_idx = step.arguments
                ref_idx = int(ref_idx.replace("#", ""))
                ref_type = type_phrases[ref_idx]
                type_phrases[i + 1] = ref_phrase.replace("#REF", ref_type)
            elif op in ["filter", "aggregate", "superlative", "comparative", \
                        "sort", "discard", "intersection", "union"]:
                ref_idx = step.arguments[1] if op in ["aggregate", "superlative"] else step.arguments[0]
                ref_idx = int(ref_idx.replace("#", ""))
                type_phrases[i + 1] = type_phrases[ref_idx]
            else:
                type_phrases[i + 1] = None
        return type_phrases

    def get_step_type_phrase(self, step_num):
        type_phrases = self.step_type_phrases()
        return type_phrases[step_num]

    def remove_unused_steps(self):
        """remove all steps not used in the computation
        of the last step.

        E.g.:
            return US State; return total population of #1; return #1 that the Missouri river bisects
            --> return US State; return #1 that the Missouri river bisects
        """
        used_references = []
        qdmr_text = self.get_qdmr_text()
        steps = parse_decomposition(qdmr_text)
        n = len(steps)
        used_steps_refs = self.get_last_step_pointers(steps) + [n]
        new_steps = []
        for i in reversed(range(len(steps))):
            # always prune the last steps first because of refs!
            idx = i + 1
            if idx not in used_steps_refs:
                new_steps = self.refs_minus_one(idx, n, new_steps)
            else:
                new_steps = [steps[i]] + new_steps
        return new_steps

    def refs_minus_one(self, removed_idx, end_idx, steps):
        target_refs_map = {}
        new_ref_steps = []
        for i in range(removed_idx, end_idx + 1):
            target_refs_map["#%s" % i] = "#%s" % (i - 1)
        for step in steps:
            new_qdmr_step = ""
            for tok in step.split():
                if tok in target_refs_map.keys():
                    new_qdmr_step += "%s " % target_refs_map[tok]
                else:
                    new_qdmr_step += "%s " % tok
            new_ref_steps += [new_qdmr_step.strip()]
        return new_ref_steps

    def get_last_step_pointers(self, qdmr_steps_strs):
        all_references = []
        last_step = qdmr_steps_strs[-1]
        refs = self.extract_references(last_step)
        all_references += refs
        for idx in refs:
            all_references += self.get_last_step_pointers(qdmr_steps_strs[:idx])
        return list(set(all_references))

    def extract_references(self, step):
        """Extracts a list of references to previous steps"""
        # make sure decomposition does not contain a mere '# ' other than a reference.
        step = step.replace("# ", "hashtag ")
        references = []
        l = step.split(REF)
        for chunk in l[1:]:
            if len(chunk) > 1:
                ref = chunk.split()[0]
                ref = int(ref)
                references += [ref]
            if len(chunk) == 1:
                ref = int(chunk)
                references += [ref]
        return references

    def get_qdmr_text(self):
        qdmr = ""
        for i in range(len(self.qdmr_steps)):
            qdmr += "return %s; " % self.qdmr_steps[i + 1]
        return qdmr.strip()[:-1]
